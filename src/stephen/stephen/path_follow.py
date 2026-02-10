#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from time import perf_counter
import numpy as np
import math
from math import pi
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import signal
import sys
from collections import defaultdict
from types import SimpleNamespace
from simple_pid import PID
import threading


# TODO:
# - Range limit could be adaptive based on amount of path taken up or geometric distance
# - More effiction wall extension (try cupy or only 720 indexes)
# - Fix gap definition
# - Handle potential bottle neck disparity overlap
# - Tune choose path
# - Smoothing function sometimes gets stuck at offset
# - Create angle planning to account for little information know about front
# - Create custom maps for sim
# - Create tracking line for sim to compare paths

publish_marker = True
publish_virtual_scan = True
publish_v2 = False
file_output = False
fast_print = False

cycle_time = 1e-5

flat_speed = 1.0
smoothing_exp = 2.0
disparity_threshold = 0.5

file_info = SimpleNamespace(**{
    'path_info':{},
    'virtual_scan_info':{},
    'mean_times': defaultdict(lambda:[0.0, 0]),
    'other':{}
})

class PathFollow(Node):
    
    def __init__(self):
        super().__init__('gap_follow')
        signal.signal(signal.SIGINT, handler)
        if publish_marker:
            self.laser_scan_sub = self.create_subscription(LaserScan, '/scan', self.save_scan, 10)
        else:
            self.laser_scan_sub = self.create_subscription(LaserScan, '/scan', self.adjust, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        if publish_virtual_scan:
            self.scan_pub = self.create_publisher(LaserScan, '/virtual_scan', 10)
        if publish_v2:
            self.v2_pub = self.create_publisher(Float32MultiArray, '/v2_ranges', 10)
        if publish_marker:
            self.marker_pub = self.create_publisher(Marker, '/viz/projected_line', 10)
            self.timer = self.create_timer(0.025, self.adjust_wrapper)
        self.overhead_start = 0.0
        self.cycle_start = perf_counter()
        self.prev_speed = 0.0
        self.prev_steering_angle = 0.0


        Vehicle.setup(
            wheelbase=0.33,
            radius=0.5,
            max_steering_angle = 0.4
        )

        self.planner = Planner(
            disparity_threshold=0.5,
            extension=0.3,
            track_direction='ccw'
            )
        
        self.steering = Steering(
            # | arc | line |
            method='arc'
        )
        
        global smoothing_exp
        # Smoothing functions domain and range [0, 1]
        f0 = lambda x : x
        f1 = lambda x : (x**smoothing_exp)
        A = .2
        f2 = lambda x : (A * math.tan(math.atan(1/A) * x))
        a = 1.7
        b = 14
        c = -9
        f3 = lambda x : (1/(1+pow(a, -(b*x+c))) - 1/(1 + pow(a, -c)))
            
        # Smoothing
        self.smooth = Smooth(
            func=f0,
            use_filter=False,
            use_pid=False,
            use_slew=False,
            tau=.05,
            slew_rate=5.0,
            pid=PID(
                Kp=0.8,
                Ki=0.0,
                Kd=0.0,
                setpoint=0.0,
                sample_time=0.025,
                proportional_on_measurement=False,
                differential_on_measurement=True
            ),
        )

        # Speed
        self.speed = Speed(
            # | flat | fast |
            method='flat',
            flat_speed=1.,
            max_speed=100.0,
            a_slide=14.0,
            a_tip=10000.0,
            )

        # File output
        if file_output:
            file_info.other['turning_radius'] = Vehicle.turning_radius
            file_info.virtual_scan_info['max_range'] = 0.0
            file_info.virtual_scan_info['min_range'] = math.inf
            file_info.path_info['max_range'] = 0.0
            file_info.path_info['min_range'] = math.inf
            file_info.other['max_pipeline_time'] = 0.0

    def save_scan(self, scan: LaserScan):
        self.scan = scan

    def adjust_wrapper(self):
        if not self.scan is None:
            self.adjust(self.scan)

    def adjust(self, scan: LaserScan):
        global cycle_time
        start_time = perf_counter()
        cycle_time = (start_time - self.cycle_start)
        self.cycle_start = start_time
        adjust_start = start_time
        overhead_time = (start_time - self.overhead_start) * 1_000

        if not Scan.initialized:
            Scan.setup(scan)
            print('Running...')
        
        if fast_print:
            print(f'\n{"="*24}')

        ranges = np.asarray(scan.ranges)

        if file_output:
            file_info.path_info['max_range'] = max(np.max(ranges), file_info.path_info['max_range'])
            file_info.path_info['min_range'] = min(np.min(ranges), file_info.path_info['min_range'])

        # =================== Pipeline ========================
        
        pipeline_start = perf_counter()

        # self.planner.apply_range_limit(ranges, 10.0)

        virtual = self.planner.get_virtual(ranges)
        
        pos_disps, neg_disps = self.planner.disparities(ranges)

        pos_disps, neg_disps = self.planner.get_paths(ranges, virtual, pos_disps, neg_disps)

        # choose should be based on pre-resolve ranges
        goal_idx, sign = self.planner.choose(ranges, pos_disps, neg_disps)
        goal_depth = ranges[goal_idx]

        if goal_idx is None:
            print('NO GOAL POINT')
            steering_angle = self.prev_steering_angle
            speed = self.prev_speed
        else:
            steering_angle = self.steering.get(goal_idx, sign, goal_depth)

            steering_angle = self.smooth.get(steering_angle, self.prev_speed)

            speed = self.speed.get(steering_angle,  goal_depth + Vehicle.radius - Vehicle.turning_radius)
            self.prev_speed = speed
        
        pipeline_time = (perf_counter() - pipeline_start) * 1_000
        # =====================================================

        self.prev_speed = speed
        self.prev_steering_angle = steering_angle

        # Publish messages
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)
        if publish_virtual_scan:
            self.scan_pub.publish(scan)
        if publish_v2:
            msg = Float32MultiArray()
            msg.data = virtual.tolist()
            self.v2_pub.publish(msg)
        if (publish_marker):
            # Marker
            L = 20.0
            front_offset = 0.4
            ang = steering_angle
            p0 = Point(x=front_offset, y=0.0, z=0.0)
            p1 = Point(x=L*math.cos(ang) + front_offset, y=L*math.sin(ang), z=0.0)
            m = Marker()
            m.pose.orientation.w = 1.0
            m.header.frame_id = '/ego_racecar/base_link'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'project_line'
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.05
            m.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
            m.points = [p0, p1]
            self.marker_pub.publish(m)

        if fast_print:
            # Basic info
            print(f'\n{speed=:.2}')
            print(f'{steering_angle=:.2}')
            # Timing info
        adjust_time = (perf_counter() - adjust_start)*1_000
        if fast_print:
            print(f'{adjust_time=:.3f}')

        # File output
        if file_output:
            file_info.mean_times['pipeline_mean'][0] += pipeline_time
            file_info.mean_times['pipeline_mean'][1] += 1
            file_info.other['max_pipeline_time'] = max(pipeline_time, file_info.other['max_pipeline_time'])
            file_info.mean_times['adjust_mean'][0] += adjust_time
            file_info.mean_times['adjust_mean'][1] += 1
            if self.overhead_start > 0:
                file_info.mean_times['overhead_mean'][0] += overhead_time
                file_info.mean_times['overhead_mean'][1] += 1
            file_info.virtual_scan_info['max_range'] = max(np.max(ranges), file_info.virtual_scan_info['max_range'])
            file_info.virtual_scan_info['min_range'] = min(np.min(ranges), file_info.virtual_scan_info['min_range'])

        self.overhead_start = perf_counter()

class Vehicle:

    @classmethod
    def setup(cls, wheelbase, radius, max_steering_angle):
        cls.wheelbase = wheelbase
        cls.radius = radius
        cls.max_steering_angle = max_steering_angle
        cls.turning_radius = cls.wheelbase/math.tan(cls.max_steering_angle)

class Scan:

    initialized = False

    @classmethod
    def valid_scan(cls, scan):
        is_valid = True
        # Check size
        if cls.size != len(scan.ranges):
            print(f'len(scan.ranges = {len(scan.ranges)}, expected {cls.size})')
            is_valid = False
        for key in cls.__dict__:
            # Check for other differences
            if getattr(scan, key, None) and cls.__dict__[key] != getattr(scan, key):
                print(f'Invalid: scan.{key} = {getattr(scan, key)}, expected {cls.__dict__[key]}')
                is_valid = False
        return is_valid
    
    @classmethod
    def setup(cls, scan):
        cls.initialized = True
        cls.size = len(scan.ranges)

        # Scan message attributes
        cls.angle_min = scan.angle_min
        cls.angle_max = scan.angle_max
        assert scan.angle_max == abs(scan.angle_min)
        cls.angle_increment = scan.angle_increment
        cls.time_increment = scan.time_increment
        cls.scan_time = scan.scan_time
        cls.range_min = scan.range_min
        cls.range_max = scan.range_max

        cls.increment = cls.angle_increment
        cls.index_to_angle_array = np.arange(cls.size)*cls.increment + cls.angle_min
        cls.fov = cls.angle_max - cls.angle_min
        if file_output:
            for key, value in cls.__dict__.items():
                file_info.path_info[key] = value
    
    @classmethod
    def angle_to_index(cls, angle):
        abs_angle = angle + cls.angle_min
        total_angle = cls.angle_max - cls.angle_min
        return round(cls.size * (abs_angle / total_angle))
    
    @classmethod
    def index_to_angle(cls, index):
        return cls.index_to_angle_array[index]
    
    @classmethod
    def index_to_degrees(cls, index):
        return round(math.degrees(cls.index_to_angle(index)), 4)
    
    @classmethod
    def fov_slice(cls, ranges, fov):
        fov = math.radians(fov)
        fov_offset = cls.angle_to_index(-fov / 2)
        return ranges[fov_offset: len(ranges) - fov_offset]
    
    @classmethod
    def revert_fov_indexes(cls, fov, indexes):
        fov = math.radians(fov)
        reverted = []
        for i in indexes:
            reverted.append(i + round((cls.fov - fov) / (2 * cls.increment)))
        return reverted
    
    @classmethod
    def span_to_angle(cls, span, depth):
        return 2 * math.asin(span / (2 * depth))
    
class Planner:

    def __init__(self, disparity_threshold, extension, track_direction):
        self.disparity_threshold = disparity_threshold # threshold defining disparity
        self.extension = extension # wall extension distance
        self.track_direction = track_direction # ccw or cw

    def apply_range_limit(self, ranges, limit):
        safe_gap_value = 100.0
        ranges[ranges > limit] = safe_gap_value

    def get_virtual(self, ranges):
        extension = self.extension
        n = len(ranges)
        range_matrix = np.full((n, n), np.inf, dtype=np.float32)
        ratio = extension / (2 * ranges)
        ratio = np.clip(ratio, -1.0, 1.0)
        index_extensions = abs(np.floor(2*np.arcsin(ratio)/Scan.angle_increment).astype(np.int32))
        rows = np.arange(n)[:, None]
        cols = np.arange(n)[None, :]
        mask = np.abs(cols-rows) <= index_extensions[:, None]
        range_matrix = np.where(mask, ranges[:, None], np.inf).astype(np.float32)
        col_mins = range_matrix.min(axis=0)
        return col_mins
    
    def resolve_disparities(self, ranges, virtual, pos_disps, neg_disps):
        for disp in pos_disps:
            # Walk gap until intersection
            disp_range = ranges[disp]
            points = np.flatnonzero((ranges[disp+1:] <= disp_range) | (virtual[disp+1:] > disp_range))
            if points.size:
                left_intersect = points[0] + disp + 1
            else:
                continue
            # Drop to intersection range
            if virtual[left_intersect] > disp_range:
                left_intersect -= 1
                new_ext = virtual[left_intersect]
            else:
                new_ext = disp_range
            # Backtrack until second intersection
            points = np.flatnonzero((ranges[:disp] <= new_ext) | (virtual[:disp] > new_ext))
            right_intersect = points[-1] if points.size else disp
            if virtual[right_intersect] > new_ext:
                right_intersect += 1
            ranges_section = ranges[right_intersect: left_intersect + 1]
            np.minimum(ranges_section, new_ext, out=ranges_section)

        for disp in neg_disps:
            # Walk gap until intersection
            disp_range = ranges[disp]
            points = np.flatnonzero((ranges[:disp] <= disp_range) | (virtual[:disp] > disp_range))
            if points.size:
                right_intersect = points[-1]
            else:
                continue
            # Drop to intersection range
            if virtual[right_intersect] > disp_range:
                right_intersect += 1
                new_ext = virtual[right_intersect]
            else:
                new_ext = disp_range
            # Backtrack until second intersection
            points = np.flatnonzero((ranges[disp+1:] <= new_ext) | (virtual[disp+1:] > new_ext))
            left_intersect = points[0] + disp + 1 if points.size else disp
            if virtual[left_intersect] > new_ext:
                left_intersect -= 1
            ranges_section = ranges[right_intersect: left_intersect + 1]
            np.minimum(ranges_section, new_ext, out=ranges_section)

    def disparities(self, ranges):
        global disparity_threshold
        diffs = np.diff(ranges)
        pos_disp = np.flatnonzero(diffs >= disparity_threshold)
        neg_disp = np.flatnonzero(diffs <= -disparity_threshold)
        return (pos_disp, neg_disp + 1)
    
    def get_paths(self, ranges, virtual, pos_disps, neg_disps):
        # Satisfiability: 1. disp radius, 2. arc path, 3. minimum arc radius
        # Filter by disp radius
        valid_pos_disps = []
        for disp in pos_disps:
            start = disp + 1
            end = disp + 2 * Scan.angle_to_index(Scan.span_to_angle(self.extension, ranges[disp]))
            section = slice(start, end + 1)
            min_radius = np.min(np.sqrt(
                ranges[disp]**2 + ranges[section]**2 - 2 * ranges[disp] * ranges[section]))
            if min_radius > 2 * self.extension:
                valid_pos_disps.append(disp)

        valid_neg_disps = []
        for disp in neg_disps:
            start = disp - 1
            end = disp - 2 * Scan.angle_to_index(Scan.span_to_angle(self.extension, ranges[disp]))
            section = slice(end, start + 1)
            min_radius = np.min(np.sqrt(
                ranges[disp]**2 + ranges[section]**2 - 2 * ranges[disp] * ranges[section]))
            if min_radius > 2 * self.extension:
                valid_neg_disps.append(disp)
            
        self.resolve_disparities(ranges, virtual, pos_disps, neg_disps)

        pos_disps, neg_disps = self.disparities(ranges)

        return pos_disps, neg_disps

    def choose(self, ranges, pos_disps, neg_disps):
        # HARDCODED
        steps = (0, 90, 180, 360, 539)
        N = Scan.size
        right_start = int(N/2 - 1) if N % 2 == 0 else int(N/2)
        left_start = int(N/2)
        right_sections = [(right_start - steps[i+1], right_start - steps[i]) for i in (range(len(steps) - 1))]
        left_sections = [(left_start + steps[i], left_start + steps[i+1]) for i in (range(len(steps) - 1))]
        # Create section list
        if self.track_direction == 'ccw':
            sections = [x for pair in zip(left_sections, right_sections) for x in pair]
        elif self.track_direction == 'cw':
            sections = [x for pair in zip(right_sections, left_sections) for x in pair]
        else:
            raise ValueError('Invalid track_direction')
        # Prevents sudden choice swapping due to dead on heading, also mid index
        sections.insert(1, (530, 550))

        for section in sections:
            goal_idx, sign = self.check_gap(ranges, pos_disps, neg_disps, section)
            if not goal_idx is None:
                    return goal_idx, sign
        print('No path...')
        return None, None

    def check_gap(self, ranges, pos_disps, neg_disps, section):
        lo, hi = section
        disps = np.concatenate(neg_disps, pos_disps)
        mask = (disps >= lo) & (disps <= hi)
        if not np.any(mask):
            return None, None
        local_map = np.where(mask)[0]
        section_ranges = ranges[local_map]
        local_goal_idx = np.nanargmax(section_ranges)
        goal_idx = local_map[local_goal_idx]
        sign = 1 if goal_idx in pos_disps else 0
        return goal_idx, sign

class Steering:

    def __init__(self, method):
        if method == 'arc':
            self.get = self.get_arc_angle
        elif method == 'line':
            self.get = self.get_line_angle
        else:
            raise ValueError('Invalid steering method')
        
    def get_arc_angle(self, disp, sign, depth):
        theta = Scan.index_to_angle(disp)
        y = math.cos(theta)
        gamma = 2 * y / depth**2
        delta = np.arctan(Vehicle.wheelbase * gamma)
        self.angle = np.clip(delta, -Vehicle.max_steering_angle, Vehicle.max_steering_angle)

    def get_line_angle(self, disp, sign, depth):
        raise NotImplementedError()

class Smooth:

    def __init__(self, func, use_filter, use_pid, use_slew, tau, slew_rate, pid):
        self.func = func
        self.use_filter = use_filter
        self.use_pid = use_pid
        self.use_slew = use_slew
        self.limit = Vehicle.max_steering_angle
        self.pid = pid
        self.pid.output_limits = (-self.limit, self.limit)
        self.start_time = 0.0
        # Max time interval
        self.max_dt = .1
        # Filter timescale (.025, 0.25)
        self.tau = tau
        # Max rate of change in steering
        self.slew_rate = slew_rate
        # Steering state
        self.theta = 0.0
        self.prev_filtered = 0.0

    def get(self, target, speed):
        dt = min(perf_counter() - self.start_time, self.max_dt)
        self.start_time = perf_counter()

        # Low pass filter
        if self.use_filter:
            alpha = self.tau/(self.tau + dt)
            target = (alpha * self.prev_filtered) + ((1 - alpha) * target)
            self.prev_filtered = target

        # Saturation (min and max limits)
        target = np.clip(target, -self.limit, self.limit)
        
        # Apply func
        sign = -1 if target < 0 else 1
        target = sign * self.limit * self.func(abs(target)/self.limit)

        # PID
        if self.use_pid:
            target = self.pid(-target)
            if target is None:
                raise ValueError('Failed to get PID')
            
        # Saturation (min and max limits)
        before_clip = abs(target)
        target = np.clip(target, -self.limit, self.limit)
        if (before_clip != self.limit and abs(target) == self.limit):
            print('CLIPPED 2')

        # Slew rate (limit on rate of change)
        if self.use_slew:
            delta = target - self.theta
            max_step = self.slew_rate * dt
            if abs(delta) > max_step:
                target = self.theta + (max_step if delta > 0 else -max_step)

        # Saturation (min and max limits)
        before_clip = abs(target)
        target = np.clip(target, -self.limit, self.limit)
        if (before_clip != self.limit and abs(target) == self.limit):
            print('CLIPPED 3')

        self.theta = target
        return target
        
class Speed:

    def __init__(self, method, flat_speed, max_speed, a_slide, a_tip):
        self.flat_speed = flat_speed
        self.min_speed = math.sqrt(min(a_slide, a_tip) * Vehicle.wheelbase / math.tan(Vehicle.max_steering_angle))
        file_info.other['min_speed'] = self.min_speed
        self.max_speed = max_speed
        self.a_slide = a_slide
        self.a_tip = a_tip
        self.wheelbase = Vehicle.wheelbase
        self.max_turning_angle = Vehicle.max_steering_angle
        self.turning_radius = Vehicle.wheelbase / math.tan(Vehicle.max_steering_angle)
        self.last_speed = 0.0

        if method == 'flat':
            self.get = self.flat
        elif method == 'fast':
            self.get = self.fast
        else:
            raise ValueError('Invalid Speed method')

    def flat(self, steering_angle, gap_depth):
        global flat_speed
        if math.isnan(steering_angle) or not gap_depth:
            return 0.0
        return flat_speed

    def fast(self, steering_angle, gap_depth):
        if math.isnan(steering_angle) or not gap_depth:
            print('FAILED SPEED')
            return 0.0
        v_min = self.min_speed
        s = gap_depth
        if s < 0:
            print('Zero depth path...')
            return 0.0
        k = abs(math.tan(steering_angle)) / self.wheelbase
        k += 1e-6 # Prevent division by zero
        v_min_2 = v_min * v_min
        v_min_4 = v_min_2 * v_min_2
        s_2 = s * s
        k_2 = k * k
        p1 = 1 + 4*s_2*k_2
        p2 = self.a_slide**2 * p1 - k_2*v_min_4
        if p2 < 0:
            return 0.0
        p3 = 2*s*math.sqrt(self.a_slide**2 * p1 - k_2*v_min_4)
        diff = v_min_2 - p3
        if diff >= 0:
            v_tot = math.sqrt(diff/p1)
        else:
            v_tot = math.sqrt(-diff/p1)
        v_lat = math.sqrt(self.a_tip/k)
        speed = min(v_tot, v_lat, self.max_speed)
        self.last_speed = speed
        return speed
    
def input_thread():
    global flat_speed, smoothing_exp, disparity_threshold
    print('Command (x=stop, s=-speed, d=+speed, j=-exp, k=+exp, h-=threshold, l+=threshold)')
    while True:
        cmd = input()
        if cmd == 's':
            flat_speed -= .1
            print(f'speed = {flat_speed:.2}')
        elif cmd == 'd':
            flat_speed += .1
            print(f'speed = {flat_speed:.2}')
        elif cmd == 'S':
            flat_speed -= 1.0
            print(f'speed = {flat_speed:.2}')
        elif cmd == 'D':
            flat_speed += 1.0
            print(f'speed = {flat_speed:.2}')
        elif cmd == 'j':
            smoothing_exp -= .1
            print(f'smoothing exp = {smoothing_exp:.2}')
        elif cmd == 'k':
            smoothing_exp += .1
            print(f'smoothin exp = {smoothing_exp:.2}')
        elif cmd == 'h':
            disparity_threshold -= .1
            print(f'disparity threshold = {disparity_threshold:.2}')
        elif cmd == 'l':
            disparity_threshold += .1
            print(f'disparity threshold = {disparity_threshold:.2}')
        elif cmd == 'x':
            print('stopped')
            flat_speed = 0.0
                       
def handler(sig, frame):
    if file_output:
        for key, value in file_info.mean_times.items():
            mean_time = value[0] / value[1]
            file_info.mean_times[key] = mean_time
        with open('path_follow_info.txt', 'w', encoding='utf-8') as f:
            for key, subdict in file_info.__dict__.items():
                f.write(f'\n{key.upper()}\n')
                for subkey, value in subdict.items():
                    f.write(f'{subkey} : {value}\n')
    sys.exit(0)

def main(args=None):
    threading.Thread(target=input_thread, daemon=True).start()
    rclpy.init(args=args)
    node = PathFollow()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

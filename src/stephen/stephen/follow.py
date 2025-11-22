#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
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
import inspect

file_output = True
fast_print = False
publish_virtual_scan = True
is_simulation = True

file_info = SimpleNamespace(**{
    'scan_info':{},
    'virtual_scan_info':{},
    'mean_times': defaultdict(lambda:[0.0, 0]),
    'other':{}
})

class PathFollow(Node):
    
    def __init__(self):
        super().__init__('gap_follow')
        signal.signal(signal.SIGINT, handler)
        self.laser_scan_sub = self.create_subscription(LaserScan, '/scan', self.adjust, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        if publish_virtual_scan:
            self.scan_pub = self.create_publisher(LaserScan, '/virtual_scan', 10)
        if is_simulation:
            self.marker_pub = self.create_publisher(Marker, '/viz/projected_line', 10)
        self.overhead_start = 0.0
        self.is_set = False

        # ====================== Options ======================

        # Constants
        self.orientation = 'cw'
        self.car_radius = 0.2
        self.wheelbase = 0.3
        self.max_steering_angle = math.radians(45)
        self.min_turning_radius = self.wheelbase/math.tan(self.max_steering_angle)

        # Speed
        self.speed = Speed(
            method='flat',
            flat_speed=1.0,
            max_speed=16.0,
            a_slide=8.0,
            a_tip=1000.0,
            wheelbase=self.wheelbase,
            max_turning_angle=self.max_steering_angle
            )
        
        # Extension
        self.extension = Extension(
            extension=0.3,
            threshold=0.5,
            # cos | none
            coeff_func='cos'
            )
        
        # Path
        self.path = Path(
            # | disparity | threshold |
            gap_method='',
            # | array of steps |
            preference=[]
        )

        self.steering  = Steering(
            point_method='center',
            path_method='line'
        )

        # Smoothing
        self.smooth = Smooth(
            use_filter=True,
            use_pid=True,
            use_slew=True,
            limit=np.pi/2,
            tau=.05,
            slew_rate=3.0,
            pid=PID(
                Kp=-.8,
                Ki=0.0,
                Kd=0.0,
                setpoint=0.0,
                sample_time=0.025,
                proportional_on_measurement=False,
                differential_on_measurement=True
            ),
        )

        # =====================================================

    def adjust(self, scan: LaserScan):
        adjust_start = perf_counter()
        overhead_time = (perf_counter() - self.overhead_start) * 1_000

        if not self.is_set:
            self.scan = Scan(scan, self.orientation)
            print('Running...')
        
        if fast_print:
            print(f'\n{"="*24}')

        ranges = np.asarray(scan.ranges)

        if file_output:
            file_info.scan_info['max_range'] = max(np.max(ranges), file_info.scan_info['max_range'])
            file_info.scan_info['min_range'] = min(np.min(ranges), file_info.scan_info['min_range'])

        # ==================== Pipeline =======================

        pipeline_start = perf_counter()

        # Extension
        self.extension.apply(ranges)

        # Steering
        steering_angle, depth, width = self.steering.get(range)

        # Smoothing
        steering_angle = self.smooth.get(steering_angle)

        # Speed
        speed = self.speed.get(steering_angle, depth, width)
        
        pipeline_time = (perf_counter() - pipeline_start) * 1_000

        # =====================================================

        # Publish messages
        if publish_virtual_scan:
            self.scan_pub.publish(scan)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

        if (is_simulation):
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
            file_info.mean_times['adjust_mean'][0] += adjust_time
            file_info.mean_times['adjust_mean'][1] += 1
            if self.overhead_start > 0:
                file_info.mean_times['overhead_mean'][0] += overhead_time
                file_info.mean_times['overhead_mean'][1] += 1
            file_info.virtual_scan_info['max_range'] = max(np.max(ranges), file_info.virtual_scan_info['max_range'])
            file_info.virtual_scan_info['min_range'] = min(np.min(ranges), file_info.virtual_scan_info['min_range'])

        self.overhead_start = perf_counter()

class Scan:
    def __init__(self, scan: LaserScan, orientation):
        self.orientation = orientation
        # Basic scan properties
        Scan.size = len(scan.ranges)
        Scan.angle_min = scan.angle_min
        Scan.angle_max = scan.angle_max
        assert scan.angle_max == abs(scan.angle_min)
        Scan.angle_increment = scan.angle_increment
        Scan.time_increment = scan.time_increment
        Scan.scan_time = scan.scan_time
        Scan.range_min = scan.range_min
        Scan.range_max = scan.range_max
        Scan.fov = self.angle_max - self.angle_min
        # Additional properties
        Scan.increment = (self.angle_max - self.angle_min) / self.size
        Scan.angles = np.linspace(self.angle_min, self.angle_max, self.size, dtype=np.float32)
        if file_output:
            for key, value in Scan.__dict__.items():
                if not key.startswith('__') and not inspect.isroutine(value):
                    file_info.scan_info[key] = value

    @classmethod
    def angle_to_index(cls, angle: float) -> int:
        if angle < cls.angle_min or angle > cls.angle_max:
            raise ValueError('Out of range angle')
        return round((angle - cls.angle_min) / cls.increment)

    @classmethod
    def index_to_angle(cls, index: int) -> float:
        if index < 0 or index >= cls.size:
            raise ValueError('Out of bounds angle index')
        return cls.angles[index]

    @classmethod
    def index_to_degrees(cls, index) -> float:
        if index < 0 or index >= cls.size:
            raise ValueError('Out of bounds angle index')
        return round(math.degrees(cls.angles[index]))

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

class Extension(Scan):
    
    def __init__(self, threshold, extension, coeff_func):
        self.threshold = threshold
        self.extension = extension
        self.coeff_func = coeff_func

    def apply(self, ranges):
        extension = self.extension
        # Extend from direction of closer range
        diffs = np.diff(ranges)
        pos_disp = np.flatnonzero(diffs >= self.threshold)
        neg_disp = np.flatnonzero(diffs <= -self.threshold)
        pos_wall_ranges = ranges[pos_disp]
        neg_wall_ranges = ranges[neg_disp + 1]

        # Positive extends in the positive direction starting with i + 1
        for i in range(pos_disp.size):
            disp = pos_disp[i]
            wall_range = pos_wall_ranges[i]
            # TODO
            if extension > (2 * wall_range): # Validate asin domain
                continue
            # Dist_count = Index count to extend from wall
            coeff = self.coeff_func(disp)
            # coeff = 1
            dist_count = math.ceil(coeff * 2 * math.asin(extension/(2*wall_range)) / self.increment)
            gap_start = disp + 1
            if (dist_count > 0):
                n = min(dist_count, ranges.size - gap_start)
                idxs = slice(gap_start, gap_start + n)
                np.minimum(ranges[idxs], wall_range, out=ranges[idxs])

        # Negative extends in the negative direction starting with i
        for i in range(neg_disp.size):
            disp = neg_disp[i]
            wall_range = neg_wall_ranges[i]
            if extension > (2 * wall_range): # Validate asin domain
                continue
            # Dist_count = Index count to extend from wall
            coeff = self.coeff_func(disp)
            # coeff = 1
            dist_count = math.ceil(coeff * 2 * math.asin(extension/(2*wall_range)) / self.increment)
            gap_start = disp
            if (dist_count > 0):
                n = min(dist_count, gap_start)
                idxs = slice(gap_start - n, gap_start + 1)
                np.minimum(ranges[idxs], wall_range, out=ranges[idxs])

class Path(Scan):
    def gaps(self, ranges, pos_starts, neg_starts):
        widths = []
        depths = []
        for i in pos_starts:
            wall_range = ranges[i-1]
            index_count = 0
            j = i
            while j < ranges.size and ranges[j] > wall_range:
                index_count += 1
                j += 1
            # Calculate distance
            theta = index_count * self.increment
            dist = 2 * math.sin(theta / 2) * wall_range
            widths.append(dist)
            depths.append(wall_range)
            
        for i in neg_starts:
            wall_range = ranges[i+1]
            index_count = 0
            j = i
            while j >= 0 and ranges[j] > wall_range:
                index_count += 1
                j -= 1
            # Calculate distance
            theta = index_count * self.increment
            dist = 2 * math.sin(theta / 2) * wall_range
            widths.append(-dist)
            depths.append(wall_range)

        widths, depths = np.array(widths), np.array(depths)

        # with np.printoptions(precision=2, floatmode='fixed', suppress=True):
        #     print(f'\n{depths=}')
        #     print(f'{widths=}')

        return widths, depths

    def choose(self, starts, widths, depths):
        left = 1
        right = -1
        # HARDCODED
        # 0, 67.5, 135 degrees
        steps = (0, 270, 540 if self.size == 1081 else 539)
        mid = self.mid_index
        right_sections = [(mid - steps[i+1], mid - steps[i]) for i in (range(len(steps) - 1))]
        left_sections = [(mid + steps[i], mid + steps[i+1]) for i in (range(len(steps) - 1))]
        
        if self.track_direction == 'ccw':
            for i in range(len(right_sections)):
                gap_idx = self.check_gap(starts, widths, depths, left_sections[i])
                if not gap_idx is None:
                    return (starts[gap_idx], widths[gap_idx], depths[gap_idx])
                gap_idx = self.check_gap(starts, widths, depths, right_sections[i])
                if not gap_idx is None:
                    return (starts[gap_idx], widths[gap_idx], depths[gap_idx])
            return None, None, None
            raise ValueError('No valid gap was found')

        elif self.track_direction == 'cw':
            for i in range(len(right_sections)):
                gap_idx = self.check_gap(starts, widths, depths, right_sections[i])
                if not gap_idx is None:
                    return (starts[gap_idx], widths[gap_idx], depths[gap_idx])
                gap_idx = self.check_gap(starts, widths, depths, left_sections[i])
                if not gap_idx is None:
                    return (starts[gap_idx], widths[gap_idx], depths[gap_idx])
            return None, None, None
            raise ValueError('No valid gap was found')

        else:
            raise ValueError('Invalid track_direction')

    def check_gap(self, starts, widths, depths, index_range):
        lo, hi = index_range
        mask = (starts >= index_range[0]) & (starts <= hi)
        if not np.any(mask):
            return None
        local_map = np.where(mask)[0]
        section_depths = depths[mask]
        for i in range(section_depths.size):
            gap_local_idx = np.nanargmax(section_depths)
            section_depths[gap_local_idx] = 0
            gap_idx = local_map[gap_local_idx]
            gap_width = widths[gap_idx]
            if abs(gap_width) >= self.min_gap_width:
                return gap_idx
        return None

class Steering(Scan):
    def __init__(self, point_method, line_method):
        if point_method == 'center':
            self.get_point = self.get_center
        elif point_method == 'disp':
            self.get_point = self.get_disp
        elif point_method == 'furthest':
            self.get_point = self.get_furthest
    def get_disp(self, start, width, depth):
        if width > 0: sign = 1
        elif width < 0: sign = -1
        else: raise ValueError('Invalid path steering length of 0')
        # TODO OPTIMIZE EXTENSION DISTANCE TO MINIMIZE TURNING ANGLE
        if self.steering_extension > 2 * depth: # Validate asin domain
            return 0.0
        index_width = round(2 * math.asin(self.steering_extension / (2 * depth)) / self.increment)
        steering_idx = start + sign * index_width
        steering_angle = self.index_to_angle(steering_idx)
        return steering_angle

class Smooth:
    def __init__(self, use_filter, use_pid, use_slew, limit, tau, slew_rate, pid):
        self.use_filter = use_filter
        self.use_pid = use_pid
        self.use_slew = use_slew
        self.limit = limit
        self.pid = pid
        self.pid.output_limits = (-limit, limit)
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

    def get(self, target):
        dt = min(perf_counter() - self.start_time, self.max_dt)
        self.start_time = perf_counter()

        # Low pass filter
        if self.use_filter:
            alpha = self.tau/(self.tau + dt)
            # alpha = math.exp(-dt/self.tau)
            # target = self.prev_filtered + alpha * (target - self.prev_filtered)
            target = (alpha * self.prev_filtered) + ((1 - alpha) * target)
            # = alpha * pre_filtered + target - alpha * target
            # = target + alpha * (prefilterd - target)
            self.prev_filtered = target

        # PID
        if self.use_pid:
            target = self.pid(target)
            if target is None:
                raise ValueError('Failed to get PID')
            
        # Saturation (min and max limits)
        target = np.clip(target, -self.limit, self.limit)

        # Slew rate (limit on rate of change)
        if self.use_slew:
            delta = target - self.theta

            # Constant
            max_step = self.slew_rate * dt
            if abs(delta) > max_step:
                target = self.theta + (max_step if delta > 0 else -max_step)

            # Relative to speed

        # Saturation (min and max limits)
        target = np.clip(target, -self.limit, self.limit)

        self.theta = target
        return target

class Speed(Scan):
    # Method:
    #     1. flat - gets flat speed
    #     2. fast - uses steering angle and gap_depth

    def __init__(self, method, flat_speed, max_speed, a_slide, a_tip, wheelbase, max_turning_angle):
        self.flat_speed = flat_speed
        self.min_speed = math.sqrt(min(a_slide, a_tip) * wheelbase / math.tan(max_turning_angle))
        file_info.other['min_speed'] = self.min_speed
        self.max_speed = max_speed
        self.a_slide = a_slide
        self.a_tip = a_tip
        self.wheelbase = wheelbase
        self.max_turning_angle = max_turning_angle
        self.turning_radius = wheelbase / math.tan(max_turning_angle)
        self.last_speed = 0.0

        if method == 'flat':
            self.get = self.flat
        elif method == 'fast':
            self.get = self.fast
        else:
            raise ValueError('Invalid Speed method')

    def flat(self, steering_angle, gap_depth):
        if math.isnan(steering_angle) or not gap_depth:
            return 0.0
        return self.flat_speed

    def fast(self, steering_angle, gap_depth):
        if math.isnan(steering_angle) or not gap_depth:
            print('FAILED SPEED')
            return 0.0
        v_min = self.min_speed
        s = gap_depth
        if s < 0:
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
    rclpy.init(args=args)
    node = PathFollow()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
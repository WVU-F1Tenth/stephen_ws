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

# TODO BUGS:
# - 270 max gap being called inappropriately (probably ok)

# TODO:
# - Prefer inside turns for 270 max gap
# - Optimize binary_search_max_gap by assuming that previous threshold changes little
# - Graph f(threshold) -> gap

is_simulation = False
publish_virtual_scan = True
file_output = True
fast_print = False
# | counter | clockwise |
track_direction = 'counter'

cycle_time = 1e-5

file_info = SimpleNamespace(**{
    'scan_info':{},
    'virtual_scan_info':{},
    'mean_times': defaultdict(lambda:[0.0, 0]),
    'other':{}
})

class GapFollow(Node):
    
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
        self.scan_info = None
        self.cycle_start = perf_counter()

        # Constants
        self.car_radius = 0.4
        self.wheelbase = 0.3
        self.max_steering_angle = 0.42
        self.turning_radius = self.wheelbase/math.tan(self.max_steering_angle)

        # Speed
        self.speed = Speed(
            # | flat | fast |
            method='flat',
            flat_speed=1.0,
            max_speed=16.0,
            a_slide=8.0,
            a_tip=1000.0,
            wheelbase=self.wheelbase,
            max_turning_angle=self.max_steering_angle,
            extension=0.0
            )

        # Steering
        self.steering = Steering(
            # | center | furthest |
            method='center', 
            limit=self.max_steering_angle
            )
        
        # Smoothing functions domain and range [0, 1]
        f0 = lambda x : x
        f1 = lambda x : (x**2)
        A = .2
        f2 = lambda x : (A * math.tan(math.atan(1/A) * x))
        a = 1.7
        b = 14
        c = -9
        f3 = lambda x : (1/(1+pow(a, -(b*x+c))) - 1/(1 + pow(a, -c)))
            
        # Smoothing
        self.smooth = Smooth(
            func=f1,
            use_filter=False,
            use_pid=False,
            use_slew=True,
            limit=self.max_steering_angle,
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

        # Max Gap
        self.max_gap = MaxGap(
            # | index | geometric |
            method='geometric',
            desired_gap=.2
            )

        # Disparity Extension
        self.disparity_extension = DisparityExtension(
            extension=0.3,
            threshold=0.5,
            turning_radius=self.turning_radius,
            car_radius=self.car_radius
            )

        # File output
        if file_output:
            file_info.other['turning_radius'] = self.turning_radius
            file_info.virtual_scan_info['max_range'] = 0.0
            file_info.virtual_scan_info['min_range'] = math.inf
            file_info.scan_info['max_range'] = 0.0
            file_info.scan_info['min_range'] = math.inf

    def adjust(self, scan: LaserScan):
        global cycle_time
        start_time = perf_counter()
        cycle_time = (start_time - self.cycle_start)
        self.cycle_start = start_time
        adjust_start = start_time
        overhead_time = (start_time - self.overhead_start) * 1_000

        if fast_print:
            print(f'\n{"="*24}')

        # Scan info initialization
        if not self.scan_info:
            self.scan_info = ScanInfo(scan)
            print('Running...')

        ranges = np.asarray(scan.ranges)

        if file_output:
            file_info.scan_info['max_range'] = max(np.max(ranges), file_info.scan_info['max_range'])
            file_info.scan_info['min_range'] = min(np.min(ranges), file_info.scan_info['min_range'])

        # =================== Pipeline ========================
        pipeline_start = perf_counter()

        # Disparity Extension
        disparity_start = perf_counter()
        possible_180_path = self.disparity_extension.apply(ranges, self.scan_info)
        disparity_time = (perf_counter() - disparity_start) * 1_000

        # Max Gap
        max_gap_start = perf_counter()
        max_gap, gap_depth = self.max_gap.get(ranges, possible_180_path, self.scan_info)
        max_gap_time = (perf_counter() - max_gap_start) * 1_000

        # Steering
        steering_angle_start = perf_counter()
        steering_angle = self.steering.get(ranges, max_gap, self.scan_info)
        steering_angle_time = (perf_counter() - steering_angle_start) * 1_000

        # Smoothing
        steering_angle = self.smooth.get(steering_angle)

        # Speed
        speed_start = perf_counter()
        speed = self.speed.get(steering_angle, gap_depth)
        speed_time = (perf_counter() - speed_start) * 1_000

        pipeline_time = (perf_counter() - pipeline_start) * 1_000
        # =====================================================

        # Publish messages
        if publish_virtual_scan and max_gap is not None:
            ranges[max_gap[0] : max_gap[1] + 1] = gap_depth
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
            print(f'\n{disparity_time=:.3f}')
            print(f'{max_gap_time=:.3f}')
            print(f'{steering_angle_time=:.3f}')
            print(f'{speed_time=:.3f}')

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
        
class Speed:
    # Method:
    #     1. flat - gets flat speed
    #     2. fast - uses steering angle and gap_depth

    def __init__(self, method, flat_speed, max_speed, a_slide, a_tip, wheelbase, max_turning_angle, extension):
        self.flat_speed = flat_speed
        self.min_speed = math.sqrt(min(a_slide, a_tip) * wheelbase / math.tan(max_turning_angle))
        file_info.other['min_speed'] = self.min_speed
        self.max_speed = max_speed
        self.a_slide = a_slide
        self.a_tip = a_tip
        self.wheelbase = wheelbase
        self.max_turning_angle = max_turning_angle
        self.extension = extension
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
        s = gap_depth + self.extension
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

class Steering:
    # Method:
    #     1. center - steers to center of gap
    #     2. farthest - steers to farthest point in gap

    def __init__(self, method, limit):
        # Angular velocity and angle state
        self.theta: float = 0.0
        self.w = 0.0

        # Inertia
        self.alpha = 0.0

        # Exponential decay / Low pass filter
        self.tau = 0.12

        # Low pass filter
        self.low_pass_filter_tau = 0.03
        self.target_smoothed = 0.0

        # PID
        self.kp = 1.0
        self.ki = 0.01
        self.kd = 0.02
        self.udot_max = 4.0
        self.u_prev = 0.0
        
        # Mass spring damper
        # (wn * dt) typically ranges 0.3 to 0.5
        self.wn = 80.0 
        self.zeta = 1.0

        self.limit = limit

        if method == 'center':
            self.get = self.get_center
        elif method == 'farthest':
            self.get = self.get_farthest
        else:
            raise ValueError('Invalid Steering method')

    def get_center(self, ranges, max_gap, scan_info) -> float:
        # aim for center of max_gap
        if not max_gap:
            return math.nan
        middle_index = max_gap[0] + math.ceil((max_gap[1] - max_gap[0]) / 2)
        steering_angle = scan_info.index_to_angle(middle_index)
        return steering_angle

    def get_farthest(self, ranges, max_gap, scan_info) -> float:
        # aim for farthest
        if not max_gap:
            return math.nan
        max_index = max_gap[0] + np.argmax(ranges[max_gap[0]:max_gap[1]])
        steering_angle = scan_info.index_to_angle(max_index)
        return steering_angle

class Smooth:
    def __init__(self, func, use_filter, use_pid, use_slew, limit, tau, slew_rate, pid):
        self.func = func
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
        
class MaxGap:

    def __init__(self, method, desired_gap):
        self.desired_gap = desired_gap
        # Can be used for optimizing binary search
        self.threshold = None
        if method == 'index':
            self.get_gap = self.index_max_gap_binary_search
        elif method == 'geometric':
            self.get_gap = self.geometric_max_gap_binary_search
        else:
            raise ValueError('Invalid MaxGap method')

    def get(self, ranges, possible_180_path, scan_info):
        if possible_180_path:
            max_gap, threshold = self.get_gap(scan_info.fov_slice(ranges, 180), scan_info)
            return scan_info.revert_fov_indexes(180, max_gap), threshold
        else:
            max_gap, threshold = self.get_gap(scan_info.fov_slice(ranges, 270), scan_info)
            if max_gap:
                return scan_info.revert_fov_indexes(270, max_gap), threshold

        if not is_simulation:
            raise NotImplementedError('Max_gap not found within parameters...')
        else:
            return None, None

    def index_max_gap_binary_search(self, ranges, scan_info):
        max_iterations = 8
        max_threshold = np.max(ranges)
        min_threshold = np.min(ranges)

        iterations = 0
        max_gap = 0
        threshold = (max_threshold + min_threshold) / 2
        while (iterations < max_iterations):
            iterations += 1
            ranges_diff = np.diff((ranges >= threshold).astype(int), prepend=0, append=0)
            starts = np.where(ranges_diff == 1)[0]
            ends = np.where(ranges_diff == -1)[0]
            range_list = list(zip(starts, ends))
            if range_list:
                lengths = ends - starts
                max_index = np.argmax(lengths)
                max_gap = lengths[max_index]
                max_gap_indexes = range_list[max_index]
            else:
                # Nothing found need to make threshold smaller
                max_gap = 0

            if (max_gap > self.desired_gap):
                min_threshold = threshold
            elif (max_gap < self.desired_gap):
                max_threshold = threshold
            else:
                break

            # Set new threshold
            threshold = (max_threshold + min_threshold) / 2

        if not max_gap_indexes:
            return None, None

        return max_gap_indexes, threshold

    def geometric_max_gap_binary_search(self, ranges, scan_info):
        max_iterations = 8
        max_threshold = np.max(ranges)
        min_threshold = np.min(ranges)

        iterations = 0
        max_gap = 0.0
        threshold = (max_threshold + min_threshold) / 2
        while (iterations < max_iterations):
            iterations += 1
            ranges_diff = np.diff((ranges >= threshold).astype(int), prepend=0, append=0)
            starts = np.where(ranges_diff == 1)[0]
            ends = np.where(ranges_diff == -1)[0]
            range_list = list(zip(starts, ends))
            if range_list:
                index_lengths = ends - starts
                lengths = 2 * threshold * np.sin(index_lengths * scan_info.increment / 2)
                max_index = np.argmax(lengths)
                max_gap = lengths[max_index]
                max_gap_indexes = range_list[max_index]
            else:
                # Nothing found need to make threshold smaller
                max_gap = 0.0

            if (max_gap > self.desired_gap):
                min_threshold = threshold
            elif (max_gap < self.desired_gap):
                max_threshold = threshold
            else:
                break

            # Set new threshold
            threshold = (max_threshold + min_threshold) / 2

        if not max_gap_indexes:
            return None, None

        return max_gap_indexes, threshold

class DisparityExtension:
    
    def __init__(self, extension, threshold, turning_radius, car_radius):
        self.extension = extension
        self.threshold = threshold
        self.turning_radius = turning_radius
        self.car_radius = car_radius

    def apply(self, ranges, scan_info):
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
            if self.extension > (2 * wall_range): # Validate asin domain
                continue
            # Dist_count = Index count to extend from wall
            coeff = math.sqrt(abs(math.cos(scan_info.index_to_angle(disp))))
            dist_count = math.ceil(coeff * 2 * math.asin(self.extension/(2*wall_range)) / scan_info.increment)
            gap_start = disp + 1
            if (dist_count > 0):
                n = min(dist_count, ranges.size - gap_start)
                idxs = slice(gap_start, gap_start + n)
                np.minimum(ranges[idxs], wall_range, out=ranges[idxs])

       # Negative extends in the negative direction starting with i
        for i in range(neg_disp.size):
            disp = neg_disp[i]
            wall_range = neg_wall_ranges[i]
            if self.extension > (2 * wall_range): # Validate asin domain
                continue
            # Dist_count = Index count to extend from wall
            coeff = math.sqrt(abs(math.cos(scan_info.index_to_angle(disp))))
            dist_count = math.ceil(coeff * 2 * math.asin(self.extension/(2*wall_range)) / scan_info.increment)
            gap_start = disp
            if (dist_count > 0):
                n = min(dist_count, gap_start)
                idxs = slice(gap_start - n, gap_start + 1)
                np.minimum(ranges[idxs], wall_range, out=ranges[idxs])

        indexes = np.concatenate((pos_disp, neg_disp))
        return ((indexes > 180) & (indexes < 900)).any()
                
class ScanInfo:
    
    def __init__(self, scan):
        self.size = len(scan.ranges)
        self.angle_min = scan.angle_min
        self.angle_max = scan.angle_max
        self.angle_increment = scan.angle_increment

        self.time_increment = scan.time_increment
        self.scan_time = scan.scan_time
        self.range_min = scan.range_min
        self.range_max = scan.range_max
        self.fov = self.angle_max - self.angle_min

        self.mid_index = math.ceil((self.size - 1) / 2)
        self.increment = (self.angle_max - self.angle_min) / self.size

        if file_output:
            for key, value in self.__dict__.items():
                file_info.scan_info[key] = value

    def valid(self, scan):
        is_valid = True
        # Check size
        if self.size != len(scan.ranges):
            print(f'len(scan.ranges = {len(scan.ranges)}, expected {self.size})')
            is_valid = False
        for key in self.__dict__:
            # Check for other differences
            if getattr(scan, key, None) and self.__dict__[key] != getattr(scan, key):
                print(f'Invalid: scan.{key} = {getattr(scan, key)}, expected {self.__dict__[key]}')
                is_valid = False
        return is_valid

    def angle_to_index(self, angle) -> int:
        return self.mid_index + round(angle / self.increment)

    def index_to_angle(self, index) -> float:
        return (index - self.mid_index) * self.increment

    def index_to_degrees(self, index) -> float:
        return round(math.degrees((index - self.mid_index) * self.increment), 4)

    def fov_slice(self, ranges, fov):
        fov = math.radians(fov)
        fov_offset = self.angle_to_index(-fov / 2)
        return ranges[fov_offset: len(ranges) - fov_offset]
    
    def revert_fov_indexes(self, fov, indexes):
        fov = math.radians(fov)
        reverted = []
        for i in indexes:
            reverted.append(i + round((self.fov - fov) / (2 * self.increment)))
        return reverted

class SafetyBubble:
    #  1. Safety Bubble: (only needed within threshold)
    #     a. put bubble around nearest lidar point
    #     b. set all points in bubble to 0
    # Radius - radius of safety bubble
    # Fov - fov in which the safety bubble will be applied
    # Max distace at which bubble will be applied

    def __init__(self, radius, fov, threshold):
        self.radius = radius
        self.fov = fov
        self.threshold = threshold

    def apply(self, ranges, scan_info):
        # fov adjustment
        ranges = scan_info.fov_slice(ranges, self.fov)

        # find minimum and check if far enough
        min_index = np.nanargmin(ranges)
        min_range = ranges[min_index]
        if min_range > self.threshold:
            return
        
        # make bubble around point
        radius_offset_max = round(math.atan(self.radius / min_range) / scan_info.increment)
        
        ranges[min_index] = 0

        # caculate circle for both sides around minimum range
        for radius_offset in range(1, radius_offset_max + 1):
            # new_range = min_range - math.sqrt(
            #     self.bubble_radius**2 - (self.bubble_radius * radius_offset / radius_offset_max)**2)
            i = min_index + radius_offset
            if i < len(ranges):
                ranges[i] = 0.0
            j = min_index - radius_offset
            if j >= 0:
                ranges[j] = 0.0

class CornerSafety:
    # 1. scan beyond +- 90 degrees
    # 2. if any point is below safe distance on side of car in direction car is turning,
    #    stop turning and go straight
    #
    # Safe_wall_distance - disables turning towards wall if within this distance

    def __init__(self, safe_wall_distance):
        self.safe_wall_distance = safe_wall_distance

    def get(self, ranges, steering_angle, scan_info):
        if math.isnan(steering_angle):
            return 0.0
        if self.too_close_to_wall(ranges, steering_angle, scan_info):
            return 0.0
        else:
            return steering_angle

    def too_close_to_wall(self, ranges, steering_angle, scan_info):
        if (steering_angle < 0 and ranges[scan_info.angle_to_index(-pi/2)] < self.safe_wall_distance):
            return True
        if (steering_angle > 0 and ranges[scan_info.angle_to_index(pi/2)] < self.safe_wall_distance):
            return True
        return False

def handler(sig, frame):
    if file_output:
        for key, value in file_info.mean_times.items():
            mean_time = value[0] / value[1]
            file_info.mean_times[key] = mean_time
        with open('gap_follow_info.txt', 'w', encoding='utf-8') as f:
            for key, subdict in file_info.__dict__.items():
                f.write(f'\n{key.upper()}\n')
                for subkey, value in subdict.items():
                    f.write(f'{subkey} : {value}\n')
    sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = GapFollow()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

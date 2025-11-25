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
# - Off by one errors
# - Fix index-angle functions
# - Disparity moving on convex curvature causes wiggling
# - False disparity switching when alternating to either side of heading
# - wall_extension creates false disparities
# - Handle potential bottle neck disparity overlap
# - Tune choose path
# - Smoothing function sometimes gets stuck at offset
# - Smooth virtual lidar
# - Fix negative curve problem
#       Handled by wall_extend
# - Create angle planning to account for little information know about front
# - Create custom maps for sim
# - Create tracking line for sim to compare paths

publish_marker = True
publish_virtual_scan = True
publish_v2 = True
file_output = True
fast_print = False

cycle_time = 1e-5

flat_speed = 0.0
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
        self.scan = None
        self.prev_speed = 0.0
        self.prev_steering_angle = 0.0

        # Constants
        self.car_radius = 0.
        self.wheelbase = 0.33
        self.max_steering_angle = 0.42
        self.turning_radius = self.wheelbase/math.tan(self.max_steering_angle)

        # Path
        self.path = Path(
            disparity_threshold=0.5,
            extension=0.35,
            steering_extension=0.0,
            min_gap_width=0.0,
            max_steering_angle=self.max_steering_angle,
            track_direction='ccw',
            # | disparity | center |
            steering_method='disparity',
            # | max_gap | furthest |
            path_method='furthest'
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

        # Speed
        self.speed = Speed(
            # | flat | fast |
            method='flat',
            flat_speed=1.,
            max_speed=100.0,
            a_slide=14.0,
            a_tip=10000.0,
            wheelbase=self.wheelbase,
            max_turning_angle=self.max_steering_angle,
            extension=0.2
            )

        # File output
        if file_output:
            file_info.other['turning_radius'] = self.turning_radius
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

        if not self.path.is_set:
            self.path.setup(scan)
            print('Running...')
        
        if fast_print:
            print(f'\n{"="*24}')

        ranges = np.asarray(scan.ranges)

        if file_output:
            file_info.path_info['max_range'] = max(np.max(ranges), file_info.path_info['max_range'])
            file_info.path_info['min_range'] = min(np.min(ranges), file_info.path_info['min_range'])

        # =================== Pipeline ========================
        pipeline_start = perf_counter()

        #self.path.apply_range_limit(ranges, 10.0)

        extensions = self.path.get_wall_extensions(ranges)
        self.path.disparity_extend2(ranges, extensions)
        
        pos_disp, neg_disp = self.path.disparities(ranges)
        pos_starts, neg_starts = pos_disp + 1, neg_disp

        widths, depths = self.path.gaps(ranges, pos_starts, neg_starts)
        starts = np.concatenate((pos_starts, neg_starts))

        start, width, depth = self.path.choose(starts, widths, depths)

        if start is None or width is None or depth is None:
            steering_angle = self.prev_steering_angle
            speed = self.prev_speed
        else:
            steering_angle = self.path.steering(start, width, depth)

            steering_angle = self.smooth.get(steering_angle, self.prev_speed)

            speed = self.speed.get(steering_angle, depth + self.car_radius - self.turning_radius)
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
            msg.data = extensions.tolist()
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

class Path:
    def __init__(self, disparity_threshold, extension, steering_extension,
                  min_gap_width, track_direction, max_steering_angle, steering_method, path_method):
        self.disparity_threshold = disparity_threshold
        self.extension = extension
        self.steering_extension = steering_extension
        self.min_gap_width = min_gap_width
        self.track_direction = track_direction
        self.max_steering_angle = max_steering_angle
        self.steering_method = steering_method
        self.path_method = path_method
        self.is_set = False

    def apply_range_limit(self, ranges, limit):
        safe_gap_value = 100.0
        ranges[ranges > limit] = safe_gap_value

    def get_wall_extensions(self, ranges):
        extension = self.extension
        n = len(ranges)
        range_matrix = np.full((n, n), np.inf, dtype=np.float32)

        ratio = extension / (2 * ranges)
        ratio = np.clip(ratio, -1.0, 1.0)
        index_extensions = abs(np.floor(2*np.arcsin(ratio)/self.angle_increment).astype(np.int32))

        rows = np.arange(n)[:, None]
        cols = np.arange(n)[None, :]
        mask = np.abs(cols-rows) <= index_extensions[:, None]

        range_matrix = np.where(mask, ranges[:, None], np.inf).astype(np.float32)

        col_mins = range_matrix.min(axis=0)

        return col_mins
    
    def disparity_extend2(self, ranges, extensions):

        # Maybe use min gap because not all points extended (wall of curvature)

        # Neg_disp extending pos_disp

        # Roughly Parallel disps fluctuate causing random switching when in same choice section

        global disparity_threshold
        diffs = np.diff(ranges)
        # Disps are wall points
        pos_disp = np.flatnonzero(diffs >= disparity_threshold)
        neg_disp = np.flatnonzero(diffs <= -disparity_threshold) + 1

        for disp in pos_disp:
            # Walk gap until intersection
            disp_range = ranges[disp]
            points = np.flatnonzero((ranges[disp+1:] <= disp_range) | (extensions[disp+1:] > disp_range))
            if points.size:
                left_intersect = points[0] + disp+1
            else:
                continue
            # Drop to intersection range
            if extensions[left_intersect] > disp_range:
                new_ext = extensions[left_intersect-1]
            else:
                new_ext = ranges[left_intersect]
            # Backtrack until second intersection
            points = np.flatnonzero((ranges[:disp] <= new_ext) | (extensions[:disp] > new_ext))
            right_intersect = points[-1] if points.size else disp
            if extensions[right_intersect] > new_ext:
                right_intersect += 1
            ranges_slice = ranges[right_intersect: left_intersect + 1]
            np.minimum(ranges_slice, new_ext, out=ranges_slice)

        for disp in neg_disp:
            # Walk gap until intersection
            disp_range = ranges[disp]
            points = np.flatnonzero((ranges[:disp] <= disp_range) | (extensions[:disp] > disp_range))
            if points.size:
                right_intersect = points[-1]
            else:
                continue
            # Drop to intersection range
            if extensions[right_intersect] > disp_range:
                new_ext = extensions[right_intersect+1]
            else:
                new_ext = ranges[right_intersect]
            # Backtrack until second intersection
            points = np.flatnonzero((ranges[disp+1:] <= new_ext) | (extensions[disp+1:] > new_ext))
            left_intersect = points[0] + disp + 1 if points.size else disp
            if extensions[left_intersect] > new_ext:
                left_intersect -= 1
            ranges_slice = ranges[right_intersect: left_intersect + 1]
            np.minimum(ranges_slice, new_ext, out=ranges_slice)

    def index_extend(self, ranges):
        global disparity_threshold
        # Extend from direction of closer range
        diffs = np.diff(ranges)
        pos_disp = np.flatnonzero(diffs >= disparity_threshold)
        neg_disp = np.flatnonzero(diffs <= -disparity_threshold)
        pos_wall_ranges = ranges[pos_disp]
        neg_wall_ranges = ranges[neg_disp + 1]
        
        index_offset = 67

        # Positive extends in the positive direction starting with i + 1
        for i in range(pos_disp.size):
            disp = pos_disp[i]
            wall_range = pos_wall_ranges[i]
            # Dist_count = Index count to extend from wall
            coeff = math.sqrt(abs(math.cos(self.index_to_angle(disp))))
            # coeff = 1
            dist_count = math.ceil(coeff * index_offset)
            gap_start = disp + 1
            if (dist_count > 0):
                n = min(dist_count, ranges.size - gap_start)
                idxs = slice(gap_start, gap_start + n)
                np.minimum(ranges[idxs], wall_range, out=ranges[idxs])

        # Negative extends in the negative direction starting with i
        for i in range(neg_disp.size):
            disp = neg_disp[i]
            wall_range = neg_wall_ranges[i]
            # Dist_count = Index count to extend from wall
            coeff = math.sqrt(abs(math.cos(self.index_to_angle(disp))))
            # coeff = 1
            dist_count = math.ceil(coeff * index_offset)
            gap_start = disp
            if (dist_count > 0):
                n = min(dist_count, gap_start)
                idxs = slice(gap_start - n, gap_start + 1)
                np.minimum(ranges[idxs], wall_range, out=ranges[idxs])


    def disparity_extend(self, ranges):
        global disparity_threshold
        extension = self.extension
        # Extend from direction of closer range
        diffs = np.diff(ranges)
        pos_disp = np.flatnonzero(diffs >= disparity_threshold)
        neg_disp = np.flatnonzero(diffs <= -disparity_threshold)
        pos_wall_ranges = ranges[pos_disp]
        neg_wall_ranges = ranges[neg_disp + 1]

        # Positive extends in the positive direction starting with i + 1
        for i in range(pos_disp.size):
            disp = pos_disp[i]
            wall_range = pos_wall_ranges[i]
            ratio = extension/(2*wall_range)
            ratio = np.clip(ratio, -1.0, 1.0)
            # Dist_count = Index count to extend from wall
            # coeff = math.sqrt(abs(math.cos(self.index_to_angle(disp))))
            coeff = 1
            dist_count = math.ceil(coeff * 2 * math.asin(ratio) / self.increment)
            gap_start = disp + 1
            if (dist_count > 0):
                n = min(dist_count, ranges.size - gap_start)
                idxs = slice(gap_start, gap_start + n)
                np.minimum(ranges[idxs], wall_range, out=ranges[idxs])

        # Negative extends in the negative direction starting with i
        for i in range(neg_disp.size):
            disp = neg_disp[i]
            wall_range = neg_wall_ranges[i]
            ratio = extension/(2*wall_range)
            ratio = np.clip(ratio, -1.0, 1.0)
            # Dist_count = Index count to extend from wall
            # coeff = math.sqrt(abs(math.cos(self.index_to_angle(disp))))
            coeff = 1
            dist_count = math.ceil(coeff * 2 * math.asin(ratio) / self.increment)
            gap_start = disp
            if (dist_count > 0):
                n = min(dist_count, gap_start)
                idxs = slice(gap_start - n, gap_start + 1)
                np.minimum(ranges[idxs], wall_range, out=ranges[idxs])

    def disparities(self, ranges):
        global disparity_threshold
        diffs = np.diff(ranges)
        return (np.flatnonzero(diffs >= disparity_threshold),
        np.flatnonzero(diffs <= -disparity_threshold))
    
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

        return widths, depths

    def choose(self, starts, widths, depths):
        left = 1
        right = -1
        # HARDCODED
        steps = (0, 180, 360, 540 if self.size == 1081 else 539)
        mid = self.mid_index
        right_sections = [(mid - steps[i+1], mid - steps[i]) for i in (range(len(steps) - 1))]
        left_sections = [(mid + steps[i], mid + steps[i+1]) for i in (range(len(steps) - 1))]
        
        if self.track_direction == 'ccw':
            sections = [x for pair in zip(left_sections, right_sections) for x in pair]
        elif self.track_direction == 'cw':
            sections = [x for pair in zip(right_sections, left_sections) for x in pair]
        else:
            raise ValueError('Invalid track_direction')
        
        # Prevents sudden choice swapping due to dead on heading
        sections.insert(1, (535, 545))
        
        for section in sections:
            gap_idx = self.check_gap(starts, widths, depths, section)
            if not gap_idx is None:
                    return (starts[gap_idx], widths[gap_idx], depths[gap_idx])
        
        print('No path...')
        return None, None, None

    def check_gap(self, starts, widths, depths, index_range):
        lo, hi = index_range
        mask = (starts >= lo) & (starts <= hi)
        if not np.any(mask):
            return None
        local_map = np.where(mask)[0]

        if self.path_method == 'furthest':
            section_depths = depths[mask]
            for i in range(section_depths.size):
                gap_local_idx = np.nanargmax(section_depths)
                section_depths[gap_local_idx] = 0
                gap_idx = local_map[gap_local_idx]
                gap_width = widths[gap_idx]
                if abs(gap_width) >= self.min_gap_width:
                    return gap_idx
        elif self.path_method == 'max_gap':
            section_widths = widths[mask]
            for i in range(section_widths.size):
                gap_local_idx = np.nanargmax(section_widths)
                gap_idx = local_map[gap_local_idx]
                gap_width = widths[gap_idx]
                if abs(gap_width) >= self.min_gap_width:
                    return gap_idx
        else:
            raise ValueError('Invalid path_method')
        return None
        
    def steering(self, start: int, width: float, depth: float):
        if width > 0: sign = 1
        elif width < 0: sign = -1
        else: raise ValueError('Invalid path steering length of 0')
        # if self.steering_extension > 2 * depth: # Validate asin domain
        #     return 0.0
        # ext_index_width = round(2 * math.asin(self.steering_extension / (2 * depth)) / self.increment)
        index_width = round(2 * math.asin(abs(width) / (2 * depth)) / self.increment)
        if self.steering_method == 'center':
            steering_idx = start + (sign * index_width / 2)
            steering_angle = self.index_to_angle(steering_idx)
        elif self.steering_method == 'disparity':
            # steering_idx = start + sign * ext_index_width
            steering_idx = start
            steering_angle = self.index_to_angle(steering_idx)
        else:
            raise ValueError('Invalid steering method')
        return steering_angle

    def valid_scan(self, scan):
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

    def setup(self, scan):
        self.is_set = True
        self.size = len(scan.ranges)
        self.angle_min = scan.angle_min
        self.angle_max = scan.angle_max
        assert scan.angle_max == abs(scan.angle_min)
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
                file_info.path_info[key] = value

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
        global flat_speed
        if math.isnan(steering_angle) or not gap_depth:
            return 0.0
        return flat_speed

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
    
def input_thread():
    global flat_speed, smoothing_exp, disparity_threshold
    print('Command (s=-speed, d=+speed, j=-exp, k=+exp)')
    while True:
        cmd = input()
        if cmd == 's':
            flat_speed -= .1
            print(f'speed = {flat_speed:.2}')
        elif cmd == 'd':
            flat_speed += .1
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

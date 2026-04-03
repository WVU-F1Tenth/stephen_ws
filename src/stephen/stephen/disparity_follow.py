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
import sys
from collections import defaultdict
from types import SimpleNamespace
from simple_pid import PID
import select
import termios
import tty

# TODO:
# - Make markers for virtual on car
# - Check algorithm on smoothed map
# - Implement path radius filter
# - Combine with global planning
# - Implment arc steering and arc marker
# - Range limit could be adaptive based on amount of path taken up or geometric distance
# - More effiction wall extension (try cupy or only 720 indexes)
# - Handle potential bottle neck disparity overlap
#       Base extension must be set to car radius or this could be a problem
# - Tune choose path
# - Smoothing function sometimes gets stuck at offset
# - Create angle planning to account for little information know about front
#       Could use difference between depth and vdepth
# - Create tracking line for sim to compare paths

HERTZ = 0.0
VISUALS = True
LINE_STEERING_MARKER = True
ARC_STEERING_MARKER = False
PUBLISH_POINTS1 = True
PUBLISH_POINTS2 = True
PUBLISH_POINTS3 = True
PUBLISH_V1 = False
PUBLISH_V2 = False
VISUAL_HERTZ = 5.0
FILE_OUTPUT = True
FAST_PRINT = False


params = SimpleNamespace(
    speed=SimpleNamespace(v=0.0, key='s', name='speed'),
    velocities_coeff=SimpleNamespace(v=0.1, key='v', name='velocity coefficient'),
    velocities_mode=SimpleNamespace(v=False, key=None),
    disparity_threshold=SimpleNamespace(v=0.5, key='t', name='disparity threshold'),
    steering_velocity=SimpleNamespace(v=0.0, key='w', name='steering velocity'),
    map_extension=SimpleNamespace(v=0.3, key='e', name='map extension'),
)
SELECTED = params.speed

file_info = SimpleNamespace(**{
    'path_info':{},
    'virtual_scan_info':{},
    'mean_times': defaultdict(lambda:[0.0, 0]),
    'other':{}
})

class PathFollow(Node):
    
    def __init__(self):
        super().__init__('gap_follow')
        print(f'TTY: {sys.stdin.isatty()}')
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        if HERTZ == 0.0:
            self.laser_scan_sub = self.create_subscription(LaserScan, '/scan', self.adjust, 10)
        else:
            self.laser_scan_sub = self.create_subscription(LaserScan, '/scan', self.save_scan, 10)
            self.timer = self.create_timer(1.0 / HERTZ, self.adjust_wrapper)
        if  VISUALS:
            self.create_timer(1.0 / VISUAL_HERTZ, self.publish_markers)
        if PUBLISH_V1:
            self.v1_pub = self.create_publisher(Float32MultiArray, '/v1_ranges', 10)
        if PUBLISH_V2:
            self.v2_pub = self.create_publisher(Float32MultiArray, '/v2_ranges', 10)
        if LINE_STEERING_MARKER:
            self.line_marker_pub = self.create_publisher(Marker, '/viz/goal', 10)
        if ARC_STEERING_MARKER:
            self.arc_marker_pub = self.create_publisher(Marker, '/viz/steering_arc', 10)
        if PUBLISH_POINTS1:
            self.points1_pub = self.create_publisher(Marker, '/viz/points1', 10)
        if PUBLISH_POINTS2:
            self.points2_pub = self.create_publisher(Marker, '/viz/points2', 10)
        if PUBLISH_POINTS3:
            self.points3_pub = self.create_publisher(Marker, '/viz/points3', 10)
        self.keyboard_timer = self.create_timer(.2, self.check_input)
        self.steering_angle = 0.0
        self.speed = 0.0
        self.path = None
        self.scan = None
        self.v1 = np.zeros((1,))
        self.v2 = np.zeros((1,))
        self.paths = []
        self.section = None

        self.fd = sys.stdin.fileno()
        self.terminal_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

        Vehicle.setup(
            wheelbase=0.33,
            radius=0.33,
            max_steering_angle = 0.38
        )

        self.planner = Planner(
            disparity_threshold=0.5,
            extension=0.3,
            track_direction='ccw'
            )
        
        self.steering = Steering(
            # | arc | line |
            method='line'
        )
        
        # Smoothing
        self.smoothing = Smoothing(
            limit=Vehicle.max_steering_angle
        )

        # Speed
        self.speed_controller = SpeedController(
            # | flat | fast |
            method='flat',
            max_speed=100.0,
            a_slide=14.0,
            a_tip=10000.0,
            )
        
        # Print key bindings
        command_bindings = '\n'.join(
            [f'  {param.key}={param.name}' for param in vars(params).values() if param.key])
        print(f'Commands:\n  space = stop\n{command_bindings}')

        # File output
        if FILE_OUTPUT:
            file_info.other['turning_radius'] = Vehicle.turning_radius
            file_info.virtual_scan_info['max_range'] = 0.0
            file_info.virtual_scan_info['min_range'] = math.inf
            file_info.path_info['max_range'] = 0.0
            file_info.path_info['min_range'] = math.inf
            file_info.other['max_pipeline_time'] = 0.0

    def save_scan(self, scan: LaserScan):
        self.scan = scan

    def adjust_wrapper(self):
        if self.scan:
            self.adjust(self.scan)

    def adjust(self, scan: LaserScan):

        adjust_start = perf_counter()

        if not Scan.initialized:
            Scan.setup(scan)
            self.v1 = np.zeros(Scan.size)
            self.v2 = np.zeros(Scan.size)
        
        if FAST_PRINT:
            print(f'\n{"="*24}')

        ranges = np.asarray(scan.ranges)

        if FILE_OUTPUT:
            file_info.path_info['max_range'] = max(np.max(ranges), file_info.path_info['max_range'])
            file_info.path_info['min_range'] = min(np.min(ranges), file_info.path_info['min_range'])

        # =================== Pipeline ========================
        
        pipeline_start = perf_counter()

        # self.planner.apply_range_limit(ranges, 10.0)

        virtual = self.planner.get_virtual(ranges)

        self.v1[:] = virtual
        
        pos_disps, neg_disps = self.planner.disparities(ranges)

        self.paths = self.planner.get_paths(ranges, virtual, pos_disps, neg_disps)

        path = self.planner.choose(ranges, self.paths)

        if path is None:
            print('NO GOAL POINT')
            steering_angle = self.steering_angle
            speed = self.speed
        else:
            steering_angle = self.steering.get(path)

            steering_angle, steering_velocity = self.smoothing.get(steering_angle, self.speed)

            speed = self.speed_controller.get(steering_angle,  path.depth + Vehicle.radius - Vehicle.turning_radius)
        
        pipeline_time = (perf_counter() - pipeline_start) * 1_000
        # =====================================================

        # DEBUG
        # print(Scan.span_to_angle(1.0, 20.0)/Scan.angle_increment)
        
        if path:
            self.path = path
            self.speed = speed
            self.steering_angle = steering_angle
            self.virtual = virtual
       
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.steering_angle_velocity = steering_velocity
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

        adjust_time = (perf_counter() - adjust_start) * 1000

        if FAST_PRINT:
            print(f'\n{speed=:.2}')
            print(f'{steering_angle=:.2}')
            print(f'{adjust_time=:.3f}')

        # File output
        if FILE_OUTPUT:
            file_info.mean_times['pipeline_mean'][0] += pipeline_time
            file_info.mean_times['pipeline_mean'][1] += 1
            file_info.other['max_pipeline_time'] = max(pipeline_time, file_info.other['max_pipeline_time'])
            file_info.mean_times['adjust_mean'][0] += adjust_time
            file_info.mean_times['adjust_mean'][1] += 1
            file_info.virtual_scan_info['max_range'] = max(np.max(ranges), file_info.virtual_scan_info['max_range'])
            file_info.virtual_scan_info['min_range'] = min(np.min(ranges), file_info.virtual_scan_info['min_range'])

        self.overhead_start = perf_counter()
    
    def publish_markers(self):
        if PUBLISH_V1:
            msg = Float32MultiArray()
            msg.data = self.v1.tolist()
            self.v1_pub.publish(msg)
        if PUBLISH_V2:
            msg = Float32MultiArray()
            msg.data = self.v2.tolist()
            self.v2_pub.publish(msg)
        if LINE_STEERING_MARKER and self.path:
            L = self.path.vdepth
            theta = self.steering_angle
            p0 = Point(x=0.0, y=0.0, z=0.0)
            p1 = Point(x=L*math.cos(theta), y=L*math.sin(theta), z=0.0)
            m = Marker()
            m.pose.orientation.w = 1.0
            m.header.frame_id = '/ego_racecar/laser'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'steering_line'
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.05
            m.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
            m.points = [p0, p1]
            self.line_marker_pub.publish(m)
        if ARC_STEERING_MARKER and self.path:
            goal_depth = self.path.depth
            delta = self.steering_angle
            L = Vehicle.wheelbase
            if goal_depth <= 0.0:
                return
            # Handle straight case
            if abs(delta) < 1e-6:
                s = np.linspace(0.0, goal_depth, 30)
                x = s
                y = np.zeros_like(s)
            else:
                R = L / math.tan(delta)
                # Arc length to goal
                s = np.linspace(0.0, goal_depth, 30)
                # Convert arc length to circle angle
                phi = s / R
                x = R * np.sin(phi)
                y = R * (1 - np.cos(phi))
            points = [Point(x=float(px), y=float(py), z=0.0)
                    for px, py in zip(x, y)]
            m = Marker()
            m.header.frame_id = '/ego_racecar/laser'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'steering_arc'
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.05
            m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            m.pose.orientation.w = 1.0
            m.points = points
            self.arc_marker_pub.publish(m)
        if PUBLISH_POINTS1:
            points1 = [Point(x=path.depth*math.cos(path.angle),
                            y=path.depth*math.sin(path.angle),
                            z=0.0) for path in self.paths]
            m = Marker()
            m.header.frame_id = '/ego_racecar/laser'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'points1'
            m.id = 0
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            m.points = points1
            self.points1_pub.publish(m)
        if PUBLISH_POINTS2:
            points2 = [Point(x=path.vdepth*math.cos(path.vangle),
                            y=path.vdepth*math.sin(path.vangle),
                            z=0.0) for path in self.paths]
            m = Marker()
            m.header.frame_id = '/ego_racecar/laser'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'points2'
            m.id = 0
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            m.points = points2
            self.points2_pub.publish(m)
        if PUBLISH_POINTS3 and not self.planner.vdisps is None:
            points3 = [Point(x=self.virtual[disp]*math.cos(Scan.index_to_angle(disp)),
                            y=self.virtual[disp]*math.sin(Scan.index_to_angle(disp)),
                            z=0.0) for disp in self.planner.vdisps]
            m = Marker()
            m.header.frame_id = '/ego_racecar/laser'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'points3'
            m.id = 0
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.5)
            m.points = points3
            self.points3_pub.publish(m)

    def check_input(self):
        global params, SELECTED
        key = self.get_key()
        if not key:
            return
        if key == ' ':
            params.speed.v = 0.0
            params.velocities_coeff.v = 0.0
            print('stopped')
        elif key in ('i', 'o', 'j', 'k', 'n', 'm'):
            if key == 'i':
                SELECTED.v -= 1.0
            elif key == 'o':
                SELECTED.v += 1.0
            elif key == 'j':
                SELECTED.v -= 0.1
            elif key == 'k':
                SELECTED.v += 0.1
            elif key == 'n':
                SELECTED.v -= 0.01
            elif key == 'm':
                SELECTED.v += 0.01
            print(f'{SELECTED.name} = {SELECTED.v:.2f}')
        else:
            for param in vars(params).values():
                if key == param.key:
                    if isinstance(param.v, float):
                        SELECTED = param
                        print(f'{param.name} selected')
                        break
                    elif isinstance(param.v, bool):
                        param.v = not param.v
            if key == 'v':
                print('Velocities Mode')
                params.velocities_mode.v = True
            elif key == 's':
                print('Manual Speed')
                params.velocities_mode.v = False
            elif key == 'c':
                print('Curvature Lookahead Mode')
                params.curvature_lookahead_mode.v = True
            elif key == 'l':
                print('Manual Lookahead')
                params.curvature_lookahead_mode.v = False

    def get_key(self):
        rlist, _, _ = select.select([sys.stdin], [], [], 0.005)
        if rlist:
            return sys.stdin.read(1)
        return None

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
        if FILE_OUTPUT:
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
        if depth <= 0.0:
            return 0.0
        return 2 * math.asin(span / (2 * depth))
    
class Path:
    def __init__(self, index, depth, sign):
        self.valid = True
        self.index = index
        self.depth = depth
        self.sign = sign
        self.angle = Scan.index_to_angle(index)
        self.vindex = 0
        self.vdepth = 0.0
        self.vangle = 0.0
            
class Planner:

    def __init__(self, disparity_threshold, extension, track_direction):
        self.disparity_threshold = disparity_threshold # threshold defining disparity
        self.extension = extension # wall extension distance
        self.track_direction = track_direction # ccw or cw
        self.vdisps = None

    def apply_range_limit(self, ranges, limit):
        safe_gap_value = 100.0
        ranges[ranges > limit] = safe_gap_value

    def get_virtual(self, ranges):
        extension = params.map_extension.v
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
    
    def disparities(self, ranges):
        diffs = np.diff(ranges)
        threshold = params.disparity_threshold.v
        pos_disp = np.flatnonzero(diffs >= threshold)
        neg_disp = np.flatnonzero(diffs <= -threshold)
        return (pos_disp, neg_disp + 1)
    
    def get_paths(self, ranges, virtual, pos_disps, neg_disps):
        # Satisfiability: 1. disp radius, 2. arc path, 3. minimum arc radius
        paths = ([Path(disp, ranges[disp], 1) for disp in pos_disps] +
                 [Path(disp, ranges[disp], -1) for disp in neg_disps])
        
        self.resolve_virtual(virtual, paths)
        
        self.resolve_radii(ranges, paths, params.map_extension.v)

        valid_paths = [path for path in paths if path.valid]

        return valid_paths
    
    def resolve_virtual(self, virtual, paths):
        pdisps, ndisps = self.disparities(virtual)
        vdisps = np.sort(np.concatenate((pdisps, ndisps)))
        for path in paths:
            diff = np.abs(vdisps - path.index)
            nearest = vdisps[np.argmin(diff)]
            neighborhood = vdisps[np.abs(vdisps - nearest) <= 10]
            goal = neighborhood[np.argmax(virtual[neighborhood])]
            path.vindex = goal
            path.vdepth = virtual[goal]
            path.vangle = Scan.index_to_angle(goal)
            self.vdisps = vdisps

    def resolve_radii(self, ranges, paths, min_radius):
        pass
        # disp = self.index
        # depth = self.depth
        # if self.sign > 0:
        #     start = disp + 1
        #     end = int(disp + 2 * Scan.span_to_angle(max_radius, depth) / Scan.angle_increment)
        #     if start >= end or start < 0 or end >= len(ranges) or end < 0:
        #         self.radius = 0.0
        #         return False
        # if self.sign < 0:
        #     end = disp - 1
        #     start = int(disp - 2 * Scan.span_to_angle(max_radius, depth) / Scan.angle_increment)
        #     if start >= end or start < 0 or end >= len(ranges) or end < 0:
        #         self.radius = 0.0
        #         return False
        # section = slice(start, end + 1)
        # section_ranges = ranges[section]
        # radii = np.sqrt(ranges[disp]**2 + section_ranges**2 - 2 * ranges[disp] * section_ranges)
        # min_radius = np.min(radii)
        # self.radius = min_radius
        # return True

    def choose(self, ranges, paths):
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
            path = self.check_section(ranges, section, paths)
            if path:
                self.section = section
                return path
        print('No path...')
        return None

    def check_section(self, ranges, section, paths):
        lo, hi = section
        disps = np.asarray([path.index for path in paths])
        mask = (disps >= lo) & (disps <= hi)
        if not np.any(mask):
            return None
        idx_map = np.flatnonzero(mask)
        section_disps = disps[mask]
        section_ranges = ranges[section_disps]
        section_idx = np.nanargmax(section_ranges)
        path = paths[idx_map[section_idx]]
        return path

class Steering:

    def __init__(self, method):
        if method == 'arc':
            self.get = self.get_arc_angle
        elif method == 'line':
            self.get = self.get_line_angle
        else:
            raise ValueError('Invalid steering method')
        
    def get_arc_angle(self, path):
        theta = Scan.index_to_angle(path.vindex)
        y = math.sin(theta)
        gamma = 2 * y / path.vdepth**2
        delta = np.arctan(Vehicle.wheelbase * gamma)
        angle = np.clip(delta, -Vehicle.max_steering_angle, Vehicle.max_steering_angle)
        return angle

    def get_line_angle(self, path):
        theta = Scan.index_to_angle(path.vindex)
        angle = np.clip(theta, -Vehicle.max_steering_angle, Vehicle.max_steering_angle)
        return angle

class Smoothing:
    def __init__(self, limit):
        self.limit = limit
        self.start_time = None

    def get(self, theta, speed):
        if self.start_time:
            dt = perf_counter() - self.start_time
        else:
            dt = 0.025
        self.start_time = perf_counter()

        # Low pass filter
        if False:
            alpha = self.tau/(self.tau + dt)
            theta = (alpha * self.prev_filtered) + ((1 - alpha) * theta)
            self.prev_filtered = theta

        # Velocity calulation
        theta_velocity = params.steering_velocity.v * theta
            
        # Slew rate (limit on rate of change)
        if False:
            delta = theta - self.theta
            max_step = self.slew_rate * dt
            if abs(delta) > max_step:
                theta = self.theta + (max_step if delta > 0 else -max_step)

        theta = np.clip(theta, -self.limit, self.limit)
        self.theta = theta
        return theta, theta_velocity
        
class SpeedController:

    def __init__(self, method, max_speed, a_slide, a_tip):
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
        if math.isnan(steering_angle) or not gap_depth:
            return 0.0
        return params.speed.v

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
                       
def create_file():
    if FILE_OUTPUT:
        for key, value in file_info.mean_times.items():
            mean_time = value[0] / value[1]
            file_info.mean_times[key] = mean_time
        with open('path_follow_info.txt', 'w', encoding='utf-8') as f:
            for key, subdict in file_info.__dict__.items():
                f.write(f'\n{key.upper()}\n')
                for subkey, value in subdict.items():
                    f.write(f'{subkey} : {value}\n')

def main(args=None):
    rclpy.init(args=args)
    node = PathFollow()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        create_file()
    finally:
        termios.tcsetattr(node.fd, termios.TCSADRAIN, node.terminal_settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

# def modify_ranges_for_disparities(ranges, virtual, pos_disps, neg_disps):
#         for disp in pos_disps:
#             # Walk gap until intersection
#             disp_range = ranges[disp]
#             points = np.flatnonzero((ranges[disp+1:] <= disp_range) | (virtual[disp+1:] > disp_range))
#             if points.size:
#                 left_intersect = points[0] + disp + 1
#             else:
#                 continue
#             # Drop to intersection range
#             if virtual[left_intersect] > disp_range:
#                 left_intersect -= 1
#                 new_ext = virtual[left_intersect]
#             else:
#                 new_ext = disp_range
#             # Backtrack until second intersection
#             points = np.flatnonzero((ranges[:disp] <= new_ext) | (virtual[:disp] > new_ext))
#             right_intersect = points[-1] if points.size else disp
#             if virtual[right_intersect] > new_ext:
#                 right_intersect += 1
#             ranges_section = ranges[right_intersect: left_intersect + 1]
#             np.minimum(ranges_section, new_ext, out=ranges_section)

#         for disp in neg_disps:
#             # Walk gap until intersection
#             disp_range = ranges[disp]
#             points = np.flatnonzero((ranges[:disp] <= disp_range) | (virtual[:disp] > disp_range))
#             if points.size:
#                 right_intersect = points[-1]
#             else:
#                 continue
#             # Drop to intersection range
#             if virtual[right_intersect] > disp_range:
#                 right_intersect += 1
#                 new_ext = virtual[right_intersect]
#             else:
#                 new_ext = disp_range
#             # Backtrack until second intersection
#             points = np.flatnonzero((ranges[disp+1:] <= new_ext) | (virtual[disp+1:] > new_ext))
#             left_intersect = points[0] + disp + 1 if points.size else disp
#             if virtual[left_intersect] > new_ext:
#                 left_intersect -= 1
#             ranges_section = ranges[right_intersect: left_intersect + 1]
#             np.minimum(ranges_section, new_ext, out=ranges_section)
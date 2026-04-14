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
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import ColorRGBA
from dataclasses import dataclass
from .io_utils import Binding, DualBinding, KeyBindings
from .disp_utils import Scan, get_virtual, nearest_object_intersect
from pathlib import Path as FilePath
import os
from .utils import Raceline
import pandas as pd
from numba import njit

# Criteria: use steering angle
# maximize distance seen when stable
# steer to center of space

map_path = os.environ.get('MAP_PATH')
if map_path is None:
        raise RuntimeError('MAP_PATH not set')
CSV_PATH = FilePath(map_path+'_raceline.csv')
if not CSV_PATH.exists():
    raise RuntimeError("Waypoint file doesn't exist")

@dataclass
class Config:
    simulation: bool = True
    ccw: bool = True
    # Algorithm parameters
    disparity_threshold: float = 0.5
    map_extension: float = 0.25
    steering_velocity: float = 0.0
    # Speed parameters
    speed_method: str = 'fast' # 'flat' or 'fast'
    max_speed: float = 100.0
    v_min: float = 1.0
    a_slide: float = 14.0
    a_tip: float = 10000.0
    # Vehicle parameters
    wheelbase: float = 0.33
    max_steer: float = 0.38
    # Output parameters
    viz_rate: float = 0.2
    file_output: bool = True
    publish_points1: bool = True
    publish_points2: bool = True
    publish_points3: bool = True
    publish_v1: bool = True
    publish_v2: bool = True
config = Config()

params = KeyBindings(
    acceleration=Binding('acceleration', 'a', 0.0),
    velocities_coeff=Binding('velocity coefficient', 'v', 0.1),
    velocities_mode=DualBinding('Velocities Mode', 'v', 's', False),
    disparity_threshold=Binding('disparity threshold', 't', 0.5),
    steering_velocity=Binding('steering velocity', 'w', 0.0),
    map_extension=Binding('map extension', 'e', 0.45),
)

@dataclass
class Path:
    index: int
    depth:np.float32
    sign: int
    angle: np.float32 = np.float32(0.0)
    vindex: int = 0
    vdepth: np.float32 = np.float32(0.0)
    vangle: np.float32 = np.float32(0.0)
    valid: bool = True

class DisparityFollow(Node):
    def __init__(self):
        super().__init__('gap_follow')
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.laser_scan_sub = self.create_subscription(LaserScan, '/scan', self.adjust, 10)
        if config.simulation:
            self.sub_odom = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback,  1)
        else:
            self.sub_pose = self.create_subscription(PoseStamped, '/pf/viz/inferred_pose', self.pose_callback, 1)
        # Accessory sub/pub
        self.viz_timer = self.create_timer(config.viz_rate, self.publish_markers)
        self.raceline_viz = self.create_publisher(Marker, '/viz/raceline', 10)
        self.keyboard_timer = self.create_timer(.2, params.check_input)
        self.print_timer = self.create_timer(1.0, self.print_info)
        self.line_marker_pub = self.create_publisher(Marker, '/viz/goal', 10)
        if config.publish_v1:
            self.v1_pub = self.create_publisher(Float32MultiArray, '/v1_ranges', 10)
        if config.publish_v2:
            self.v2_pub = self.create_publisher(Float32MultiArray, '/v2_ranges', 10)
        if config.publish_points1:
            self.points1_pub = self.create_publisher(Marker, '/viz/points1', 10)
        if config.publish_points2:
            self.points2_pub = self.create_publisher(Marker, '/viz/points2', 10)
        if config.publish_points3:
            self.points3_pub = self.create_publisher(Marker, '/viz/points3', 10)

        df = pd.read_csv(CSV_PATH, header=0, comment='#', sep=';')
        self.raceline = Raceline(df)
        if not config.ccw:
            self.raceline.reverse()
        
        self.path = None
        self.steering_angle: np.float32 = np.float32(0.0)
        self.speed: float = 0.0
        self.v1: np.ndarray
        self.v2: np.ndarray
        self.start_time = 0.0
        self.scan_flag = False
        self.ready_flag = False

        self.publish_raceline(self.raceline.x_ref, self.raceline.y_ref)
        
    def adjust(self, scan: LaserScan):
        if not self.scan_flag:
            self.scan_flag = True
            self.scan = Scan(scan)
            self.v1 = np.zeros(self.scan.size)
            self.v2 = np.zeros(self.scan.size)
        
        self.ranges = np.asarray(scan.ranges, dtype=np.float32)

        try:
        # =================== Pipeline ========================
            
            pipeline_start = perf_counter()

            # self.apply_range_limit(ranges, 10.0)

            get_virtual_start = perf_counter()
            self.virtual = get_virtual(self.ranges, np.float32(self.scan.increment), np.float32(params.map_extension.v))
            self.get_virtual_time = (perf_counter() - get_virtual_start) * 1_000

            pos_disps, neg_disps = self.disparities(self.ranges)

            self.paths = self.get_paths(pos_disps, neg_disps)

            self.path = self.choose(self.paths)

            steering_angle = self.get_steering(self.path)

            self.steering_angle, self.steering_velocity = self.get_smooth(steering_angle, self.speed)

            self.speed = self.get_speed(self.path)
            
            self.pipeline_time = (perf_counter() - pipeline_start) * 1_000

        # =====================================================
        except Exception as e:
            print(f'{e}')
            self.publish_drive(self.speed, 1.0, self.steering_angle, 1.0)
            return

        self.publish_drive(self.speed, 1.0, self.steering_angle, self.steering_velocity)
        self.v1 = self.ranges
        self.v2 = self.virtual

        self.ready_flag = True

    def print_info(self):
        if not self.ready_flag:
            return
        print(f'pipeline load {self.pipeline_time/0.025:.2f}%')
        print(f'virtual time load {self.get_virtual_time/0.025:.2f}%\n')

    def publish_drive(self, velocity, acceleration, steering_angle, steering_velocity):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = float(steering_angle)
        drive_msg.drive.steering_angle_velocity = steering_velocity
        drive_msg.drive.speed = velocity
        drive_msg.drive.acceleration = acceleration
        self.drive_pub.publish(drive_msg)

    def odom_callback(self, odom_info):
        self.pose = odom_info.pose.pose

    def pose_callback(self, pose_stamped):
        self.pose = pose_stamped.pose
    
    def apply_range_limit(self, ranges, limit):
        safe_gap_value = 100.0
        ranges[ranges > limit] = safe_gap_value
    
    def disparities(self, ranges):
        diffs = np.diff(ranges)
        threshold = params.disparity_threshold.v
        pos_disp = np.flatnonzero(diffs >= threshold)
        neg_disp = np.flatnonzero(diffs <= -threshold)
        return (pos_disp, neg_disp + 1)
    
    def get_paths(self, pos_disps: np.ndarray, neg_disps: np.ndarray):
        # Satisfiability: 1. disp radius, 2. arc path, 3. minimum arc radius
        paths = ([Path(disp, self.ranges[int(disp)], 1) for disp in pos_disps] +
                 [Path(disp, self.ranges[int(disp)], -1) for disp in neg_disps])
        
        self.resolve_virtual(self.virtual, paths)
        
        valid_paths = [path for path in paths if path.valid]

        if not valid_paths:
            raise RuntimeError('No valid paths found.')

        for path in valid_paths:
            path.angle = np.float32(self.scan.index_to_angle(path.index))
            path.vangle = np.float32(self.scan.index_to_angle(path.vindex))

        return valid_paths
    
    def resolve_virtual(self, virtual, paths):
        pdisps, ndisps = self.disparities(virtual)
        if pdisps.size == 0 and ndisps.size == 0:
            raise RuntimeError('No virtual disparities found.')
        vdisps = np.sort(np.concatenate((pdisps, ndisps)))
        for path in paths:
            diff = np.abs(vdisps - path.index)
            nearest = vdisps[np.argmin(diff)]
            neighborhood = vdisps[np.abs(vdisps - nearest) <= 10]
            goal = neighborhood[np.argmax(virtual[neighborhood])]
            path.vindex = goal
            path.vdepth = virtual[goal]
            path.vangle = self.scan.index_to_angle(goal)
            self.vdisps = vdisps
    
    def choose(self, paths):
        if not hasattr(self, 'pose'):
            raise RuntimeError('Pose not set yet')
        # Pick disp closest to raceline
        x_car = self.pose.position.x
        y_car = self.pose.position.y
        idx = nearest_object_intersect(
            self.scan.angles,
            self.ranges,
            np.vstack((self.raceline.x_ref, self.raceline.y_ref)),
            (x_car, y_car)
        )
        self.intersect_idx = idx
        self.intersect_range = self.ranges[idx]
        path_idxs = np.asarray([path.index for path in paths])
        nearest_path = paths[np.argmin(np.abs(path_idxs - idx))]
        if nearest_path == -1:
            raise RuntimeError('Nearest path failure')
        return nearest_path

    def get_steering(self, path):
        if path is None:
            return self.steering_angle
        theta = self.scan.index_to_angle(path.vindex)
        angle = np.clip(theta, -config.max_steer, config.max_steer)
        return angle

    def get_smooth(self, theta, speed):
        if self.start_time:
            dt = perf_counter() - self.start_time
        else:
            dt = 0.025
        self.start_time = perf_counter()

        # Velocity calulation
        theta_velocity = params.steering_velocity.v * theta**2
            
        theta = np.clip(theta, -config.max_steer, config.max_steer)
        self.theta = theta
        return theta, theta_velocity
    
    def get_speed(self, path):
        return self.flat(path.vangle, path.vdepth)
        
    def flat(self, steering_angle, gap_depth):
        if math.isnan(steering_angle) or not gap_depth:
            return 0.0
        return params.speed.v

    def fast(self, steering_angle, gap_depth):
        if math.isnan(steering_angle) or not gap_depth:
            print('FAILED SPEED')
            return 0.0
        v_min = config.v_min
        s = gap_depth
        if s < 0:
            print('Zero depth path...')
            return 0.0
        k = abs(math.tan(steering_angle)) / config.wheelbase
        k += 1e-6 # Prevent division by zero
        v_min_2 = v_min * v_min
        v_min_4 = v_min_2 * v_min_2
        s_2 = s * s
        k_2 = k * k
        p1 = 1 + 4*s_2*k_2
        p2 = config.a_slide**2 * p1 - k_2*v_min_4
        if p2 < 0:
            return 0.0
        p3 = 2*s*math.sqrt(config.a_slide**2 * p1 - k_2*v_min_4)
        diff = v_min_2 - p3
        if diff >= 0:
            v_tot = math.sqrt(diff/p1)
        else:
            v_tot = math.sqrt(-diff/p1)
        v_lat = math.sqrt(config.a_tip/k)
        speed = min(v_tot, v_lat, config.max_speed)
        self.last_speed = speed
        return speed
    
    def publish_raceline(self, x, y):
        raceline = Marker()
        raceline.header.frame_id = "map"
        raceline.id = 0
        raceline.type = Marker.POINTS
        raceline.action = Marker.ADD
        raceline.pose.orientation.w = 1.0
        raceline.scale.x = 0.1
        raceline.scale.y = 0.1
        raceline.color.a = 1.0
        raceline.color.r = 0.0
        raceline.color.g = 0.0
        raceline.color.b = 1.0
        raceline.points = [Point(x=float(x), y=float(y), z=0.0) for x, y in zip(x, y)]
        self.raceline_viz.publish(raceline)
    
    def publish_markers(self):
        if not self.ready_flag:
            return
        if hasattr(self, 'path') and hasattr(self.path, 'vdepth'):
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
        if config.publish_v1 and hasattr(self, 'v1'):
            msg = Float32MultiArray()
            msg.data = self.v1.tolist()
            self.v1_pub.publish(msg)
        if config.publish_v2 and hasattr(self, 'v2'):
            msg = Float32MultiArray()
            msg.data = self.v2.tolist()
            self.v2_pub.publish(msg)
        if config.publish_points1 and hasattr(self, 'paths'):
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
        if config.publish_points2 and hasattr(self, 'paths'):
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
        if config.publish_points3 and hasattr(self, 'intersect_idx'):
            points3 = [Point(x=self.intersect_range*math.cos(self.scan.index_to_angle(idx)),
                            y=self.intersect_range*math.sin(self.scan.index_to_angle(idx)),
                            z=0.0) for idx in [self.intersect_idx]]
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

def main(args=None):
    rclpy.init(args=args)
    node = DisparityFollow()
    try:
        rclpy.spin(node)
    finally:
        params.restore_terminal()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

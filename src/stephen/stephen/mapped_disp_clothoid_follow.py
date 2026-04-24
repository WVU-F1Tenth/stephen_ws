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
from .disp_utils import (Scan, get_virtual, nearest_object_intersect,
                         max_point_radius, radial_extension_to_path)
from pathlib import Path as FilePath
import os
from .utils import (Raceline, car_to_map, map_to_car, quat_to_yaw, threshold_index_cumulative,
                    idx_nearest_point)
import pandas as pd
from pyclothoids import Clothoid

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
    publish_points1: bool = True
    publish_points2: bool = True
    publish_points3: bool = True
    publish_points4: bool = True
config = Config()

if config.simulation:
    config.wheelbase = 0.28

params = KeyBindings(
    acceleration=Binding('acceleration', 'a', 0.0),
    velocities_coeff=Binding('velocity coefficient', 'v', 0.1),
    velocities_mode=DualBinding('Velocities Mode', 'v', 's', False),
    disparity_threshold=Binding('disparity threshold', 't', 0.5),
    steering_velocity=Binding('steering velocity', 'w', 0.0),
    map_extension=Binding('map extension', 'e', 0.45),
    lookahead=Binding('lookahead', 'l', 1.0),
    clothoid_lookahead=Binding('clothoid lookahead', 'q', 0.1),
    intersect_threshold=Binding('intersect threshold', 'x', 1.0)
)

@dataclass
class Path:
    sign: int
    index: int
    depth:np.float32
    angle: np.float32 = np.float32(0.0)
    vsign: int = 0
    vindex: int = 0
    vdepth: np.float32 = np.float32(0.0)
    vangle: np.float32 = np.float32(0.0)
    valid: bool = True
    ref_idx: int = -1

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
        self.print_timer = self.create_timer(5.0, self.print_info)
        self.line_marker_pub = self.create_publisher(Marker, '/viz/goal', 10)
        if config.publish_points1:
            self.points1_pub = self.create_publisher(Marker, '/viz/points1', 10)
        if config.publish_points2:
            self.points2_pub = self.create_publisher(Marker, '/viz/points2', 10)
        if config.publish_points3:
            self.points3_pub = self.create_publisher(Marker, '/viz/points3', 10)
        if config.publish_points4:
            self.points4_pub = self.create_publisher(Marker, '/viz/points4', 10)

        df = pd.read_csv(CSV_PATH, header=0, comment='#', sep=';')
        self.raceline = Raceline(df)
        if not config.ccw:
            self.raceline.reverse()
        
        self.path = None
        self.steering_angle: np.float32 = np.float32(0.0)
        self.speed: float = 0.0
        self.start_time = 0.0
        self.scan_flag = False
        self.ready_flag = False

        self.publish_raceline(self.raceline.x_ref, self.raceline.y_ref)
        
    def adjust(self, scan: LaserScan):
        if not self.scan_flag:
            self.scan_flag = True
            self.scan = Scan(scan)

        if not hasattr(self, 'pose'):
            print('Waiting on pose')
            return
        
        self.ranges = np.asarray(scan.ranges, dtype=np.float32)

        try:
        # =================== Pipeline ========================
            
            pipeline_start = perf_counter()

            # Car pose in map frame
            self.x_car, self.y_car, self.yaw_car = self.car_xyyaw()

            # self.apply_range_limit(ranges, 10.0)

            get_virtual_start = perf_counter()
            self.virtual = get_virtual(self.ranges, np.float32(self.scan.increment), np.float32(params.map_extension.v))
            self.get_virtual_time = (perf_counter() - get_virtual_start) * 1_000

            pos_disps, neg_disps = self.disparities(self.ranges)
            self.xr, self.xtheta = nearest_object_intersect(
            self.scan.angles,
            self.virtual,
            np.vstack((self.raceline.x_ref, self.raceline.y_ref)),
            (self.x_car, self.y_car, self.yaw_car)
            )
            # If intersect is close, choose path
            if self.xr < params.intersect_threshold.v:
                # Potential paths
                self.paths = self.get_paths(pos_disps, neg_disps)

                # Choose path
                self.path = self.choose(self.paths)

                # Steering for path
                delta = self.path_steering(self.path)
                self.steering_angle, self.steering_velocity = self.get_smooth(delta, self.speed)
                self.speed = self.get_path_speed(self.path)

            # Else intersect is far, steer using reference
            else:
                dx_car = self.raceline.x_ref - self.pose.position.x
                dy_car = self.raceline.y_ref - self.pose.position.y
                d_car = np.hypot(dx_car, dy_car)
                nearest_idx = np.argmin(d_car)
                dx = np.diff(self.raceline.x_ref_closed)
                dy = np.diff(self.raceline.y_ref_closed)
                d = np.hypot(dx, dy)
                goal_idx, _ = threshold_index_cumulative(d,
                                                      nearest_idx,
                                                      params.lookahead.v)
                x_goal_map = self.raceline.x_ref[goal_idx]
                y_goal_map = self.raceline.y_ref[goal_idx]
                yaw_goal_map = self.raceline.yaw_ref[goal_idx]
                x_goal_car, y_goal_car, yaw_goal_car = map_to_car(
                    x_goal_map, y_goal_map, yaw_goal_map, self.x_car, self.y_car, self.yaw_car)
                self.goal_x, self.goal_y = x_goal_car, y_goal_car
                c = Clothoid.G1Hermite(0.0, 0.0, self.steering_angle, x_goal_car, y_goal_car, yaw_goal_car)
                self.clothoid = c.SampleXY(20)
                clookahead = params.clothoid_lookahead.v
                x = c.X(clookahead)
                y = c.Y(clookahead)
                yaw = c.Theta(clookahead)
                delta = np.arctan2(y, x)
                self.steering_angle, self.steering_velocity = self.get_smooth(delta, self.speed)
                self.speed = self.get_ref_speed()

            self.pipeline_time = (perf_counter() - pipeline_start) * 1_000

        # =====================================================
        except RuntimeError as e:
            print(f'{e}')
            self.publish_drive(self.speed, 1.0, self.steering_angle, 1.0)
            return

        self.publish_drive(self.speed, 1.0, self.steering_angle, self.steering_velocity)

        self.ready_flag = True

    def print_info(self):
        if not self.ready_flag:
            return
        print(f'pipeline load {100*self.pipeline_time/0.025:.3f}%\n')
        print(f'virtual time load {100*self.get_virtual_time/0.025:.3f}%')

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

    def car_xyyaw(self):
        yaw = quat_to_yaw(self.pose.orientation)
        x = self.pose.position.x + config.wheelbase * math.cos(yaw)
        y = self.pose.position.y + config.wheelbase * math.sin(yaw)
        return np.float32(x), np.float32(y), np.float32(yaw)
    
    def disparities(self, ranges):
        diffs = np.diff(ranges)
        threshold = params.disparity_threshold.v
        pos_disp = np.flatnonzero(diffs >= threshold)
        neg_disp = np.flatnonzero(diffs <= -threshold)
        return (pos_disp, neg_disp + 1)
    
    def get_paths(self, pos_disps: np.ndarray, neg_disps: np.ndarray):
        paths = ([Path(1, disp, self.ranges[int(disp)]) for disp in pos_disps] +
                 [Path(-1, disp, self.ranges[int(disp)]) for disp in neg_disps])
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
        vdisps = np.concatenate((pdisps, ndisps))
        for path in paths:
            diff = np.abs(vdisps - path.index)
            nearest_idx = np.argmin(diff)
            vsign = 1 if nearest_idx < pdisps.size else -1
            nearest = vdisps[nearest_idx]
            neighborhood = vdisps[np.abs(vdisps - nearest) <= 10]
            goal = neighborhood[np.argmax(virtual[neighborhood])]
            path.vsign = vsign
            path.vindex = goal
            path.vdepth = virtual[goal]
            path.vangle = self.scan.index_to_angle(goal)
            self.vdisps = vdisps

    def choose(self, paths):
        # Pick disp closest to raceline
        path_angles = np.asarray([path.angle for path in paths])
        p = paths[np.argmin(np.abs(path_angles - self.xtheta))]
        # Find path info
        pvx = p.vdepth*np.cos(p.vangle)
        pvy = p.vdepth*np.sin(p.vangle)
        pvx, pvy, _ = car_to_map(pvx, pvy, 0.0, self.x_car, self.y_car, self.yaw_car)
        self.pvx = pvx
        self.pvy = pvy
        # Find ref nearest path
        dx = self.raceline.x_ref - pvx
        dy = self.raceline.y_ref - pvy
        ref_idx = np.argmin(np.hypot(dx, dy))
        p.ref_idx = ref_idx
        self.ref_idx = ref_idx
        return p

    def path_steering(self, path):
        # Transform to car frame
        ref_x, ref_y, ref_yaw = map_to_car(
            self.raceline.x_ref,
            self.raceline.y_ref,
            self.raceline.yaw_ref,
            self.x_car,
            self.y_car,
            self.yaw_car
        )
        v_r = path.vdepth
        v_theta = path.vangle
        self.pv_theta, self.pv_r = v_theta, v_r
        # Set goal points in car frame
        self.goal_x = v_r*np.cos(v_theta)
        self.goal_y = v_r*np.sin(v_theta)
        self.goal_yaw = ref_yaw[idx_nearest_point(
            self.goal_x,
            self.goal_y,
            ref_x,
            ref_y
        )]
        c = Clothoid.G1Hermite(0.0, 0.0, self.steering_angle, self.goal_x, self.goal_y, self.goal_yaw)
        self.clothoid = c.SampleXY(20)
        clookahead = params.clothoid_lookahead.v
        x = c.X(clookahead)
        y = c.Y(clookahead)
        yaw = c.Theta(clookahead)
        return np.arctan2(y, x)

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
    
    def get_ref_speed(self):
        return params.speed.v
    
    def get_path_speed(self, path):
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
        if hasattr(self, 'path') and hasattr(self, 'goal_yaw'):
            L = self.path.vdepth # type: ignore
            theta = self.steering_angle
            p0 = Point(x=0.0, y=0.0, z=0.0)
            p1 = Point(x=3*math.cos(self.goal_yaw), y=3*math.sin(self.goal_yaw), z=0.0)
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
        if config.publish_points1 and hasattr(self, 'ref_idx'):
            points1 = [Point(x=float(self.raceline.x_ref[self.ref_idx]),
                            y=float(self.raceline.y_ref[self.ref_idx]),
                            z=0.1)]
            m = Marker()
            m.header.frame_id = '/map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'points1'
            m.id = 0
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
            m.points = points1
            self.points1_pub.publish(m)
        if config.publish_points2 and hasattr(self, 'clothoid'):
            points2 = [Point(x=point[0],
                            y=point[1],
                            z=0.2) for point in zip(self.clothoid[0], self.clothoid[1])]
            m = Marker()
            m.header.frame_id = '/ego_racecar/laser'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'points2'
            m.id = 0
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = 0.1
            m.scale.y = 0.1
            m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            m.points = points2
            self.points2_pub.publish(m)
        if config.publish_points3 and hasattr(self, 'goal_x'):
            points3 = [Point(x=float(self.goal_x),
                             y=float(self.goal_y),
                             z=0.1)]
            m = Marker()
            m.header.frame_id = '/ego_racecar/laser'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'points3'
            m.id = 0
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            m.points = points3
            self.points3_pub.publish(m)
        if config.publish_points4 and hasattr(self, 'pv_theta'):
            points4 = [Point(x=float(self.pv_r*np.cos(self.pv_theta)),
                            y=float(self.pv_r*np.sin(self.pv_theta)),
                            z=0.2)]
            m = Marker()
            m.header.frame_id = '/ego_racecar/laser'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'points4'
            m.id = 0
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            m.points = points4
            self.points4_pub.publish(m)

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

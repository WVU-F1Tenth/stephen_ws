#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
import numpy as np
import pandas as pd
import tf2_ros
from pathlib import Path
import math
import os
from .io_utils import Binding, DualBinding, KeyBindings
from dataclasses import dataclass
from .utils import quat_to_yaw, RacelineSpline, Raceline
from scipy.interpolate import splprep, splev
from time import perf_counter

# Notes:
#   spline cw doesn't work
#   Angles are off, check car axes

map_path = os.environ.get('MAP_PATH')
if map_path is None:
        raise RuntimeError('MAP_PATH not set')
CSV_PATH = Path(map_path+'_raceline.csv')
if not CSV_PATH.exists():
    raise RuntimeError("Waypoint file doesn't exist")

@dataclass
class Config:
    simulation: bool = False
    ccw: bool = True
    wheelbase: float = 0.33
    max_steer: float = 0.33
    viz_rate: float = 5.0
    use_spline: bool = True
config = Config()

# Numeric parameters adjustable by keyboard
params = KeyBindings(
    lookahead=Binding('lookahead', 'l', 0.05),
    acceleration=Binding('acceleration', 'a', 0.0),
    velocities_coeff=Binding('velocities coefficient', 'v', 0.0),
    k_error=Binding('cross-track error gain', 'e', 0.5),
    k_heading=Binding('heading error gain', 'h', 0.6),
    velocities_mode=DualBinding('Velocities Mode', 'v', 's', False),
    proportional_lookahead=Binding('velocity proportional lookahead', 'p', False)
)

class Stanley(Node):
    def __init__(self):
        super().__init__('stanley_node')
        
        # Create ROS subscribers and publishers
        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.raceline_viz = self.create_publisher(Marker, '/viz/raceline', 10)
        self.goal_viz = self.create_publisher(Marker, '/viz/goal', 10)
        if config.simulation:
            self.sub_odom = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback,  1)
        else:
            self.sub_pose = self.create_subscription(PoseStamped, '/pf/viz/inferred_pose', self.pose_callback, 1)
        self.viz_timer = self.create_timer(1.0 / config.viz_rate, self.publish_markers)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.keyboard_timer = self.create_timer(.2, params.check_input)
        self.print_timer = self.create_timer(1.0, self.print_info)
        self.ready_flag = False

        self.angle = 0.0
        self.speed = 0.0
        
        df = pd.read_csv(CSV_PATH, header=0, comment='#', sep=';')
        raceline = Raceline(df)
        if not config.ccw:
            raceline.reverse()
            
        self.x_ref = raceline.x_ref
        self.y_ref = raceline.y_ref
        self.yaw_ref = raceline.yaw_ref
        self.velocities = raceline.v_ref
        # dists[0] = distance from point 0 to point 1
        self.dists = np.hypot(np.diff(self.x_ref), np.diff(self.y_ref))
        self.dist_sums = np.cumsum(np.append(self.dists, self.dists))
        u, self.tck = splprep((self.x_ref, self.y_ref))
        self.u_max = u[-1]

        if config.use_spline:
            self.raceline_spline = RacelineSpline(self.x_ref, self.y_ref, np.float32)
        
        self.publish_raceline(self.x_ref, self.y_ref)

    def print_info(self):
        if not self.ready_flag:
            return
        print(f'pipeline load = {100*self.callback_time/0.025:.3f}%')
        print(f'raceline conversion load = {100*self.race_conv_time/0.025:.3f}%')

    def odom_callback(self, odometry_info: Odometry):
        self.pose_callback(odometry_info.pose)

    def pose_callback(self, pose_stamped):
        callback_time_start = perf_counter()
        pose = pose_stamped.pose
        x_car_map = pose.position.x
        y_car_map = pose.position.y
        heading_car_map = quat_to_yaw(pose.orientation)

        lookahead = (params.lookahead.v * self.speed
                     if params.proportional_lookahead.v
                     else params.lookahead.v)

        # Project to front axle
        x_car_map = x_car_map + config.wheelbase * math.cos(heading_car_map)
        y_car_map = y_car_map + config.wheelbase * math.sin(heading_car_map)
        
        # Find goal point (arc length lookahead)
        if config.use_spline:
            # Find nearest point
            race_conv_start = perf_counter()
            nearest_s = self.raceline_spline.xy_to_s([x_car_map, y_car_map])
            goal_s = nearest_s + params.lookahead.v
            x_goal_map, y_goal_map = self.raceline_spline.s_to_xy(goal_s)
            heading_goal_map = self.raceline_spline.s_to_heading(goal_s)
            self.goal = (x_goal_map, y_goal_map)
            self.v_ref = 5.0
            self.race_conv_time = perf_counter() - race_conv_start
        else:
            # Find nearest point
            race_conv_start = perf_counter()
            dx = x_car_map - self.x_ref
            dy = y_car_map - self.y_ref
            d = np.hypot(dx, dy)
            self.nearest_index = np.argmin(d)
            relative_dists = self.dist_sums - self.dist_sums[self.nearest_index]
            goal_index = np.searchsorted(relative_dists, lookahead) % d.size
            # Get goal index params
            x_goal_map, y_goal_map = self.x_ref[goal_index], self.y_ref[goal_index]
            heading_goal_map = self.yaw_ref[goal_index]
            self.v_ref = self.velocities[goal_index]
            self.goal = self.x_ref[goal_index], self.y_ref[goal_index]
            self.race_conv_time = perf_counter() - race_conv_start

        # ===================================================================================

        # feedforward_term = 

        # yaw_damping = 

        if config.ccw:
            heading_error = math.atan2(
                math.cos(-(heading_goal_map - heading_car_map)),
                math.sin(-(heading_goal_map - heading_car_map))
            )
        else:
            heading_error = math.atan2(
                -math.cos(-(heading_goal_map - heading_car_map)),
                -math.sin(-(heading_goal_map - heading_car_map))
            )
        heading_term = params.k_heading.v * heading_error

        crosstrack_error = ((x_car_map - x_goal_map) * math.sin(heading_car_map) - 
                            (y_car_map - y_goal_map) * math.cos(heading_car_map))
        cross_track_term = math.atan2((params.k_error.v * crosstrack_error), (1.0 + self.speed))

        delta = heading_term + cross_track_term # + yaw_damping + feedforward_term

        # ===================================================================================

        self.angle = np.clip(delta, -config.max_steer, config.max_steer)

        self.speed = self.get_speed()

        self.publish_drive()

        self.callback_time = perf_counter() - callback_time_start
        self.ready_flag = True

    def get_speed(self):
        if params.velocities_mode.v:
            return params.velocities_coeff.v * self.v_ref
        else:
            return params.speed.v
        
    def publish_drive(self):
        ackermann_drive_result = AckermannDriveStamped()
        ackermann_drive_result.header.stamp = self.get_clock().now().to_msg()
        ackermann_drive_result.drive.steering_angle = self.angle
        ackermann_drive_result.drive.speed = self.speed
        acc = 0.0 if params.acceleration.v < 0.05 else params.acceleration.v
        ackermann_drive_result.drive.acceleration = acc
        self.pub_drive.publish(ackermann_drive_result)

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
        point = Point()
        goal_marker = Marker()
        point.x = float(self.goal[0])
        point.y = float(self.goal[1])
        goal_marker.points = []
        goal_marker.points.append(point)
        goal_marker.header.frame_id = "map"
        goal_marker.id = 1
        goal_marker.type = Marker.POINTS
        goal_marker.action = Marker.ADD
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = 0.2
        goal_marker.scale.y = 0.2
        goal_marker.color.a = 1.0
        goal_marker.color.r = 1.0
        self.goal_viz.publish(goal_marker)

def main(args=None):
    rclpy.init(args=args)
    print("Stanley Initialized")
    node = Stanley()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('Keyboard interrupt')
    finally:
        params.restore_terminal()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

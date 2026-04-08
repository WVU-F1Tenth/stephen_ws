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

map_path = os.environ.get('MAP_PATH')
if map_path is None:
        raise RuntimeError('MAP_PATH not set')
CSV_PATH = Path(map_path+'_raceline.csv')
if not CSV_PATH.exists():
    raise RuntimeError("Waypoint file doesn't exist")

@dataclass
class Config:
    simulation: bool = True
    ccw: bool = True
    wheelbase: float = 0.33
    max_steer: float = 0.33
    viz_rate: float = 5.0
config = Config()

# Numeric parameters adjustable by keyboard
params = KeyBindings(
    speed=Binding('speed', 's', 0.0),
    lookahead=Binding('lookahead', 'l', 0.05),
    acceleration=Binding('acceleration', 'a', 0.0),
    velocities_coeff=Binding('velocities coefficient', 'v', 0.0),
    k_error=Binding('cross-track error gain', 'e', 1.0),
    k_heading=Binding('heading error gain', 'h', 1.0),
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

        self.angle = 0.0
        self.speed = 0.0
        self.goal_index = 0
        
        df = pd.read_csv(CSV_PATH, header=0, comment='#', sep=';')
        self.waypoints_x = df.iloc[:, 1].to_numpy(dtype=float)
        self.waypoints_y = df.iloc[:, 2].to_numpy(dtype=float)
        self.waypoints_heading = df.iloc[:, 3].to_numpy(dtype=float)
        self.velocities = df.iloc[:, 5]
        # dists[0] = distance from point 0 to point 1
        self.dists = np.hypot(np.diff(self.waypoints_x), np.diff(self.waypoints_y))
        self.dist_sums = np.cumsum(np.append(self.dists, self.dists))
        
        self.publish_raceline(self.waypoints_x, self.waypoints_y)

    def odom_callback(self, odometry_info: Odometry):
        self.pose_callback(odometry_info.pose)

    def pose_callback(self, pose_stamped):
        pose = pose_stamped.pose
        x_car_map = pose.position.x
        y_car_map = pose.position.y
        heading_car_map = self.quaternion_to_heading(pose.orientation)
        lookahead = (params.lookahead.v * self.speed 
                     if params.proportional_lookahead.v
                     else params.lookahead.v)

        # Project to front axle
        x_car_map = x_car_map + config.wheelbase * math.cos(heading_car_map)
        y_car_map = y_car_map + config.wheelbase * math.sin(heading_car_map)

        # Find nearest point
        dx = x_car_map - self.waypoints_x
        dy = y_car_map - self.waypoints_y
        d = np.hypot(dx, dy)
        self.nearest_index = np.argmin(d)
        
        # Find goal point (arc length lookahead)
        relative_dists = self.dist_sums - self.dist_sums[self.nearest_index]
        self.goal_index = np.searchsorted(relative_dists, lookahead) % d.size
        
        # Get goal index params
        goal_index = self.goal_index
        x_goal_map, y_goal_map = self.waypoints_x[goal_index], self.waypoints_y[goal_index]
        heading_goal_map = self.waypoints_heading[goal_index]

        # ===================================================================================

        # feedforward_term = 

        # yaw_damping = 

        angle_diff = (heading_car_map - heading_goal_map)
        heading_error = math.atan2(math.cos(angle_diff), math.sin(angle_diff))
        heading_term = params.k_heading.v * heading_error

        crosstrack_error = ((x_car_map - x_goal_map) * math.sin(heading_car_map) - 
                            (y_car_map - y_goal_map) * math.cos(heading_car_map))
        cross_track_term = math.atan2((params.k_error.v * crosstrack_error), (1.0 + self.speed))

        delta = heading_term + cross_track_term # + yaw_damping + feedforward_term

        # ===================================================================================

        self.angle = np.clip(delta, -config.max_steer, config.max_steer)

        self.speed = self.get_speed()

        self.publish_drive()

    def get_speed(self):
        if params.velocities_mode.v:
            return params.velocities_coeff.v * self.velocities[self.goal_index]
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

    def map_to_car_point(self, car_pose, map_x, map_y):
        dx = map_x - car_pose.position.x
        dy = map_y - car_pose.position.y
        theta = self.quaternion_to_heading(car_pose.orientation)
        x_car =  math.cos(theta) * dx + math.sin(theta) * dy
        y_car = -math.sin(theta) * dx + math.cos(theta) * dy
        return x_car, y_car
        
    def quaternion_to_heading(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)
    
    def heading_to_quaternion(self, yaw):
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q
    
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
        point = Point()
        goal_marker = Marker()
        point.x = self.waypoints_x[self.goal_index]
        point.y = self.waypoints_y[self.goal_index]
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

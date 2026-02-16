#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PointStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
import numpy as np
import pandas as pd
import tf2_ros
from rclpy.duration import Duration
from rclpy.time import Time
from typing import Tuple
from pathlib import Path
import tf2_geometry_msgs
from rclpy.time import Time
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy
import os

map_path = os.environ.get('MAP_PATH')
if map_path is None:
        raise RuntimeError('MAP_PATH not set')
CSV_PATH = Path(map_path+'_raceline.csv')
if not CSV_PATH.exists():
    raise RuntimeError("Waypoint file doesn't exist")

LOOKAHEAD_DISTANCE = 1.20
WHEELBASE = 0.3
MAX_STEER = 0.42
VIS_RATE = 5.0

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        
        # Create ROS subscribers and publishers
        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.pub_env_viz = self.create_publisher(Marker, '/env_viz', 10)
        self.pub_dynamic_viz = self.create_publisher(Marker, '/dynamic_viz', 10)
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.sub_odom = self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback,  qos)
        self.vis_timer = self.create_timer(1.0 / VIS_RATE, self.publish_markers)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Reading CSV data
        self.path_marker = Marker()
        self.path_marker.header.frame_id = "map"
        self.path_marker.id = 0
        self.path_marker.type = Marker.POINTS
        self.path_marker.action = Marker.ADD
        self.path_marker.pose.position.x = 0.0
        self.path_marker.pose.position.y = 0.0
        self.path_marker.pose.position.z = 0.0
        self.path_marker.pose.orientation.x = 0.0
        self.path_marker.pose.orientation.y = 0.0
        self.path_marker.pose.orientation.z = 0.0
        self.path_marker.pose.orientation.w = 1.0
        self.path_marker.scale.x = 0.1
        self.path_marker.scale.y = 0.1
        self.path_marker.color.a = 1.0
        self.path_marker.color.r = 0.0
        self.path_marker.color.g = 0.0
        self.path_marker.color.b = 1.0
        self.path_published = False

        self.angle = 0.0
        self.x_odom = 0.0
        self.y_odom = 0.0
        self.heading_odom = 0.0

        df = pd.read_csv(CSV_PATH, header=None, comment='#', sep=',')
        self.waypoints_x = df.iloc[:, 0].to_numpy(dtype=float)
        self.waypoints_y = df.iloc[:, 1].to_numpy(dtype=float)
        self.waypoints_heading = None
        self.path_marker.points = [Point(x=float(x), y=float(y), z=0.0)
                              for x, y in zip(self.waypoints_x, self.waypoints_y)]

    def pose_callback(self, odometry_info):
        self.x_odom = odometry_info.pose.pose.position.x
        self.y_odom = odometry_info.pose.pose.position.y
        siny_cosp = 2.0 * (odometry_info.pose.pose.orientation.w * odometry_info.pose.pose.orientation.z + 
                           odometry_info.pose.pose.orientation.x * odometry_info.pose.pose.orientation.y)
        cosy_cosp = 1.0 - 2.0 * (odometry_info.pose.pose.orientation.y * odometry_info.pose.pose.orientation.y + 
                                 odometry_info.pose.pose.orientation.z * odometry_info.pose.pose.orientation.z)
        self.heading_odom = np.arctan2(siny_cosp, cosy_cosp)
        
        # Get goal point
        x_map, y_map = self.x_odom, self.y_odom
        dx = x_map - self.waypoints_x
        dy = y_map - self.waypoints_y
        d = np.hypot(dx, dy)
        start_index = np.argmin(d)
        for i in range(d.size):
            if d[(start_index + i) % d.size] > LOOKAHEAD_DISTANCE:
                break
        if i == d.size - 1:
            raise RuntimeError('Exhausted waypoints')
        self.goal_index = (start_index + i) % d.size

        # Transform goal point to vehicle frame of reference
        x_goal_base_link, y_goal_base_link = self.tfxy(
            'map', 'ego_racecar/base_link', self.waypoints_x[self.goal_index], self.waypoints_y[self.goal_index])
        if x_goal_base_link is None:
            return

        # Calculate curvature/steering angle
        L = np.hypot(x_goal_base_link, y_goal_base_link)
        gamma = 2*y_goal_base_link/L**2
        delta = np.arctan(WHEELBASE*gamma)
        self.angle = np.clip(delta, -MAX_STEER, MAX_STEER)

        self.reactive_control()
        
    def publish_markers(self):
        # Visualization marker for current goal point
        point = Point()
        goal_marker = Marker()
        point.x = self.waypoints_x[self.goal_index]
        point.y = self.waypoints_y[self.goal_index]
        point.z = 0.0
        goal_marker.points.append(point) # type: ignore
        goal_marker.header.frame_id = "map"
        goal_marker.id = 1
        goal_marker.type = Marker.POINTS
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = 0.0
        goal_marker.pose.position.y = 0.0
        goal_marker.pose.position.z = 0.0
        goal_marker.pose.orientation.x = 0.0
        goal_marker.pose.orientation.y = 0.0
        goal_marker.pose.orientation.z = 0.0
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = 0.2
        goal_marker.scale.y = 0.2
        goal_marker.color.a = 1.0
        goal_marker.color.r = 1.0
        goal_marker.color.g = 0.0
        goal_marker.color.b = 0.0

        if not self.path_published:
            self.pub_env_viz.publish(self.path_marker)
            self.path_published = True
        self.pub_dynamic_viz.publish(goal_marker)

    def reactive_control(self):
        ackermann_drive_result = AckermannDriveStamped()
        ackermann_drive_result.header.stamp = self.get_clock().now().to_msg()
        ackermann_drive_result.drive.steering_angle = self.angle
        if abs(self.angle) > np.radians(20.0):
            ackermann_drive_result.drive.speed = 2.0
        elif abs(self.angle) > np.radians(10.0):
            ackermann_drive_result.drive.speed = 4.0
        else:
            ackermann_drive_result.drive.speed = 6.0
        self.pub_drive.publish(ackermann_drive_result)

    def tfxy(self, from_frame, to_frame, x_from: float, y_from: float) -> Tuple[float, float]:
        try:
            p1 = PointStamped()
            p1.header.frame_id = from_frame
            p1.header.stamp = Time().to_msg()
            p1.point.x = float(x_from)
            p1.point.y = float(y_from)
            p1.point.z = 0.0
            p2 = self.tf_buffer.transform(
                p1,
                to_frame,
                timeout=Duration(seconds=0.05), # type: ignore
            )
            return p2.point.x, p2.point.y
        except tf2_ros.TransformException as e: # type: ignore
            self.get_logger().warn(f"TF unavailable: {e}")
            return None, None # type: ignore

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)
    pure_pursuit_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

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

LOOKAHEAD_DISTANCE = 1.20
CSV_PATH = Path(__file__).resolve().parent.parent / 'pursuit' / 'data.csv'
if not CSV_PATH.exists():
    raise RuntimeError("Waypoint file doesn't exist")
WHEELBASE = 1.2
MAX_STEER = .42

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        
        # Create ROS subscribers and publishers
        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.pub_env_viz = self.create_publisher(Marker, '/env_viz', 10)
        self.pub_dynamic_viz = self.create_publisher(Marker, '/dynamic_viz', 10)
        self.sub_odom = self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Reading CSV data
        self.marker = Marker()
        self.marker.header.frame_id = "map"
        self.marker.id = 0
        self.marker.type = Marker.POINTS
        self.marker.action = Marker.ADD
        self.marker.pose.position.x = 0.0
        self.marker.pose.position.y = 0.0
        self.marker.pose.position.z = 0.0
        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0
        self.marker.scale.x = 0.1
        self.marker.scale.y = 0.1
        self.marker.color.a = 1.0
        self.marker.color.r = 0.0
        self.marker.color.g = 0.0
        self.marker.color.b = 1.0

        df = pd.read_csv(CSV_PATH, header=None)
        self.waypoints_x = df.iloc[:, 0].to_numpy(dtype=float)
        self.waypoints_y = df.iloc[:, 1].to_numpy(dtype=float)
        self.waypoints_heading = df.iloc[:, 2].to_numpy(dtype=float)
        self.marker.points = [Point(x=float(x), y=float(y), z=0.0)
                              for x, y in zip(self.waypoints_x, self.waypoints_y)]

        self.angle = 0.0
        self.x_odom = 0.0
        self.y_odom = 0.0
        self.heading_odom = 0.0
        self.flag = False

    def pose_callback(self, odometry_info):
        self.x_odom = odometry_info.pose.pose.position.x
        self.y_odom = odometry_info.pose.pose.position.y
        siny_cosp = 2.0 * (odometry_info.pose.pose.orientation.w * odometry_info.pose.pose.orientation.z + 
                           odometry_info.pose.pose.orientation.x * odometry_info.pose.pose.orientation.y)
        cosy_cosp = 1.0 - 2.0 * (odometry_info.pose.pose.orientation.y * odometry_info.pose.pose.orientation.y + 
                                 odometry_info.pose.pose.orientation.z * odometry_info.pose.pose.orientation.z)
        self.heading_current = np.arctan2(siny_cosp, cosy_cosp)
        
        # Get goal point
        x_map, y_map = self.tfxy('odom', 'map', self.x_odom, self.y_odom)
        dx = x_map - self.waypoints_x
        dy = y_map - self.waypoints_y
        d = np.hypot(dx, dy)
        i = np.argmin(d)
        while d[i] < LOOKAHEAD_DISTANCE:
            i += 1
        goal_index = i

        # Transform goal point to vehicle frame of reference
        x_goal_base_link, y_goal_base_link = self.tfxy(
            'map', 'base_link', self.waypoints_x[i], self.waypoints_y[i])

        # Calculate curvature/steering angle
        L = np.hypot(x_goal_base_link, y_goal_base_link)
        gamma = 2*abs(y_goal_base_link)/L**2
        delta = np.arctan(WHEELBASE*gamma)
        self.angle = np.clip(delta, -MAX_STEER, MAX_STEER)
        
        # Visualization marker for current goal point
        point = Point()
        marker_2 = Marker()
        point.x = self.waypoints_x[goal_index]
        point.y = self.waypoints_y[goal_index]
        point.z = 0.0
        marker_2.points.append(point)
        marker_2.header.frame_id = "map"
        marker_2.id = 0
        marker_2.type = Marker.POINTS
        marker_2.action = Marker.ADD
        marker_2.pose.position.x = 0.0
        marker_2.pose.position.y = 0.0
        marker_2.pose.position.z = 0.0
        marker_2.pose.orientation.x = 0.0
        marker_2.pose.orientation.y = 0.0
        marker_2.pose.orientation.z = 0.0
        marker_2.pose.orientation.w = 1.0
        marker_2.scale.x = 0.2
        marker_2.scale.y = 0.2
        marker_2.color.a = 1.0
        marker_2.color.r = 1.0
        marker_2.color.g = 0.0
        marker_2.color.b = 0.0

        self.reactive_control()
        self.pub_env_viz.publish(self.marker)
        self.pub_dynamic_viz.publish(marker_2)
        #self.get_logger().info('Published marker with current point ({}, {})'.format(point.x, point.y))


    # TODO: publish drive message, don't forget to limit the steering angle.

    def reactive_control(self):
        ackermann_drive_result = AckermannDriveStamped()
        ackermann_drive_result.drive.steering_angle = self.angle
        if abs(self.angle) > np.radians(20.0):
            ackermann_drive_result.drive.speed = 0.5
        elif abs(self.angle) > np.radians(10.0):
            ackermann_drive_result.drive.speed = 1.0
        else:
            ackermann_drive_result.drive.speed = 1.5
        self.pub_drive.publish(ackermann_drive_result)

    def tfxy(self, from_frame, to_frame, x_odom: float, y_odom: float) -> Tuple[float, float]:
        p1 = PointStamped()
        p1.header.frame_id = from_frame
        p1.header.stamp = Time().to_msg()
        p1.point.x = float(x_odom)
        p1.point.y = float(y_odom)
        p1.point.z = 0.0
        p2 = self.tf_buffer.transform(
            p1,
            to_frame,
            timeout=Duration(nanoseconds=200_000_000),
        )
        return p2.point.x, p2.point.y

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)
    pure_pursuit_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

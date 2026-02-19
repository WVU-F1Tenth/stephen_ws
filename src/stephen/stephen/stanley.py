#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PointStamped, TransformStamped, Quaternion
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
MAX_STEER = 0.38
VIS_RATE = 5.0

K_ERROR = 1.0 # Cross-track error gain - main error
K_HEADING = 1.0 # Heading gain - helps smooth higher speeds
K_SOFTENING = 1.0 # Softening contant - helps smooth low speeds
K_DAMPING = 1.0 # Alternative to heading gain - helps smooth high speeds

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
        self.speed = 0.0

        df = pd.read_csv(CSV_PATH, header=None, comment='#', sep=',')
        self.waypoints_x = df.iloc[:, 0].to_numpy(dtype=float)
        self.waypoints_y = df.iloc[:, 1].to_numpy(dtype=float)
        self.headings = df.iloc[:, 2].tonumpy(dtype=float)
        self.path_marker.points = [Point(x=float(x), y=float(y), z=0.0)
                              for x, y in zip(self.waypoints_x, self.waypoints_y)]
        
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

    def pose_callback(self, odometry_info):
        self.x_car_odom = odometry_info.pose.pose.position.x
        self.y_car_odom = odometry_info.pose.pose.position.y
        self.heading_car_odom = self.quaternion_to_heading(odometry_info.pose.pose.orientation)
        
        self.speed = self.get_speed()

        # Get goal point
        x_car_map, y_car_map = self.x_car_odom, self.y_car_odom
        dx = x_car_map - self.waypoints_x
        dy = y_car_map - self.waypoints_y
        distances = np.hypot(dx, dy)
        start_index = np.argmin(distances)
        for i in range(distances.size):
            if distances[(start_index + i) % distances.size] > LOOKAHEAD_DISTANCE:
                break
        if i == distances.size - 1:
            raise RuntimeError('Exhausted waypoints')
        self.goal_index = (start_index + i) % distances.size

        # Transform goal point to vehicle frame of reference
        result = self.transform(
            'map',
            'ego_racecar/laser',
            self.waypoints_x[self.goal_index],
            self.waypoints_y[self.goal_index],
            self.headings[self.goal_index]
            )
        if result is None:
            return
        x_goal_laser, y_goal_laser, heading_goal_laser = result
        
        # Transform to front axle
        goal_heading = self.headings[self.goal_index]
        error = d[self.goal_index]

        feedforward_term = math.atan(WHEELBASE * goal_heading)

        heading_term = K_HEADING * (self.heading_odom - goal_heading)

        cross_track_term = math.atan((K_ERROR * error)/(K_SOFTENING + self.speed))

        # yaw_damping = -K_DAMPING * YAW_RATE
        
        delta = feedforward_term + heading_term + cross_track_term # + yaw_damping

        self.angle = np.clip(delta, -MAX_STEER, MAX_STEER)

        self.publish_drive()
        
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

    def get_speed(self):
        if abs(self.angle) > np.radians(20.0):
            return 2.0
        elif abs(self.angle) > np.radians(10.0):
            return 4.0
        else:
            return 6.0

    def publish_drive(self):
        ackermann_drive_result = AckermannDriveStamped()
        ackermann_drive_result.header.stamp = self.get_clock().now().to_msg()
        ackermann_drive_result.drive.steering_angle = self.angle
        ackermann_drive_result.drive.speed = self.speed
        self.pub_drive.publish(ackermann_drive_result)

    def transform(self, from_frame, to_frame, x_from, y_from, yaw):
        try:
            t1 = TransformStamped()
            t1.header.frame_id = from_frame
            t1.header.stamp = Time().to_msg()
            t1.transform.translation.x = float(x_from)
            t1.transform.translation.y = float(y_from)
            t1.transform.translation.z = 0.0
            quat = self.heading_to_quaternion(yaw)
            t1.transform.rotation.z = quat.z
            t1.transform.rotation.w = quat.w
            t2 = self.tf_buffer.transform(
                t1,
                to_frame,
                timeout=Duration(seconds=0.05), # type: ignore
            )
            return 
        except tf2_ros.TransformException as e: # type: ignore
            self.get_logger().warn(f"TF unavailable: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)
    pure_pursuit_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

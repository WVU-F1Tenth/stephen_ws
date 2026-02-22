#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion
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
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy
import os

map_path = os.environ.get('MAP_PATH')
if map_path is None:
        raise RuntimeError('MAP_PATH not set')
CSV_PATH = Path(map_path+'_raceline.csv')
if not CSV_PATH.exists():
    raise RuntimeError("Waypoint file doesn't exist")

SIMULATOR = True

LOOKAHEAD = 1.20
WHEELBASE = 0.3
MAX_STEER = 0.38
VIS_RATE = 5.0

K_ERROR = 1.0 # Cross-track error gain - main error
K_HEADING = 0.5 # Heading gain - helps smooth higher speeds
K_SOFTENING = 1.0 # Softening contant - helps smooth low speeds
K_DAMPING = 1.0 # Alternative to heading gain - helps smooth high speeds

class Stanley(Node):
    def __init__(self):
        super().__init__('stanley_node')
        
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
        self.speed = 0.0

        self.x_car_odom = 0.0
        self.y_car_odom = 0.0
        self.heading_car_odom = 0.0
        self.goal_index = 0
        
        df = pd.read_csv(CSV_PATH, header=None, comment='#', sep=',')
        self.waypoints_x = df.iloc[:, 0].to_numpy(dtype=float)
        self.waypoints_y = df.iloc[:, 1].to_numpy(dtype=float)
        self.waypoints_heading = df.iloc[:, 2].to_numpy(dtype=float)
        self.path_marker.points = [Point(x=float(x), y=float(y), z=0.0)
                              for x, y in zip(self.waypoints_x, self.waypoints_y)]

    def pose_callback(self, odometry_info):
        self.x_car_odom = odometry_info.pose.pose.position.x
        self.y_car_odom = odometry_info.pose.pose.position.y
        self.heading_car_odom = self.quaternion_to_heading(odometry_info.pose.pose.orientation)

        if SIMULATOR:
            x_car_map, y_car_map = self.x_car_odom, self.y_car_odom
            heading_car_map = self.heading_car_odom
        else:
            map_car_point = self.transform(
                'odom',
                'map',
                self.x_car_odom,
                self.y_car_odom,
                self.heading_car_odom
                )
            if map_car_point is None:
                return
            x_car_map, y_car_map, heading_car_map = map_car_point # type: ignore

        # Project to front axle
        x_car_map = x_car_map + WHEELBASE * math.cos(heading_car_map)
        y_car_map = y_car_map + WHEELBASE * math.sin(heading_car_map)
    
        dx = x_car_map - self.waypoints_x
        dy = y_car_map - self.waypoints_y
        distances = np.hypot(dx, dy)
        goal_index = (np.argmin(distances) + 5) % len(self.waypoints_x)
        # for i in range(distances.size):
        #     if distances[(start_index + i) % distances.size] > LOOKAHEAD_DISTANCE:
        #         break
        # if i == distances.size - 1:
        #     raise RuntimeError('Exhausted waypoints')
        # self.goal_index = (start_index + i) % distances.size

        self.goal_index = goal_index
        x_goal_map, y_goal_map = self.waypoints_x[goal_index], self.waypoints_y[goal_index]
        heading_goal_map = self.waypoints_heading[goal_index]

        crosstrack_error = (-(x_goal_map - x_car_map) * math.sin(heading_goal_map) + 
                            (y_goal_map - y_car_map) * math.cos(heading_goal_map))
        
        # feedforward_term = heading_goal_laser

        heading_error = self.normalize_angle(heading_goal_map - heading_car_map)
        heading_term = K_HEADING * heading_error

        cross_track_term = math.atan2((K_ERROR * crosstrack_error), (K_SOFTENING + self.speed))

        # yaw_damping = -K_DAMPING * YAW_RATE
        
        delta = heading_term + cross_track_term # + yaw_damping + feedforward_term

        self.angle = np.clip(delta, -MAX_STEER, MAX_STEER)

        self.speed = self.get_speed()

        self.publish_drive()
        
    def publish_markers(self):
        # Visualization marker for current goal point
        point = Point()
        goal_marker = Marker()
        point.x = self.waypoints_x[self.goal_index]
        point.y = self.waypoints_y[self.goal_index]
        point.z = 0.0
        goal_marker.points = []
        goal_marker.points.append(point)
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
            p1 = PoseStamped()
            p1.header.frame_id = from_frame
            p1.header.stamp = Time().to_msg()
            p1.pose.position.x = float(x_from)
            p1.pose.position.y = float(y_from)
            p1.pose.position.z = 0.0
            quat = self.heading_to_quaternion(yaw)
            p1.pose.orientation.z = quat.z
            p1.pose.orientation.w = quat.w
            p2 = self.tf_buffer.transform(
                p1,
                to_frame,
                timeout=Duration(seconds=0.05), # type: ignore
            )
            heading = self.quaternion_to_heading(p2.pose.orientation)
            x, y = p2.pose.position.x, p2.pose.position.y
            return x, y, heading
        except tf2_ros.TransformException as e: # type: ignore
            self.get_logger().warn(f"TF unavailable: {e}")
            return None
        
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
    
    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

def main(args=None):
    rclpy.init(args=args)
    print("Stanley Initialized")
    stanley_node = Stanley()
    rclpy.spin(stanley_node)
    stanley_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

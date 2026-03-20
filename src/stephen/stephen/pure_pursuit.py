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
import select
import termios
import sys
import tty
from types import SimpleNamespace

map_path = os.environ.get('MAP_PATH')
if map_path is None:
        raise RuntimeError('MAP_PATH not set')
CSV_PATH = Path(map_path+'_raceline.csv')
if not CSV_PATH.exists():
    raise RuntimeError("Waypoint file doesn't exist")

SIMULATOR = True
VELOCITIES=True

LOOKAHEAD = 0.8
WHEELBASE = 0.33
MAX_STEER = 0.38
VIS_RATE = 5.0

params = SimpleNamespace(
    speed=SimpleNamespace(v=0.0, keys=('s', 'd')),
    )

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        
        # Create ROS subscribers and publishers
        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.raceline_viz = self.create_publisher(Marker, '/viz/raceline', 10)
        self.goal_viz = self.create_publisher(Marker, '/viz/goal', 10)
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.sub_odom = self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback,  qos)
        self.viz_timer = self.create_timer(1.0 / VIS_RATE, self.publish_markers)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.keyboard_timer = self.create_timer(.2, self.check_input)

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
        self.goal_index = 0
        self.nearest_index = 0
        
        df = pd.read_csv(CSV_PATH, header=0, comment='#', sep=';')
        self.waypoints_x = df.iloc[:, 1].to_numpy(dtype=float)
        self.waypoints_y = df.iloc[:, 2].to_numpy(dtype=float)
        if VELOCITIES:
            self.velocities = df.iloc[:, 5].to_numpy(dtype=float)
        self.waypoints_heading = None
        self.path_marker.points = [Point(x=float(x), y=float(y), z=0.0)
                              for x, y in zip(self.waypoints_x, self.waypoints_y)]
        
        print('Command (space=stop, sd=speed)')
        self.fd = sys.stdin.fileno()
        self.terminal_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def pose_callback(self, odometry_info):
        x_car_odom = odometry_info.pose.pose.position.x
        y_car_odom = odometry_info.pose.pose.position.y
        heading_car_odom = self.quaternion_to_heading(odometry_info.pose.pose.orientation)

        if SIMULATOR:
            x_car_map, y_car_map = x_car_odom, y_car_odom
            heading_car_map = heading_car_odom
        else:
            point_car_map = self.transform(
                'odom',
                'map',
                x_car_odom,
                y_car_odom,
                heading_car_odom
                )
            if point_car_map is None:
                return
            x_car_map, y_car_map, heading_car_map = point_car_map # type: ignore

        # ===================================================================================

        # Find nearest point
        dx = x_car_map - self.waypoints_x
        dy = y_car_map - self.waypoints_y
        d = np.hypot(dx, dy)
        start_index = np.argmin(d)
        self.nearest_index = start_index
        for i in range(d.size):
            if d[(start_index + i) % d.size] > LOOKAHEAD:
                break
        if i == d.size - 1:
            raise RuntimeError('Exhausted waypoints')
        self.goal_index = (start_index + i) % d.size

        # Transform goal point to vehicle frame of reference
        point_goal_car = self.transform(
            'map', 
            'ego_racecar/base_link', 
            self.waypoints_x[self.goal_index], 
            self.waypoints_y[self.goal_index],
            0.0)
        if point_goal_car is None:
            return
        x_goal_car, y_goal_car, heading_goal_car = point_goal_car # type: ignore

        # Calculate curvature/steering angle
        L = np.hypot(x_goal_car, y_goal_car)
        gamma = 2*y_goal_car/L**2
        delta = np.arctan(WHEELBASE*gamma)

        # ===================================================================================

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
            self.raceline_viz.publish(self.path_marker)
            self.path_published = True
        self.goal_viz.publish(goal_marker)

    def get_speed(self): 
        if VELOCITIES and params.speed.v > 0.0:
            return self.velocities[self.nearest_index]
        else:
            return params.speed.v

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

    def check_input(self):
        global params
        key = self.get_key()
        if not key:
            return
        if key == ' ':
            params.speed.v = 0.0
            print('stopped')
        for name, d in vars(params).items():
            if d.keys is None:
                continue
            k1, k2 = d.keys
            if key == k1:
                d.v -= 0.1
            elif key == k1.upper():
                d.v -= 1.0
            elif key == k2:
                d.v += 0.1
            elif key == k2.upper():
                d.v += 1.0
            else:
                continue
            print(name, '=', d.v)
            break

    def get_key(self):
        rlist, _, _ = select.select([sys.stdin], [], [], 0.005)
        if rlist:
            return sys.stdin.read(1)
        return None

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    node = PurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('Keyboard interrupt')
    finally:
        termios.tcsetattr(node.fd, termios.TCSADRAIN, node.terminal_settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan
import numpy as np
import pandas as pd
from pathlib import Path
import os
from .io_utils import Binding, DualBinding, KeyBindings
from dataclasses import dataclass
from .utils import quat_to_yaw, RacelineSpline, Raceline
from scipy.interpolate import splprep, splev
from time import perf_counter


# Numeric parameters adjustable by keyboard
params = KeyBindings(
    steering_angle=Binding('steering angle', 'd', 0.0),
    acceleration=Binding('acceleration', 'a', 0.0),
)

class Testing(Node):
    def __init__(self):
        super().__init__('testing_node')
        # Create ROS subscribers and publishers
        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 4)
        self.laser_scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_handler, 1)
        self.keyboard_timer = self.create_timer(.2, params.check_input)
        
    def scan_handler(self, scan):
        self.publish_drive()

    def publish_drive(self):
        ackermann_drive_result = AckermannDriveStamped()
        ackermann_drive_result.header.stamp = self.get_clock().now().to_msg()
        steering_angle = np.clip(params.steering_angle.v, -0.38, 0.38)
        ackermann_drive_result.drive.steering_angle = steering_angle
        ackermann_drive_result.drive.speed = params.speed.v
        acc = 0.0 if params.acceleration.v < 0.05 else params.acceleration.v
        ackermann_drive_result.drive.acceleration = acc
        self.pub_drive.publish(ackermann_drive_result)
            
def main(args=None):
    rclpy.init(args=args)
    print("Testing Initialized")
    node = Testing()
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

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
import numpy as np
import pandas as pd
from pathlib import Path
import os
from .utils import threshold_index_cumulative
from time import perf_counter
from dataclasses import dataclass
from .io_utils import Binding, DualBinding, KeyBindings
from .utils import quat_to_heading, map_to_car_point, Raceline

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

params = KeyBindings(
    lookahead=Binding('lookahead', 'l', 0.8),
    acceleration=Binding('acceleration', 'a', 0.0),
    curvature_lookahead=Binding('curvature lookahead', 'c', 0.2),
    velocities_coeff=Binding('velocity coefficient', 'v', 0.1),
    velocities_mode=DualBinding('Velocities Mode', 'v', 's', False),
    curvature_lookahead_mode=DualBinding('Curvature Lookhead Mode', 'c', 'l', False),
)

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        
        # Publishers and Subscribers
        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.raceline_viz = self.create_publisher(Marker, '/viz/raceline', 10)
        self.goal_viz = self.create_publisher(Marker, '/viz/goal', 10)
        if config.simulation:
            self.sub_odom = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback,  1)
        else:
            self.sub_pose = self.create_subscription(PoseStamped, '/pf/viz/inferred_pose', self.pose_callback, 1)
        self.viz_timer = self.create_timer(1.0 / config.viz_rate, self.publish_markers)
        self.keyboard_timer = self.create_timer(.2, params.check_input)

        self.angle = 0.0
        self.speed = 0.0
        self.goal_index = 0
        self.nearest_index = 0
        
        # Track attribues
        df = pd.read_csv(CSV_PATH, header=0, comment='#', sep=';')
        self.track = Raceline(df)
        if not config.ccw:
            self.track.reverse()

        # dists[0] = distance from point 0 to point 1
        self.dists = np.hypot(np.diff(self.track.x_ref_closed), np.diff(self.track.y_ref_closed))
        self.raceline_spacing = np.mean(self.dists)
        # Cumulative distances
        self.dist_sums = np.cumsum(np.append(self.dists, self.dists))
        # Curvature
        self.abs_weighted_curvatures = np.abs(self.track.k_ref * self.raceline_spacing)

        self.publish_raceline(self.track.x_ref, self.track.y_ref)

    def odom_callback(self, odometry_info: Odometry):
        self.pose_callback(odometry_info.pose)

    def pose_callback(self, pose_stamped):
        pose = pose_stamped.pose
        x_car_map = pose.position.x
        y_car_map = pose.position.y
        heading_car_map = quat_to_heading(pose.orientation)

        # ===================================================================================

        # Find nearest point
        dx = x_car_map - self.track.x_ref
        dy = y_car_map - self.track.y_ref
        d = np.hypot(dx, dy)
        self.nearest_index = np.argmin(d)

        # Find lookahead relative goal
        if params.curvature_lookahead_mode.v:
            lookahead_starttime = perf_counter()
            x = 1
            if x == 1:
                threshold = params.curvature_lookahead.v
                sum = 0
                nearest = self.nearest_index
                offset = 0
                while (sum <= threshold and offset < self.track.point_count):
                    index = (nearest + offset) % self.track.point_count
                    sum += self.abs_weighted_curvatures[index]
                    offset += 1
            elif x == 2:
                _, offset = threshold_index_cumulative(
                    self.abs_weighted_curvatures, self.nearest_index, params.curvature_lookahead.v)
            if  self.goal_index == self.nearest_index:
                print('Failed to find lookahead')
            max_lookahead = 5.0
            min_lookahead = 0.5
            min_offset = min_lookahead / self.raceline_spacing
            max_offset = max_lookahead / self.raceline_spacing
            offset = int(np.clip(offset, min_offset, max_offset))
            self.goal_index = (self.nearest_index + offset) % self.track.point_count
            # print(f'{perf_counter() - lookahead_starttime:f}')
        else:
            lookahead = params.lookahead.v
            # Arc length lookahead
            relative_dists = self.dist_sums - self.dist_sums[self.nearest_index]
            self.goal_index = np.searchsorted(relative_dists, lookahead) % self.track.point_count

        # Transform goal point to vehicle frame of reference
        x_goal_car, y_goal_car = map_to_car_point(
            pose, 
            (self.track.x_ref[self.goal_index], self.track.y_ref[self.goal_index]))

        # Calculate curvature/steering angle
        L = np.hypot(x_goal_car, y_goal_car)
        if L < 1e-6:
            return
        gamma = 2*y_goal_car/L**2
        delta = np.arctan(config.wheelbase*gamma)

        # ===================================================================================

        self.angle = np.clip(delta, -config.max_steer, config.max_steer)

        self.speed = self.get_speed()

        self.publish_drive()
        
    def get_speed(self):
        if params.velocities_mode.v:
            return params.velocities_coeff.v * self.track.v_ref[self.goal_index]
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
        point = Point()
        goal_marker = Marker()
        point.x = self.track.x_ref[self.goal_index]
        point.y = self.track.y_ref[self.goal_index]
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
    print("PurePursuit Initialized")
    node = PurePursuit()
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

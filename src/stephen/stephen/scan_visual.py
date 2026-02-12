#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray
import numpy as np
import math
import vispy
vispy.use('glfw')
from vispy import app, scene
from vispy.scene.visuals import Line
import threading

# Displays visual representation of scan
# Assumes 270 fov
class ScanVisual(Node):

    def __init__(self, disparity_threshold=0.5):
        super().__init__('scan_visual')
        self.disparity_threshold = disparity_threshold
        self.initialized = False
        self.lock = threading.Lock()

        self.laser_scan_sub = self.create_subscription(LaserScan, '/scan', self.update_ranges, 10)
        self.v1_sub = self.create_subscription(Float32MultiArray, '/v1_ranges', self.update_v1, 10)
        self.v2_sub = self.create_subscription(Float32MultiArray, '/v2_ranges', self.update_v2, 10)
        self.steering_sub = self.create_subscription(AckermannDriveStamped, '/drive', self.update_drive, 10)

        # Visual options
        self.line_width = 3 # Pixels / Not used for glfw
        self.scan_color = (.8, .8, .8, 1.)
        self.disparity_color = (1.0, 0.2, 0.2, 1.0)
        self.disparity_color2 = (1.0, 0.4, 0.6, 1.0)
        self.v1_color = (0.0, 1.0, 0.2, 0.5)
        self.v2_color = (1.0, 0.6, 0.2, 1.0)
        self.steering_color = (0.4, 0.4, 1.0, 1.0)
        
        # Canvas + view
        self.canvas = scene.SceneCanvas(keys='interactive',
                                    show=True,
                                    vsync=True,
                                    bgcolor='black',
                                    title='Scan Visual',
                                    # size=(1280, 720),
                                    size=(1920, 1080))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(aspect=1)
        self.view.camera.set_range(x=(-10, 10), y=(-10, 10))  # set to your range (meters)

    def init(self, scan):
        view = self.view
        N = len(scan.ranges)
        self.scan_size = N

        # --- Scan geometry (example values; replace with your LaserScan msg) ---
        # angle_min = -3*np.pi/4 + np.pi/2
        angle_min = scan.angle_min + np.pi/2
        angle_increment = scan.angle_increment
        self.theta = angle_min + np.arange(N, dtype=np.float32)*angle_increment

        # Precompute unit vectors ONCE
        self.ux = np.cos(self.theta).astype(np.float32)
        self.uy = np.sin(self.theta).astype(np.float32)

        # Laser Scan
        self.ranges = np.zeros(N)
        self.colors = np.ones((N+2, 4), dtype=np.float32)
        self.pos = np.empty((N+2, 2), dtype=np.float32)
        self.pos[0] = (0,0)
        self.pos[-1] = (0,0)
        self.line = Line(pos=self.pos, color=self.colors, width=self.line_width, method='gl', connect='strip')
        self.line.set_gl_state(depth_test=False, blend=True)
        self.line.order = 3.0
        view.add(self.line)

        # v1 Scan
        self.v1_ranges = np.zeros(N)
        self.v1_colors = np.ones((N+2, 4), dtype=np.float32)
        self.v1_colors[0:2] = (.3, .3, .3, 1)
        self.v1_colors[-2:] = (.3, .3, .3, 1)
        self.v1_pos = np.empty((N+2, 2), dtype=np.float32)
        self.v1_pos[0] = (0,0)
        self.v1_pos[-1] = (0,0)
        self.v1_line = Line(pos=self.v1_pos, color=self.v1_colors,
                                  width=self.line_width, method='gl', connect='strip')
        self.v1_line.set_gl_state(depth_test=False, blend=True)
        self.v1_line.order = 2.0
        view.add(self.v1_line)

        # v2 scan
        self.v2_ranges = np.zeros(N)
        self.v2_colors = np.ones((N+2, 4), dtype=np.float32)
        self.v2_colors[0:2] = (.3, .3, .3, 1)
        self.v2_colors[-2:] = (.3, .3, .3, 1)
        self.v2_pos = np.empty((N+2, 2), dtype=np.float32)
        self.v2_pos[0] = (0,0)
        self.v2_pos[-1] = (0,0)
        self.v2_line = Line(pos=self.v2_pos, color=self.v2_colors,
                                  width=self.line_width, method='gl', connect='strip')
        self.v2_line.set_gl_state(depth_test=False, blend=True)
        self.v2_line.order = 1.0
        view.add(self.v2_line)
        
        # Horizontal
        horizontal = Line(
            pos=((-100, 0), (100, 0)), color=(0.3, 0.3, 0.3, 1.0), method='gl', connect='strip')
        horizontal.set_gl_state(depth_test=False, blend=True)
        horizontal.order = 0.0
        view.add(horizontal)
        
        # 10 meter circle
        circ10m= Line(
            pos=np.column_stack((np.cos(self.theta)*10, np.sin(self.theta)*10)).astype(np.float32),
            color=(0.3, 0.3, 0.3, 1.0), method='gl', connect='strip')
        circ10m.set_gl_state(depth_test=False, blend=True)
        circ10m.order = 0.0
        view.add(circ10m)
        
        # 30 meter circle
        circ30m= Line(
            pos=np.column_stack((np.cos(self.theta)*30, np.sin(self.theta)*30)).astype(np.float32),
            color=(0.3, 0.3, 0.3, 1.0), method='gl', connect='strip')
        circ30m.set_gl_state(depth_test=False, blend=True)
        circ30m.order = 0.0
        view.add(circ30m)

        # # Heading line
        # center_line = Line(
        #     pos=((0.0, 0.0), (0.0, 100.0)), color=(.3,.3,.3,1.),
        #     method='gl', connect='strip')
        # center_line.set_gl_state(depth_test=False, blend=True)
        # center_line.order = 1.0
        # view.add(center_line)

        # Forward increment lines
        self.increment_lines = []
        if N % 2 == 0: # As in sim case of 1080
            mid_right = int(N/2) - 1
            mid_left = int(N/2)
            angle_idxs = [mid_right, mid_left]
        else:
            mid = int(N/2)
            mid_right = int(N/2) - 1
            mid_left = int(N/2) + 1
            angle_idxs = [mid_right, mid, mid_left]

        for idx in angle_idxs:
            self.increment_lines.append(
                Line(
                    pos=((0.0, 0.0), (100*self.ux[idx], 100*self.uy[idx])),
                    color=(.3,.3,.3,1.), method='gl',
                    connect='strip', width=self.line_width)
            )

        for line in self.increment_lines:
            view.add(line)
            line.order = 1.0
            line.set_gl_state(depth_test=False, blend=True)

        # Inner circle representing car
        car_circ = Line(
            pos=np.column_stack((.3*np.cos(self.theta), .3*np.sin(self.theta))).astype(np.float32),
            color=(0.6, 0.3, 0.5, 1.0), method='gl', connect='strip', width=self.line_width)
        view.add(car_circ)
        car_circ.order = 4.0
        car_circ.set_gl_state(depth_test=False, blend=True)

        # Steering
        self.steering_pos = np.zeros((2, 2))
        self.steering_line = Line(
            pos=self.steering_pos, color=self.steering_color, method='gl', connect='strip', width=self.line_width)
        view.add(self.steering_line)
        self.steering_line.order = 2.0
        self.steering_line.set_gl_state(depth_test=False, blend=True)
        self.steering = 0.0

        self.initialized = True

        self.timer = app.Timer(interval=1/30, connect=self.update_visual, start=True)
    
    def update_ranges(self, scan:LaserScan):
        if not self.initialized:
            self.init(scan)
        with self.lock:
            self.ranges[:] = scan.ranges

    def update_v1(self, msg):
        if self.initialized:
            with self.lock:
                self.v1_ranges[:] = msg.data

    def update_v2(self, msg):
        if self.initialized:
            with self.lock:
                self.v2_ranges[:] = msg.data

    def update_drive(self, drive:AckermannDriveStamped):
        with self.lock:
            self.drive = drive
            self.steering = drive.drive.steering_angle

    def color_disparities(self, ranges, colors, disparity_color):
        diffs = np.diff(ranges)
        seg_idx = np.where(np.abs(diffs) >= self.disparity_threshold)[0]
        if seg_idx.size:
            vert_idx = np.unique(np.concatenate([seg_idx, seg_idx+1]))
            colors[vert_idx] = disparity_color

    def update_visual(self, _):
        with self.lock:
            # Scan
            r = np.where(np.isfinite(self.ranges), self.ranges, 0).astype(np.float32)
            x = r*self.ux
            y = r*self.uy
            self.pos[1:-1, 0] = x
            self.pos[1:-1, 1] = y
            self.colors[:] = self.scan_color
            self.color_disparities(r, self.colors[1:-1], self.disparity_color)

            # v1 ranges
            rv = np.where(np.isfinite(self.v1_ranges), self.v1_ranges, 0).astype(np.float32)
            xv = rv*self.ux
            yv = rv*self.uy
            self.v1_pos[1:-1, 0] = xv
            self.v1_pos[1:-1, 1] = yv
            self.v1_colors[2:-2] = self.v1_color
            self.color_disparities(rv, self.v1_colors[1:-1], self.disparity_color2)

            # v2 ranges
            rv2 = np.where(np.isfinite(self.v2_ranges), self.v2_ranges, 0).astype(np.float32)
            xv2 = rv2*self.ux
            yv2 = rv2*self.uy
            self.v2_pos[1:-1, 0] = xv2
            self.v2_pos[1:-1, 1] = yv2
            self.v2_colors[2:-2] = self.v2_color

            # Steering
            self.steering_pos[1] = (
                30 * math.cos(self.steering + math.pi/2),
                30 * math.sin(self.steering + math.pi/2)
                )

            # Set lines
            self.line.set_data(pos=self.pos, color=self.colors) 
            self.v1_line.set_data(pos=self.v1_pos, color=self.v1_colors)
            self.v2_line.set_data(pos=self.v2_pos, color=self.v2_colors)
            self.steering_line.set_data(pos=self.steering_pos, color=self.steering_color)

def main(args=None):
    rclpy.init(args=args)
    node = ScanVisual()
    spin_t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_t.start()
    try:
        app.run()   # blocks until window closed
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
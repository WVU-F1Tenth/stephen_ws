from numba import njit
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Quaternion, Pose
import math
from typing import Tuple
import numpy as np
from sensor_msgs.msg import LaserScan
from dataclasses import dataclass

class Scan:
    def __init__(self, scan: LaserScan):
        self.size = len(scan.ranges)
        # Scan message attributes
        self.angle_min = scan.angle_min
        self.angle_max = scan.angle_max
        self.angle_increment = scan.angle_increment
        self.time_increment = scan.time_increment
        self.scan_time = scan.scan_time
        self.range_min = scan.range_min
        self.range_max = scan.range_max
        self.increment = self.angle_increment
        self.index_to_angle_array = np.arange(self.size)*self.increment + self.angle_min
        self.fov = self.angle_max - self.angle_min
        self.angles = scan.angle_min + np.arange(self.size) * scan.angle_increment

    def angle_to_index(self, angle):
        abs_angle = angle + self.angle_min
        total_angle = self.angle_max - self.angle_min
        return round(self.size * (abs_angle / total_angle))
    
    def index_to_angle(self, index):
        return self.index_to_angle_array[index]
    
    def index_to_degrees(self, index):
        return round(math.degrees(self.index_to_angle(index)), 4)
    
    def fov_slice(self, ranges, fov):
        fov = math.radians(fov)
        fov_offset = self.angle_to_index(-fov / 2)
        return ranges[fov_offset: len(ranges) - fov_offset]
    
    def revert_fov_indexes(self, fov, indexes):
        fov = math.radians(fov)
        reverted = []
        for i in indexes:
            reverted.append(i + round((self.fov - fov) / (2 * self.increment)))
        return reverted
    
    def span_to_angle(self, span, depth):
        if depth <= 0.0:
            return 0.0
        return 2 * math.asin(span / (2 * depth))
    
def get_virtual2(ranges, angle_increment, extension):
        n = len(ranges)
        range_matrix = np.full((n, n), np.inf, dtype=np.float32)
        ratio = extension / (2 * ranges)
        ratio = np.clip(ratio, -1.0, 1.0)
        index_extensions = abs(np.floor(2*np.arcsin(ratio)/angle_increment).astype(np.int32))
        rows = np.arange(n)[:, None]
        cols = np.arange(n)[None, :]
        mask = np.abs(cols-rows) <= index_extensions[:, None]
        range_matrix = np.where(mask, ranges[:, None], np.inf).astype(np.float32)
        col_mins = range_matrix.min(axis=0)
        return col_mins

@njit
def get_virtual(ranges: np.ndarray, angle_increment: np.float32, extension: np.float32) -> np.ndarray:
    n = len(ranges)
    ratio = extension / (2 * ranges)
    ratio = np.clip(ratio, -1.0, 1.0)
    index_extensions = np.abs(2*np.arcsin(ratio)/angle_increment).astype(np.int32)
    new_ranges = ranges.copy()
    for i in range(n):
        j = index_extensions[i]
        new_ranges[max(0, i-j): min(n, i+j+1)] = np.minimum(new_ranges[max(0, i-j): min(n, i+j+1)], ranges[i])
    return new_ranges

@njit
def nearest_object_intersect(scan_angles, scan_ranges, ref, car_xyyaw):
    """
    Returns scan index of nearest forward object intersect from car.
    """
    x_car_map = car_xyyaw[0]
    y_car_map = car_xyyaw[1]
    yaw_map = car_xyyaw[2]
    dx_map = (ref[0] - x_car_map)
    dy_map = (ref[1] - y_car_map)
    yaw_map = car_xyyaw[2]
    d = np.hypot(dx_map, dy_map)
    dx = np.cos(yaw_map)*dx_map + np.sin(yaw_map)*dy_map
    dy = -np.sin(yaw_map)*dx_map + np.cos(yaw_map)*dy_map
    nearest = np.argmin(d)
    idx = nearest
    for _ in range(dx.size):
        r = np.hypot(dx[idx], dy[idx])
        theta = np.arctan2(dy[idx], dx[idx])
        scan_r = np.interp(theta, scan_angles, scan_ranges)
        if r > scan_r:
            return r, theta
        idx = (idx + 1) % dx.size
    raise RuntimeError('No object intersection found')

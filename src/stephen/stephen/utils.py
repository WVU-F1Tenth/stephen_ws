from numba import njit
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Quaternion, Pose
import math
from typing import Tuple
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.optimize import minimize_scalar

@njit
def threshold_index_cumulative(ar, start_index, threshold):
    """
    Returns first index from start index where
    (sum of array[start_index:index+1] > threshold) on circular array.
    If none found returns 0.
    """
    sum = 0
    offset = 0
    index = start_index
    while offset < ar.size and sum <= threshold:
        offset += 1
        index = (start_index + offset) % ar.size
        sum += ar[index]
    if offset == ar.size:
        return 0, offset
    else:
        return index, offset

@njit
def threshold_index(ar, start_index, threshold):
    """
    Returns first index from start index where (array[index] > threshold) on circular array.
    If none found returns 0.
    """
    offset = 0
    index = start_index
    while offset < ar.size and ar[index] <= threshold:
        offset += 1
        index = (start_index + offset) % ar.size
    if offset == ar.size:
        return 0, offset
    else:
        return index, offset
    
def quat_to_heading(orientation: Quaternion):
    quat = [orientation.x, orientation.y, orientation.z, orientation.w]
    return Rotation.from_quat(quat).as_euler('xyz')[2]

def heading_to_quat(yaw):
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q

def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

def map_to_car_point(car_pose: Pose, map_point: Tuple[float, float]):
    map_x, map_y = map_point
    dx = map_x - car_pose.position.x
    dy = map_y - car_pose.position.y
    theta = quat_to_heading(car_pose.orientation)
    x_car =  math.cos(theta) * dx + math.sin(theta) * dy
    y_car = -math.sin(theta) * dx + math.cos(theta) * dy
    return x_car, y_car


def nearest_spline_sample(tck, point, levels=3, n=100):
    """
    Returns spline progress closest to x, y
    """
    u1_min = 0
    u1_max = n//2
    u2_min = n//2 + 1
    u2_max = n-1
    point = np.asarray(point)
    for level in range(levels):
        sample_space1 = np.linspace(u1_min, u1_max, n//2)
        sample_space2 = np.linspace(u2_min, u2_max, n//2)
        u = splev(np.concatenate((sample_space1, sample_space2)), tck)
        diff =  u - point
        err = np.hypot(diff[:, 0], diff[:, 1])
        # err = diff[:, 0]**2 + diff[:, 1]**2
        closest = np.argmin(err)
        u1_min = closest - 1 if closest != 0 else n - 1
        u1_max = closest if closest != 0 else n
        u2_min = closest
        u2_max = closest + 1
    return closest


def nearest_spline_point(tck, umax, point):
    """
    Returns spline progress closest to x, y
    """
    pass
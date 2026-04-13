from numba import njit
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Quaternion, Pose
import math
from typing import Tuple
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.optimize import minimize_scalar
from typing import Tuple

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

class RacelineSpline:
    def __init__(self, xref, yref, smooth=0.0, n_dense=5000, dtype=np.float64):
        self.dtype = dtype
        self.tck, self.u = splprep([xref, yref], per=1, s=smooth)
        self.u0, self.u1 = self.u[0], self.u[-1]
        # TODO: check if this is always 0 to 1
        self.u_period = self.u1 - self.u0
        self.u_dense = np.linspace(self.u0, self.u1, n_dense, endpoint=False, dtype=self.dtype)
        x_dense, y_dense = splev(self.u_dense, self.tck)
        self.xy_dense = np.column_stack([x_dense, y_dense])
        dx = np.diff(x_dense, append=x_dense[0])
        dy = np.diff(y_dense, append=y_dense[0])
        ds = np.hypot(dx, dy)
        self.s_dense = np.empty(n_dense, dtype=self.dtype)
        self.s_dense[0] = 0.0
        self.s_dense[1:] = np.cumsum(ds[:-1])
        self.length = ds.sum()

    def wrap_u(self, u):
        return self.u0 + ((np.asarray(u) - self.u0) % self.u_period)
    
    def s_to_u(self, s):
        s = np.asarray(s) % self.length
        return np.interp(s, self.s_dense, self.u_dense)
    
    def u_to_s(self, u):
        u = self.wrap_u(u)
        return np.interp(u, self.u_dense, self.s_dense)
    
    def s_to_xy(self, s) -> Tuple[float, float]:
        uu = self.s_to_u(s)
        x, y = splev(uu, self.tck)
        return (float(x), float(y)) # type: ignore
    
    def s_to_heading(self, s):
        uu = self.s_to_u(s)
        dx_du, dy_du = splev(uu, self.tck, der=1)
        return -np.arctan2(dx_du, dy_du)

    def xy_to_s(self, point, levels=2, n=50):
        """
        Returns spline progress closest to x, y
        """
        point = np.asarray(point, dtype=self.dtype)
        diff = self.xy_dense - point
        idx0 = np.argmin(np.sum(diff * diff, axis=1))
        best_u = self.u_dense[idx0]
        du = self.u_period / len(self.u_dense)
        for _ in range(levels):
            samples = best_u + np.linspace(-du, du, n, dtype=self.dtype)
            samples = self.wrap_u(samples)
            xy = np.asarray(splev(samples, self.tck), dtype=self.dtype).T
            err = np.sum((xy - point) ** 2, axis=1)
            idx = np.argmin(err)
            best_u = samples[idx]
            du *= 0.5
        return self.u_to_s(best_u)
    
    def nearest_xy(self, point):
        return self.s_to_xy(self.xy_to_s(point))
    
class Raceline:
    def __init__(self, df, dtype=np.float64):
        self.x_ref_closed = df.iloc[:, 1].to_numpy(dtype=dtype)
        self.x_ref = self.x_ref_closed[:-1]
        self.y_ref_closed = df.iloc[:, 2].to_numpy(dtype=dtype)
        self.y_ref = self.y_ref_closed[:-1]
        self.yaw_ref = df.iloc[:-1, 3].to_numpy(dtype=dtype)
        self.k_ref = df.iloc[:-1, 4].to_numpy(dtype=dtype)
        self.v_ref = df.iloc[:-1, 5].to_numpy(dtype=dtype)
        self.point_count = self.x_ref.size

    def reverse(self):
        self.x_ref_closed = self.x_ref_closed[::-1]
        self.x_ref = self.x_ref_closed[:-1]
        self.y_ref_closed = self.y_ref_closed[::-1]
        self.y_ref = self.y_ref_closed[:-1]
        self.yaw_ref = (self.yaw_ref[::-1] + math.pi) % (2 * math.pi) - math.pi
        self.k_ref = -self.k_ref[::-1]
        self.v_ref = self.v_ref[::-1]

        
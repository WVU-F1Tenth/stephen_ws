from numba import njit
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Quaternion, Pose
import math
from typing import Tuple
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.optimize import minimize_scalar
from typing import Tuple


@njit(cache=True)
def threshold_index_cumulative(ar, start_index, threshold):
    """
    Returns first index from start index where
    (sum of array[start_index:index+1] > threshold) on circular array.
    If none found returns 0.
    """
    sum = 0.0
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

@njit(cache=True)
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
    
def quat_to_yaw(orientation: Quaternion):
    quat = [orientation.x, orientation.y, orientation.z, orientation.w]
    return Rotation.from_quat(quat).as_euler('xyz')[2]

def yaw_to_quat(yaw):
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q

def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

class RacelineSpline:
    def __init__(self, xref, yref, dtype, smooth=20.0, n_dense=1000):
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
    
    def s_to_xy(self, s):
        uu = self.s_to_u(s)
        x, y = splev(uu, self.tck)
        return x, y
    
    def u_to_xy(self, u):
        return splev(u, self.tck)
    
    def s_to_heading(self, s):
        uu = self.s_to_u(s)
        dx_du, dy_du = splev(uu, self.tck, der=1)
        return -np.arctan2(dx_du, dy_du)
    
    def xy_to_u(self, point, levels=2, n=50):
        """
        Returns spline progress as u closest to x, y
        """
        point = np.asarray(point, dtype=self.dtype)
        diff = self.xy_dense - point
        idx0 = np.argmin(np.sum(diff * diff, axis=1))
        best_u = self.u_dense[idx0]
        du = self.u_period / len(self.u_dense)
        for _ in range(int(levels)):
            samples = best_u + np.linspace(-du, du, n, dtype=self.dtype)
            samples = self.wrap_u(samples)
            xy = np.asarray(splev(samples, self.tck), dtype=self.dtype).T
            err = np.sum((xy - point) ** 2, axis=1)
            idx = np.argmin(err)
            best_u = samples[idx]
            du  = (2 * du) / (n - 1)
        return best_u
    
    def xy_to_ue(self, point, levels=2, n=50):
        """
        Returns spline progress as u closest to x, y
        """
        point = np.asarray(point, dtype=self.dtype)
        diff = self.xy_dense - point
        idx0 = np.argmin(np.sum(diff * diff, axis=1))
        best_u = self.u_dense[idx0]
        du = self.u_period / len(self.u_dense)
        for _ in range(int(levels)):
            samples = best_u + np.linspace(-du, du, n, dtype=self.dtype)
            samples = self.wrap_u(samples)
            xy = np.asarray(splev(samples, self.tck), dtype=self.dtype).T
            err = np.sum((xy - point) ** 2, axis=1)
            idx = np.argmin(err)
            best_u = samples[idx]
            du  = (2 * du) / (n - 1)
        return best_u, err[idx]

    def xy_to_s(self, point, levels=2, n=50):
        """
        Returns spline progress as s closest to x, y
        """
        best_u = self.xy_to_u(point, levels, n)
        return self.u_to_s(best_u)
    
    def nearest_xy(self, point):
        return self.s_to_xy(self.xy_to_s(point))

    def progress_at(self, x, y):
        u = self.xy_to_u((x, y))
        return u, self.u_to_s(u)

    def relative_from(self, origin_s, x, y):
        """
        Return relative progress and signed cross-track error from a cached origin.
        """
        point = np.asarray((x, y), dtype=self.dtype)
        target_u = self.xy_to_u(point)
        target_s = self.u_to_s(target_u)
        s_rel = target_s - origin_s
        if s_rel < -0.5 * self.length:
            s_rel += self.length
        elif s_rel > 0.5 * self.length:
            s_rel -= self.length
        cx, cy = splev(target_u, self.tck)
        dx_du, dy_du = splev(target_u, self.tck, der=1)
        norm = np.hypot(dx_du, dy_du)
        if norm == 0:
            d = np.hypot(x - cx, y - cy)
        else:
            tx = dx_du / norm
            ty = dy_du / norm
            d = tx * (y - cy) - ty * (x - cx)
        return s_rel, d
    
    def relative(self, origin_x, origin_y, x, y):
        """
        Return relative progress and signed cross-track error.
        Positive s_rel means target is ahead along the shortest track direction.
        Positive d means target is left of the spline tangent.
        """
        _, origin_s = self.progress_at(origin_x, origin_y)
        return self.relative_from(origin_s, x, y)

class Raceline:
    def __init__(self, df, dtype=np.float32):
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

    def nearest(self, point):
        dx = self.x_ref - point[0]
        dy = self.y_ref - point[1]
        d = np.hypot(dx, dy)
        return np.argmin(d)

# def map_to_car(x, y, yaw, x_c, y_c, yaw_c):
#     dx = x - x_c
#     dy = y - y_c
#     c = np.cos(yaw_c)
#     s = np.sin(yaw_c)
#     x_car = c * dx + s * dy
#     y_car = -s * dx + c * dy
#     offset = -np.pi/2
#     yaw_car = np.arctan2(np.sin(yaw - yaw_c + offset), np.cos(yaw - yaw_c + offset))
#     return x_car, y_car, yaw_car

def map_to_car(x, y, yaw, x_c, y_c, yaw_c):
    dx = x - x_c
    dy = y - y_c
    c = np.cos(yaw_c)
    s = np.sin(yaw_c)
    x_car = c * dx + s * dy
    y_car = -s * dx + c * dy
    # Transform the heading vector itself
    hx = np.cos(yaw)
    hy = np.sin(yaw)
    hx_car = c * hx + s * hy
    hy_car = -s * hx + c * hy
    yaw_car = np.arctan2(hy_car, hx_car)
    return x_car, y_car, yaw_car

def car_to_map(x, y, yaw, x_c, y_c, yaw_c):
    c = np.cos(yaw_c)
    s = np.sin(yaw_c)
    x_map = x_c + c * x - s * y
    y_map = y_c + s * x + c * y
    yaw_map = np.arctan2(np.sin(yaw + yaw_c), np.cos(yaw + yaw_c))
    return x_map, y_map, yaw_map

def idx_nearest_point(x, y, path_x, path_y):
    dx = x - path_x
    dy = y - path_y
    d = np.hypot(dx, dy)
    return np.argmin(d)

def car_xyyaw(pose,  wheelbase):
    """
    Front axis projection and axes correction
    """
    yaw = quat_to_yaw(pose.orientation)
    x = pose.position.x + wheelbase * math.cos(yaw)
    y = pose.position.y + wheelbase * math.sin(yaw)
    return x, y, yaw

def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

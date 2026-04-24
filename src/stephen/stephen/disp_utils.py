from dataclasses import dataclass

from numba import njit
import math
import numpy as np
from sensor_msgs.msg import LaserScan

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

@njit(cache=True)
def get_virtual(ranges: np.ndarray, angle_increment: np.float32, extension: np.float32,
                min_range: np.float32 = np.float32(0.3)) -> np.ndarray:
    ranges = ranges.copy() / min_range
    new_ranges = ranges.copy() / min_range
    extension /= min_range
    n = ranges.shape[0]
    ratio = extension / (2 * ranges)
    ratio = np.clip(ratio, -1.0, 1.0)
    index_extensions = np.abs(2*np.arcsin(ratio)/angle_increment).astype(np.int32)
    for i in range(n):
        j = index_extensions[i]
        new_ranges[max(0, i-j): min(n, i+j+1)] = np.minimum(new_ranges[max(0, i-j): min(n, i+j+1)], ranges[i])
    return new_ranges * min_range

# @njit(cache=True)
# def get_virtual(ranges: np.ndarray, angle_increment: np.float32, extension: np.float32,
#                 min_range: np.float32 = np.float32(1e-3)) -> np.ndarray:
#     ranges = np.asarray(ranges, dtype=np.float32)
#     new_ranges = ranges.copy()
#     n = ranges.shape[0]
#     valid = np.isfinite(ranges) & (ranges > np.float32(0.0))
#     valid_indices = np.flatnonzero(valid)
#     safe_ranges = np.maximum(ranges[valid], min_range).astype(np.float32)
#     half_extension = np.float32(0.5) * extension
#     half_angles = 2*np.arctan(half_extension / safe_ranges).astype(np.float32)
#     index_extensions = np.ceil(half_angles / angle_increment).astype(np.int32)
#     for idx, j in zip(valid_indices, index_extensions):
#         lo = max(0, idx - j)
#         hi = min(n, idx + j + 1)
#         new_ranges[lo:hi] = np.minimum(new_ranges[lo:hi], ranges[idx]).astype(np.float32)
#     return new_ranges.astype(np.float32)

@njit(cache=True)
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

@njit(cache=True)
def max_point_radius(dir, ranges, angle_increment, point_idx):
    if dir > 0:
        side_a = ranges[point_idx]
        side_a_2 = side_a**2
        idx = point_idx + 1
        radius = ranges[idx]
        theta = angle_increment
        while radius > (2 * side_a * np.sin(theta/2)) and idx < ranges.size - 1:
            idx += 1
            theta += angle_increment
            side_b = ranges[idx]
            r = np.sqrt(side_a_2 + side_b**2 - 2*side_a*side_b*np.cos(theta))
            if r < radius:
                radius = r
    elif dir < 0:
        side_a = ranges[point_idx]
        side_a_2 = side_a**2
        idx = point_idx - 1
        radius = ranges[idx]
        theta = angle_increment
        while radius > (2 * side_a * np.sin(theta/2)) and idx > 0:
            idx -= 1
            theta += angle_increment
            side_b = ranges[idx]
            r = np.sqrt(side_a_2 + side_b**2 - 2*side_a*side_b*np.cos(theta))
            if r < radius:
                radius = r
    # print(dir)
    # print(radius)
    # print()
    return radius, theta
    
@njit(cache=True)
def radial_extension_to_path(dir, r_start, theta_start, r_ref, theta_ref):
    # Sort ref points
    sort = np.argsort(theta_ref)
    r_ref = r_ref[sort]
    theta_ref = theta_ref[sort]
    # Get limit in angle
    idx = np.searchsorted(theta_ref, theta_start)
    if dir > 0:
        while not np.isclose(r_start, r_ref[idx], atol=0.1) and idx < theta_ref.size:
            idx += 1
    elif dir < 0:
        while not np.isclose(r_start, r_ref[idx], atol=0.1) and idx > 0:
            idx -= 1
    return theta_ref[idx]

import numpy as np
from dataclasses import dataclass

@dataclass
class ProgressResult:
    s: float           # continuous unwrapped progress
    s_raw: float       # wrapped progress in [0, track_length)
    d: float
    seg_idx: int
    t: float
    proj_x: float
    proj_y: float
    lap: int

class LocalFrenetProgress:
    def __init__(self, x_ref, y_ref):
        x_ref = np.asarray(x_ref, dtype=float)
        y_ref = np.asarray(y_ref, dtype=float)
        pts = np.column_stack((x_ref, y_ref))
        # Always closed
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack((pts, pts[0]))
        self.closed = True
        self.pts = pts
        self.p0 = pts[:-1]
        self.p1 = pts[1:]
        self.seg = self.p1 - self.p0
        self.seg_len = np.linalg.norm(self.seg, axis=1)
        self.seg_len_sq = self.seg_len ** 2
        if np.any(self.seg_len_sq == 0):
            raise ValueError("Consecutive duplicate raceline points found.")
        self.s_cum = np.concatenate(([0.0], np.cumsum(self.seg_len)))
        self.track_length = self.s_cum[-1]
        self.num_seg = len(self.seg)
        self.last_seg_idx = 0
        self.initialized = False
        # Internal continuous progress state
        self.s_continuous = None

    def _candidate_indices(self, center_idx, window):
        return [(center_idx + k) % self.num_seg for k in range(-window, window + 1)]

    def _project_to_segment(self, qx, qy, i):
        p0 = self.p0[i]
        seg = self.seg[i]
        seg_len_sq = self.seg_len_sq[i]
        wx = qx - p0[0]
        wy = qy - p0[1]
        t = (wx * seg[0] + wy * seg[1]) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        proj_x = p0[0] + t * seg[0]
        proj_y = p0[1] + t * seg[1]
        dx = qx - proj_x
        dy = qy - proj_y
        dist_sq = dx * dx + dy * dy
        cross_z = seg[0] * dy - seg[1] * dx
        d = np.sign(cross_z) * np.hypot(dx, dy)
        s_raw = self.s_cum[i] + t * self.seg_len[i]
        return dist_sq, s_raw, d, t, proj_x, proj_y

    def _unwrap_from_previous(self, s_raw):
        """
        Convert wrapped s_raw in [0, track_length) into continuous progress
        using the previously stored continuous value.
        """
        if self.s_continuous is None:
            return s_raw
        # previous wrapped position corresponding to stored continuous progress
        s_prev_raw = self.s_continuous % self.track_length
        ds = s_raw - s_prev_raw
        if ds > 0.5 * self.track_length:
            ds -= self.track_length
        elif ds < -0.5 * self.track_length:
            ds += self.track_length
        return self.s_continuous + ds

    def _build_result(self, s_raw, d, seg_idx, t, proj_x, proj_y):
        s_cont = self._unwrap_from_previous(s_raw)
        self.s_continuous = s_cont
        lap = int(np.floor(s_cont / self.track_length))
        return ProgressResult(
            s=s_cont,
            s_raw=s_raw,
            d=d,
            seg_idx=seg_idx,
            t=t,
            proj_x=proj_x,
            proj_y=proj_y,
            lap=lap,
        )

    def initialize_global(self, qx, qy):
        best = None
        best_i = None
        for i in range(self.num_seg):
            result = self._project_to_segment(qx, qy, i)
            dist_sq = result[0]
            if best is None or dist_sq < best[0]:
                best = result
                best_i = i
        self.last_seg_idx = best_i
        self.initialized = True
        _, s_raw, d, t, proj_x, proj_y = best # type: ignore
        return self._build_result(s_raw, d, best_i, t, proj_x, proj_y)

    def update(self, qx, qy, window=20):
        if not self.initialized:
            return self.initialize_global(qx, qy)
        idxs = self._candidate_indices(self.last_seg_idx, window)
        best = None
        best_i = None
        for i in idxs:
            result = self._project_to_segment(qx, qy, i)
            dist_sq = result[0]
            if best is None or dist_sq < best[0]:
                best = result
                best_i = i
        self.last_seg_idx = best_i
        _, s_raw, d, t, proj_x, proj_y = best # type: ignore
        return self._build_result(s_raw, d, best_i, t, proj_x, proj_y)

    def reset_progress(self):
        self.s_continuous = None
        self.initialized = False
        self.last_seg_idx = 0
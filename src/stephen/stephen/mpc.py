#!/usr/bin/env python3
import math
from dataclasses import dataclass, field
import cvxpy
import numpy as np
import rclpy
from ackermann_msgs.msg import  AckermannDriveStamped
from rclpy.node import Node
# from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from .mpc_utils import nearest_point
from numpy import typing as npt
from typing import Any
import os
from pathlib import Path
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
import pandas as pd
from time import perf_counter
from .utils import quat_to_yaw
from .io_utils import Binding, DualBinding, KeyBindings

SIMULATOR = False

params = KeyBindings()

map_path = os.environ.get('MAP_PATH')
if map_path is None:
        raise RuntimeError('MAP_PATH not set')
CSV_PATH = Path(map_path+'_raceline.csv')
if not CSV_PATH.exists():
    raise RuntimeError("Waypoint file doesn't exist")

@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [acceleration, delta]
    TK: int = 8  # finite time horizon length kinematic
    # ---------------------------------------------------
    # TODO: you may need to tune the following matrices
    Rk: npt.NDArray[Any] = field(
        default_factory=lambda: np.diag([0.01, 80.0])
    )  # input cost matrix, penalty for inputs - [accel, steering]
    Rdk: npt.NDArray[Any] = field(
        default_factory=lambda: np.diag([0.01, 80.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering]
    Qk: npt.NDArray[Any] = field(
        default_factory=lambda: (np.diag([60.0, 60.0, 20.0, 2.0]))
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, v, yaw]
    Qfk: npt.NDArray[Any] = field(
        default_factory=lambda: (np.diag([60.0, 60.0, 20.0, 2.0]))
    )  # final state error matrix, penalty  for the final state constraints: [x, y, v, yaw]
    # ---------------------------------------------------

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # Time step used
    dlk: float = 0.1  # Distance between reference points
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4  # minimum steering angle [rad]
    MAX_STEER: float = 0.4  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(90)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 4.0  # maximum acceleration [m/ss]

@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0

class MPC(Node):
    
    def __init__(self):
        super().__init__('mpc_node')
        # Create ROS subscribers and publishers
        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.raceline_viz = self.create_publisher(Marker, '/viz/raceline', 10)
        if SIMULATOR:
            self.sub_odom = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback,  1)
        else:
            self.sub_pose = self.create_subscription(Odometry, '/pf/odom', self.pose_callback, 1)
        self.keyboard_timer = self.create_timer(.5, params.check_input)
        self.print_timer = self.create_timer(1.0, self.print_info)
        self.mpc_solve_time = 0.0
        self.mpc_total_time = 0.0
        
        # Reading CSV data
        df = pd.read_csv(CSV_PATH, header=0, comment='#', sep=';')
        waypoints_x_closed = df.iloc[:, 1].to_numpy(dtype=float)
        self.ref_x = waypoints_x_closed[:-1]
        waypoints_y_closed = df.iloc[:, 2].to_numpy(dtype=float)
        self.ref_y = waypoints_y_closed[:-1]
        self.ref_yaw = df.iloc[:-1, 3].to_numpy(dtype=float)
        self.curvatures = df.iloc[:-1, 4].to_numpy(dtype=float)
        self.ref_v = df.iloc[:-1, 5].to_numpy(dtype=float)
        self.point_count = self.ref_x.size
        # dists[0] = distance from point 0 to point 1
        self.dists = np.hypot(np.diff(waypoints_x_closed), np.diff(waypoints_y_closed))
        self.raceline_spacing = float(np.mean(self.dists))

        self.config = mpc_config()
        self.config.dlk = self.raceline_spacing
        self.odelta = [0.0] * self.config.TK
        self.oa = [0.0] * self.config.TK
        self.odelta_input = 0.0
        self.ovel_input = 0.0
        self.init_flag = 0

        # initialize MPC problem
        self.mpc_prob_init()

        self.publish_raceline(self.ref_x, self.ref_y)

    def odom_callback(self, odometry_info: Odometry):
        self.pose_callback(odometry_info)

    def pose_callback(self, odometry_info):
        self.mpc_total_time_start = perf_counter()
        pose = odometry_info.pose.pose
        twist = odometry_info.twist.twist
        vehicle_state = State(
            x = pose.position.x,
            y = pose.position.y,
            delta = self.odelta_input,
            v = twist.linear.x,
            yaw = quat_to_yaw(pose.orientation),
            yawrate = twist.angular.z,
            beta = 0.0,
        )

        ref_path = self.calc_ref_trajectory(vehicle_state, self.ref_x, self.ref_y, self.ref_yaw, self.ref_v)
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        # TODO: solve the MPC control problem
        (
            self.oa,
            self.odelta,
            ox,
            oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control(ref_path, x0, self.oa, self.odelta)
        
        self.ovel_input = np.clip(vehicle_state.v + self.oa[0] * self.config.DTK, # type: ignore
                            self.config.MIN_SPEED,
                            self.config.MAX_SPEED)
        self.odelta_input = np.clip(self.odelta[0], # type: ignore
                            self.config.MIN_STEER,
                            self.config.MAX_STEER)
        ackermann_drive_result = AckermannDriveStamped()
        ackermann_drive_result.header.stamp = self.get_clock().now().to_msg()
        ackermann_drive_result.drive.steering_angle = self.odelta_input
        vel_input = 0.0 if params.speed.v < 0.05 else self.ovel_input
        ackermann_drive_result.drive.speed = vel_input
        self.pub_drive.publish(ackermann_drive_result)
        self.mpc_total_time = perf_counter() - self.mpc_total_time_start

    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        More QP example in CVXPY here: https://www.cvxpy.org/examples/basic/quadratic_program.html
        """
        # Initialize and create vectors for the optimization problem
        # Vehicle State Vector
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)
        )
        # Control Input vector
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)
        )
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))

        # Previous input parameter
        self.prev_u_k = cvxpy.Parameter((self.config.NU,))
        self.prev_u_k.value = np.zeros(self.config.NU)

        # Objective
        traj_error_term = cvxpy.quad_form(
            cvxpy.reshape(self.xk - self.ref_traj_k, (self.config.NXK * (self.config.TK + 1),), order='F'), 
            Q_block)
        
        control_term = cvxpy.quad_form(
            cvxpy.reshape(self.uk, (self.config.NU * self.config.TK,), order='F'), 
            R_block)
        
        du = self.uk[:, 1:] - self.uk[:, :-1]
        control_diff_term = cvxpy.quad_form(
            cvxpy.reshape(du, (self.config.NU * (self.config.TK - 1),), order='F'),
            Rd_block)
        
        prev_du = self.uk[:, 0] - self.prev_u_k
        prev_du_term = cvxpy.quad_form(prev_du, self.config.Rdk)
        
        objective = traj_error_term + control_term + control_diff_term + prev_du_term

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], self.odelta_input
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block)).tocoo()
        B_block = block_diag(tuple(B_block)).tocoo()
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # -------------------------------------------------------------
        
        constraints.append(
            cvxpy.reshape(self.xk[:, 1:], (self.config.NXK*self.config.TK,), order='F')
            == self.Ak_ @ cvxpy.reshape(self.xk[:, :-1], (self.config.NXK*self.config.TK), order='F')
            + self.Bk_ @ cvxpy.reshape(self.uk, (self.config.NU*self.config.TK), order='F')
            + self.Ck_
            )
        
        constraints.append(
            cvxpy.abs(self.uk[1, 1:] - self.uk[1, :-1]) <= self.config.MAX_DSTEER * self.config.DTK
        )

        constraints.append(self.xk[:, 0] == self.x0k)
        constraints.append(self.xk[2, :] <= self.config.MAX_SPEED)
        constraints.append(self.xk[2, :] >= self.config.MIN_SPEED)
        constraints.append(cvxpy.abs(self.uk[0, :]) <= self.config.MAX_ACCEL)
        constraints.append(cvxpy.abs(self.uk[1, :]) <= self.config.MAX_STEER)
        
        # -------------------------------------------------------------

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)
        cyaw = cyaw.copy()

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        # TODO use better velocity approximation
        travel = abs(state.v) * self.config.DTK * 1.1
        # Distance in index count
        dind = travel / self.config.dlk
        # ind_list = [ind, ind+1*dind, ind+2*dind, ...]
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        # Where greater than or equal to ncourse, then -= ncourse
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]
        # Anywere | yaw - course_yaw | > 4.5, shift by 2pi
        cyaw[cyaw - state.yaw > np.pi] -= 2 * np.pi
        cyaw[cyaw - state.yaw < -np.pi] += 2 * np.pi
        ref_traj[3, :] = cyaw[ind_list]

        return ref_traj

    def predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(self, state, a, delta):

        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state

    def get_model_matrix(self, v, phi, delta):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax +Bu + C
        State vector: x=[x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C

    def mpc_prob_solve(self, ref_traj, path_predict, x0, oa, od):
        self.x0k.value = x0

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], od[t] #type: ignore
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block)).tocoo()
        B_block = block_diag(tuple(B_block)).tocoo()
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        self.prev_u_k.value = np.array([oa[0], od[0]])

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        self.mpc_solve_time_start = perf_counter()
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)
        self.mpc_solve_time = perf_counter() - self.mpc_solve_time_start

        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def linear_mpc_control(self, ref_path, x0, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param oa: acceleration of T steps of last time
        :param od: delta of T steps of last time
        """

        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK

        # Call the Motion Prediction function: Predict the vehicle motion for x-steps
        path_predict = self.predict_motion(x0, oa, od, ref_path)
        poa, pod = oa[:], od[:]

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
            ref_path, path_predict, x0, oa, od)

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict
    
    def print_info(self):
        print(f'mpc solve time          {self.mpc_solve_time}')
        print(f'total pipeline time  {self.mpc_total_time}\n')

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
        
def main(args=None):
    rclpy.init(args=args)
    print("MPC Initialized")
    node = MPC()
    try:
        rclpy.spin(node)
    finally:
        params.restore_terminal()
        node.destroy_node()
        rclpy.shutdown()
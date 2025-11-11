import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tqdm import tqdm
import time
import functools
import mppi_eece.sim.plot_utils as plot_utils
from mppi_eece.systems import HalfCar, Unicycle
from mppi_eece.sim.grid import OccupGrid
from mppi_eece.jax_mppi.rrt_star import RRTStar
from mppi_eece.jax_mppi.ca_mpc import *
from mppi_eece.jax_mppi.collision_checker import CollisionChecker
from mppi_eece.jax_mppi.acd_unicycle import create_unicycle_solver_with_time_varying_obstacles, create_unicycle_model, solve_with_obstacles_and_goal
import casadi as ca
import os
import json
import copy 
import cProfile
import datetime
import io
import pydecomp as pdc
# matplotlib.use('Agg')
def rotate_2dvectors(x, theta, center=np.array([0.0,0.0])):
    x = np.array(x)
    if len(x.shape) == 1:
        x = np.expand_dims(x, 0)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    for i in range(len(x)):
        x[i,:2] = R.T @ x[i,:2] - R.T @ center
    return x

def rotate_polyhedral(A, b, theta, center=np.array([0.0,0.0])):
    A_rot = rotate_2dvectors(A, theta)
    b_rot = b - A @center.reshape(-1,1)
    return A_rot, b_rot

def find_controls(path, start, Nt, occupied, box, planner, solver, num_sides):
    p = np.array(path)[:,0:2]
    if len(occupied) < 1:
        A = [np.zeros((num_sides,2)) for i in range(len(p))]
        b = [np.ones((num_sides,1))*1000 for i in range(len(p))]
    else:
        A, b = pdc.convex_decomposition_2D(occupied, p, box)
    X0, idx =  planner.discretizePath(p,Nt+1)
    thetas = np.ones((Nt+1, 1)) * start[2]
    X0_f = np.hstack((np.array(X0), thetas.reshape(-1,1)))
    rX0 = rotate_2dvectors(X0_f, start[2], start[:2])
    rX0[:,2] = wrap_to_pi(rX0[:,2] - start[2])
    
    obs_halfplane = [[A[i],b[i]] for i in idx]
    A_np = []
    b_np = []

    for Ab in obs_halfplane:
        # print(len(Ab[0]))
        if len(Ab[0]) >= num_sides:
            Ai = np.array(Ab[0][:num_sides,:])
            bi = np.array(Ab[1][:num_sides,:])
        else:
            Ai = np.vstack((Ab[0], np.zeros((num_sides-len(Ab[0]),2))))
            bi = np.vstack((Ab[1], np.zeros((num_sides-len(Ab[1]),1))))

        A_rot, b_rot = rotate_polyhedral(Ai,bi,start[2],start[:2])
        b_rot[np.argwhere(np.isnan(b_rot))] =  np.inf
        A_np.append(A_rot)
        b_np.append(b_rot)
    x_sol, u_sol, _ = solve_with_obstacles_and_goal(solver,
                                                    np.array([0.0,0.0,0.0]),
                                                    rX0[-1,:2].reshape(1,2),
                                                    A_np, 
                                                    b_np,
                                                    x_ref=rX0)
    x_sol = rotate_2dvectors(x_sol, -start[2], center=start[:2])
    x_sol[:,2] = wrap_to_pi(x_sol[:,2] + start[2])
    return x_sol, u_sol

class MPCPlanner:
    def __init__(self, params, do_profiling=False):
        self.params = params
        self.do_profiling = do_profiling
        self.Nt = params['Nt']
        self.T = params['T']
        self.Q = params['Q']
        self.R = params['R']
        self.QT = params['QT']
        self.start = params['start']
        self.q_ref = params['q_ref']
        self.obs = params['obs']
        self.nlmodel = params['nlmodel']
        self.num_sides = params.get('num_sides', 9)
        self.resolution = params.get('resolution', 0.5)

    # ---------------------
    # Environment setup
    # ---------------------
    def build_environment(self):
        origin = np.array([-40, -10])
        wh = np.array([50 / self.resolution, 20 / self.resolution], dtype=np.int32)
        boundary = [[origin[0], origin[0] + wh[0] * self.resolution],
                    [origin[1], origin[1] + wh[1] * self.resolution]]
        grid = OccupGrid(boundary, self.resolution)
        grid.find_occupancy_grid(self.obs, buffer=0.0)
        occupied = grid.find_all_occupied(self.obs)
        collision_checker = CollisionChecker(
            jnp.array(grid.occup_grid), origin, self.resolution, wh, occup_value=100
        )
        return grid, occupied, collision_checker

    # ---------------------
    # Cost function
    # ---------------------
    def eval_trajectory(self, u_seq, start, q_ref):
        state = start
        cost = 0
        states = [start]
        for t in range(self.Nt):
            u = u_seq[t, :]
            for i in range(10):  # sub-steps
                state = self.nlmodel.dynamics(state, u, 0,
                                              dt=self.T / self.Nt / 10,
                                              params=self.nlmodel.nominal_params)
            cost += (state - q_ref) @ self.Q @ (state - q_ref) + u @ self.R @ u
            states.append(state)
        cost += (state - q_ref) @ self.QT @ (state - q_ref)
        return cost, states

    # ---------------------
    # MPC solver + RRT* planner
    # ---------------------
    def plan(self, start, q_ref, occupied, collision_checker):
        Nt = self.Nt
        T = self.T
        Q, QT, R = self.Q, self.QT, self.R
        num_sides = self.num_sides

        # Create CasADi solver
        lbu = np.array([-np.pi, -3.0])
        ubu = np.array([np.pi, 3.0])
        solver = create_unicycle_solver_with_time_varying_obstacles(
            lbu, ubu, N=Nt, Tf=T, num_sides=num_sides,
            q_goal_pos=QT[0, 0], q_pos=Q[0, 0], r_control=R[0, 0]
        )

        # Create RRT* planner
        planner = RRTStar(
            collision_checker=collision_checker,
            max_iter=250,
            expand_dist=1.0,
            goal_sample_rate=0.2,
            connect_circle_dist=30.0,
        )

        # Find feasible path
        if self.do_profiling:
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()

        paths, _ = planner.find_path(start, q_ref, reset=True)

        if self.do_profiling:
            profiler.disable()
            print("Profiling data collected.")

        best_cost = np.inf
        best_u_sol, best_states = None, None
        all_mpc_paths, all_planner_paths = [], []

        for path in paths:
            x_sol, u_sol = find_controls(
                path, start, Nt, occupied,
                box=np.array([[1, 2]]),
                planner=planner, solver=solver, num_sides=num_sides
            )
            all_mpc_paths.append(x_sol)
            current_cost, states = self.eval_trajectory(u_sol, start, q_ref)
            if current_cost < best_cost:
                best_cost, best_u_sol, best_states = current_cost, u_sol, states

        all_planner_paths.extend(paths)
        return best_cost, best_u_sol, best_states, all_planner_paths, all_mpc_paths

    # ---------------------
    # Main solve entry point
    # ---------------------
    def solve(self):
        grid, occupied, collision_checker = self.build_environment()
        best_cost, best_u_sol, best_states, all_planner_paths, all_mpc_paths = self.plan(
            self.start, self.q_ref, occupied, collision_checker
        )
        return best_states, best_cost, all_planner_paths, all_mpc_paths
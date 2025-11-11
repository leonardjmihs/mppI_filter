from mppi_eece.ancillary_controller.ancillary_controller.utils import rotate_2dvectors, rotate_polyhedral, wrap_to_pi
from mppi_eece.mpc_solvers.acados_solver import AcadosSolver
from mppi_eece.mpc_solvers.casadi_solver import CasadiMPCSolver
from mppi_eece.mpc_solvers.solvers_base import MPCSolver
from mppi_eece.planners.rrt_star import RRTStarPlanner
from mppi_eece.jax_mppi.collision_checker import CollisionChecker
from mppi_eece.planners.grid import OccupGrid
import numpy as np
import pydecomp as pdc
import os, json, copy

class AncillaryController:
    """
    Base class for ancillary controllers (e.g., MPC, ACADOS planners).
    """
    def __init__(self, params, solver_type='casadi'):
        self.params = params
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
        if solver_type == 'acados':
            self.solver = AcadosSolver(params)
        else:
            self.solver = CasadiMPCSolver(params)

    def plan(self, start, q_ref, occupied, collision_checker):
        if self.planner is None:
            self.planner = RRTStarPlanner(
                collision_checker=collision_checker,
                max_iter=250,
                expand_dist=1.0,
                goal_sample_rate=0.2,
                connect_circle_dist=30.0,
            )
        self.planner.collision_checker = collision_checker
        paths = self.planner.plan(start, q_ref, obstacles=occupied)

        best_cost = np.inf
        best_u_sol, best_states = None, None
        all_mpc_paths, all_planner_paths = [], []

        for path in paths:
            A, b, rX0 = self.decompose_and_rotate_env(path, start, occupied)
            x_sol, u_sol = self.solver.solve(np.zeros((self.system.n_dims, 1)), 
                                            rX0, obstacles=occupied)
            all_mpc_paths.append(x_sol)
            current_cost, states = self.eval_trajectory(u_sol, start, q_ref)
            if current_cost < best_cost:
                best_cost, best_u_sol, best_states = current_cost, u_sol, states

        all_planner_paths.extend(paths)
        return best_cost, best_u_sol, best_states, all_planner_paths, all_mpc_paths

    def decompose_and_rotate_env(self, path, start, occupied):
        p = np.array(path)[:,0:2]
        box = np.array([[1,2]])
        if len(occupied) < 1:
            A = [np.zeros((self.num_sides,2)) for i in range(len(p))]
            b = [np.ones((self.num_sides,1))*1000 for i in range(len(p))]
        else:
            A, b = pdc.convex_decomposition_2D(occupied, p, box)
        try:
            X0, idx =  self.planner.discretizePath(p,self.Nt+1)
        except Exception as e:
            breakpoint()

        thetas = np.ones((self.Nt+1, 1)) * start[2]
        X0_f = np.hstack((np.array(X0), thetas.reshape(-1,1)))
        rX0 = rotate_2dvectors(X0_f, start[2], start[:2])
        rX0[:,2] = wrap_to_pi(rX0[:,2] - start[2])
    
        obs_halfplane = [[A[i],b[i]] for i in idx]
        A_np = []
        b_np = []

        for Ab in obs_halfplane:
            # print(len(Ab[0]))
            if len(Ab[0]) >= self.num_sides:
                Ai = np.array(Ab[0][:self.num_sides,:])
                bi = np.array(Ab[1][:self.num_sides,:])
            else:
                Ai = np.vstack((Ab[0], np.zeros((self.num_sides-len(Ab[0]),2))))
                bi = np.vstack((Ab[1], np.zeros((self.num_sides-len(Ab[1]),1))))

            A_rot, b_rot = rotate_polyhedral(Ai,bi,start[2],start[:2])
            b_rot[np.argwhere(np.isnan(b_rot))] =  np.inf
            A_np.append(A_rot)
            b_np.append(b_rot)
        return A_np, b_np, rX0

    def eval_trajectory(self, u_seq, start, q_ref):
        state = start
        cost = 0
        states = [start]
        for t in range(self.Nt):
            u = u_seq[t, :]
            for i in range(10):
                state = self.nlmodel.dynamics(state, u, 0,
                                              dt=self.T / self.Nt / 10,
                                              params=self.nlmodel.nominal_params)
            cost += (state - q_ref) @ self.Q @ (state - q_ref) + u @ self.R @ u
            states.append(state)
        cost += (state - q_ref) @ self.QT @ (state - q_ref)
        return cost, states



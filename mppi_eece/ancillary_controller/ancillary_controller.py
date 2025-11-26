from mppi_eece.ancillary_controller.utils import rotate_2dvectors, rotate_polyhedral, wrap_to_pi
from mppi_eece.mpc_solvers.acados_solver import AcadosSolver
from mppi_eece.mpc_solvers.casadi_solver import CasadiMPCSolver
from mppi_eece.mpc_solvers.solvers_base import MPCSolverParams
from mppi_eece.planners.rrt_star import RRTStarPlanner
from mppi_eece.planners.topo_prm import TopoPRM
from mppi_eece.jax_mppi.collision_checker import CollisionChecker
import numpy as np
import pydecomp as pdc
import os, json, copy



class AncillaryController:
    """
    Base class for ancillary controllers (e.g., MPC, ACADOS planners).
    """
    def __init__(self, mpc_params=None, planner_params={},solver_type='casadi', planner_type='rrt'):
        if mpc_params is None:
            mpc_params = MPCSolverParams()
        self.mpc_params = mpc_params
        self.T = mpc_params['T']
        self.Nt = mpc_params['Nt']
        self.dt = mpc_params['dt']
        self.Q = mpc_params['Q']
        self.R = mpc_params['R']
        self.QT = mpc_params['QT']
        self.nlmodel = mpc_params['nlmodel']
        self.num_sides = mpc_params.get('num_sides', 9)

        self.planner_params = planner_params

        solverParams = MPCSolverParams(Nt=self.Nt, dt=self.dt, T=self.T, Q=self.Q, R=self.R, QT=self.QT,
                                      num_sides=self.num_sides, nlmodel=self.nlmodel)
        if solver_type.lower() == 'acados':
            self.solver = AcadosSolver(self.nlmodel, solverParams)
        else:
            self.solver = CasadiMPCSolver(self.nlmodel, solverParams)
        if planner_type.lower() == 'rrt':
            self.planner = RRTStarPlanner(
                    collision_checker=None,
                    max_iter=planner_params.get('max_iter', 250),
                    expand_dist=planner_params.get('expand_dist', 1.0),
                    goal_sample_rate=planner_params.get('goal_sample_rate', 0.2),
                    connect_circle_dist=planner_params.get('connect_circle_dist', 30.0),
                    footprint=planner_params.get('footprint', [[0.0, 0.0]]),
                )
        elif planner_type.lower() == 'topo_prm':
            self.planner = TopoPRM(
                    occup_grid=None,
                    sample_inflate=planner_params.get('sample_inflate', [5.0, 5.0, 0.0]),
                    origin=planner_params.get('origin', [0.0, 0.0]),
                    resolution=planner_params.get('resolution', 0.1),
                    wh=planner_params.get('wh', [10, 10]),
                    max_raw_path=planner_params.get('max_raw_path', 10),
                    max_raw_path2=planner_params.get('max_raw_path2', 5),
                    max_sample_num=planner_params.get('max_sample_num', 500),
                    reserve_num=planner_params.get('reserve_num', 50),
                    ratio_to_short=planner_params.get('ratio_to_short', 3),
                    sample_sz_p=planner_params.get('sample_sz_p', 0.5),
                    footprint=planner_params.get('footprint', [[0.0, 0.0]]),
                    occup_value=planner_params.get('occup_value', [100]),
                    max_time=planner_params.get('max_time', 0.05),
                    )

    def plan_best(self, start, q_ref, occupied, collision_checker,reset=False):
        self.planner.collision_checker = collision_checker
        paths = self.planner.plan(start, q_ref, reset=reset)

        best_cost = np.inf
        best_u_sol, best_states = None, None
        all_mpc_paths, all_planner_paths = [], []
        if paths is None or len(paths) == 0:
            return best_cost, best_u_sol, best_states, all_planner_paths, all_mpc_paths
        for path in paths:
            A, b, rX0 = self.decompose_and_rotate_env(path, start, occupied)
            x_sol, u_sol, _ = self.solver.solve(np.zeros((self.nlmodel.n_dims, )), 
                                            rX0[-1,:], obstacles={'A': A, 'b': b},
                                            reference={'x_ref':rX0,
                                                       'u_init':np.zeros((self.Nt+1, self.nlmodel.n_controls)),
                                                       'path':rX0,
                                                       'w_path':1.0})
            
            all_mpc_paths.append(x_sol)
            current_cost, states = self.eval_trajectory(u_sol, start, q_ref, dt = self.T/self.Nt)
            if current_cost < best_cost:
                best_cost, best_u_sol, best_states = current_cost, u_sol, states

        all_planner_paths.extend(paths)
        best_u_sol = upsample_mpc(best_u_sol, self.T, self.Nt, self.dt, method="step")
        best_states = upsample_mpc(best_states, self.T, self.Nt, self.dt, method="step")
        best_u_sol = best_u_sol[:self.Nt, :]
        best_states = best_states[:self.Nt, :]
        return best_cost, best_u_sol, best_states, all_planner_paths, all_mpc_paths

    def plan_multi(self, start, q_ref, occupied, collision_checker, reset=False):
        self.planner.collision_checker = collision_checker
        paths = self.planner.plan(start, q_ref, reset=reset)

        best_cost = np.inf
        best_u_sol, best_states = None, None
        all_mpc_paths, all_planner_paths, all_control_sequences = [], [], []
        for path in paths:
            A, b, rX0 = self.decompose_and_rotate_env(path, start, occupied)
            x_sol, u_sol, _ = self.solver.solve(np.zeros((self.nlmodel.n_dims, )), 
                                            rX0[-1,:], obstacles={'A': A, 'b': b},
                                            reference={'x_ref':rX0,
                                                       'u_init':np.zeros((self.Nt+1, self.nlmodel.n_controls)),
                                                       'path':rX0,
                                                       'w_path':1.0})
            
            u_sol = upsample_mpc(u_sol, self.T, self.Nt, self.dt, method="step")
            x_sol = upsample_mpc(x_sol, self.T, self.Nt, self.dt, method="step")
            u_sol = u_sol[:self.Nt, :]
            x_sol = x_sol[:self.Nt, :]

            all_mpc_paths.append(x_sol)
            all_control_sequences.append(u_sol)

        return all_planner_paths, all_mpc_paths, all_control_sequences

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
            print(f"Error in discretizing path: {e}")
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

    def eval_trajectory(self, u_seq, start, q_ref, dt=None):
        if dt is None:
            dt = self.dt
        state = start
        cost = 0
        states = [start]
        for t in range(self.Nt):
            u = u_seq[t, :]
            for i in range(10):
                state = self.nlmodel.dynamics_jax(state, u,
                                              dt=dt / 10,
                                              params=self.nlmodel.nominal_params)
            cost += (state - q_ref) @ self.Q @ (state - q_ref) + u @ self.R @ u
            states.append(state)
        cost += (state - q_ref) @ self.QT @ (state - q_ref)
        return cost, states

    def solve(self):
        grid, occupied, collision_checker = self.build_environment()
        best_cost, best_u_sol, best_states, all_planner_paths, all_mpc_paths = self.plan(
            self.start, self.q_ref, occupied, collision_checker
        )
        return best_states, best_cost, all_planner_paths, all_mpc_paths
    
def upsample_mpc(traj, T, Nt, dt, method="step"):
    """
    Upsample or interpolate a control trajectory to uniform time step `dt`.

    Args:
        traj: (Nt, control_dim) array of controls over horizon T
        T: total duration of the trajectory
        Nt: number of original control nodes
        dt: desired upsample step size
        method: "step" (zero-order hold) or "linear" (interpolation)

    Returns:
        up: (M, control_dim) array where M = floor(T / dt)
    """
    if traj is None:
        return None

    traj = np.asarray(traj)
    control_dim = traj.shape[1]

    # Original node times (uniform spacing)
    original_times = np.linspace(0, T, Nt, endpoint=True)
    
    # Desired sample times (uniform spacing up to T)
    desired_times = np.arange(0, T + 1e-9, dt)  # include last if close to T

    if method == "step":
        # For each desired time, find previous node index (step hold)
        idx = np.searchsorted(original_times, desired_times, side="right") - 1
        idx = np.clip(idx, 0, Nt - 1)
        return traj[idx, :]

    elif method == "linear":
        # Linear interpolation per control dimension
        up = np.empty((len(desired_times), control_dim))
        for d in range(control_dim):
            up[:, d] = np.interp(desired_times, original_times, traj[:, d])
        return up

    else:
        raise ValueError(f"Unknown interpolation method: {method}")
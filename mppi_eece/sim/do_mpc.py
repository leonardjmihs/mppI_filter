import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tqdm import tqdm
import time
import functools
import mppi_eece.jax_mppi.plot_utils as plot_utils
from mppi_eece.systems import HalfCar, Unicycle
from mppi_eece.jax_mppi.grid import OccupGrid
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


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    if not os.path.exists(path):
        path = filename + "0" + extension
    
    while os.path.exists(path):
        path = filename + str(counter) + extension
        counter += 1
    return path, counter-1

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
    try:
        X0, idx =  planner.discretizePath(p,Nt+1)
    except Exception as e:
        breakpoint()
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
                                                    
def wrap_to_pi(x: jnp.ndarray) -> jnp.ndarray:
    """Wraps x to lie within [-pi, pi]."""
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


def find_mpc(sim_state, q_ref, Nt, T, occupied, collision_checker, eval_trajectory, Q, QT, R, num_sides=10, lbu=None, ubu=None, solver=None):
    all_planner_paths = []
    all_mpc_paths = []
    if lbu is None:
        lbu = np.array([-np.pi, -3.0])
    if ubu is None:
        ubu = np.array([np.pi, 3.0])
    if solver is None:
        solver = create_unicycle_solver_with_time_varying_obstacles(lbu, 
                                                    ubu, 
                                                    N=Nt, 
                                                    Tf=T, 
                                                    num_sides=num_sides,
                                                    q_goal_pos = QT[0,0],
                                                    q_pos=Q[0,0],
                                                    r_control = R[0,0])

    # profiler = cProfile.Profile()
    # profiler.enable()
    planner = RRTStar( 
        collision_checker=collision_checker,    
        max_iter=200,
        expand_dist=1.0,
        goal_sample_rate=0.2,
        connect_circle_dist=10.0,
        )
    paths, _ = planner.find_path(sim_state, q_ref, reset=True) 

    # Handle the case where planner found no path (returns None or empty)
    if paths is None or len(paths) == 0:
        # disable profiler and write stats if available
        # if profiler is not None:
        #     profile_dir = os.path.join(os.getcwd(), "profiles")
        #     os.makedirs(profile_dir, exist_ok=True)

        #     profiler.disable()
        #     ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        #     base = f"test_rrt_{ts}"
        #     prof_path = os.path.join(str(profile_dir), base + ".prof")
        #     try:
        #         profiler.dump_stats(prof_path)
        #         print(f"Wrote profile .prof to: {os.path.abspath(prof_path)}")
        #     except Exception as e:
        #         print(f"Failed to write .prof file: {e}")

        # Return dummy outputs: no valid trajectory found
        best_cost = np.inf
        best_states = None
        best_u_sol = None
        all_planner_paths = []
        all_mpc_paths = []
        return best_cost, best_u_sol, best_states, all_planner_paths, all_mpc_paths, solver

    
    all_planner_paths.extend(paths)

    best_cost = np.inf
    best_states = None
    best_u_sol = None
    current_mpc_paths = []
    for i, path in enumerate(paths):
        start_time = time.time()
        x_sol, u_sol = find_controls(path, 
                                     sim_state, 
                                     Nt, occupied, 
                                     box=np.array([[1,2]]), 
                                     planner=planner, 
                                     solver=solver, 
                                     num_sides=num_sides)
        print(f"MPC solve time for path {i}: {time.time() - start_time:.4f} seconds")
        current_mpc_paths.append(x_sol)
                    
        current_cost, states = eval_trajectory(u_sol, sim_state, q_ref)
        if current_cost < best_cost:
            best_cost = current_cost
            best_u_sol = u_sol
            best_states = states

    all_mpc_paths.extend(current_mpc_paths)
                
    return best_cost, best_u_sol, best_states, all_planner_paths, all_mpc_paths, solver

def do_mpc(params):
    Nt = params['Nt']
    dt = params['dt']
    start = params['start']
    q_ref = params['q_ref']
    obs = params['obs']
    nlmodel = params['nlmodel']
    T = params['T']
    Q = params['Q']
    R = params['R']
    QT = params['QT']

    
    resolution = 0.5
    origin = np.array([-40, -10])
    wh = np.array([50/resolution, 20/resolution], dtype=np.int32)
    boundary = [[origin[0], origin[0]+wh[0]*resolution], [origin[1], origin[1]+wh[1]*resolution]]
    grid = OccupGrid(boundary, resolution)
    grid.find_occupancy_grid(obs, buffer=0.0)

    occupied = grid.find_all_occupied(obs)
    collision_checker = CollisionChecker(grid.occup_grid, origin, resolution, wh, occup_value=100)

    def eval_trajectory(u_seq, start, q_ref):
        state = start
        cost = 0
        states = [start]
        for t in range(Nt):
            u = u_seq[t, :]
            for i in range(10):
                state = nlmodel.dynamics(state, u, 0, dt=T/Nt/10, params=nlmodel.nominal_params)
            cost = cost + (state - q_ref) @ Q @ (state - q_ref) + u @ R @ u
            states.append(state)
        cost = cost + (state - q_ref) @ QT @ (state - q_ref)
        return cost, states

    best_cost, best_u_sol, best_states, all_planner_paths, all_mpc_paths, _ = find_mpc(start,
                                                    q_ref,
                                                    Nt,
                                                    T,
                                                    occupied,
                                                    Q=Q,
                                                    QT=QT,
                                                    R=R,
                                                    collision_checker=collision_checker,
                                                    eval_trajectory=eval_trajectory,
                                                    num_sides=9)

                
    return best_states, best_cost, all_planner_paths, all_mpc_paths     

def gen_and_save_mpc_results(params, outputs, foldername, counter, alg="mpc"):
    n_samples = params['n_samples']
    obs = params['obs']

    best_states = outputs[0]
    best_cost = outputs[1]
    all_planner_paths = outputs[2]
    all_mpc_paths = outputs[3] 

    states = np.array(best_states)
    all_planner_paths = np.array(all_planner_paths)
    all_mpc_paths = np.array(all_mpc_paths)

    pic = plot_utils.plot_simulation_result(states, obs, text="", max_arrows=10, planner_paths=all_planner_paths, mpc_paths=all_mpc_paths)
    pic_name = os.path.join(foldername, f'{alg}_{counter}.png')
    npz_name = os.path.join(foldername, f'{alg}_{counter}')
    
    np.savez(npz_name, 
             states=states,
             all_planner_paths=all_planner_paths,
             all_mpc_paths=all_mpc_paths
             )

    pic.savefig(pic_name)

def main():
    np.random.seed(30)
    params = {}
    params['dt'] = 0.2
    params['Nt'] = 30
    params['N_safe'] = 0
    params['N_mini'] = 0
    params['n_samples'] = 100
    params['n_mini'] = 200

    num_obs = 5
    low_val = jnp.array([-25.0, -5.0, 0.5])
    high_val = jnp.array([0.0, 5.0, 3.0])
    start = np.array([3.5, 0, np.pi])
    params['start'] = start
    q_ref = np.array([-30.0, -0.0, 0.0])
    params['q_ref'] = q_ref

    min_control = np.array([-np.pi, -3.0])
    max_control = np.array([np.pi, 3.0])
    params['obs'] = np.random.uniform(size=(num_obs, 3), low=low_val, high=high_val)
    while (np.any(np.linalg.norm(start[:2] - params['obs'][:, :2], axis=1) < np.sqrt(params['obs'][:, 2]))) or \
          (np.any(np.linalg.norm(q_ref[:2] - params['obs'][:, :2], axis=1) < np.sqrt(params['obs'][:, 2]))):
        params['obs'] = np.random.uniform(size=(num_obs, 3), low=low_val, high=high_val)

    params['nlmodel'] = Unicycle({"lb": min_control, "ub": max_control}, dt=params['dt'])
    params['T'] = 24
    sigma0 = np.diag(np.array([np.pi/4, 1.0]))
    params['sigma0'] = np.kron(np.eye(params['Nt']), sigma0)
    params['temperature'] = 1.0
    # params['Q'] = np.diag([3.0, 3.0, 0.0])
    # params['QT'] = params['Q'].copy() * 5
    params['Q'] = np.diag([1.0, 1.0, 0.0])
    params['QT'] = np.diag([5.0, 5.0, 0.0])
    params['R'] = np.diag([0.1, 0.1])

    params_copy = copy.deepcopy(params)
        
    # MPC only
    outputs_mpc =do_mpc(copy.deepcopy(params))
        
    foldername, counter = uniquify('sim_results')
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    param_name = os.path.join(foldername, f'params_{counter}')
    with open(param_name, 'w') as file:
        params_copy.pop('nlmodel')
        params_copy['start'] = params_copy['start'].tolist()
        params_copy['q_ref'] = params_copy['q_ref'].tolist()
        params_copy['obs'] = params_copy['obs'].tolist()
        params_copy['sigma0'] = params_copy['sigma0'].tolist()
        params_copy['Q'] = params_copy['Q'].tolist()
        params_copy['QT'] = params_copy['QT'].tolist()
        params_copy['R'] = params_copy['R'].tolist()
        json.dump(params_copy, file)

    gen_and_save_mpc_results(copy.deepcopy(params), outputs_mpc, foldername, counter, alg="mpc")
    # plt.close('all')
    plt.show()


if __name__ == "__main__":
    main()
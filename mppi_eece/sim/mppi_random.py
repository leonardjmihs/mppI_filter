
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mppi_eece.jax_mppi import plot_utils
from tqdm import tqdm
import time
import functools
from mppi_eece.systems import Unicycle
from mppi_eece.planners.grid import OccupGrid
from mppi_eece.planners.topo_prm import TopoPRMPlanner
from mppi_eece.planners.rrt_star import RRTStarPlanner
from mppi_eece.planners.collision_checker import CollisionChecker
from mppi_eece.planners.path_processor import PathProcessor
from mppi_eece.planners.sampler import Sampler, EllipsoidSampler, UniformSampler
from mppi_eece.jax_mppi.ca_mpc import *
import cProfile
import datetime
import casadi as ca
import os
import json
import copy
# from mppi_eece.planners.mppi_planners import MPPI_Planner_Occup
# from mppi_eece.planners.do_mpc import find_mpc, gen_and_save_mpc_results, do_mpc
from mppi_eece.jax_mppi.UKF_controller import do_ukf

matplotlib.use('Agg')

do_profiling=False
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    if not os.path.exists(path):
        path = filename + "0" + extension
    
    while os.path.exists(path):
        path = filename + str(counter) + extension
        counter += 1
    return path, counter-1

def easy_problem1():
    params = {}
    params['dt'] = 0.2
    params['Nt'] = 30
    params['n_samples'] = 100

    num_obs = 5
    low_val = jnp.array([-25.0, -5.0, 0.5])
    high_val = jnp.array([0.0, 5.0, 3.0])
    start = np.array([3.5, 0, np.pi])
    params['start'] = start
    q_ref = np.array([-30.0, -0.0, 0.0])
    params['q_ref'] = q_ref

    min_control = np.array([-np.pi, -3.0])
    max_control = np.array([np.pi, 3.0])
    params['obs'] = np.array([[-10,5.0,1.00],
                              [-10,-5.0,1.00]])

    params['nlmodel'] = Unicycle({"lb": min_control, "ub": max_control}, dt=params['dt'])
    params['T'] = 24
    sigma0 = np.diag(np.array([np.pi/4, 1.0]))
    params['sigma0'] = np.kron(np.eye(params['Nt']), sigma0)
    params['temperature'] = 1.0
    params['Q'] = np.diag([1.0, 1.0, 0.0])
    params['QT'] = np.diag([5.0, 5.0, 0.0])
    params['R'] = np.diag([0.1, 0.1])
    return params

def hard_problem1():
    params = {}
    params['dt'] = 0.2
    params['Nt'] = 30
    params['n_samples'] = 100

    num_obs = 5
    low_val = jnp.array([-25.0, -5.0, 0.5])
    high_val = jnp.array([0.0, 5.0, 3.0])
    start = np.array([3.5, 0, np.pi])
    params['start'] = start
    q_ref = np.array([-30.0, -0.0, 0.0])
    params['q_ref'] = q_ref

    min_control = np.array([-np.pi, -3.0])
    max_control = np.array([np.pi, 3.0])
    params['obs'] = np.array([[-10,0.0,5.0]])

    params['nlmodel'] = Unicycle({"lb": min_control, "ub": max_control}, dt=params['dt'])
    params['T'] = 24
    sigma0 = np.diag(np.array([np.pi/4, 1.0]))
    params['sigma0'] = np.kron(np.eye(params['Nt']), sigma0)
    params['temperature'] = 1.0
    params['Q'] = np.diag([1.0, 1.0, 0.0])
    params['QT'] = np.diag([5.0, 5.0, 0.0])
    params['R'] = np.diag([0.1, 0.1])
    return params

def rand_problem1():
    params = {}
    params['dt'] = 0.2
    params['Nt'] = 30
    params['n_samples'] = 100

    num_obs = 4
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
    params['Q'] = np.diag([1.0, 1.0, 0.0])
    params['QT'] = np.diag([5.0, 5.0, 0.0])
    params['R'] = np.diag([0.1, 0.1])
    return params

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def do_mppi(params, rng_key, do_mpc=True, ais_iters=0, base_alg=False, heuristic_weight=0.0, solver="ipopt"):
    dt = params['dt']
    Nt = params['Nt']
    n_samples = params['n_samples']
    start = params['start']
    q_ref = params['q_ref']
    obs = params['obs']
    nlmodel = params['nlmodel']
    T = params['T']
    sigma0 = params['sigma0']
    temperature = params['temperature']
    Q = params['Q']
    R = params['R']
    QT = params['QT']

    ns = 20
    ratio_sim_mppi = 10
    states = [start]
    costs = []
    min_cost = []
    total_costs = 0

    U = np.kron(np.ones((1, Nt)), [0.0, 1.0]).ravel()
    global_U = U.copy()
    global_Us = [global_U.copy()]
    global_us = [global_U.copy().reshape((-1, 2))]
    
    resolution = 0.5
    origin = np.array([-40, -10])
    wh = np.array([50/resolution, 20/resolution], dtype=np.int32)
    boundary = [[origin[0], origin[0]+wh[0]*resolution], [origin[1], origin[1]+wh[1]*resolution]]
    grid = OccupGrid(boundary, resolution)
    grid.find_occupancy_grid(obs, buffer=0.05)

    occupied = grid.find_all_occupied(obs)
    collision_checker = CollisionChecker(jnp.array(grid.occup_grid), origin, resolution, wh, occup_value=100)
    num_anci = 4
    planner = TopoPRMPlanner(collision_checker=collision_checker, resolution=resolution, 
                      max_raw_path=10, 
                      max_raw_path2=10,
                      reserve_num=num_anci, 
                      ratio_to_short=2.0,
                      sample_sz_p=0.0,
                      occup_value=100,
                      max_time=0.1)
    # planner.occup_grid = grid.occup_grid
    # planner.origin = origin
    # planner.resolution = resolution
    # planner.wh = wh
    dis = nlmodel.control_bounds[1][1] * nlmodel.dt * Nt
    ns = 10

    dxdt, state, control = nlmodel.cas_ode()
    ode = ca.Function('ode', [state, control], [dxdt]) 
    f = cas_shooting_solver(nlmodel, int(Nt/2), ns=ns, dt=nlmodel.dt*2, ode=ode, solver=solver)
    box = np.array([[1, 2]])
    timestep_reached = -1

    mppi_planner = MPPI_Planner_Occup(sigma=sigma0,
                                      Q=Q,
                                      QT=QT,
                                      R=R,
                                      temperature=temperature,
                                      system=nlmodel, num_anci=num_anci, 
                                      n_samples=n_samples, 
                                      N=Nt,
                                      tolerance=0.4,
                                      occup_value=100)
    state = start
    U = np.kron(np.ones((1, Nt)), [0.0, 1.0]).ravel()
    global_U = U.copy()
    sim_state = state.copy()
    cost = 0
    timestep_prog = tqdm(np.arange(0, T, dt))
    iter = 0

    def eval_trajectory(u_seq, start, system, Nt, q_ref):
        state = start
        cost = 0
        for t in range(Nt):
            u = u_seq[t, :]
            state = system.dynamics(state, u, 0, dt=dt, params=system.nominal_params)
            cost = cost + (state - q_ref) @ Q @ (state - q_ref) + u @ R @ u
        cost = cost + (state - q_ref) @ QT @ (state - q_ref)
        return cost

    sampled_states = []
    for t in timestep_prog:
        rng_key, subkey = jax.random.split(rng_key)
        U_anci = np.tile(np.array(global_U), [num_anci+1, 1])
        if not base_alg:
            if do_mpc:
                find_controls = functools.partial(find_Nonlin_Controls, 
                                                  start=sim_state, 
                                                  solver=f, 
                                                  dis=dis, 
                                                  system=nlmodel, 
                                                  Nt=int(Nt/2), occupied=occupied, box=box, planner=planner, ns=ns)
                paths, _ = planner.findTopoPaths(sim_state, q_ref, reset=True) 

                if paths is not None:
                    num_paths = len(paths) 
                    for i, path in enumerate(paths):
                        x_sol, u_sol, _ = find_controls(path)
                        u_sol = np.repeat(u_sol, repeats=2, axis=0)
                        try:
                            U_anci[i+1, :] = u_sol[:Nt, :].reshape((Nt*2))
                        except:
                            breakpoint()
        outputs = mppi_planner.mppi_mmodal(sim_state, global_U, U_anci, subkey, q_ref, collision_checker)
        best_u = outputs[0]
        new_u = outputs[1]
        new_U = outputs[2]
        min_cost = outputs[3]
        collision_free = outputs[4]
        all_costs = outputs[5]
        all_state_seq = outputs[6]


        cost = eval_trajectory(outputs[1], sim_state, nlmodel, Nt, q_ref)

        for i in range(ratio_sim_mppi):
            sim_state = nlmodel.dynamics(sim_state, best_u[0], 0, dt=dt/ratio_sim_mppi, params=nlmodel.nominal_params)
            dist = np.linalg.norm((sim_state - q_ref)[0:2])
            if (dist<0.5) and timestep_reached==-1:
                timestep_reached = t / dt

        sampled_states.append(all_state_seq) 
        states.append(sim_state)
        global_us.append(best_u)
        global_Us.append(global_U.copy())
        costs.append(cost)
        total_costs += cost
        iter += 1

    return (states,
            costs, 
            sampled_states,
            timestep_reached, 
            global_us,)

def upsample_mpc_controls(u_mpc, T, Nt, dt, method="step"):
    if u_mpc is None:
        return None
    u_mpc = np.asarray(u_mpc)
    control_dim = u_mpc.shape[1]
    times = np.arange(0, T, dt)
    if method == "step":
        mpc_dt = T / float(Nt)
        idx = np.floor(times / mpc_dt).astype(int)
        idx = np.clip(idx, 0, Nt - 1)
        return u_mpc[idx, :]
    else:
        # linear interpolation per control dimension
        t_mpc = np.linspace(0, T, Nt, endpoint=False)
        up = np.zeros((len(times), control_dim))
        for d in range(control_dim):
            up[:, d] = np.interp(times, t_mpc, u_mpc[:, d])
        return up

def compare_mppi_to_mpc(mppi_outputs, params, rng_key, do_mpc=True, ais_iters=0, base_alg=False, heuristic_weight=0.0, solver="ipopt"):

    Nt = params['Nt']
    start = params['start']
    q_ref = params['q_ref']
    obs = params['obs']
    nlmodel = params['nlmodel']
    T = params['T']
    Q = params['Q']
    R = params['R']
    QT = params['QT']

    states = mppi_outputs[0]
    global_us = mppi_outputs[4]
    obs = params['obs']
    resolution = 0.5
    origin = np.array([-40, -10])
    wh = np.array([50/resolution, 20/resolution], dtype=np.int32)
    boundary = [[origin[0], origin[0]+wh[0]*resolution], [origin[1], origin[1]+wh[1]*resolution]]
    grid = OccupGrid(boundary, resolution)
    grid.find_occupancy_grid(obs, buffer=0.0)

    occupied = grid.find_all_occupied(obs)
    collision_checker = CollisionChecker(grid.occup_grid, origin, resolution, wh, occup_value=100)
    optimal_us = []
    optimal_states = []
    optimal_costs = []
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

    solver = None
    for i in range(len(states)-1):
        best_cost, best_u_sol, best_states, all_planner_paths, all_mpc_paths,solver = find_mpc(states[i],
                        q_ref,
                        Nt,
                        T,
                        occupied,
                        Q=Q,
                        QT=QT,
                        R=R,
                        collision_checker=collision_checker,
                        eval_trajectory=eval_trajectory,
                        num_sides=9,
                        solver=solver)
            
        optimal_us.append(best_u_sol)
        optimal_costs.append(best_cost)
        optimal_states.append(best_states)

    # Compute RMSE between MPPI and MPC control trajectories.
    # For each timestep i, if MPC failed (best_u_sol is None) we skip that timestep.
    # MPC provides Nt controls applied over horizon T with control step T/Nt.
    # MPPI provides controls at the simulation dt. We upsample the MPC controls
    # to the MPPI dt by holding each MPC control constant over its interval.

    rmse_per_timestep = []
    for i, u_mpc in enumerate(optimal_us):
        # skip if MPC failed to produce a control sequence
        if u_mpc is None:
            continue
        u_mpc_up = upsample_mpc_controls(u_mpc, T, Nt, params['dt'], method="step")
        if u_mpc_up is None:
            continue

        # corresponding MPPI control sequence at this timestep
        try:
            u_mppi = np.asarray(global_us[i])
        except Exception:
            # if indexing fails, skip
            continue

        # align lengths (compare up to the shorter horizon)
        L = min(len(u_mpc_up), len(u_mppi))
        if L <= 0:
            continue
        err = rmse(u_mpc_up[:L], u_mppi[:L])
        rmse_per_timestep.append(err)

    # summary stats
    avg_rmse = float(np.mean(rmse_per_timestep)) if len(rmse_per_timestep) > 0 else float('nan')
    print(f"MPPI vs MPC control RMSE over {len(rmse_per_timestep)} timesteps: avg={avg_rmse:.4f}")

    return mppi_outputs, (optimal_us, optimal_states, optimal_costs, rmse_per_timestep)


def gen_and_save_mppi_results(params, outputs, foldername, counter, alg="mpc_ais"):
    obs = params['obs']
    q_ref = params['q_ref']

    states = outputs[0]
    costs = outputs[1]
    sampled_states = outputs[2]
    

    states = np.array(states)
    costs = np.array(costs)
    sampled_states = np.array(sampled_states)
    pic = plot_utils.plot_simulation_result(states, obs, goal=q_ref[:2], text="",  max_arrows=10)
    pic_name = os.path.join(foldername, f'mppi_{alg}_{counter}.png')
    gif_name = os.path.join(foldername, f'mppi_{alg}_{counter}.gif')
    percent_safe_name = os.path.join(foldername, f'mppi_{alg}_{counter}.png')
    npz_name = os.path.join(foldername, f'mppi_{alg}_{counter}')
    anim = plot_utils.animate_simulation_with_sampled_states(states, obs,goal=q_ref[:2], sampled_xs=sampled_states)
    anim.save(gif_name)
    np.savez(npz_name, 
             states=states,
             costs=costs,
             )

    pic.savefig(pic_name)

def main(args):
    solver = args.solver if hasattr(args, 'solver') else "ipopt"
    problem_type = args.problem_type if hasattr(args, 'problem_type') else "random"
    trials = 1
    rng_keys = jax.random.split(jax.random.PRNGKey(0), trials)
    np.random.seed(30)
    for trial in range(trials):
        if problem_type == "easy":
            params = easy_problem1()
        elif problem_type == "hard":
            params = hard_problem1()
        else:
            params = rand_problem1()
        params_copy = copy.deepcopy(params)
        
        # MPC only
        # outputs_ukf = do_ukf(copy.deepcopy(params))
        # breakpoint()
        outputs_mppi = do_mppi(copy.deepcopy(params), rng_keys[trial], base_alg=True, solver=solver)
        outputs_mpc = do_mpc(copy.deepcopy(params))
    
        # outputs = do_mppi(params, rng_key, do_mpc=do_mpc, ais_iters=ais_iters, base_alg=base_alg, heuristic_weight=heuristic_weight, solver=solver)

        foldername, counter = uniquify('sim_results')
        os.mkdir(foldername)   
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

        gen_and_save_mppi_results(copy.deepcopy(params), outputs_mppi, foldername, counter, alg="mppi")
        gen_and_save_mpc_results(copy.deepcopy(params), outputs_mpc, foldername, counter, alg="mpc")

        outputs_mppi, outputs_mpc = compare_mppi_to_mpc(outputs_mppi, copy.deepcopy(params), rng_keys[trial], do_mpc=True, base_alg=True, solver=solver) 
        plt.close('all')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', nargs='?', default="ipopt", help='filename')
    args = parser.parse_args()
    main(args)
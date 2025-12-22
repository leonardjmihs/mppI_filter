import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mppi_eece.sim import plot_utils
from tqdm import tqdm
import time
import functools
from mppi_eece.systems import Unicycle
from mppi_eece.sim.grid import OccupGrid
from mppi_eece.ancillary_controller import AncillaryController
from mppi_eece.jax_mppi.collision_checker import CollisionChecker
from mppi_eece.planners.sampler import Sampler, EllipsoidSampler, UniformSampler
import cProfile
import datetime
import casadi as ca
import os
import json
import copy
from mppi_eece.jax_mppi.mppi_planners import MPPI_Planner_Occup
from mppi_eece.sim.do_mpc import gen_and_save_mpc_results, do_mpc
from mppi_eece.sim.UKF_controller import do_ukf, do_ukf_with_pcrb
from mppi_eece.sim.ckf_controller import do_ckf
from mppi_eece.sim.gsf_controller import do_gsf
# from mppi_eece.sim.ukf_filterpy import do_ukf_filterpy

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

def trivial_problem1():
    params = {}
    # params['dt'] = 0.5
    # params['Nt'] = 3
    params['dt'] = 0.2
    params['Nt'] = 9
    # params['Nt'] = 30

    params['n_samples'] = 100

    num_obs = 5
    low_val = jnp.array([-25.0, -5.0, 0.5])
    high_val = jnp.array([0.0, 5.0, 3.0])
    start = np.array([3.5, 0, np.pi])
    params['start'] = start
    q_ref = np.array([-30.0, 10.0, 0.0])
    params['q_ref'] = q_ref

    min_control = np.array([-np.pi, -3.0])
    max_control = np.array([np.pi, 3.0])
    params['obs'] = np.array([[-15,5.0,4.0]])

    params['nlmodel'] = Unicycle({"lb": min_control, "ub": max_control}, dt=params['dt'])
    params['T'] = 24
    sigma0 = np.diag(np.array([np.pi/4, 1.0]))*100
    # params['sigma0'] = np.kron(np.eye(params['Nt']), sigma0)
    params['sigma0'] = np.kron(np.diag(1+np.arange(params['Nt'])[::-1]), sigma0)
    params['temperature'] = 10.0
    params['Q'] = np.diag([1.0, 1.0, 0.0])
    params['QT'] = np.diag([5.0, 5.0, 0.0])
    params['R'] = np.diag([0.1, 0.1])
    return params

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
    # return np.sqrt(np.mean((e - b) ** 2))
    return np.sqrt((a-b) @ (a-b).T)

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
    global_us = mppi_outputs[-1]
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

    ancillary_controller = AncillaryController(mpc_params=params, 
                                               solver_type='acados',
                                               planner_type='rrt')
    for i in range(len(states)-1):
        best_cost, best_u_sol, best_states, all_planner_paths, all_mpc_paths = ancillary_controller.plan_best(
                                                    start,
                                                    q_ref,
                                                    occupied,
                                                    collision_checker=collision_checker)
            
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
            rmse_per_timestep.append(np.nan)
            continue


        # corresponding MPPI control sequence at this timestep
        try:
            u_mppi = np.asarray(global_us[i]).reshape(u_mpc.shape)
        except Exception:
            # if indexing fails, skip
            rmse_per_timestep.append(np.nan)
            continue

        # align lengths (compare up to the shorter horizon)
        L = min(len(u_mpc), len(u_mppi))
        if L <= 0:
            rmse_per_timestep.append(np.nan)
            continue
        err = rmse(u_mpc[:L], u_mppi[:L])
        rmse_per_timestep.append(err)
    return optimal_us, optimal_states, optimal_costs, rmse_per_timestep

def plot_rmse_pcrg(rmse_per_timestep, pcrb_history, J_history, foldername, counter, alg="ukf"):
    plt.figure()

    pcrb_std = np.sqrt(np.array([np.linalg.inv(J).diagonal() for J in J_history]))
    for dim in range(2):
        plt.subplot(3,1,dim+1)
        plt.plot(pcrb_std[:, dim], label=f'PCRB std dim {dim}')
        plt.legend()
    
    plt.subplot(3,1,3)
    trace_rmse = []
    for rmse in rmse_per_timestep:
        if np.isscalar(rmse):
            trace_rmse.append(np.nan)
        else:
            trace_rmse.append(np.trace(rmse))
    valid_idx = ~np.isnan(trace_rmse)
    plt.plot( np.array(pcrb_history)[valid_idx] / np.array(trace_rmse)[valid_idx], label='PCRB efficiency')

    plt.legend()
    pic_name = os.path.join(foldername, f'rmse_pcrb_{alg}_{counter}.png')
    plt.savefig(pic_name)
    plt.close()

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
        elif problem_type == "trivial":
            params = trivial_problem1()
        else:
            params = rand_problem1()
        params_copy = copy.deepcopy(params)
        
        print(params)
        # MPC only
        # outputs_ukf = do_ukf(copy.deepcopy(params))
        outputs_ukf = do_ukf_with_pcrb(copy.deepcopy(params))
        # outputs_gsf = do_gsf(copy.deepcopy(params))

        # outputs_ckf = do_ckf(copy.deepcopy(params))

        # MPPI only
        # outputs_mppi = do_mppi(copy.deepcopy(params), rng_keys[trial], do_mpc=False)

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
        gen_and_save_mppi_results(copy.deepcopy(params), outputs_ukf, foldername, counter, alg="ukf")
        # Plot UKF diagnostics (covariance trace, innovations, resets) into the results folder
        try:
            # outputs_ukf expected: (states, costs, sigma_sequences, cov_norms, cov_traces, innovations, reset_flags, cov_trace_threshold, innovation_threshold)
            # cov_traces = None
            # cov_norms = None
            # innovations = None
            # reset_flags = None
            # cov_thresh = None
            # innov_thresh = None

            # if isinstance(outputs_ukf, (tuple, list)) and len(outputs_ukf) >= 9:
            #     _, _, _, cov_norms, cov_traces, innovations, reset_flags, cov_thresh, innov_thresh = outputs_ukf
            # elif isinstance(outputs_ukf, (tuple, list)) and len(outputs_ukf) >= 5:
            #     # older fallback: (states, costs, sigma_sequences, cov_norms, reset_flags)
            #     _, _, _, cov_norms, reset_flags = outputs_ukf
            states = outputs_ukf[0]
            costs = outputs_ukf[1]
            sigma_sequences = outputs_ukf[2]
            cov_norms = outputs_ukf[3]
            cov_traces = outputs_ukf[4]
            innovations = outputs_ukf[5]
            reset_flags = outputs_ukf[6]
            cov_thresh = outputs_ukf[7]
            innov_thresh = outputs_ukf[8]


            # prefer plotting cov_trace if available
            if cov_traces is not None:
                fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
                ax[0].plot(cov_traces, label='cov trace')
                if cov_thresh is not None:
                    ax[0].axhline(cov_thresh, color='r', linestyle='--', label='cov_trace_thresh')
                ax[0].set_ylabel('trace(P)')
                ax[0].legend()
                ax[0].grid(True)

                if innovations is not None:
                    ax[1].plot(np.abs(innovations), label='|innovation|')
                    if innov_thresh is not None:
                        ax[1].axhline(innov_thresh, color='r', linestyle='--', label='innovation_thresh')
                else:
                    ax[1].plot([], label='|innovation|')

                # mark resets
                if reset_flags is not None and innovations is not None:
                    reset_idx = [i for i, v in enumerate(reset_flags) if v]
                    if len(reset_idx) > 0:
                        ax[1].plot(reset_idx, [np.abs(innovations[i]) for i in reset_idx], 'rx', label='resets')

                ax[1].set_ylabel('innovation')
                ax[1].set_xlabel('timestep')
                ax[1].legend()
                ax[1].grid(True)

                fig.tight_layout()
                fig.savefig(os.path.join(foldername, 'ukf_cov_innov.png'))
                plt.close(fig)
            elif cov_norms is not None:
                # fallback: plot cov Frobenius norm and attempt to plot innovations
                fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
                ax[0].plot(cov_norms, label='cov Frobenius norm')
                ax[0].set_ylabel('||P||_F')
                ax[0].legend()
                ax[0].grid(True)

                if innovations is not None:
                    ax[1].plot(np.abs(innovations), label='|innovation|')
                else:
                    ax[1].plot([], label='|innovation|')

                if reset_flags is not None and innovations is not None:
                    reset_idx = [i for i, v in enumerate(reset_flags) if v]
                    if len(reset_idx) > 0:
                        ax[1].plot(reset_idx, [np.abs(innovations[i]) for i in reset_idx], 'rx', label='resets')

                ax[1].set_ylabel('innovation')
                ax[1].set_xlabel('timestep')
                ax[1].legend()
                ax[1].grid(True)

                fig.tight_layout()
                fig.savefig(os.path.join(foldername, 'ukf_covnorm_innov.png'))
                plt.close(fig)
        except Exception:
            # best-effort plotting; don't fail the run if plotting breaks
            pass
        # gen_and_save_mppi_results(copy.deepcopy(params), outputs_mppi, foldername, counter, alg="mppi")
        # gen_and_save_mpc_results(copy.deepcopy(params), outputs_mpc, foldername, counter, alg="mpc")

        outputs_ground_truth = compare_mppi_to_mpc(outputs_ukf, copy.deepcopy(params), rng_keys[trial], do_mpc=True, base_alg=True, solver=solver) 
        plot_rmse_pcrg(outputs_ground_truth[3], outputs_ukf[9], outputs_ukf[10], foldername, counter, alg="ukf")
        plt.close('all')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', nargs='?', default="ipopt", help='filename')
    parser.add_argument('--problem_type', nargs='?', default="trivial", help='problem type: trivial, easy, hard, random')

    args = parser.parse_args()
    main(args)
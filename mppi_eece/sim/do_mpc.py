import jax.numpy as jnp
import matplotlib.pyplot as plt
import casadi as ca
import os
import json
import copy 
import numpy as np
import time

import mppi_eece.sim.plot_utils as plot_utils
from mppi_eece.systems import Unicycle
from mppi_eece.sim.grid import OccupGrid
from mppi_eece.planners.rrt_star import RRTStarPlanner
from mppi_eece.ancillary_controller import AncillaryController
from mppi_eece.mpc_solvers.acados_solver import AcadosSolver
from mppi_eece.jax_mppi.collision_checker import CollisionChecker
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

    ancillary_controller = AncillaryController(mpc_params=params, 
                                               solver_type='acados',
                                               planner_type='rrt')
    best_cost, best_u_sol, best_states, all_planner_paths, all_mpc_paths = ancillary_controller.plan_best(start,
                                                    q_ref,
                                                    occupied,
                                                    collision_checker=collision_checker)

                
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
    import time
    np.random.seed(31)
    params = {}
    params['dt'] = 0.2
    params['Nt'] = 30
    params['N_safe'] = 0
    params['N_mini'] = 0
    params['n_samples'] = 100
    params['n_mini'] = 200

    num_obs = 8
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
    params['T'] = 12
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
    start_time = time.time()
    outputs_mpc =do_mpc(copy.deepcopy(params))
    end_time = time.time()
    print(f"MPC Time taken: {end_time - start_time} seconds")
        
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
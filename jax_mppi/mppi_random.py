import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from jax_mppi import plot_utils
from tqdm import tqdm
import time
import functools
from systems import Unicycle, Unicycle_HJ
from jax_mppi.grid import OccupGrid
from jax_mppi.topo_prm import TopoPRM
from jax_mppi.ca_mpc import *
import casadi as ca
import os
import json
import copy 
from jax_mppi.nested_mppi_planners import  MPPI_Planner_Occup
import hj_reachability as hj
# from jax_mppi.acd_halfcar import create_car_trajectory_solver, create_hcar_model, solve_trajectory
# from jax_mppi.ca_mpc import rotate_2dvectors, wrap_to_pi

# jax.config.update('jax_platform_name', 'cpu')
matplotlib.use('Agg')

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    if not  os.path.exists(path):
        path = filename + "0" + extension
    
    while os.path.exists(path):
        path = filename + str(counter) + extension
        counter += 1
    return path, counter-1

def create_elliptical_covariance(sigma_x, sigma_y, theta):
    """
    Create a covariance matrix for an elliptical Gaussian with rotation.
    
    Parameters:
        sigma_x (float): Standard deviation along the x-axis before rotation
        sigma_y (float): Standard deviation along the y-axis before rotation
        theta (float): Rotation angle in radians (counter-clockwise)
        
    Returns:
        cov (2x2 numpy array): Covariance matrix
    """
    # Create rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    # Create diagonal matrix of variances (before rotation)
    D = np.diag([sigma_x**2, sigma_y**2])
    
    # Compute rotated covariance matrix: Σ = R D R^T
    cov = R @ D @ R.T
    
    return cov

def wrap_to_pi(x: jnp.ndarray) -> jnp.ndarray:
    """Wraps x to lie within [-pi, pi]."""
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi

def rand_problem1():
    params = {}
    params['dt'] = 0.2  # Time step
    params['Nt'] = 30  # horizon for MPPI
    params['N_safe'] = 0
    params['N_mini'] = 0
    params['n_samples'] =100  # Number of samples for MPPI
    params['n_mini'] = 200

    num_obs=5
    # num_safe_zones=40
    low_val =jnp.array([-25.0,-5.0,0.5])
    high_val=jnp.array([0.0,5.0,3.0])
    resolution = 10
    start = np.array([3.5, 0, np.pi])
    params['start'] = start
    q_ref = np.array([-30.0, -0.0, 0.0])
    params['q_ref'] = q_ref  # Reference state

    min_control = np.array([-np.pi, -3.0])
    max_control = np.array([np.pi, 3.0])
    # params['obs'] = jax.random.uniform(jax.random.PRNGKey(0), (num_obs,3), minval=low_val, maxval=high_val)
    params['obs'] = np.random.uniform(size=(num_obs,3), low=low_val, high=high_val)
    while (np.any(np.linalg.norm(start[:2] - params['obs'][:,:2], axis=1) < np.sqrt(params['obs'][:,2]))) or \
          (np.any(np.linalg.norm(q_ref[:2] - params['obs'][:,:2], axis=1) < np.sqrt(params['obs'][:,2]))):
        params['obs'] = np.random.uniform(size=(num_obs,3), low=low_val, high=high_val)

    

    params['nlmodel'] =  Unicycle({"lb":min_control,
                        "ub":max_control}, dt=params['dt'])
    # params['T'] = 24 # Total time for simulation
    # params['T'] = 16 # Total time for simulation
    params['T'] = 24 # Total time for simulation
    sigma0 = np.diag(np.array([np.pi/4, 1.0]))
    params['sigma0'] = np.kron(np.eye(params['Nt']), sigma0)
    # params['temperature'] = 0.1
    params['temperature'] = 1.0
    params['Q'] = np.diag([3.0,3.0, 0.0])  # Weight for stz`te
    params['QT'] = params['Q'].copy()*5
    params['R'] = np.diag([10.0, 1.0])  # Weight for control

    dynamics = Unicycle_HJ(params['obs'],
                max_control = jnp.array([jnp.pi, 3.0]),
                 min_control = jnp.array([-jnp.pi, -3.0])
                 )
    scale = np.array((50,50,50))
    box_r = params['N_mini']*params['dt']*max_control[1]*0.8 / 2
    wh = np.array([box_r, box_r])*2
    target_time = -1.0*params['N_mini']*params['dt']

    params['safe_zones'] = np.zeros((2,3))
    params['safe_zones'][0] = q_ref
    params['safe_zones'][1, :2] = params['start'][:2]
    resolution = 0.5
    origin = np.array([-40,-10])
    topo_wh = np.array([50/resolution, 20/resolution]) # width, height
    # boundary = [[-40,10], [-10, 10]]
    boundary = [[origin[0], origin[0]+topo_wh[0]*resolution], [origin[1], origin[1]+topo_wh[1]*resolution]]
    grid = OccupGrid(boundary, resolution)
    grid.find_occupancy_grid(params['obs'], buffer=0.05)
    return params

def get_reachable_set_hjr(pos, dynamics, scale, wh, target_time, tol=0.25, solver_settings=None):
    if solver_settings is None:
        solver_settings = hj.SolverSettings.with_accuracy("low")

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(
        lo=np.array([-wh[0],-wh[1], -np.pi])+np.array([*pos[:2], 0]),
        hi=np.array([*wh, np.pi])+np.array([*pos[:2], 0])
        ),
        scale,
        periodic_dims=2)
    values = jnp.linalg.norm(grid.states[..., :2]-pos[:2], axis=-1) - tol # 0.25 is the "radius" of acceptance for safety
    target_values = hj.step(solver_settings, dynamics, grid, 0., values, target_time, progress_bar=False)
    return target_values

def check_reachability_hjr(values, center, pos, scale, wh):
    origin = center[:2] - wh/2
    dxdy = wh/scale
    index = ((pos[:2]-origin)/dxdy).astype(int)
    if index[0] < 0 or index[1] <0 or index[0] >= scale[0] or index[1] >= scale[1]:
        return False
    return min(values[index[0], index[1],:]) < 0

def check_reachability_multiple_hjr(target_values, safe_zones, pos, scale,wh):
    # diff_sz = np.linalg.norm(pos -safe_zones[:,:2])
    for i, sz in enumerate(safe_zones):
        center = sz[:2]
        values = target_values[i]
        if check_reachability_hjr(values, center, pos, scale, wh):
            return True
    return False


        

def do_mppi_ais_mpc(params, rng_key, do_mpc=True, ais_iters=0, base_alg=False, heuristic_weight=0.0,solver="ipopt"):
    dt = params['dt']
    Nt = params['Nt']
    N_safe = params['N_safe']
    N_mini = params['N_mini']
    n_samples = params['n_samples']
    n_mini = params['n_mini']
    start = params['start']
    q_ref = params['q_ref']
    safe_zones = params['safe_zones']
    obs = params['obs']
    nlmodel = params['nlmodel']
    T = params['T']
    sigma0 = params['sigma0']
    temperature = params['temperature']
    Q = params['Q']
    R = params['R']
    QT = params['QT']


    # ns = safe_zones.shape[0]
    ns = 20
    ratio_sim_mppi = 1  # how fast simulator run faster than mppi controller

    states = [start]  # `states` is a list containing the states over time
    costs = []
    min_cost = []
    total_costs = 0

    # U = np.kron(np.ones((1,Nt)), [0.0,0.0]).ravel()
    U = np.kron(np.ones((1,Nt)), [0.0,1.0]).ravel()
    global_U = U.copy()
    global_Us = [global_U.copy()]
    global_us = [global_U.copy().reshape((-1,2))]
    
    resolution = 0.5
    origin = np.array([-40,-10])
    wh = np.array([50/resolution, 20/resolution], dtype=np.int32) # width, height
    # boundary = [[-40,10], [-10, 10]]
    boundary = [[origin[0], origin[0]+wh[0]*resolution], [origin[1], origin[1]+wh[1]*resolution]]
    grid = OccupGrid(boundary, resolution)
    grid.find_occupancy_grid(obs, buffer=0.05)

    virtual_obs =  np.ones([int(wh[1]), int(wh[0])]).T
    occupied = grid.find_all_occupied(obs)

    num_anci = 3
    # planner = TopoPRM(None, max_raw_path2=num_anci)
    radius = 0.1
    footprint = np.array([[radius, 0], 
                            [0,radius],
                            [-radius,0],
                            [0,-radius]])
    # footprint = np.array([[0, 0]])
    planner = TopoPRM(None, resolution=resolution, 
                        max_raw_path=10, 
                        max_raw_path2=10,
                        reserve_num=num_anci, 
                        ratio_to_short=2.0,
                        sample_sz_p=0.0,
                        occup_value=100,
                        max_time=0.1
                        )
    if base_alg:
        planner = TopoPRM(None, resolution=resolution, 
                            max_raw_path=10, 
                            max_raw_path2=10,
                            reserve_num=num_anci, 
                            ratio_to_short=2.0,
                            sample_sz_p=0.0,
                            occup_value=100,
                            max_time=np.inf
                            )
    planner.occup_grid = grid.occup_grid
    planner.origin = origin
    planner.resolution = resolution
    planner.wh = wh
    planner.safe_zones = safe_zones

    dis = nlmodel.control_bounds[1][1]*nlmodel.dt * Nt
    # total_sampled_states = [np.ones((n_samples//(ais_iters+1), Nt, 3))*np.nan]
    # if N_mini == 0:
    #     total_con_states = [np.ones((n_samples//(ais_iters+1), 1, 3))*np.nan]
    # else:
    #     total_con_states = [np.ones((n_samples//(ais_iters+1), N_mini, 3))*np.nan]
    
    # total_sampled_states[0][0,0] = np.squeeze(states) # Closed-loop simulation
    # breakpoint()
    # total_con_states[0][0,0] = np.squeeze(states) # Closed-loop simulation

    not_inf = 0
    number_safe_avg = 0
    safety_hist = []
    start_time = time.time()
    trial_hz = 0
    ns=10

    dxdt, state, control = nlmodel.cas_ode()
    ode = ca.Function('ode', [state, control], [dxdt]) 
    f = cas_shooting_solver(nlmodel, int(Nt/2),ns=ns, dt = nlmodel.dt*2, ode=ode, solver=solver)

    # model = create_hcar_model() 
    # solver_acd = create_car_trajectory_solver(lbu=nlmodel.control_bounds[0], ubu=nlmodel.control_bounds[1],N=int(Nt/2), Tf=T, num_sides=ns)

    path_list = []
    box = np.array([[1,2]])
    timestep_reached = -1
    mppi_planner = MPPI_Planner_Occup(sigma=sigma0,
                                Q=Q,
                                QT=QT,
                                R=R,
                                temperature=temperature,
                                system=nlmodel, num_anci=num_anci, 
                                n_samples=n_samples, 
                                n_mini=n_mini,
                                N=Nt,
                                N_mini=N_mini,
                                N_safe=N_safe,
                                max_sz=len(safe_zones),
                                tolerance=0.4,
                                footprint=footprint,
                                occup_value=100,
                                heuristic_weight= heuristic_weight
                                )
    state = start  # [x, y, theta, v]
    U = np.kron(np.ones((1,Nt)), [0.0,1.0]).ravel()
    global_U = U.copy()
    # U_anci = np.tile(global_U, [num_anci,1])
    sim_state = state.copy()
    cost = 0
    timestep_prog = tqdm(np.arange(0, T, dt))
    iter = 0
    prev_U_anci = None 
    number_safe_hist = []
    percent_safe_hist = []
    num_paths = 0

    topo_times = []
    mpc_times = []
    mppi_times = []

    def eval_trajectory(u_seq, start, system, Nt, q_ref):
        state = start
        cost = 0
        for t in range(Nt):
            u = u_seq[t,:]
            state = system.dynamics(state, u, 0, dt=dt, params=system.nominal_params)
            cost = cost + (state - q_ref) @ Q @ (state - q_ref) + u @ R @ u
        cost = cost + (state - q_ref) @ QT @ (state - q_ref)
        return cost

    for t in timestep_prog:
        rng_key, subkey = jax.random.split(rng_key)
        st = time.time()
        U_anci = np.tile(np.array(global_U    ), [num_anci+1,1])
        if not base_alg:
            if do_mpc:
                # U_anci = np.tile(U, [num_anci,1])
                # U_anci = np.tile(global_U, [num_anci,1])
                find_controls = functools.partial(find_Nonlin_Controls, 
                        start=sim_state, 
                        solver=f, 
                        dis=dis, 
                        system=nlmodel, 
                        Nt=int(Nt/2),occupied=occupied, box=box,planner=planner, ns=ns)
                st_topo_time = time.time()
                paths, _ = planner.findTopoPaths(sim_state, q_ref, reset=True) 
                topo_times.append(time.time()-st_topo_time)

                if paths is not None:
                    num_paths = len(paths) 
                    start_mpc_time = time.time()
                    for i,path in enumerate(paths):
                        x_sol, u_sol, _ = find_controls(path)
                        u_sol = np.repeat(u_sol,repeats=2,axis=0)
                        try:
                            U_anci[i+1,:] = u_sol[:Nt,:].reshape((Nt*2))
                        except:
                            breakpoint()
                        print(f"MPC Time:{time.time()-st}")
                        mpc_times.append(time.time()-start_mpc_time)
            st_mppi_time = time.time()
            outputs = mppi_planner.mppi_mmodal(sim_state, global_U, U_anci, subkey, q_ref, safe_zones, jnp.array(grid.occup_grid),origin,resolution,wh=wh, ais_iters=ais_iters)
            u_mppi_cbf = outputs[0]
            global_u = outputs[1]
            global_U = outputs[2]
            min_f_cost = outputs[3]
            sampled_states = outputs[4]
            # contingency_states = outputs[5]
            # contingency_states = np.nan_to_num(contingency_states)
            number_safe = outputs[6]
            current_safe = outputs[7]
            temperature = outputs[9]
            mppi_planner.temperature = temperature
            safe_u_seqs = outputs[10]

            cost = eval_trajectory(outputs[1], sim_state, nlmodel, Nt, q_ref)

        else:
            find_controls = functools.partial(find_Nonlin_Controls, 
                    start=sim_state, 
                    solver=f, 
                    dis=dis, 
                    system=nlmodel, 
                    Nt=int(Nt/2),occupied=occupied, box=box,planner=planner, ns=ns)
            st_topo_time = time.time()
            paths, _ = planner.findTopoPaths(sim_state, q_ref, reset=True) 
            topo_times.append(time.time()-st_topo_time)
            start_mpc_time = time.time()
            st_mppi_time = time.time()

            min_f_cost = np.inf
            for i,path in enumerate(paths):
                x_sol, u_sol, _ = find_controls(path)
                u_sol = np.repeat(u_sol,repeats=2,axis=0)
                u_sol = u_sol[:Nt,:] 
                cost = eval_trajectory(u_sol, sim_state, nlmodel, Nt, q_ref)
                if min_f_cost > cost:
                    min_f_cost = cost
                    u_mppi_cbf = u_sol
                    global_u = u_sol
                    global_u = jnp.roll(global_u, -1, axis=0)
                    global_u = global_u.at[-1].set(global_u[-2])
                    global_u = global_u.reshape((1, -1)).squeeze(0)
                    global_U = u_sol.reshape((Nt*2))
                    sampled_states = x_sol
                    number_safe = 1
                    current_safe = 1
                print(f"MPC Time:{time.time()-st}")
                mpc_times.append(time.time()-start_mpc_time)
        # if do_ais:
        #     outputs = mppi_planner.mppi_mmodal(sim_state, global_U, U_anci, subkey, q_ref, safe_zones, jnp.array(grid.occup_grid),origin,resolution,wh=wh)
        # else:
        #     try:
        #         outputs = mppi_planner.mppi_mmodal_no_ais(sim_state, global_U, U_anci, subkey, q_ref, safe_zones, jnp.array(grid.occup_grid),origin,resolution,wh=wh)
        #     except Exception as e:
        #         print(e)
        # print(safe_u_seqs.shape)
        # breakpoint()
        # print(temperature)
        # u_mppi_cbf, global_u, sampled_us, global_U, min_f_cost, sampled_states, contingency_states, number_safe, current_safe = mppi_planner.mppi(sim_state, global_U, U_anci, subkey, q_ref, safe_zones, jnp.array(grid.occup_grid),np.array([-40,10]),0.5,wh=np.array(grid.occup_grid.shape))
        # u_mppi_cbf, global_u, sampled_us, global_U, min_f_cost, sampled_states, contingency_states, number_safe, current_safe = mppi_planner.mppi(sim_state, global_U, U_anci, subkey, q_ref, safe_zones, obs)
        # u_mppi_cbf, global_u, sampled_us, global_U, min_f_cost, sampled_states, contingency_states, number_safe, current_safe = mppi_planner.mppi_mmodal(sim_state, global_U, U_anci, subkey, q_ref, safe_zones, obs, )

        number_safe_hist.append(float(number_safe))
        timestep_prog.set_description(f"Time Taken: {time.time()-st}")

        percent_safe_hist.append(float(number_safe)/n_samples)
        number_safe_avg += number_safe
        min_cost.append(min_f_cost)
        safety_hist.append(current_safe)
        if current_safe:
            not_inf +=1
        else:
            print("\n" + str(iter) + " ")

        for i in range(ratio_sim_mppi):
            sim_state = nlmodel.dynamics(sim_state, u_mppi_cbf[0], 0, dt=dt/ratio_sim_mppi, params=nlmodel.nominal_params)
            dist = np.linalg.norm((sim_state - q_ref)[0:2])
            if (dist<0.5) and timestep_reached==-1:
                timestep_reached = t / dt
            # cost = cost + dist


        states.append(sim_state)
        global_us.append(global_u)
        global_Us.append(global_U.copy())
        # total_sampled_states.append(sampled_states)
        # total_con_states.append(contingency_states)
        trial_hz += timestep_prog.format_dict['rate'] if timestep_prog.format_dict['rate'] is not None else 0
        safety_hist.append(True)
        costs.append(cost)
        total_costs += cost
        if iter >0:
            mppi_times.append(time.time()-st_mppi_time)
        iter+=1
    time_taken = time.time() - start_time
    # breakpoint()
    return (states,
            # total_sampled_states, 
            number_safe_hist, 
            # total_con_states, 
            trial_hz/iter, 
            safety_hist, costs, 
            timestep_reached, 
            global_us, 
            time_taken, 
            mpc_times, 
            mppi_times,
            topo_times)

def gen_and_save_results(params, outputs, foldername, counter, alg="mpc_ais"):
    n_samples = params['n_samples']
    safe_zones = params['safe_zones']
    obs = params['obs']

    states = outputs[0]
    # total_sampled_states = outputs[1]
    number_safe_hist = outputs[1]
    # total_con_states = outputs[3] 
    trial_hz = outputs[2] 
    safety_hist = outputs[3] 
    costs = outputs[4]
    timestep_reached = outputs[5]
    global_us = outputs[6]
    time_taken = outputs[7]
    mpc_times = outputs[8]
    mppi_times = outputs[9]
    topo_times = outputs[10]
    percent_safe_hist = np.array(number_safe_hist) / n_samples

    states = np.array(states)
    # total_sampled_states = np.array(total_sampled_states)
    number_safe_hist = np.array(number_safe_hist)
    trial_hz = np.array(trial_hz) 
    safety_hist = np.array(safety_hist) 
    costs = np.array(costs)
    timestep_reached = np.array(timestep_reached)
    # global_us = np.array(global_us)
    time_taken = np.array(time_taken)
    mpc_times = np.array(mpc_times)
    mppi_times = np.array(mppi_times)
    topo_times = np.array(topo_times)
    percent_safe_hist = np.array(percent_safe_hist)

    pic = plot_utils.plot_simulation_result(states, obs, safe_zones, text="", safe_hist=safety_hist, max_arrows=10)
    pic_name = os.path.join(foldername,f'nested_mppi_{alg}_{counter}.png')
    percent_safe_name = os.path.join(foldername, f'percent_safe_trajectories_{alg}_{counter}.png')
    # contingency_gif_name = os.path.join(foldername, f'nested_mppi_con_{alg}_{counter}.gif')
    # samples_gif_name = os.path.join(foldername, f'nested_mppi_sampled_{alg}_{counter}.gif')
    npz_name = os.path.join(foldername, f'sim_results_{alg}_{counter}')
    
    np.savez(npz_name, 
             states=states,
            # total_sampled_states=total_sampled_states,
            number_safe_hist=number_safe_hist,
            # total_con_states=total_con_states,
            trial_hz=trial_hz,
            safety_hist=safety_hist,
            costs=costs,
            timestep_reached=timestep_reached,
            # global_us=global_us,
            time_taken=time_taken,
            percent_safe_hist=percent_safe_hist,
            mpc_times=mpc_times,
            mppi_times=mppi_times,
            topo_times=topo_times,
                        )

    pic.savefig(pic_name)
    fig = plt.figure()
    plt.plot(percent_safe_hist)
    plt.xlabel('Timestep')
    plt.ylabel('Percent Safe Trajectories')
    plt.savefig(percent_safe_name)

    # anim = plot_utils.animate_simulation_with_sampled_states(tmp_states, obs, 
    #                         safe_zones, 
    #                         tmp_con_states, tmp_global_us, dynamics=nlmodel.dynamics)

    # anim.save(contingency_gif_name, writer='imagemagick')
    # anim = plot_utils.animate_simulation_with_sampled_states(tmp_states, obs, 
    #                         safe_zones, 
    #                         tmp_sampled_states, tmp_global_us, dynamics=nlmodel.dynamics)
    # anim.save(samples_gif_name, writer='imagemagick')


def main(args):
    # np.random.seed(12310) 
    solver = args.solver if hasattr(args, 'solver') else "ipopt"
    trials=1
    rng_keys = jax.random.split(jax.random.PRNGKey(0), trials)

    for trial in range(trials):
        params = rand_problem1()

        params_copy = copy.deepcopy(params)
        
        # try:
        outputs5= do_mppi_ais_mpc(copy.deepcopy(params), rng_keys[trial], base_alg=True, solver=solver)
        outputs4= do_mppi_ais_mpc(copy.deepcopy(params), rng_keys[trial], do_mpc=False, ais_iters= 1, solver=solver)
        outputs2= do_mppi_ais_mpc(copy.deepcopy(params), rng_keys[trial], do_mpc=False, ais_iters= 0, solver=solver)
        outputs1= do_mppi_ais_mpc(copy.deepcopy(params), rng_keys[trial], do_mpc=True, ais_iters=0, solver=solver)

        foldername, counter = uniquify('sim_results')
        os.mkdir(foldername)   
        param_name = os.path.join(foldername, f'params_{counter}')
        with open(param_name, 'w') as file:
            params_copy.pop('nlmodel')
            params_copy['start'] = params_copy['start'].tolist()
            params_copy['q_ref'] = params_copy['q_ref'].tolist()
            params_copy['obs'] = params_copy['obs'].tolist()
            params_copy['safe_zones'] = params_copy['safe_zones'].tolist()
            params_copy['sigma0'] = params_copy['sigma0'].tolist()
            params_copy['Q'] = params_copy['Q'].tolist()
            params_copy['QT'] = params_copy['QT'].tolist()
            params_copy['R'] = params_copy['R'].tolist()
            json.dump(params_copy, file)
        gen_and_save_results(copy.deepcopy(params), outputs5, foldername, counter, alg="mpc")
        plt.close('all')
        gen_and_save_results(copy.deepcopy(params), outputs1, foldername, counter, alg="mppi_mpc")
        plt.close('all')
        gen_and_save_results(copy.deepcopy(params), outputs2, foldername, counter, alg="mppi")
        plt.close('all')
        gen_and_save_results(copy.deepcopy(params), outputs4, foldername, counter, alg="mppi_ais")
        plt.close('all')

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', nargs='?', default="ipopt", help = 'filename')
    args = parser.parse_args()
    main(args)
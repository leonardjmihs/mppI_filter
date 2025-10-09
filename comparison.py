import numpy as np
import matplotlib.pyplot as plt
import time
import jax

from jax_mppi.acd_halfcar import create_car_trajectory_solver, solve_trajectory
from jax_mppi.nmppi_global_random import do_mppi_ais_mpc, rand_problem1

def run_mpc_simulation(params):
    """
    Runs the MPC simulation.
    """
    print("Running MPC simulation...")
    start_time = time.time()

    lbu = np.array([-np.pi/4, -3.0])
    ubu = np.array([np.pi/4, 3.0])
    N = params['Nt']
    Tf = params['T']

    solver = create_car_trajectory_solver(lbu, ubu, N=N, Tf=Tf)

    x0 = params['start']
    x_ref = np.linspace(x0, params['q_ref'], N + 1)

    x_opt, u_opt = solve_trajectory(solver, x0, x_ref=x_ref)

    end_time = time.time()
    print(f"MPC simulation finished in {end_time - start_time:.2f} seconds.")

    return x_opt, u_opt

def run_mppi_simulation(params):
    """
    Runs the MPPI simulation.
    """
    print("Running MPPI simulation...")
    start_time = time.time()
    
    rng_key = jax.random.PRNGKey(0)
    states, _, _, _, _, _, _, _, _, _, _, _, _ = do_mppi_ais_mpc(params, rng_key)
    
    end_time = time.time()
    print(f"MPPI simulation finished in {end_time - start_time:.2f} seconds.")
    
    return np.array(states)

def main():
    """
    Main function to run the comparison.
    """
    params = rand_problem1()

    # Run MPC
    mpc_states, _ = run_mpc_simulation(params)

    # Run MPPI
    mppi_states = run_mppi_simulation(params)

    # Plot the results
    plt.figure(figsize=(10, 8))
    plt.plot(mpc_states[:, 0], mpc_states[:, 1], 'b-', label='MPC Trajectory')
    plt.plot(mppi_states[:, 0], mppi_states[:, 1], 'r-', label='MPPI Trajectory')
    
    # Plot start, goal, and obstacles
    plt.plot(params['start'][0], params['start'][1], 'go', markersize=10, label='Start')
    plt.plot(params['q_ref'][0], params['q_ref'][1], 'rx', markersize=10, label='Goal')
    for obs in params['obs']:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='k', alpha=0.5)
        plt.gca().add_patch(circle)

    plt.title('MPC vs. MPPI Trajectory Comparison')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('comparison.png')
    plt.show()

if __name__ == "__main__":
    main()

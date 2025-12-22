import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tqdm
import sys
from functools import partial
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def determine_frame_bounds(fig):
    ax = fig.gca()
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    return x_limits, y_limits
def plot_simulation_result(states, obs, goal=None, safe_zones=[], text="", max_arrows=30, safe_hist=None, mpc_paths=None, planner_paths=None, costmap=None):
    """
    Plot the trajectory and orientation of the car given the state history.

    Parameters:
        states (list of np.array): List of states [x, y, theta, v] at each time step.
    """
    x_vals = [state[0] for state in states]
    y_vals = [state[1] for state in states]
    theta_vals = [state[2] for state in states]

    fig = plt.figure(figsize=(6, 6))

    # Generate circle for CBF
    for circ in obs:
        circle = plt.Circle((circ[0], circ[1]), np.sqrt(circ[2]), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
        plt.gca().add_artist(circle)
    for zone in safe_zones:
        rect = plt.Rectangle((zone[0]-0.25, zone[1]-0.25), 0.5, 0.5)
        plt.gca().add_artist(rect)

    if goal is not None:
        plt.scatter(goal[0], goal[1], s=200, color="purple", alpha=0.75, label="Goal")

    # Plot the trajectory
    plt.plot(x_vals, y_vals, '-o', label='Trajectory', markersize=4, alpha=0.5)

    # Plot planner paths
    if planner_paths is not None:
        for path in planner_paths:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], 'b--', alpha=0.3, label='Planner Path')

    # Plot MPC paths
    if mpc_paths is not None:
        for path in mpc_paths:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], 'g-', alpha=0.5, label='MPC Path')

    if costmap is not None:
        occupied = costmap.get_all_nonzero()
        plt.scatter(occupied[:,0], occupied[:,1], s=5, color="black", alpha=0.5, label="obstacles")

    # Plot the orientation at each point
    for i in range(0, len(states), int(len(states)/max_arrows)):  # Only plot 20 arrows for visibility
        x, y, theta = states[i][0:3]
        dx = 0.1 * np.cos(theta)
        dy = 0.1 * np.sin(theta)
        plt.arrow(x, y, dx, dy, head_width=0.3, head_length=0.3, fc='red', ec='red')
    
    if safe_hist is not None and len(safe_hist)==len(states):
        # breakpoint()
        not_safe_x = np.take(x_vals, np.nonzero(1-np.array(safe_hist)))
        not_safe_y = np.take(y_vals, np.nonzero(1-np.array(safe_hist)))
        # plt.plot
        plt.plot(not_safe_x, not_safe_y, 'ro',  markersize=4, alpha=0.5)


    # plot start and end point
    # plt.scatter(3.5, 3.5, s=200, color="green", alpha=0.75, label="init. position")
    plt.scatter(states[0][0], states[0][1], s=200, color="green", alpha=0.75, label="init. position")

    plt.text(0.0,0.0, text, transform=plt.gca().transAxes,verticalalignment="bottom")

    plt.title('Simulation Result with Car Orientation')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    # plt.legend()
    plt.grid(True)
    # plt.axis('equal')
    # x_range = np.max(safe_zones[:,0]) - np.min(safe_zones[:,0])
    # y_range = np.max(safe_zones[:,1]) - np.min(safe_zones[:,1])
    # x_low = np.min(safe_zones[:,0])- x_range/2
    # y_low = np.min(safe_zones[:,1])- y_range/
    x_range = np.max(x_vals) - np.min(x_vals)
    y_range = np.max(y_vals) - np.min(y_vals)
    x_low = np.min(x_vals)-1
    x_max = np.max(x_vals)+1
    y_low = np.minimum(np.min(y_vals), -10)
    y_max = np.maximum(np.max(y_vals), 10)
    plt.xlim([x_low-1, x_max+1])
    plt.ylim([y_low-1, y_max+1])
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.show(block=False)
    plt.pause(0.5)
    # plt.savefig('asdf.png')
    return fig


def animate_simulation_with_sampled_states(states, obs, goal=None, safe_zones=[], sampled_xs=[], optimal_us=None, dynamics=None, costmap=None):
    # if dynamics is None:
    #     print("errorr")
    def update(frame, states):
        plt.gca().cla()  # Clear the current axes
        # Set axis limits
        x_vals = [state[0] for state in states]
        y_vals = [state[1] for state in states]

        x_low = np.min(x_vals)-1
        y_low = np.min(y_vals)-1
        x_max = np.max(x_vals)+1
        y_max = np.max(y_vals)+1
        plt.xlim([x_low-1, x_max+1])
        plt.ylim([y_low-1, y_max+1])

        # x_low = np.min(safe_zones[:,0])-1
        # y_low = np.min(safe_zones[:,1])- 2
        # x_low = np.min(x_vals)
        # y_low = np.min(y_vals)
        # x_max = np.max(x_vals)
        # y_max = np.max(y_vals)
        # plt.xlim([x_low, x_max])
        # plt.ylim([y_low, y_max])
        # plt.xlim([x_low, 4.5])
        # plt.ylim([y_low, 4.5])
        # plt.xlim([-5, 4.5])
        # plt.ylim([-5, 4.5])
        if goal is not None:
            plt.scatter(goal[0], goal[1], s=200, color="purple", alpha=0.75, label="Goal")

        # Set aspect ratio to be equal, so each cell will be square-shaped
        plt.gca().set_aspect('equal', adjustable='box')
        plt.text(0.0,0.0, f"Frame={frame}", transform=plt.gca().transAxes, verticalalignment="bottom")

        # Generate circle for CBF
        # circle = plt.Circle((2, 2), np.sqrt(1), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
        # plt.gca().add_artist(circle)
        for circ in obs:
            circle = plt.Circle((circ[0], circ[1]), np.sqrt(circ[2]), color='grey', fill=True, linestyle='--', linewidth=2, alpha=0.5)
            plt.gca().add_artist(circle)

        for zone in safe_zones:
            rect = plt.Rectangle((zone[0]-0.25, zone[1]-0.25), 0.5, 0.5)
            plt.gca().add_artist(rect)
        if costmap is not None:
            occupied = costmap.get_all_nonzero()
            plt.scatter(occupied[:,0], occupied[:,1], s=5, color="black", alpha=0.5, label="obstacles")
        # Plot MPPI trajectories
        # breakpoint()
        if len(sampled_xs) > 0:
          # pred_trajs = [get_predicted_trajectories(states[frame], sampled_us[frame][i]) for i in range(len(sampled_us))]
          num_trajs_plotted = np.minimum(100,  sampled_xs[frame].shape[0])  # plot maximum 50 trajs per step
          for idx in range(num_trajs_plotted):
            # pred_traj = get_predicted_trajectories(states[frame], sampled_us[frame][idx], dynamics)
            x_pos, y_pos = sampled_xs[frame][idx, :, 0], sampled_xs[frame][idx, :, 1]
            plt.plot(x_pos, y_pos, color="g", alpha=0.5)

        # plot optimal predicted trajectory
        if optimal_us is not None:
          opt_pred_traj = get_predicted_trajectories(states[frame], optimal_us[frame], dynamics)
          x_pos, y_pos = opt_pred_traj[:, 0], opt_pred_traj[:, 1]
          plt.plot(x_pos, y_pos, color="orange", alpha=0.8, label="optimized traj.")


        x, y, theta = states[frame][0:3]
        dx = 0.1 * np.cos(theta)
        dy = 0.1 * np.sin(theta)

        # Plot the trajectory up to the current frame
        plt.plot([state[0] for state in states[:frame+1]], [state[1] for state in states[:frame+1]], '-o', markersize=4, alpha=0.5)

        # Plot the orientation at the current frame
        plt.arrow(x, y, dx, dy, head_width=0.3, head_length=0.3, fc='red', ec='red')

        plt.scatter(states[0][0], states[0][1], s=200, color="green", alpha=0.75, label="init. position")
        # plt.title('Simulation Result with Car Orientation')
        # plt.xlabel('X Position')
        # plt.ylabel('Y Position')
        plt.grid(True)
        # plt.legend(loc="upper left")

    fig = plt.figure(figsize=(6, 6))
    anim = FuncAnimation(fig, update, frames=tqdm.tqdm(range(len(states)-1), file=sys.stdout), fargs=(states,), interval=100, blit=False)
    # anim.save('simulation.gif', writer='imagemagick')
    return anim

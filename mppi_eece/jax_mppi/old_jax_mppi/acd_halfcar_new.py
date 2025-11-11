# save as acados_time_opt.py (or paste into your script)
import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
# reuse create_hcar_model and auto_xdot from your file (acd_halfcar.py)
# from acd_halfcar import create_hcar_model, auto_xdot
# If acd_halfcar is in the same dir:
from acd_halfcar import create_hcar_model, auto_xdot, sim_car

def create_hcar_solver_with_time_varying_obstacles(
    lbu, ubu,
    N=40,
    Tf=8.0,
    num_sides=6,
    v_max=1.0,
    q_goal_pos=500.0,
    q_pos=1.0,
    r_control=0.1
):
    """
    Build an acados OCP for the halfcar model that:
    - accepts per-node obstacle halfspaces (A, b) as parameters (time-varying)
    - terminal cost pushes position to goal (set via yref_e in the solver call)
    - running cost includes term encouraging speed close to v_max
    Args:
        lbu, ubu: control bounds (arrays length nu)
        N: number of control intervals
        Tf: final time (seconds)
        num_sides: max number of halfplanes for obstacles at each node
        v_max: maximum forward speed (used in incentive term)
        q_goal_pos: terminal weight for positional error
    Returns:
        acados solver object
    """
    model = create_hcar_model()  # uses your function; model.x = [X,Y,theta], model.u = [delta, v]
    ocp = AcadosOcp()
    # copy model fields into ocp.model
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.z = model.z
    model_ac.name = model.name  
    # --- Add obstacle parameters ---
    # obs_A: num_sides x 2, obs_b: num_sides
    obs_A_sym = ca.MX.sym('obs_A', num_sides*2)  # flattened rows (num_sides * 2 entries)
    obs_b_sym = ca.MX.sym('obs_b', num_sides)    # each row has one b
    # pack into single p
    model_ac.p = ca.vertcat(obs_A_sym, obs_b_sym)

    # con_h_expr: for each side: a_i^T [X;Y] - b_i  --> length num_sides
    Xpos = model.x[0]
    Ypos = model.x[1]
    # reconstruct obs_A matrix from flattened vector
    obs_A_mat = ca.reshape(obs_A_sym, num_sides, 2)
    pos_vec = ca.vertcat(Xpos, Ypos)
    h_obs = ca.mtimes(obs_A_mat, pos_vec) - obs_b_sym  # vector of length num_sides
    # set nonlinear constraint expression
    # model_ac.con_h_expr = h_obs
    # model_ac.con_h_expr_e = h_obs  # allow terminal constraints too (we'll set bounds so they are enforced)
    ocp.model = model_ac

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ocp.dims.N = N
    ocp.solver_options.tf = Tf

    # Cost: linear LS formulation (Vx, Vu)
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ny = nx + nu
    # running weights
    Q = np.diag([q_pos, q_pos, 0.0])   # penalize X,Y error in yref
    R = np.diag([r_control, r_control])
    W = np.block([[Q, np.zeros((nx,nu))],
                  [np.zeros((nu,nx)), R]])
    ocp.cost.W = W
    # mapping Vx, Vu
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[nx:, :] = np.eye(nu)
    ocp.cost.Vu = Vu

    # terminal cost heavier on position
    ocp.cost.W_e = np.diag([q_goal_pos, q_goal_pos, 1.0])  # heavier on X,Y
    ocp.cost.Vx_e = np.eye(nx)

    # initial condition placeholder (will be set at solve time)
    ocp.constraints.x0 = model.x0

    # control bounds
    ocp.constraints.lbu = lbu
    ocp.constraints.ubu = ubu
    ocp.constraints.idxbu = np.arange(nu)

    # Constraint bounds for obstacle halfspaces:
    # we want A x - b <= 0  => h_obs <= 0  (i.e. upper bound = 0)
    # no lower bound (set to large negative)
    # ocp.constraints.lh = -1e6 * np.ones(num_sides)  # big negative allowed
    # ocp.constraints.uh = np.zeros(num_sides)        # must be <= 0
    # ocp.constraints.lh_e = -1e6 * np.ones(num_sides)
    # ocp.constraints.uh_e = np.zeros(num_sides)

    # Solver options
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.print_level = 0

    np_ = num_sides * 3
    ocp.parameter_values = np.zeros(np_)

    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(nx)
    print(model.name)
    solver = AcadosOcpSolver(ocp)
    # Save meta-info on solver object to help solve wrapper
    solver._num_sides = num_sides
    solver._v_max = v_max
    solver._N = N
    solver._nx = nx
    solver._nu = nu
    return solver


def solve_with_obstacles_and_goal(solver, x0, goal, A_per_node, b_per_node, x_ref=None, u_init=None):
    """
    Fill solver with initial condition, per-node obstacle parameters and goal,
    then solve.

    Args:
      solver: returned from create_hcar_solver_with_time_varying_obstacles
      x0: initial state [X,Y,theta]
      goal: [Xg, Yg] goal position (theta not required)
      A_per_node: shape (N+1, num_sides, 2)
      b_per_node: shape (N+1, num_sides)
      x_ref: optional (N+1, nx) reference trajectory (for smoothness); if None, zero
      u_init: optional initial guess for U (N x nu)
    Returns:
      x_opt (N+1, nx), u_opt (N, nu), status
    """
    N = solver._N
    num_sides = solver._num_sides
    nx = solver._nx
    nu = solver._nu

    # initial state
    solver.constraints_set(0, 'lbx', x0)
    solver.constraints_set(0, 'ubx', x0)

    # build running yref sequence that penalizes position error to goal
    if x_ref is None:
        x_ref = np.tile(np.zeros(nx), (N+1, 1))
    # set xref and u reference for each stage (linear LS expects a yref of size nx+nu)
    for i in range(N):
        yref_i = np.concatenate([x_ref[i], np.zeros(nu)])
        solver.set(i, 'yref', yref_i)
        # optionally set nominal u to current v ~= desired cruise (0.8*v_max) to help solve
        if u_init is not None:
            solver.set(i, 'u', u_init[i])
    # terminal yref pushing to goal (only position components used)
    yref_e = np.zeros(nx)
    yref_e[0:2] = goal[0:2]
    solver.set(N, 'yref', yref_e)

    # pack and set obstacle parameters for each node
    # expected param layout: first (num_sides*2) entries = flattened A row-wise, then num_sides entries b
    for i in range(N+1):
        A_i = np.array(A_per_node[i])   # shape (num_sides,2)
        b_i = np.array(b_per_node[i]).reshape(-1)
        # flatten
        A_flat = A_i.reshape(-1)
        p = np.concatenate([A_flat, b_i])
        solver.set(i, 'p', p)

    # solve
    status = solver.solve()
    x_opt = np.array([solver.get(i, 'x') for i in range(N+1)])
    u_opt = np.array([solver.get(i, 'u') for i in range(N)])
    return x_opt, u_opt, status


def find_minimum_feasible_Tf(
    lbu, ubu,
    x0, goal,
    A_per_node_fn, b_per_node_fn,
    N=40,
    Tf_max=12.0,
    Tf_min=1.0,
    tol=0.2,
    tries=8,
    **solver_kwargs
):
    """
    Try to find a near-minimal feasible Tf by binary / shrink search.
    A_per_node_fn, b_per_node_fn: functions that produce per-node A, b arrays given N and Tf:
        A_per_node, b_per_node = A_per_node_fn(N, Tf)
    Returns: x_opt, u_opt, final_Tf, status
    Note: recreates the solver for each tested Tf (acados solver encodes Tf in ocp)
    """
    lo = Tf_min
    hi = Tf_max
    best = None
    for it in range(tries):
        mid = 0.5*(lo+hi)
        # create solver with this Tf
        sol = create_hcar_solver_with_time_varying_obstacles(lbu, ubu, N=N, Tf=mid, **solver_kwargs)
        A_nodes, b_nodes = A_per_node_fn(N, mid)
        try:
            x_opt, u_opt, status = solve_with_obstacles_and_goal(sol, x0, goal, A_nodes, b_nodes)
            # status 0 means success
            if status == 0:
                best = (x_opt, u_opt, mid, status)
                # try shorter Tf
                hi = mid
            else:
                # infeasible -> need longer Tf
                lo = mid
        except Exception as e:
            # treat as infeasible
            lo = mid
        # stop when interval small
        if hi - lo < tol:
            break
    return best

# --- Utility: convert your A_np, b_np (list per path node) to arrays ---
def pack_A_b_for_acados(A_np_list, b_np_list, N):
    """
    Take your A_np, b_np lists (size N+1 each, each A is (num_sides,2) or padded)
    and return arrays shaped (N+1, num_sides, 2) and (N+1, num_sides)
    """
    # find num_sides (max)
    num_sides = max([np.array(Ai).shape[0] for Ai in A_np_list])
    A_nodes = np.zeros((N+1, num_sides, 2))
    b_nodes = np.zeros((N+1, num_sides))
    for i in range(N+1):
        Ai = np.array(A_np_list[i])
        bi = np.array(b_np_list[i]).reshape(-1)
        ni = Ai.shape[0]
        A_nodes[i, :ni, :] = Ai
        b_nodes[i, :ni] = bi
        # any unused rows remain zero which becomes trivial constraints (0<=0)
    return A_nodes, b_nodes

def print_initial_constraint_violations(x0, A_nodes, b_nodes, tol=1e-8):
    # x0: full state [x,y,theta]
    Np1 = A_nodes.shape[0]
    max_violation = 0.0
    infeasible_stages = []
    for i in range(Np1):
        Ai = A_nodes[i]    # (num_sides, 2)
        bi = b_nodes[i]    # (num_sides,)
        vals = Ai @ x0[:2] - bi
        stage_max = np.max(vals)
        if stage_max > tol:
            infeasible_stages.append((i, float(stage_max)))
        max_violation = max(max_violation, float(stage_max))
    print(f"max constraint violation at x0 over all stages = {max_violation:.6g}")
    if len(infeasible_stages) > 0:
        print("Infeasible stages (index, violation):")
        for s, v in infeasible_stages[:10]:
            print(f"  stage {s}: {v:.6g}")
    else:
        print("No initial-stage violations detected.")
    return infeasible_stages


def main():
    import time
    import matplotlib.pyplot as plt
    model = create_hcar_model()
    print(model)
    lbu = np.array([-np.pi/4, -1.0])
    ubu = np.array([np.pi/4, 1.0])
    N = 30
    Tf = 10.0

    num_sides = 4  # rectangle obstacles
        # solver, n_p_original, constants = create_car_trajectory_solver(N=N, Tf=Tf, ns=ns)
    solver = create_hcar_solver_with_time_varying_obstacles(lbu, 
                                                ubu, 
                                                N=N, 
                                                Tf=Tf, 
                                                num_sides=num_sides)

    # Initial state
    x0 = np.zeros(3)
    x0[0] = 2
    x0[1] = 3
        
    # Reference trajectory (straight line with slight curve)
    x_ref = np.ones((N+1, 3))
    x_ref[:, 0] = 2+np.linspace(0, 10, N+1)  # X position
    x_ref[:, 1] = 3+0.5*np.sin(np.linspace(0, np.pi, N+1))  # Y position
    # Get heading angles
    heading = np.arctan2(np.gradient(x_ref[:,1]), np.gradient(x_ref[:,0]))
    x_ref[:,2]= heading

    # Create time-varying obstacle constraints (moving corridor)
    obs_A = np.zeros((N+1, num_sides, 2))
    obs_b = np.zeros((N+1, num_sides))
        
    for i in range(N+1):
        progress = i / N
        width = 2.0 - 1.5 * progress  # Narrowing corridor
        half = width / 2.0

        # Left wall: y >= -half --> -y <= half  -> a = [0, -1], b = +half
        obs_A[i, 0, :] = [0.0, -1.0]
        obs_b[i, 0] = +half

        # Right wall: y <= +half -->  y <= half -> a = [0, 1], b = +half
        obs_A[i, 1, :] = [0.0, 1.0]
        obs_b[i, 1] = +half

        # Front wall: x <= 12   -> a = [1,0], b = 12
        obs_A[i, 2, :] = [1.0, 0.0]
        obs_b[i, 2] = 12.0

        # Back wall: x >= -1 -> -x <= 1 -> a = [-1, 0], b = 1
        obs_A[i, 3, :] = [-1.0, 0.0]
        obs_b[i, 3] = 1.0

        # Pad the remaining sides with benign constraints: 0*x <= big_number
        for j in range(4, num_sides):
            obs_A[i, j, :] = [0.0, 0.0]
            obs_b[i, j] = 1e6

        # Diagnostic: check initial feasibility at x0
    infeas = print_initial_constraint_violations(x0, obs_A, obs_b, tol=1e-8)
    if len(infeas) > 0:
        print("WARNING: initial state infeasible for current corridor. Consider moving x0 or relaxing constraints.")

    # Warm-start guesses for x and u
    for i in range(N+1):
        solver.set(i, 'x', x_ref[i])
    # constant nominal control: small steering, moderate speed
    u_guess = np.zeros((N, solver._nu))
    u_guess[:, 0] = 0.0
    u_guess[:, 1] = 0.6 * ubu[1]
    for i in range(N):
        solver.set(i, 'u', u_guess[i])

    # extra solver robustness options
    try:
        solver.options_set("print_level", 1)
        solver.options_set("levenberg_marquardt", 1e-2)
    except Exception:
        # some acados versions use different option names; ignore if not present
        pass


    x_opt, u_opt, _ = solve_with_obstacles_and_goal(solver, x0, goal=x_ref[-1,:], A_per_node=obs_A, b_per_node=obs_b, x_ref=x_ref)
    print("Optimization successful!")
    print("Final position:", x_opt[-1, :2])
    # print("Steering angles (rear L/R):", u_opt[-1, 3], u_opt[-1, 4])

    # time solver
    # x_ref = np.ones((N+1, 3))
    # x_ref[:, 0] = 2+np.linspace(0, 10, N+1)  # X position
    # x_ref[:, 1] = 3+0.5*np.sin(np.linspace(0, np.pi, N+1))  # Y position
    # # Get heading angles
    # heading = np.arctan2(np.gradient(x_ref[:,1]), np.gradient(x_ref[:,0]))
    # x_ref[:,2]= heading
    # x_ref = np.array([[ 0.        ,  0.        ,  0.        ],
    #    [ 0.55194163,  0.23528799,  0.        ],
    #    [ 1.10388325,  0.47057599,  0.        ],
    #    [ 1.65582488,  0.70586398,  0.        ],
    #    [ 2.20776651,  0.94115198,  0.        ],
    #    [ 2.75970814,  1.17643997,  0.        ],
    #    [ 3.31164976,  1.41172797,  0.        ],
    #    [ 3.86359139,  1.64701596,  0.        ],
    #    [ 4.41553302,  1.88230395,  0.        ],
    #    [ 4.96747464,  2.11759195,  0.        ],
    #    [ 5.51941627,  2.35287994,  0.        ],
    #    [ 6.0713579 ,  2.58816794,  0.        ],
    #    [ 6.62329953,  2.82345593,  0.        ],
    #    [ 7.17524115,  3.05874393,  0.        ],
    #    [ 7.72718278,  3.29403192,  0.        ],
    #    [ 8.27912441,  3.52931991,  0.        ],
    #    [ 8.86084129,  3.62434966,  0.        ],
    #    [ 9.46077012,  3.63359088,  0.        ],
    #    [10.06069895,  3.6428321 ,  0.        ],
    #    [10.66062778,  3.65207332,  0.        ],
    #    [11.26055661,  3.66131453,  0.        ],
    #    [11.86048544,  3.67055575,  0.        ],
    #    [12.46041426,  3.67979697,  0.        ],
    #    [13.06034309,  3.68903819,  0.        ],
    #    [13.66027192,  3.69827941,  0.        ],
    #    [14.26020075,  3.70752062,  0.        ],
    #    [14.86012958,  3.71676184,  0.        ],
    #    [15.46005841,  3.72600306,  0.        ],
    #    [16.05998724,  3.73524428,  0.        ],
    #    [16.65991607,  3.7444855 ,  0.        ],
    #    [17.2598449 ,  3.75372671,  0.        ]])




    # start_time =  time.time()
    # x_opt, u_opt = solve_trajectory(solver, np.array([0.0,0.0,0.0]), x_ref=x_ref, obs_A=obs_A, obs_b=obs_b)
    # end_time = time.time()
    # print("Time taken for optimization:", end_time - start_time)

    plt.figure()
    plt.plot(x_opt[:, 0], x_opt[:, 1], label='Optimized Path')
    plt.plot(x_ref[:, 0], x_ref[:, 1], 'r--', label='Reference Path')
    # plot heading as arrows
    for i in range(0, len(x_ref), int(len(x_ref)/20)):  # Plot 20 arrows
        x, y, theta = x_ref[i]
        dx = 0.2 * np.cos(theta)
        dy = 0.2 * np.sin(theta)
        plt.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    plt.legend()

    # # plt.figure()
    # # bike_model, bike_c = create_bike_model()
    # # bike_traj = sim_bike(bike_model, bike_c)
    # # bike_traj = np.array(bike_traj)
    # # plt.plot(bike_traj[:, 0], bike_traj[:, 1], label="bike path")

    plt.figure()
    x_traj = sim_car(solver.acados_ocp.model, x0, u_opt, Tf)
    x_traj = np.array(x_traj)
    plt.plot(x_traj[:, 0], x_traj[:, 1], label="car path")
    plt.legend()
    plt.show()

    # # x_traj = sim_car(solver.acados_ocp.model, constants)
    # # x_traj = np.array(x_traj)
    # # plt.plot(x_traj[:, 0], x_traj[:, 1], label="car path")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()
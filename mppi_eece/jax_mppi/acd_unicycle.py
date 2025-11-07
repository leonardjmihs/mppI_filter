from casadi import sin, cos, MX, vertcat, atan, sign
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosOcpIterates, AcadosSim, AcadosSimSolver
import casadi as ca
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import os

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

def create_unicycle_model(constants=None, model_name="unicycle"):
    model = AcadosModel()
    model.name = model_name
    X = MX.sym("X")
    Y = MX.sym("Y")
    theta = MX.sym("theta")
    v = MX.sym("v_x")
    r = MX.sym("r")  # r = yaw rate, often called omega as well
    model.u = vertcat(
        r,
        v,
    )
    model.x = vertcat(
        X,
        Y,
        theta,
    )
    model.p = vertcat([])
    model.x0 = np.array([0, 0, 0])
    model.xdot = auto_xdot(model.x)
    # right hand side for differential equations
    model.f_expl_expr = vertcat(
        v * ca.cos(theta),  # = X_dot
        v * ca.sin(theta),  # = Y_dot
        r,  # = phi_dot
    )
    model.f_impl_expr = model.xdot - model.f_expl_expr
    return model

def create_unicycle_solver_with_time_varying_obstacles(
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
    model = create_unicycle_model()  # uses your function; model.x = [X,Y,theta], model.u = [delta, v]
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
        sol = create_unicycle_solver_with_time_varying_obstacles(lbu, ubu, N=N, Tf=mid, **solver_kwargs)
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

def generate_car_sim(model, code_export_dir, time_step=0.001, num_stages=1, num_steps=1, parameters=None):
    # model, c, q = create_car_model()
    sim = AcadosSim()
    sim.model = model
    if parameters is not None:
        # Directly assign to the parameter attribute 'p'
        sim.parameter_values = parameters
    else:
        sim.parameter_values = np.zeros(model.p.shape)
    sim.solver_options.T = time_step
    sim.solver_options.integrator_type = "ERK"
    sim.solver_options.num_stages = num_stages
    sim.solver_options.num_steps = num_steps
    sim.code_export_directory = code_export_dir

    return AcadosSimSolver(sim, json_file=os.path.join(sim.code_export_directory, "acados_sim.json"))

def sim_car(model, x0, u, tf):
    N = len(u)
    dt = tf/N
    sim_dt = 0.01
    sim_solver = generate_car_sim(model, "/tmp/generated", time_step=sim_dt, num_stages=1, num_steps=1)
    times = np.linspace(0, tf, N+1)
    sim_solver.set("u", np.array([0.0, 1.0]))
    sim_solver.set("x", x0)
    def step(sim_solver, u):
        sim_solver.set("x", sim_solver.get("x"))
        sim_solver.set("u", u)
        sim_solver.solve()
    x_traj = []

    for i in range(int(tf/sim_dt)):
        t = int(i*sim_dt/dt)
        # figure out which u to use from the trajectory based off on t
        # step(sim_solver, u[t])
        x0 = sim_solver.simulate(x=x0, u=u[t])
        x_traj.append(sim_solver.get("x"))
    return x_traj   

def auto_xdot(model_x):
    """
        For a vector of scalar casadi state variables, generate a corresponding vector of {name}_dot casadi symbols.
    """
    xdot = ca.vertcat([])
    for i in range(model_x.size()[0]):
        xdot = ca.vertcat(xdot, ca.MX.sym(model_x[i].name() + "_dot"))
    return xdot

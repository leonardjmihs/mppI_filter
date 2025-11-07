import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from casadi import MX, vertcat
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver


def auto_xdot(x):
    return MX.sym("xdot", x.shape[0])


# ------------------------------------------------------------------
# Global bicycle model (for simulation only)
# ------------------------------------------------------------------
def create_hcar_model(constants=None, model_name="halfcar"):
    model = AcadosModel()
    model.name = model_name

    if constants is None:
        constants = {"L": 2.0}
    L = constants["L"]

    # states
    X = MX.sym("X")
    Y = MX.sym("Y")
    theta = MX.sym("theta")
    model.x = vertcat(X, Y, theta)

    # controls
    delta = MX.sym("delta")
    v = MX.sym("v")
    model.u = vertcat(delta, v)

    model.p = vertcat([])
    model.x0 = np.array([0, 0, 0])
    model.xdot = auto_xdot(model.x)

    # dynamics
    X_dot = v * ca.cos(theta)
    Y_dot = v * ca.sin(theta)
    theta_dot = v / L * ca.tan(delta)

    model.f_expl_expr = vertcat(X_dot, Y_dot, theta_dot)
    model.f_impl_expr = model.xdot - model.f_expl_expr

    return model


# ------------------------------------------------------------------
# Deviation model (MPC)
# States: [s, e_y, e_theta]
# Controls: [delta, v]
# Params: [kappa] (reference curvature)
# ------------------------------------------------------------------
def create_car_deviation_model(constants=None, model_name="car_deviation"):
    model = AcadosModel()
    model.name = model_name

    if constants is None:
        constants = {"L": 2.0}
    L = constants["L"]

    # states
    s = MX.sym("s")          # path progress
    e_y = MX.sym("e_y")      # lateral deviation
    e_theta = MX.sym("e_theta")  # heading error
    model.x = vertcat(s, e_y, e_theta)

    # controls
    delta = MX.sym("delta")
    v = MX.sym("v")
    model.u = vertcat(delta, v)

    # parameters (curvature at current path point)
    kappa = MX.sym("kappa")
    model.p = vertcat(kappa)

    model.x0 = np.zeros(3)
    model.xdot = auto_xdot(model.x)

    # Deviation dynamics - FIXED: proper handling of denominator
    # Use small epsilon to avoid division by zero
    epsilon = 1e-6
    denominator = 1 - kappa * e_y
    # Safeguard against division by zero
    safe_denominator = ca.if_else(ca.fabs(denominator) < epsilon, epsilon, denominator)
    # safe_denominator = denominator
    
    s_dot = v * ca.cos(e_theta) / safe_denominator
    e_y_dot = v * ca.sin(e_theta)
    e_theta_dot = v / L * ca.tan(delta) - kappa * s_dot

    model.f_expl_expr = vertcat(s_dot, e_y_dot, e_theta_dot)
    model.f_impl_expr = model.xdot - model.f_expl_expr

    return model


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def compute_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    kappa = (dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5
    # Remove NaN/inf values
    kappa = np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
    return kappa


def global_to_path_states(xg, yg, heading, kappa):
    """
    Convert global reference path to deviation space.
    Reference deviation states: [s, e_y=0, e_theta=0]
    """
    # Calculate actual arc length
    dx = np.diff(xg)
    dy = np.diff(yg)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate(([0], np.cumsum(ds)))
    
    e_y = np.zeros_like(xg)
    e_theta = np.zeros_like(xg)
    x_ref = np.vstack([s, e_y, e_theta]).T
    p_ref = kappa.reshape(-1, 1)
    return x_ref, p_ref


# ------------------------------------------------------------------
# Solver creation - FIXED cost matrices
# ------------------------------------------------------------------
def create_car_trajectory_solver(lbu, ubu, N=25, Tf=5.0, constants=None):
    model = create_car_deviation_model(constants=constants)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = N

    # horizon
    ocp.solver_options.tf = Tf

    # cost: quadratic (track deviation to zero)
    # We want e_y and e_theta to be zero, s can vary
    Q = np.diag([0.0001, 10.0, 10.0])   # weights for [s, e_y, e_theta]
    R = np.diag([0.5, 0.5])         # weights for [delta, v]

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    
    # Correct cost matrices
    ocp.cost.W = np.block([[Q, np.zeros((nx, nu))],
                           [np.zeros((nu, nx)), R]])
    ocp.cost.W_e = Q

    # Vx should be [ny, nx]
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    
    # Vu should be [ny, nu]
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:, :] = np.eye(nu)
    
    ocp.cost.Vx_e = np.eye(nx)

    # Reference is zero for all states (we want to track the path, so deviations should be zero)
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(nx)

    # constraints
    ocp.constraints.x0 = np.zeros(3)
    ocp.constraints.lbu = lbu
    ocp.constraints.ubu = ubu
    ocp.constraints.idxbu = np.array([0, 1])

    # params (curvature)
    ocp.parameter_values = np.array([0.0])

    # solver options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.nlp_solver_max_iter = 100

    return AcadosOcpSolver(ocp, json_file="acados_ocp.json")


# ------------------------------------------------------------------
# Trajectory solve - FIXED reference handling
# ------------------------------------------------------------------
def solve_trajectory(solver, x0, x_ref, p_ref):
    N = solver.acados_ocp.dims.N
    nx = x0.shape[0]
    nu = solver.acados_ocp.dims.nu

    # Set parameters and initial state
    for i in range(N):
        solver.set(i, "p", p_ref[i])
        solver.set(i, 'yref', np.concatenate([np.array([N, 0,0]), np.zeros(solver.acados_ocp.dims.nu)]))
    solver.set(N, "p", p_ref[N])

    # Set initial state constraint
    solver.set(N, 'yref', np.array([1, 0, 0]))  # Final state should be at s=1, e_y=0, e_theta=0
    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)

    status = solver.solve()
    if status != 0:
        print(f"Warning: acados solver returned status {status}")
        # Try to get solution anyway

    x_opt = np.array([solver.get(i, "x") for i in range(N + 1)])
    u_opt = np.array([solver.get(i, "u") for i in range(N)])

    return x_opt, u_opt


# ------------------------------------------------------------------
# Simulation in global model
# ------------------------------------------------------------------
def generate_car_sim(model, code_export_dir, time_step):
    sim = AcadosSim()
    sim.model = model
    sim.solver_options.T = time_step
    sim.solver_options.integrator_type = "ERK"
    return AcadosSimSolver(sim, json_file="acados_sim.json")


def sim_car_global(x0, u_traj, Tf, dt=0.01, constants=None):
    model = create_hcar_model(constants=constants)
    sim_solver = generate_car_sim(model, "/tmp/generated_global", time_step=dt)

    X = []
    x = np.array(x0)

    N = len(u_traj)
    horizon_dt = Tf / N

    for i in range(int(Tf / dt)):
        t = int(i * dt / horizon_dt)
        if t >= N:
            t = N - 1
        u = u_traj[t]

        sim_solver.set("x", x)
        sim_solver.set("u", u)
        sim_solver.solve()
        x = sim_solver.get("x").flatten()
        X.append(x)

    return np.array(X)


# ------------------------------------------------------------------
# Convert deviation states back to global coordinates for plotting
# ------------------------------------------------------------------
def deviation_to_global(x_dev, x_ref_global, heading_ref):
    """
    Convert deviation states back to global coordinates
    x_dev: [s, e_y, e_theta] 
    x_ref_global: reference global path points
    """
    # For simplicity, interpolate reference position based on s
    # In a real implementation, you'd need the actual path representation
    s = x_dev[0]
    e_y = x_dev[1]
    e_theta = x_dev[2]
    
    # Simple linear interpolation (replace with proper path interpolation)
    idx = min(int(s), len(x_ref_global) - 2)
    alpha = s - idx
    x_ref = x_ref_global[idx, 0] + alpha * (x_ref_global[idx+1, 0] - x_ref_global[idx, 0])
    y_ref = x_ref_global[idx, 1] + alpha * (x_ref_global[idx+1, 1] - x_ref_global[idx, 1])
    
    # Calculate global position using deviation
    theta_ref = heading_ref[idx] + alpha * (heading_ref[idx+1] - heading_ref[idx])
    x_global = x_ref - e_y * np.sin(theta_ref)
    y_global = y_ref + e_y * np.cos(theta_ref)
    theta_global = theta_ref + e_theta
    
    return np.array([x_global, y_global, theta_global])


# ------------------------------------------------------------------
# MAIN - FIXED reference handling
# ------------------------------------------------------------------
def main():
    import time

    # Problem setup
    lbu = np.array([-np.pi/4, 0.1])  # steering, velocity (min velocity > 0)
    ubu = np.array([ np.pi/4,  1.0])
    N = 25
    Tf = 5.0

    # Build solver
    solver = create_car_trajectory_solver(lbu, ubu, N=N, Tf=Tf)

    # Global reference path (sinusoidal)
    xg = 2 + np.linspace(0, 10, N+1)
    yg = 3 + 0.5*np.sin(np.linspace(0, 2*np.pi, N+1))  # More interesting path
    heading = np.arctan2(np.gradient(yg), np.gradient(xg))
    kappa = compute_curvature(xg, yg)

    # Convert to path-relative states + curvature
    x_ref, p_ref = global_to_path_states(xg, yg, heading, kappa)

    # Initial state in deviation space should be zero if starting on path
    x0_dev = np.array([0.0, 0.0, 0.0])  # Start on path

    # Solve MPC
    start_time = time.time()
    x_opt, u_opt = solve_trajectory(solver, x0_dev, x_ref=x_ref, p_ref=p_ref)
    end_time = time.time()

    print("Optimization completed!")
    print("Final deviation state:", x_opt[-1, :])
    print("Time taken for optimization:", end_time - start_time)

    # Simulate in GLOBAL coordinates
    x0_global = np.array([xg[0], yg[0], heading[0]])
    x_traj_global = sim_car_global(x0_global, u_opt, Tf)

    # Convert MPC solution to global coordinates for comparison
    x_opt_global = []
    for i in range(len(x_opt)):
        global_state = deviation_to_global(x_opt[i], np.column_stack([xg, yg]), heading)
        x_opt_global.append(global_state)
    x_opt_global = np.array(x_opt_global)

    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(xg, yg, "r--", label="Reference Path", linewidth=2)
    plt.plot(x_traj_global[:, 0], x_traj_global[:, 1], "b-", label="Car Path (sim)")
    plt.plot(x_opt_global[:, 0], x_opt_global[:, 1], "g--", label="MPC Planned Path")
    plt.plot(xg[0], yg[0], "ro", markersize=8, label="Start")
    plt.axis("equal")
    plt.legend()
    plt.title("Path Tracking")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    plt.subplot(2, 2, 2)
    plt.plot(x_opt[:, 1], label="Lateral Error (e_y)")
    plt.plot(x_opt[:, 2], label="Heading Error (e_theta)")
    plt.legend()
    plt.title("Deviation States")
    plt.xlabel("Time step")
    plt.ylabel("Error")
    
    plt.subplot(2, 2, 3)
    plt.plot(u_opt[:, 0], label="Steering (delta)")
    plt.legend()
    plt.title("Control Input - Steering")
    plt.xlabel("Time step")
    plt.ylabel("rad")
    
    plt.subplot(2, 2, 4)
    plt.plot(u_opt[:, 1], label="Velocity (v)")
    plt.legend()
    plt.title("Control Input - Velocity")
    plt.xlabel("Time step")
    plt.ylabel("m/s")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
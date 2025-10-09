from casadi import sin, cos, MX, vertcat, atan, sign
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosOcpIterates, AcadosSim, AcadosSimSolver
import casadi as ca
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import os


def create_hcar_model(constants=None, model_name="halfcar"):
    model = AcadosModel()
    model.name = model_name
    # CasADi Model
    # states
    if constants is None:
        constants = {'L': 2.0} 
    L = constants["L"]
    X = MX.sym("X")
    Y = MX.sym("Y")
    theta = MX.sym("theta")
    v = MX.sym("v_x")
    delta = MX.sym("delta")  # r = yaw rate, often called omega as well
    model.u = vertcat(
        delta,
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
        v*ca.tan(delta)/L,  # = phi_dot
    )
    model.f_impl_expr = model.xdot - model.f_expl_expr
    return model

# def create_car_trajectory_solver(lbu, ubu, N=20, Tf=2.0, dt=None, ns=6):
def create_car_trajectory_solver(lbu, ubu, N=50, Tf=2.0, dt=None, num_sides=6):

    """
    Create an ACADOS solver with proper time-varying obstacle constraints
    that correctly reference position states at each node.
    """
    # Create car model
    # model, constants = create_car_model()
    model = create_hcar_model()
    # Create OCP
    ocp = AcadosOcp()
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.z = model.z
    model_ac.p = model.p
    model_ac.name = model.name
    ocp.model = model_ac
    
    # Dimensions
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    # n_p_original = model.p.size()[0]  # Original parameters (Fz, mu_x, mu_y)
    
    # Set dimensions
    ocp.dims.N = N
    # ocp.dims.nh = ns + 3  # Obstacle + steering constraints
    # ocp.dims.nh_e = ns    # Only obstacles at terminal
    # ocp.dims.nh = num_sides # Obstacle + steering constraints
    # ocp.dims.nh_e = num_sides    # Only obstacles at terminal
    
    # Time step
    if dt is None:
        dt = Tf / N
    ocp.solver_options.tf = Tf
    
    # Cost function
    Q = np.diag([1.0, 1.0, 1.0])
    R = np.diag([0.1,0.1 ])
    
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ny = nx + nu
    
    ocp.cost.yref = np.zeros(ny)  # Reference trajectory
    ocp.cost.W = np.block([[Q, np.zeros((nx, nu))], 
                          [np.zeros((nu, nx)), R]])

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx,:nx] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[nx:,:nu] = np.eye(nu)
    ocp.cost.Vu = Vu 

    # Terminal cost
    ocp.cost.W_e = Q*2
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref_e = np.zeros((nx, ))


    # Initial state
    ocp.constraints.x0 = model.x0 
    
    # Control bounds
    # max_steer = np.pi/4
    ocp.constraints.lbu = lbu
    ocp.constraints.ubu = ubu
    ocp.constraints.idxbu = np.arange(nu)

    # --- Critical Fix: Proper State Reference for Obstacle Constraints ---
    # Original model parameters
    # p_orig = model.p
    
    # # # Additional parameters for obstacles
    # obs_A = ca.MX.sym('obs_A', num_sides, 2)
    # obs_b = ca.MX.sym('obs_b', num_sides)
    
    # # Combine all parameters
    # ocp.model.p = ca.vertcat(p_orig, obs_A.reshape((-1, 1)), obs_b)
    
    # # --- Corrected Constraint Implementation ---
    # # Get position states symbolically
    X = model.x[0]  # X position state
    Y = model.x[1]  # Y position state
    
    # # Obstacle constraints for ALL nodes
    # h_obs = obs_A @ ca.vertcat(X, Y) - obs_b
    
    # # Steering constraints for ALL nodes
    
    # # Combined constraints for path (nodes 0 to N-1)
    # ocp.model.con_h_expr = ca.vertcat(h_obs)
    
    # # Terminal constraints (only obstacles at node N)
    # ocp.model.con_h_expr_e = h_obs
    
    # # Constraint bounds
    # ocp.constraints.lh = np.concatenate([-100.0*np.ones(ns), np.zeros(3)])
    # ocp.constraints.uh = np.concatenate([np.zeros(ns), np.zeros(3)])
    # ocp.constraints.lh_e = -100.0*np.ones(ns)
    # ocp.constraints.uh_e = np.zeros(ns)
    
    # Solver options
    # ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_OSQP'
    # ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    # ocp.solver_options.integrator_type = 'IRK'
    # ocp.solver_options.nlp_solver_type = 'SQP'
    # ocp.solver_options.sim_method_num_stages = 4
    # ocp.solver_options.sim_method_num_steps = 3
    # ocp.solver_options.tol = 1e-4
    # ocp.solver_options.print_level = 0
    # ocp.solver_options.sim_method_newton_iter = 10

    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.tf = Tf
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    # ocp.solver_options.collocation_type = 'GAUSS_RADAU_IIA'
    # ocp.solver_options.time_steps = time_steps
    # ocp.solver_options.shooting_nodes = shooting_nodes

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"#"PARTIAL_CONDENSING_HPIPM" #"FULL_CONDENSING_HPIPM" #"PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx =  "GAUSS_NEWTON"#"EXACT",
    # ocp.solver_options.cost_discretization ="INTEGRATOR"
    ocp.solver_options.qp_solver_cond_N = int(N/100)
    # ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tol = 1e-3
    
    # Create solver
    solver = AcadosOcpSolver(ocp)
    
    # return solver, n_p_original, constants
    return solver

# def solve_trajectory(solver, n_p_original, x0, x_ref=None, obs_A=None, obs_b=None):
def solve_trajectory(solver, x0, x_ref=None, obs_A=None, obs_b=None):
    """Solve with proper time-varying obstacle constraints."""
    N = solver.acados_ocp.dims.N
    # Set initial state
    solver.constraints_set(0, 'lbx', x0)
    solver.constraints_set(0, 'ubx', x0)
    # Set reference trajectory
    if x_ref is not None:
        for i in range(N):
            solver.set(i, 'yref', np.concatenate([x_ref[i], np.zeros(solver.acados_ocp.dims.nu)]))
            solver.set(i,'x', x_ref[i])
            # solver.set(i,'u', np.array([0.5, 0.0,0.0,0.0,0.0]))
            solver.set(i,'u', np.array([0.0, 1.0]))
        solver.set(N, 'yref', x_ref[N])
        solver.set(N,'x', x_ref[N])
        solver.set(i,'u', np.array([0.0, 0.0]))
    
    # Set time-varying obstacle constraints at ALL nodes
    # if obs_A is not None and obs_b is not None:
    #      for# i in range(N+1):
    #         params = np.zeros(n_p_original)
    #          par#ams = np.concatenate([
    #             params,
    #             obs_A[i].flatten(),
    #             obs_b[i]
    #         ])
    #         solver.set(i, 'p', params)
    
    # Solve
    status = solver.solve()

    # if status != 0:
    #     print(f"Solver status: {status}")
    #     solver.print_statistics() 
    # Get solution
    x_opt = np.array([solver.get(i, 'x') for i in range(N+1)])
    u_opt = np.array([solver.get(i, 'u') for i in range(N)])
    
    return x_opt, u_opt

def generate_car_sim(model, code_export_dir, time_step=0.001, num_stages=1, num_steps=1):
    # model, c, q = create_car_model()
    sim = AcadosSim()
    sim.model = model

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

def get_symbol(symbol_vector, name):
    """
        Find a casadi symbol by its name in a vertcat (vertical concatenation) vector of symbols.
        This method can also find vector-shaped symbols that span across a part of the full symbol vector.
    """
    idx = get_symbol_idx(symbol_vector, name)
    if idx is None:
        return None
    elif isinstance(idx, tuple):
        return symbol_vector[idx[0]:idx[1]]
    else:
        return symbol_vector[idx]


def get_symbol_idx(symbol_vector, name):
    """
        Find the index of a casadi symbol by its name in a vertcat (vertical concatenation) of symbols.
        If the symbol is a vector instead of a scalar, this method returns the index range of the symbol.
    """
    v_len = symbol_vector.size()[0]
    slice_start = 0
    for i in range(v_len):
        info = symbol_vector[slice_start:i + 1].info()
        if "slice" not in info:
            if symbol_vector[slice_start:i + 1].name() == name:
                return (slice_start, i + 1) if i != slice_start else i
            else:
                slice_start = i + 1
    return None

def main():
    import time
    model = create_hcar_model()
    print(model)
    lbu = np.array([-np.pi/4, -1.0])
    ubu = np.array([np.pi/4, 1.0])
    N = 30
    Tf = 10.0

    num_sides = 4  # rectangle obstacles
        # solver, n_p_original, constants = create_car_trajectory_solver(N=N, Tf=Tf, ns=ns)
    solver = create_car_trajectory_solver(lbu, ubu, N=N, Tf=Tf, num_sides=num_sides)

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
        progress = i/N
        width = 2.0 - 1.5*progress  # Narrowing corridor
            
        # Left wall (y >= -width/2)
        obs_A[i, 0, :] = [0, -1]
        obs_b[i, 0] = -width/2
            
        # Right wall (y <= width/2)
        obs_A[i, 1, :] = [0, 1]
        obs_b[i, 1] = width/2
            
        # Front wall (x <= 12)
        obs_A[i, 2, :] = [1, 0]
        obs_b[i, 2] = 12
            
        # Back wall (x >= -1)
        obs_A[i, 3, :] = [-1, 0]
        obs_b[i, 3] = 1
            
        # # Other sides (not used)
        # obs_A[i, 4:, :] = 0
        # obs_b[i, 4:] = 1000  # Large number
            
    # Solve with time-varying obstacles
    x_opt, u_opt = solve_trajectory(solver, x0, x_ref=x_ref, obs_A=obs_A, obs_b=obs_b)
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
    x_ref = np.array([[ 0.        ,  0.        ,  0.        ],
       [ 0.55194163,  0.23528799,  0.        ],
       [ 1.10388325,  0.47057599,  0.        ],
       [ 1.65582488,  0.70586398,  0.        ],
       [ 2.20776651,  0.94115198,  0.        ],
       [ 2.75970814,  1.17643997,  0.        ],
       [ 3.31164976,  1.41172797,  0.        ],
       [ 3.86359139,  1.64701596,  0.        ],
       [ 4.41553302,  1.88230395,  0.        ],
       [ 4.96747464,  2.11759195,  0.        ],
       [ 5.51941627,  2.35287994,  0.        ],
       [ 6.0713579 ,  2.58816794,  0.        ],
       [ 6.62329953,  2.82345593,  0.        ],
       [ 7.17524115,  3.05874393,  0.        ],
       [ 7.72718278,  3.29403192,  0.        ],
       [ 8.27912441,  3.52931991,  0.        ],
       [ 8.86084129,  3.62434966,  0.        ],
       [ 9.46077012,  3.63359088,  0.        ],
       [10.06069895,  3.6428321 ,  0.        ],
       [10.66062778,  3.65207332,  0.        ],
       [11.26055661,  3.66131453,  0.        ],
       [11.86048544,  3.67055575,  0.        ],
       [12.46041426,  3.67979697,  0.        ],
       [13.06034309,  3.68903819,  0.        ],
       [13.66027192,  3.69827941,  0.        ],
       [14.26020075,  3.70752062,  0.        ],
       [14.86012958,  3.71676184,  0.        ],
       [15.46005841,  3.72600306,  0.        ],
       [16.05998724,  3.73524428,  0.        ],
       [16.65991607,  3.7444855 ,  0.        ],
       [17.2598449 ,  3.75372671,  0.        ]])




    start_time =  time.time()
    x_opt, u_opt = solve_trajectory(solver, np.array([0.0,0.0,0.0]), x_ref=x_ref, obs_A=obs_A, obs_b=obs_b)
    end_time = time.time()
    print("Time taken for optimization:", end_time - start_time)

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

    # plt.figure()
    # bike_model, bike_c = create_bike_model()
    # bike_traj = sim_bike(bike_model, bike_c)
    # bike_traj = np.array(bike_traj)
    # plt.plot(bike_traj[:, 0], bike_traj[:, 1], label="bike path")

    plt.figure()
    x_traj = sim_car(solver.acados_ocp.model, x0, u_opt, Tf)
    x_traj = np.array(x_traj)
    plt.plot(x_traj[:, 0], x_traj[:, 1], label="car path")
    plt.legend()
    plt.show()

    # x_traj = sim_car(solver.acados_ocp.model, constants)
    # x_traj = np.array(x_traj)
    # plt.plot(x_traj[:, 0], x_traj[:, 1], label="car path")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
from .solvers_base import MPCSolver, MPCSolverParams
import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

class AcadosSolver(MPCSolver):
    """
    ACADOS-based MPC solver implementation.
    params: dictionary of solver options (horizon, weights, bounds, etc.), NOT system/dynamics parameters.
    System parameters and model creation are handled by the system object.
    """
    def __init__(self, system, params: MPCSolverParams=None):
        self.system = system
        self.params = params if params is not None else MPCSolverParams()
        self.model = None
        self.solver = None
        self.last_params = None
        self.last_model = None

    def get_solver(self, model=None):
        # Use provided model or get from system
        # if model is None and hasattr(self.system, 'create_acados_model'):
        if model is not None:
            pass
        elif self.model is None:
            model = self.system.create_acados_model()
        self.model = model
        # Only recreate solver if params or model have changed
        if self.solver is None or self.last_params != self.params or self.last_model is not model:
            self.solver = self._create_solver(model, self.params)
            self.last_params = self.params
            self.last_model = model
        return self.solver

    def _create_solver(self, model, params):
        lbu = self.system.control_bounds[0]
        ubu = self.system.control_bounds[1]
        N = params.Nt
        Tf = params.T 
        # dt = params.dt
        num_sides = params.num_sides
        # q_goal_pos = params.q_goal_pos
        # q_pos = params.get('q_pos', 1.0)
        # r_control = params.get('r_control', 0.1)
        Q = params.Q
        R = params.R
        QT = params.QT

        ocp = AcadosOcp()
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.f_impl_expr
        model_ac.f_expl_expr = model.f_expl_expr
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.name = model.name
        obs_A_sym = ca.MX.sym('obs_A', num_sides*2)
        obs_b_sym = ca.MX.sym('obs_b', num_sides)
        model_ac.p = ca.vertcat(obs_A_sym, obs_b_sym)
        Xpos = model.x[0]
        Ypos = model.x[1]
        obs_A_mat = ca.reshape(obs_A_sym, num_sides, 2)
        pos_vec = ca.vertcat(Xpos, Ypos)
        ocp.model = model_ac
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ocp.dims.N = N
        ocp.solver_options.tf = Tf
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ny = nx + nu
        # Q = np.diag([q_pos, q_pos, 0.0])
        # R = np.diag([r_control, r_control])
        W = np.block([[Q, np.zeros((nx,nu))], [np.zeros((nu,nx)), R]])
        ocp.cost.W = W
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        Vu = np.zeros((ny, nu))
        Vu[nx:, :] = np.eye(nu)
        ocp.cost.Vu = Vu
        ocp.cost.W_e = QT
        ocp.cost.Vx_e = np.eye(nx)
        ocp.constraints.x0 = model.x0
        ocp.constraints.lbu = lbu
        ocp.constraints.ubu = ubu
        ocp.constraints.idxbu = np.arange(nu)
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
        solver = AcadosOcpSolver(ocp)
        return solver

    def solve(self, x0, x_goal, reference=None, obstacles=None):
        solver = self.get_solver(self.model)
        N = self.params.Nt
        num_sides = self.params.num_sides
        nx = self.system.n_dims
        nu = self.system.n_controls

        # Extract goal from ref (assume ref contains goal position)
        goal = x_goal if isinstance(x_goal, (list, np.ndarray)) else np.zeros(nx)

        # Extract obstacle parameters
        if obstacles is not None:
            A_per_node = obstacles.get('A')
            b_per_node = obstacles.get('b')
        else:
            # Default: no obstacles
            A_per_node = np.zeros((N+1, num_sides, 2))
            b_per_node = np.zeros((N+1, num_sides))

        # Optional reference trajectory and initial guess for controls
        x_ref = reference.get('x_ref') if reference and 'x_ref' in reference else None
        u_init = reference.get('u_init') if reference and 'u_init' in reference else None

        # Initial state
        solver.constraints_set(0, 'lbx', x0)
        solver.constraints_set(0, 'ubx', x0)

        # Set running yref sequence
        if x_ref is None:
            x_ref = np.tile([*goal[0:2], 0], (N+1, 1))
        for i in range(N):
            yref_i = np.concatenate([x_ref[i], np.zeros(nu)])
            solver.set(i, 'yref', yref_i)
            if u_init is not None:
                solver.set(i, 'u', u_init[i])
        yref_e = np.zeros(nx)
        yref_e[0:2] = goal[0:2]
        solver.set(N, 'yref', yref_e)

        # Set obstacle parameters for each node
        for i in range(N+1):
            A_i = np.array(A_per_node[i])
            b_i = np.array(b_per_node[i]).reshape(-1)
            A_flat = A_i.reshape(-1)
            p = np.concatenate([A_flat, b_i])
            solver.set(i, 'p', p)

        # Solve
        status = solver.solve()
        x_opt = np.array([solver.get(i, 'x') for i in range(N+1)])
        u_opt = np.array([solver.get(i, 'u') for i in range(N)])
        # x_res = upsample_mpc(x_opt, self.params.T, self.params.Nt, self.params.dt, method="step")
        # u_res = upsample_mpc(u_opt, self.params.T, self.params.Nt, self.params.dt, method="step")
        # return x_res[:self.params.Nt,:], u_res[:self.params.Nt,:], status
        return x_opt, u_opt, status

from .solvers_base import MPCSolver
import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

class AcadosSolver(MPCSolver):
    """
    ACADOS-based MPC solver implementation.
    params: dictionary of solver options (horizon, weights, bounds, etc.), NOT system/dynamics parameters.
    System parameters and model creation are handled by the system object.
    """
    def __init__(self, system, params=None):
        self.system = system
        self.params = params if params is not None else {}
        self.model = None
        self.solver = None
        self.last_params = None
        self.last_model = None

    def get_solver(self, model=None):
        # Use provided model or get from system
        if model is None and hasattr(self.system, 'create_acados_model'):
            model = self.system.create_acados_model()
        # Only recreate solver if params or model have changed
        if self.solver is None or self.last_params != self.params or self.last_model is not model:
            self.model = model
            self.solver = self._create_solver(model, self.params)
            self.last_params = self.params.copy()
            self.last_model = model
        return self.solver

    def _create_solver(self, model, params):
        lbu = params.get('lbu', np.array([-1.0, 0.0]))
        ubu = params.get('ubu', np.array([1.0, 2.0]))
        N = params.get('N', 40)
        Tf = params.get('Tf', 8.0)
        num_sides = params.get('num_sides', 6)
        v_max = params.get('v_max', 1.0)
        q_goal_pos = params.get('q_goal_pos', 500.0)
        q_pos = params.get('q_pos', 1.0)
        r_control = params.get('r_control', 0.1)
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
        h_obs = ca.mtimes(obs_A_mat, pos_vec) - obs_b_sym
        ocp.model = model_ac
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ocp.dims.N = N
        ocp.solver_options.tf = Tf
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ny = nx + nu
        Q = np.diag([q_pos, q_pos, 0.0])
        R = np.diag([r_control, r_control])
        W = np.block([[Q, np.zeros((nx,nu))], [np.zeros((nu,nx)), R]])
        ocp.cost.W = W
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        Vu = np.zeros((ny, nu))
        Vu[nx:, :] = np.eye(nu)
        ocp.cost.Vu = Vu
        ocp.cost.W_e = np.diag([q_goal_pos, q_goal_pos, 1.0])
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
        # Save meta-info on solver object to help solve wrapper
        solver._num_sides = num_sides
        solver._v_max = v_max
        solver._N = N
        solver._nx = nx
        solver._nu = nu
        return solver

    def solve(self, x0, ref, parameters=None, obstacles=None):
        solver = self.get_solver(self.model)
        N = solver._N
        num_sides = solver._num_sides
        nx = solver._nx
        nu = solver._nu

        # Extract goal from ref (assume ref contains goal position)
        goal = ref if isinstance(ref, (list, np.ndarray)) else np.zeros(nx)

        # Extract obstacle parameters
        if obstacles is not None:
            A_per_node = obstacles.get('A_per_node')
            b_per_node = obstacles.get('b_per_node')
        else:
            # Default: no obstacles
            A_per_node = np.zeros((N+1, num_sides, 2))
            b_per_node = np.zeros((N+1, num_sides))

        # Optional reference trajectory and initial guess for controls
        x_ref = parameters.get('x_ref') if parameters and 'x_ref' in parameters else None
        u_init = parameters.get('u_init') if parameters and 'u_init' in parameters else None

        # Initial state
        solver.constraints_set(0, 'lbx', x0)
        solver.constraints_set(0, 'ubx', x0)

        # Set running yref sequence
        if x_ref is None:
            x_ref = np.tile(np.zeros(nx), (N+1, 1))
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
        return x_opt, u_opt, status

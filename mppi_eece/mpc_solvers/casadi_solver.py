from .solvers_base import MPCSolver
import numpy as np
import casadi as ca

class CasadiMPCSolver(MPCSolver):
    """
    CasADi-based MPC solver implementation.
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
        if model is None and hasattr(self.system, 'create_casadi_model'):
            dxdt, state, control = self.system.create_casadi_model()
            ode = ca.Function('ode', [state, control], [dxdt])
            model = (dxdt, state, control, ode)
        # Only recreate solver if params or model have changed
        if self.solver is None or self.last_params != self.params or self.last_model != model:
            self.model = model
            self.solver = self._create_solver(model, self.params)
            self.last_params = self.params.copy()
            self.last_model = model
        return self.solver

    def _create_solver(self, model, params):
        # Fully integrate cas_shooting_solver logic
        Nt = params.get('N', 40)
        ns = params.get('num_sides', 6)
        dt = params.get('dt', getattr(self.system, 'dt', 0.1))
        solver_type = params.get('solver', 'ipopt')
        dxdt, state, control, ode = model
        opti = ca.Opti()
        x = opti.variable((Nt + 1)*3)
        u = opti.variable((Nt+1)*2)
        goal = opti.parameter(1,3)
        start = opti.parameter(1,3)
        obs_A = opti.parameter((Nt+1)*ns, 2)
        obs_b = opti.parameter((Nt+1)*ns)
        path = opti.parameter((Nt+1)*3)
        w_path = opti.parameter()
        goal_vector = ca.repmat(goal[:2].T, Nt+1,1)
        indicies = []
        for t in range(Nt+1):
            indicies.append(3*t)
            indicies.append(3*t+1)
        delta_goal = x[indicies] - goal_vector
        delta_path = x - path
        opti.minimize(((1-w_path)*ca.sumsqr(delta_goal) + w_path *ca.sumsqr(delta_path))/100)
        opti.subject_to(x[0] == start[0])
        opti.subject_to(x[1] == start[1])
        opti.subject_to(x[2] == start[2])
        for t in range(Nt):
            opti.subject_to(u[2*t] < self.system.control_bounds[1][0])
            opti.subject_to(u[2*t] > self.system.control_bounds[0][0])
            opti.subject_to(u[2*t+1] < self.system.control_bounds[1][1])
            opti.subject_to(u[2*t+1] > self.system.control_bounds[0][1])
            opti.subject_to(obs_A[ns*t:ns*(t+1),:] @ x[3*t:3*t+2] < obs_b[ns*t:ns*(t+1)])
        for t in range(Nt):
            fx = ode(x[3*t:3*(t+1)], u[2*t:2*(t+1)])
            ct = ca.cos(x[3*t+2])
            st = ca.sin(x[3*t+2])
            R = ca.MX(2,2)
            w = ca.MX(2,2)
            R[0,0] = ct
            R[0,1] = -st
            R[1,0] = st
            R[1,1] = ct
            ct_1 = ca.cos(x[3*(t+1)+2])
            st_1 = ca.sin(x[3*(t+1)+2])
            ct_w = ca.cos(fx[2]*dt)
            st_w = ca.sin(fx[2]*dt)
            w[0,0] = ct_w
            w[0,1] = -st_w
            w[1,0] = st_w
            w[1,1] = ct_w
            opti.subject_to(x[3*(t+1)] == x[3*t] + fx[0] *dt)
            opti.subject_to(x[3*(t+1)+1] == x[3*t+1] + fx[1] *dt)
            R_1s = w @ R
            opti.subject_to( (ca.arctan2(st_1, ct_1) - ca.arctan2(R_1s[1,0], R_1s[0,0])) == 0.0)
        jit_options = {"flags": ["-O3", "-march=native"], "compiler":"ccache gcc"}
        p_opts = {"expand": 1, "print_time": False, "verbose": False,  "jit": True, "compiler":"shell", "jit_options":jit_options,'jit_cleanup':True, 'jit_temp_suffix':False}
        if solver_type.lower() == "ipopt":
            s_opts = {"tol":1e-5, "max_iter": 5000, "mu_strategy":"adaptive", "print_level":0}
            opti.solver("ipopt", p_opts, s_opts)
        elif solver_type.lower() == "snopt":
            s_opts = {"Minor print level":0, "Major print level":0, "Summary file": 0, "Solution": "no", "Suppress options listings":1, "Timing level":3, "Scale option": 2, "Major iterations limit":1000}
            opti.solver("snopt", p_opts, s_opts)
        else:
            opti.solver(solver_type.lower(),p_opts)
        return opti.to_function('F', [x, u, start, goal,  obs_A, obs_b, path, w_path], [x, u], ['x', 'u', 'start', 'goal', 'obs_A', 'obs_b', 'path', 'w_path'], ['x_opt', 'u_opt'])

    def solve(self, x0, ref, parameters=None, obstacles=None):
        # Use self.solver (CasADi function) to solve the MPC problem
        solver = self.get_solver(self.model)
        Nt = self.params.get('N', 40)
        ns = self.params.get('num_sides', 6)
        # Prepare initial guess and reference trajectory
        x_ref = parameters.get('x_ref') if parameters and 'x_ref' in parameters else None
        u_init = parameters.get('u_init') if parameters and 'u_init' in parameters else None
        path = parameters.get('path') if parameters and 'path' in parameters else np.zeros((Nt+1, 3))
        w_path = parameters.get('w_path') if parameters and 'w_path' in parameters else 0.0
        # Prepare obstacles
        if obstacles is not None:
            A_per_node = obstacles.get('A_per_node')
            b_per_node = obstacles.get('b_per_node')
        else:
            A_per_node = np.zeros((Nt+1, ns, 2))
            b_per_node = np.zeros((Nt+1, ns))
        # Flatten obstacle arrays for solver
        obs_A_flat = np.array(A_per_node).reshape((Nt+1)*ns, 2)
        obs_b_flat = np.array(b_per_node).reshape((Nt+1)*ns)
        # Initial guess for controls
        if u_init is None:
            u_init = np.kron(np.ones((1, Nt+1)), [0.0, self.system.control_bounds[1][1]]).ravel()
        # Initial guess for states
        if x_ref is None:
            x_ref = np.tile(np.zeros(3), (Nt+1, 1)).ravel()
        # Call solver function
        x_opt, u_opt = solver(
            x_ref,
            u_init,
            x0.reshape(1, -1),
            ref.reshape(1, -1),
            obs_A_flat,
            obs_b_flat,
            path.ravel(),
            w_path
        )
        x_opt = np.array(x_opt).reshape((-1,3))
        u_opt = np.array(u_opt).reshape((-1,2))
        return x_opt, u_opt, 0

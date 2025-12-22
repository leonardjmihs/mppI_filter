import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mppi_eece.sim.grid import OccupGrid
from mppi_eece.jax_mppi.mppi_planners import MPPI_Planner_Occup
from mppi_eece.jax_mppi.collision_checker import CollisionChecker
import time
from mppi_eece.sim.pcrb import PCRBController

class UKF_Controller:
    def __init__(self, system, mppi_planner, alpha=1e-2, beta=2, kappa=0,
                 auto_reset: bool = True,
                 cov_trace_threshold: float = 1e8,
                 innovation_threshold: float = 1e10,
                 reset_P_scale: float = 1.0):
        """
        UKF for control sequence estimation (controls-only formulation)
        
        Args:
            system: Your Unicycle system
            mppi_planner: Your MPPI_Planner_Occup instance (for cost evaluation)
            alpha, beta, kappa: UKF tuning parameters
        """
        self.system = system
        self.planner = mppi_planner
        self.N = mppi_planner.N  # Horizon length
        # prefer planner/system-provided control dimension, fallback to 2
        self.n_u = getattr(mppi_planner, "n_u", getattr(system, "n_u", 2))
        self.n_theta = self.N * self.n_u  # total control sequence dimension
        
        # UKF parameters
        self.n_sigma = 2 * self.n_theta + 1
        self.lambda_ = alpha**2 * (self.n_theta + kappa) - self.n_theta
        
        # Weights
        self.Wm = jnp.zeros(self.n_sigma)
        self.Wc = jnp.zeros(self.n_sigma)
        
        self.Wm = self.Wm.at[0].set(self.lambda_ / (self.n_theta + self.lambda_))
        self.Wc = self.Wc.at[0].set(self.lambda_ / (self.n_theta + self.lambda_) + 
                                     (1 - alpha**2 + beta))
        
        for i in range(1, self.n_sigma):
            w = 1 / (2 * (self.n_theta + self.lambda_))
            self.Wm = self.Wm.at[i].set(w)
            self.Wc = self.Wc.at[i].set(w)

        # Auto-reset configuration for numerical stability
        self.auto_reset = bool(auto_reset)
        self.cov_trace_threshold = float(cov_trace_threshold)
        self.innovation_threshold = float(innovation_threshold)
        self.reset_P_scale = float(reset_P_scale)
    
    def shift_controls(self, theta):
        """
        Shift operator: Phi(theta)
        Moves controls forward, duplicates last control
        
        Args:
            theta: [2N] control sequence
        Returns:
            theta_new: [2N] shifted control sequence
        """
        theta_reshaped = theta.reshape((-1, self.n_u))
        theta_shifted = jnp.roll(theta_reshaped, -1, axis=0)
        # Keep last control
        theta_shifted = theta_shifted.at[-1].set(theta_reshaped[-1])
        return theta_shifted.ravel()
    
    def generate_sigma_points(self, mu, P):
        """
        Generate sigma points using Cholesky decomposition
        
        Args:
            mu: mean [n_theta]
            P: covariance [n_theta, n_theta]
        Returns:
            sigma_points: [n_sigma, n_theta]
        """
        n = self.n_theta
        
        # Regularization for numerical stability
        P_reg = P + jnp.eye(n) * 1e-9
        
        try:
            L = jnp.linalg.cholesky(P_reg)
        except:
            # Fallback to eigendecomposition
            eigvals, eigvecs = jnp.linalg.eigh(P_reg)
            eigvals = jnp.maximum(eigvals, 1e-9)
            L = eigvecs @ jnp.diag(jnp.sqrt(eigvals))
        
        L_scaled = jnp.sqrt(n + self.lambda_) * L
        
        # Generate sigma points
        sigma_points = jnp.zeros((self.n_sigma, n))
        sigma_points = sigma_points.at[0].set(mu)
        
        for i in range(n):
            sigma_points = sigma_points.at[i+1].set(mu + L_scaled[:, i])
            sigma_points = sigma_points.at[n+i+1].set(mu - L_scaled[:, i])
        
        return sigma_points
    

    def measurement_function(self, theta, x_current, q_ref, cost_map):
        """
        Measurement: h(theta) = -C(theta | x_current)
        
        Args:
            theta: [n_theta] control sequence
            x_current: [3] current observed system state
            q_ref: [3] reference state
            cost_map: collision checker
        Returns:
            y: scalar (negative cost)
        """
        u_seq = theta.reshape((-1, self.n_u))
        
        # Evaluate trajectory cost starting from known x_current
        outputs = self.planner.eval_U_seq(
            u_seq, 
            theta,  # original_u for regularization
            x_current,  # known current state
            q_ref, 
            cost_map,
        )
        cost = outputs[0] 
        return -cost
    
    # @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, mu, P, Q):
        """
        UKF Prediction step
        
        Args:
            mu: current mean [n_theta]
            P: current covariance [n_theta, n_theta]
            Q: process noise covariance [n_theta, n_theta]
        Returns:
            mu_pred, P_pred, sigma_points_pred
        """
        # Generate sigma points
        sigma_points = self.generate_sigma_points(mu, P)
        
        # Propagate through shift dynamics
        sigma_points_pred = jax.vmap(self.shift_controls)(sigma_points)
        
        # Compute predicted mean
        mu_pred = jnp.sum(self.Wm[:, None] * sigma_points_pred, axis=0)
        
        # Compute predicted covariance (use symmetric / jittered form for stability)
        diff = sigma_points_pred - mu_pred
        P_pred = jnp.sum(self.Wc[:, None, None] *
                        (diff[:, :, None] @ diff[:, None, :]),
                        axis=0) + Q

        # enforce symmetry and add tiny jitter to preserve PD
        P_pred = (P_pred + P_pred.T) / 2.0
        P_pred = P_pred + jnp.eye(self.n_theta) * 1e-9

        # defensive NaN check (kept but non-blocking)
        if jnp.any(jnp.isnan(mu_pred)):
            # don't breakpoint in library code; raise to signal upstream
            raise RuntimeError("NaN in UKF predicted mean")

        return mu_pred, P_pred, sigma_points_pred
    
    def update(self, mu_pred, P_pred, sigma_points_pred, 
               y_meas, R, x_current, q_ref, cost_map):
        """
        UKF Update step
        
        Args:
            mu_pred: predicted mean
            P_pred: predicted covariance
            sigma_points_pred: predicted sigma points
            y_meas: measurement (0 for Option B)
            R: measurement noise variance (scalar)
            x_current: current observed system state
            q_ref: reference state
            cost_map: collision checker
        Returns:
            mu_updated, P_updated, innovation
        """
        # Propagate sigma points through measurement function
        meas_sigma = jax.vmap(
            lambda theta: self.measurement_function(theta, x_current, q_ref, cost_map)
        )(sigma_points_pred)
        
        # Clip infinite costs
        meas_sigma = jnp.clip(meas_sigma, -1e10, 0)

        # Predicted measurement mean
        y_pred = jnp.sum(self.Wm * meas_sigma)

        # Innovation covariance (scalar)
        meas_diff = meas_sigma - y_pred
        S = jnp.sum(self.Wc * meas_diff**2) + R

        # Cross-covariance between state and measurement
        state_diff = sigma_points_pred - mu_pred
        P_xy = jnp.sum(self.Wc[:, None] * state_diff * meas_diff[:, None], axis=0)

        # Kalman gain
        K = P_xy / S

        # Update mean and covariance using Joseph / stabilized form
        innovation = y_meas - y_pred
        mu_updated = mu_pred + K * innovation
        P_updated = P_pred - jnp.outer(K, K) * S

        # enforce symmetry and add tiny jitter to maintain PD
        P_updated = (P_updated + P_updated.T) / 2.0
        P_updated = P_updated + jnp.eye(self.n_theta) * 1e-9

        return mu_updated, P_updated, innovation

    # ------------------ Reset / recovery helpers ------------------
    def reset_to_identity(self, mu=None, scale=None):
        """
        Reset to zero-mean with scaled identity covariance (returns jnp arrays)
        """
        mu0 = jnp.zeros(self.n_theta) if mu is None else jnp.array(mu)
        scl = self.reset_P_scale if scale is None else float(scale)
        P0 = jnp.eye(self.n_theta) * scl
        return mu0, P0

    def reset_to_prior(self, mu_prior, P_prior):
        """
        Reset to provided prior (converts to jnp arrays)
        """
        return jnp.array(mu_prior), jnp.array(P_prior)

    def check_and_reset(self, mu, P, innovation=None):
        """
        Check for numerical issues and optionally reset. Returns (mu,P,reset_flag)
        Criteria:
          - any NaN in mu or P
          - trace(P) > cov_trace_threshold
          - |innovation| > innovation_threshold (if innovation provided)
        """
        # convert to host numpy for robust checks
        try:
            mu_np = np.array(mu)
            P_np = np.array(P)
        except Exception:
            # if conversion fails, perform a safe reset
            return self.reset_to_identity(), True

        # NaN check
        if np.any(np.isnan(mu_np)) or np.any(np.isnan(P_np)):
            mu0, P0 = self.reset_to_identity()
            return (mu0, P0, True)

        # trace check
        tr = float(np.trace(P_np))
        if tr > self.cov_trace_threshold:
            mu0, P0 = self.reset_to_identity()
            return (mu0, P0, True)

        # innovation check
        if innovation is not None:
            try:
                innov_val = float(innovation)
                if abs(innov_val) > self.innovation_threshold:
                    mu0, P0 = self.reset_to_identity()
                    return (mu0, P0, True)
            except Exception:
                # if cannot convert, skip innovation check
                pass

        return (jnp.array(mu), jnp.array(P), False)
    
    def step(self, mu, P, Q, R, x_current, q_ref, cost_map, y_meas=0.0):
        """
        Full UKF prediction + update step
        
        Args:
            mu: current control sequence estimate
            P: current covariance
            Q: process noise
            R: measurement noise
            x_current: current observed system state
            q_ref: reference state
            cost_map: collision checker
            y_meas: measurement (default 0 for Option B)
        Returns:
            mu_new, P_new, innovation
        """
        # Predict
        mu_pred, P_pred, sigma_pred = self.predict(mu, P, Q)

        # Update
        mu_new, P_new, innov = self.update(
            mu_pred, P_pred, sigma_pred,
            y_meas, R, x_current, q_ref, cost_map
        )

        # Automatic reset checks for numerical stability
        if self.auto_reset:
            mu_new, P_new, was_reset = self.check_and_reset(mu_new, P_new, innov)
            if was_reset:
                # regenerate sigma_pred from reset state for consistency
                try:
                    sigma_pred = self.generate_sigma_points(mu_new, P_new)
                except Exception:
                    # fallback to zeros if generation fails
                    sigma_pred = jnp.zeros((self.n_sigma, self.n_theta))

        # Clip controls to bounds
        theta_new = mu_new.reshape((-1, self.n_u))
        theta_new = jnp.clip(theta_new, 
                            self.system.control_bounds[0],
                            self.system.control_bounds[1])
        mu_new = theta_new.ravel()
        print(f"x_current: {x_current}, q_ref: {q_ref}, mu_new: {mu_new}")  # Debug print

        # Return sigma points as well so callers can record them
        return mu_new, P_new, innov, sigma_pred, (bool(was_reset) if self.auto_reset else False)

def do_ukf(params):
    dt = params['dt']
    Nt = params['Nt']
    start = params['start']
    q_ref = params['q_ref']
    obs = params['obs']
    nlmodel = params['nlmodel']
    T = params['T']
    sigma0 = params['sigma0']
    Q = params['Q']
    QT = params['QT']
    
    # Setup collision checker
    resolution = 0.5
    origin = np.array([-40, -10])
    wh = np.array([50/resolution, 20/resolution], dtype=np.int32)
    boundary = [[origin[0], origin[0]+wh[0]*resolution], 
                [origin[1], origin[1]+wh[1]*resolution]]
    

    
    grid = OccupGrid(boundary, resolution)
    grid.find_occupancy_grid(obs, buffer=0.05)
    
    collision_checker = CollisionChecker(
        jnp.array(grid.occup_grid), origin, resolution, wh, occup_value=100
    )
    
    # Create MPPI planner (for cost evaluation only)
    mppi_planner = MPPI_Planner_Occup(
        sigma=sigma0, Q=Q, QT=QT, R=params['R'],
        temperature=params['temperature'],
        system=nlmodel, num_anci=0, 
        n_samples=0,  # not sampling
        N=Nt, tolerance=0.4, occup_value=100
    )
    
    
    # Create UKF controller
    ukf = UKF_Controller(nlmodel, mppi_planner)
    
    # Initialize
    U_init = np.kron(np.ones((1, Nt)), [0.0, 1.0]).ravel()
    mu = jnp.array(U_init)
    P = sigma0 # Control uncertainty only
    
    # Process and measurement noise
    Q_process = sigma0*100  # Control drift
    R_meas = 10.0 # Measurement noise variance
    
    # Simulate
    states = [start]
    costs = []
    sigma_sequences = []
    x_current = start.copy()
    
    cov_norms = []
    cov_traces = []
    innovations = []
    reset_flags = []

    for t in tqdm(np.arange(0, T, dt)):
        # UKF step - note x_current is observed, not estimated!
        mu, P, innovation, sigma_points, was_reset = ukf.step(
            mu, P, Q_process, R_meas, x_current, q_ref, collision_checker, y_meas=0.0
        )

        # record diagnostics
        try:
            P_np = np.array(P)
            cov_norm = float(np.linalg.norm(P_np, ord='fro'))
            cov_trace = float(np.trace(P_np))
        except Exception:
            cov_norm = float('nan')
            cov_trace = float('nan')
        cov_norms.append(cov_norm)
        cov_traces.append(cov_trace)
        innovations.append(float(innovation))
        reset_flags.append(bool(was_reset))
        
        # Extract first control
        u_opt = mu[:ukf.n_u]
        
        # Simulate system
        x_current = nlmodel.dynamics_jax(x_current, u_opt,
                                     dt=dt, params=nlmodel.nominal_params)
        states.append(x_current)
        # convert sigma points to state trajectories and store them
        # sigma_points: (n_sigma, n_theta) where n_theta = Nt * n_u
        sigma_np = np.array(sigma_points)
        n_sigma = sigma_np.shape[0]
        state_dim = x_current.shape[0]
        # trajectories shape: (n_sigma, Nt+1, state_dim)
        trajs = np.zeros((n_sigma, Nt+1, state_dim))
        for si in range(n_sigma):
            state = x_current.copy()
            trajs[si, 0, :] = state
            u_seq = sigma_np[si].reshape((-1, 2))
            for tt in range(Nt):
                # simulate one step using the system dynamics (jax version)
                state = nlmodel.dynamics_jax(state, u_seq[tt], dt=dt, params=nlmodel.nominal_params)
                trajs[si, tt+1, :] = state
        sigma_sequences.append(trajs)
        
        # Evaluate cost
        outputs = mppi_planner.eval_U_seq(
            mu.reshape((-1, 2)), mu, x_current, q_ref, collision_checker
        )
        cost = outputs[0]
        costs.append(float(cost))
        
        # Check if reached goal
        if np.linalg.norm(x_current[:2] - q_ref[:2]) < 0.5:
            print(f"Reached goal at t={t:.2f}")
            break

    # Return simulation results and diagnostics (plotting moved to caller)
    return states, costs, sigma_sequences, cov_norms, cov_traces, innovations, reset_flags, ukf.cov_trace_threshold, ukf.innovation_threshold



def do_ukf_with_pcrb(params):
    dt = params['dt']
    Nt = params['Nt']
    start = params['start']
    q_ref = params['q_ref']
    obs = params['obs']
    nlmodel = params['nlmodel']
    T = params['T']
    sigma0 = params['sigma0']
    Q = params['Q']
    QT = params['QT']
    
    # Setup collision checker
    resolution = 0.5
    origin = np.array([-40, -10])
    wh = np.array([50/resolution, 20/resolution], dtype=np.int32)
    boundary = [[origin[0], origin[0]+wh[0]*resolution], 
                [origin[1], origin[1]+wh[1]*resolution]]
    
    
    grid = OccupGrid(boundary, resolution)
    grid.find_occupancy_grid(obs, buffer=0.05)
    
    collision_checker = CollisionChecker(
        jnp.array(grid.occup_grid), origin, resolution, wh, occup_value=100
    )
    
    # Create MPPI planner (for cost evaluation only)
    mppi_planner = MPPI_Planner_Occup(
        sigma=sigma0, Q=Q, QT=QT, R=params['R'],
        temperature=params['temperature'],
        system=nlmodel, num_anci=0, 
        n_samples=0,  # not sampling
        N=Nt, tolerance=0.4, occup_value=100
    )
    
    
    # Create UKF controller
    ukf = UKF_Controller(nlmodel, mppi_planner)
    
    # Initialize
    U_init = np.kron(np.ones((1, Nt)), [0.0, 1.0]).ravel()
    mu = jnp.array(U_init)
    P = sigma0/100 # Control uncertainty only
    
    # Process and measurement noise
    # Q_process = sigma0*100  # Control drift
    Q_process = sigma0  # Control drift
    R_meas = 10.0 # Measurement noise variance
    
    # ============ PCRB INITIALIZATION ============
    nu = 2  # Control dimension
    dim = Nt * nu
    sigma_r = np.sqrt(R_meas)
    n_pcrb_samples = 1000
    epsilon = 1e-4
    
    # Create PCRB controller instance
    pcrb_controller = PCRBController(
        Nt=Nt,
        nu=nu,
        # Q=np.eye(dim) * Q_process,
        Q=np.eye(dim) *10000,
        sigma_r=sigma_r,
        n_samples=n_pcrb_samples
    )
    
    # Initialize Fisher Information Matrix
    J_k = jnp.eye(dim) * 1e-6
    
    pcrb_history = []
    J_history = []
    gradient_norms = []
    # ============================================
    
    
    # Simulate
    states = [start]
    costs = []
    sigma_sequences = []
    x_current = start.copy()
    
    cov_norms = []
    cov_traces = []
    innovations = []
    reset_flags = []
    mus = [] 

    for t in tqdm(np.arange(0, T, dt)):
        # UKF step - note x_current is observed, not estimated!
        mu, P, innovation, sigma_points, was_reset = ukf.step(
            mu, P, Q_process, R_meas, x_current, q_ref, collision_checker, y_meas=0.0
        )
        
        # ============ PCRB UPDATE ============
        # Complete PCRB step using PCRBController
        J_k, pcrb_bound, grad_norms = pcrb_controller.pcrb_step(
            J_k=J_k,
            mu=mu,
            P=jnp.eye(Nt * 2) * 1e-6,
            x_current=x_current,
            q_ref=q_ref,
            mppi_planner=mppi_planner,
            collision_checker=collision_checker,
            epsilon=epsilon
        )
            
        # Store
        pcrb_history.append(pcrb_bound)
        J_history.append(np.array(J_k))
        gradient_norms.append(np.mean(grad_norms))
        # ============================================ 
        try:
            P_np = np.array(P)
            cov_norm = float(np.linalg.norm(P_np, ord='fro'))
            cov_trace = float(np.trace(P_np))
        except Exception:
            cov_norm = float('nan')
            cov_trace = float('nan')
        cov_norms.append(cov_norm)
        cov_traces.append(cov_trace)
        innovations.append(float(innovation))
        reset_flags.append(bool(was_reset))
        
        # Extract first control
        u_opt = mu[:ukf.n_u]
        mus.append(mu)
        
        # Simulate system
        x_current = nlmodel.dynamics_jax(x_current, u_opt,
                                     dt=dt, params=nlmodel.nominal_params)
        states.append(x_current)
        # convert sigma points to state trajectories and store them
        # sigma_points: (n_sigma, n_theta) where n_theta = Nt * n_u
        sigma_np = np.array(sigma_points)
        n_sigma = sigma_np.shape[0]
        state_dim = x_current.shape[0]
        # trajectories shape: (n_sigma, Nt+1, state_dim)
        trajs = np.zeros((n_sigma, Nt+1, state_dim))
        for si in range(n_sigma):
            state = x_current.copy()
            trajs[si, 0, :] = state
            u_seq = sigma_np[si].reshape((-1, 2))
            for tt in range(Nt):
                # simulate one step using the system dynamics (jax version)
                state = nlmodel.dynamics_jax(state, u_seq[tt], dt=dt, params=nlmodel.nominal_params)
                trajs[si, tt+1, :] = state
        sigma_sequences.append(trajs)
        
        # Evaluate cost
        outputs = mppi_planner.eval_U_seq(
            mu.reshape((-1, 2)), mu, x_current, q_ref, collision_checker
        )
        cost = outputs[0]
        costs.append(float(cost))
        
        # Check if reached goal
        if np.linalg.norm(x_current[:2] - q_ref[:2]) < 0.5:
            print(f"Reached goal at t={t:.2f}")
            break
    # Return simulation results with PCRB data
    return (states, costs, sigma_sequences, cov_norms, cov_traces, innovations, 
            reset_flags, ukf.cov_trace_threshold, ukf.innovation_threshold,
            pcrb_history, J_history, gradient_norms, mus)
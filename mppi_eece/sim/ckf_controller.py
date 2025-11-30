import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple


class CKF_Controller:
    """
    Cubature Kalman Filter (CKF) for control-sequence estimation (controls-only formulation).

    Uses 2n cubature points: mu +/- sqrt(n) * columns_of_cholesky(P)
    Weights are uniform 1/(2n).
    """

    def __init__(self, system, mppi_planner, auto_reset: bool = True,
                 cov_trace_threshold: float = 1e8,
                 innovation_threshold: float = 1e12,
                 reset_P_scale: float = 1.0):
        self.system = system
        self.planner = mppi_planner
        self.N = mppi_planner.N
        self.n_u = getattr(mppi_planner, "n_u", getattr(system, "n_u", 2))
        self.n = self.n_theta = self.N * self.n_u

        # cubature points
        self.n_pts = 2 * self.n
        self.W = 1.0 / (2.0 * self.n)  # uniform weight

        # reset options
        self.auto_reset = bool(auto_reset)
        self.cov_trace_threshold = float(cov_trace_threshold)
        self.innovation_threshold = float(innovation_threshold)
        self.reset_P_scale = float(reset_P_scale)

    def shift_controls(self, theta: jnp.ndarray) -> jnp.ndarray:
        theta_reshaped = theta.reshape((-1, self.n_u))
        theta_shifted = jnp.roll(theta_reshaped, -1, axis=0)
        theta_shifted = theta_shifted.at[-1].set(theta_reshaped[-1])
        return theta_shifted.ravel()

    def generate_cubature_points(self, mu: jnp.ndarray, P: jnp.ndarray) -> jnp.ndarray:
        n = self.n
        P_reg = P + jnp.eye(n) * 1e-9
        try:
            L = jnp.linalg.cholesky(P_reg)
        except Exception:
            eigvals, eigvecs = jnp.linalg.eigh(P_reg)
            eigvals = jnp.maximum(eigvals, 1e-9)
            L = eigvecs @ jnp.diag(jnp.sqrt(eigvals))

        scaled = jnp.sqrt(n) * L
        pts = jnp.zeros((2 * n, n))
        for i in range(n):
            col = scaled[:, i]
            pts = pts.at[i].set(mu + col)
            pts = pts.at[n + i].set(mu - col)
        return pts

    def measurement_function(self, theta, x_current, q_ref, cost_map) -> float:
        # ensure numpy types for planner

        u_seq = theta.reshape((-1, self.n_u))
        outputs = self.planner.eval_U_seq(u_seq, theta, np.array(x_current), np.array(q_ref), cost_map)
        cost = outputs[0]
        return -cost

    def predict(self, mu: jnp.ndarray, P: jnp.ndarray, Q: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        sigma = self.generate_cubature_points(mu, P)
        sigma_pred = jax.vmap(self.shift_controls)(sigma)
        mu_pred = jnp.sum(self.W * sigma_pred, axis=0)
        diff = sigma_pred - mu_pred
        P_pred = jnp.sum(self.W * (diff[:, :, None] @ diff[:, None, :]), axis=0) + Q
        P_pred = (P_pred + P_pred.T) / 2.0
        P_pred = P_pred + jnp.eye(self.n) * 1e-9
        return mu_pred, P_pred, sigma_pred

    def update(self, mu_pred: jnp.ndarray, P_pred: jnp.ndarray, sigma_pred: jnp.ndarray,
               y_meas: float, R: float, x_current, q_ref, cost_map) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        # evaluate measurements on host
        sigma_np = np.array(sigma_pred)
        meas_sigma = jax.vmap(
            lambda theta: self.measurement_function(theta, x_current, q_ref, cost_map)
        )(sigma_pred)
        y_pred = jnp.sum(self.W * meas_sigma)
        meas_diff = meas_sigma - y_pred
        S = jnp.sum(self.W * meas_diff**2) + R
        state_diff = sigma_pred - mu_pred
        P_xy = jnp.sum(self.W * state_diff * meas_diff[:, None], axis=0)
        K = P_xy / S
        innovation = y_meas - y_pred
        mu_upd = mu_pred + K * innovation
        P_upd = P_pred - jnp.outer(K, K) * S
        P_upd = (P_upd + P_upd.T) / 2.0
        P_upd = P_upd + jnp.eye(self.n) * 1e-9
        return mu_upd, P_upd, float(innovation)

    # reset helpers (simple)
    def reset_to_identity(self):
        mu0 = jnp.zeros(self.n)
        P0 = jnp.eye(self.n) * self.reset_P_scale
        return mu0, P0

    def check_and_reset(self, mu, P, innovation=None):
        try:
            P_np = np.array(P)
            mu_np = np.array(mu)
        except Exception:
            return self.reset_to_identity(), True
        if np.any(np.isnan(P_np)) or np.any(np.isnan(mu_np)):
            return self.reset_to_identity(), True
        tr = float(np.trace(P_np))
        if tr > self.cov_trace_threshold:
            return self.reset_to_identity(), True
        if innovation is not None:
            try:
                if abs(float(innovation)) > self.innovation_threshold:
                    return self.reset_to_identity(), True
            except Exception:
                pass
        return (jnp.array(mu), jnp.array(P)), False

    def step(self, mu, P, Q, R, x_current, q_ref, cost_map, y_meas=0.0):
        mu_pred, P_pred, sigma_pred = self.predict(mu, P, Q)
        mu_new, P_new, innovation = self.update(mu_pred, P_pred, sigma_pred, y_meas, R, x_current, q_ref, cost_map)
        was_reset = False
        if self.auto_reset:
            (mu_new, P_new), was_reset = self.check_and_reset(mu_new, P_new, innovation)
            if was_reset:
                try:
                    sigma_pred = self.generate_cubature_points(mu_new, P_new)
                except Exception:
                    sigma_pred = jnp.zeros((self.n_pts, self.n))
        # clip controls
        theta_new = mu_new.reshape((-1, self.n_u))
        theta_new = jnp.clip(theta_new, self.system.control_bounds[0], self.system.control_bounds[1])
        mu_new = theta_new.ravel()
        print(f"x_current: {x_current}, q_ref: {q_ref}, mu_new: {mu_new}")  # Debug print

        return mu_new, P_new, float(innovation), sigma_pred, bool(was_reset)


def do_ckf(params):
    # mostly a copy of do_ukf but using CKF_Controller
    dt = params['dt']
    Nt = params['Nt']
    start = params['start']
    q_ref = params['q_ref']
    obs = params['obs']
    nlmodel = params['nlmodel']
    T = params['T']
    sigma0 = params['sigma0']

    from mppi_eece.sim.grid import OccupGrid
    from mppi_eece.jax_mppi.mppi_planners import MPPI_Planner_Occup
    from mppi_eece.jax_mppi.collision_checker import CollisionChecker

    resolution = 0.5
    origin = np.array([-40, -10])
    wh = np.array([50 / resolution, 20 / resolution], dtype=np.int32)
    boundary = [[origin[0], origin[0] + wh[0] * resolution], [origin[1], origin[1] + wh[1] * resolution]]
    grid = OccupGrid(boundary, resolution)
    grid.find_occupancy_grid(obs, buffer=0.05)
    collision_checker = CollisionChecker(jnp.array(grid.occup_grid), origin, resolution, wh, occup_value=100)

    mppi_planner = MPPI_Planner_Occup(
        sigma=sigma0, Q=params.get('Q'), QT=params.get('QT'), R=params.get('R'),
        temperature=params.get('temperature'), system=nlmodel, num_anci=0, n_samples=0, N=Nt, tolerance=0.4, occup_value=100
    )

    ckf = CKF_Controller(nlmodel, mppi_planner)

    # init
    U_init = np.kron(np.ones((1, Nt)), [0.0, 1.0]).ravel()
    mu = jnp.array(U_init)
    P = sigma0 * 100
    Q_process = sigma0 * 3
    R_meas = 1.0

    states = [start]
    costs = []
    sigma_sequences = []
    x_current = start.copy()

    cov_norms = []
    cov_traces = []
    innovations = []
    reset_flags = []

    from tqdm import tqdm
    for t in tqdm(np.arange(0, T, dt)):
        mu, P, innovation, sigma_points, was_reset = ckf.step(mu, P, Q_process, R_meas, x_current, q_ref, collision_checker, y_meas=0.0)
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

        u_opt = mu[:ckf.n_u]
        x_current = nlmodel.dynamics_jax(x_current, u_opt, dt=dt, params=nlmodel.nominal_params)
        states.append(x_current)

        sigma_np = np.array(sigma_points)
        n_sigma = sigma_np.shape[0]
        state_dim = x_current.shape[0]
        trajs = np.zeros((n_sigma, Nt + 1, state_dim))
        for si in range(n_sigma):
            state = x_current.copy()
            trajs[si, 0, :] = state
            u_seq = sigma_np[si].reshape((-1, ckf.n_u))
            for tt in range(Nt):
                state = nlmodel.dynamics_jax(state, u_seq[tt], dt=dt, params=nlmodel.nominal_params)
                trajs[si, tt + 1, :] = state
        sigma_sequences.append(trajs)

        outputs = mppi_planner.eval_U_seq(mu.reshape((-1, ckf.n_u)), mu, x_current, q_ref, collision_checker)
        cost = outputs[0]
        costs.append(float(cost))

        if np.linalg.norm(x_current[:2] - q_ref[:2]) < 0.5:
            break

    return states, costs, sigma_sequences, cov_norms, cov_traces, innovations, reset_flags, ckf.cov_trace_threshold, ckf.innovation_threshold

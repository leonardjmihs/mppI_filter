"""UKF implementation that uses the `filterpy` library.

This provides a drop-in alternative controller that exposes a similar
`step(mu, P, Q, R, x_current, q_ref, cost_map)` API and a `do_ukf_filterpy(params)`
runner that mirrors the existing `do_ukf` signature/diagnostics.

Notes:
- `filterpy` must be installed in the runtime environment for this file to import.
- The measurement function calls the (non-JAX) planner on the host.
"""
from typing import Tuple
import numpy as np

try:
    from filterpy.kalman import UnscentedKalmanFilter as UKF_py
    from filterpy.kalman import MerweScaledSigmaPoints
except Exception:  # pragma: no cover - environment-specific
    UKF_py = None
    MerweScaledSigmaPoints = None


class FilterPyUKFController:
    """UKF wrapper powered by filterpy.

    The controller is controls-only: the "state" is the flattened control
    sequence theta of length n_theta = N * n_u.
    """

    def __init__(self, system, mppi_planner, alpha=1e-3, beta=2.0, kappa=0.0,
                 auto_reset: bool = False,
                 cov_trace_threshold: float = 1e8,
                 innovation_threshold: float = 1e12,
                 reset_P_scale: float = 1.0):
        if UKF_py is None or MerweScaledSigmaPoints is None:
            raise ImportError("filterpy is required for FilterPyUKFController")

        self.system = system
        self.planner = mppi_planner
        self.N = int(mppi_planner.N)
        self.n_u = getattr(mppi_planner, "n_u", getattr(system, "n_u", 2))
        self.n_theta = self.N * self.n_u

        # sigma point generator for filterpy (Merwe scaled sigma points)
        self.points = MerweScaledSigmaPoints(n=self.n_theta, alpha=alpha, beta=beta, kappa=kappa)

        # placeholder UKF instance (created on first use)
        self.ukf = None

        # reset / diagnostic thresholds
        self.auto_reset = bool(auto_reset)
        self.cov_trace_threshold = float(cov_trace_threshold)
        self.innovation_threshold = float(innovation_threshold)
        self.reset_P_scale = float(reset_P_scale)

    # --- dynamics & measurement used by the filterpy UKF ---
    def fx(self, x, dt):
        # shift operator: move controls forward and duplicate last control
        theta = x.reshape((-1, self.n_u)).copy()
        theta = np.roll(theta, -1, axis=0)
        theta[-1, :] = x.reshape((-1, self.n_u))[-1, :]
        return theta.ravel()

    def measurement_function(self, theta, x_current, q_ref, cost_map) -> float:
        # convert to numpy and call planner (planner is non-jax)
        theta_np = np.array(theta)
        u_seq = theta_np.reshape((-1, self.n_u))
        outputs = self.planner.eval_U_seq(u_seq, theta_np, np.array(x_current), np.array(q_ref), cost_map)
        cost = outputs[0]
        return float(-cost)

    def _ensure_ukf(self):
        if self.ukf is None:
            # create UKF instance
            self.ukf = UKF_py(dim_x=self.n_theta, dim_z=1, fx=self.fx, hx=lambda x: 0.0,
                              dt=1.0, points=self.points)

    def step(self, mu: np.ndarray, P: np.ndarray, Q: np.ndarray, R: float,
             x_current, q_ref, cost_map, y_meas: float = 0.0) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Perform one UKF predict+update using filterpy.

        Returns: (mu_new, P_new, innovation, sigma_points, was_reset)
        """
        self._ensure_ukf()

        # set filter state
        self.ukf.x = np.array(mu).astype(float)
        self.ukf.P = np.array(P).astype(float)
        # set process & measurement noise
        self.ukf.Q = np.array(Q).astype(float)
        # filterpy expects scalar R as 1x1 or scalar
        self.ukf.R = np.array([[float(R)]]) if np.ndim(R) == 0 else np.array(R)

        # predict step (uses fx)
        self.ukf.predict()

        # predicted measurement at predicted mean
        x_prior = np.array(self.ukf.x).astype(float)
        y_pred = self.measurement_function(x_prior, x_current, q_ref, cost_map)

        # perform update with a dynamic measurement function (hx) via passing hx to update
        def hx_local(x):
            return np.array([self.measurement_function(x, x_current, q_ref, cost_map)], dtype=float)

        z = np.array([float(y_meas)])
        self.ukf.update(z, hx=hx_local)

        mu_new = np.array(self.ukf.x).astype(float)
        P_new = np.array(self.ukf.P).astype(float)

        # get sigma points for diagnostics from the points object (2n+1)
        try:
            sigma_points = np.array(self.points.sigma_points(mu_new, P_new))
        except Exception:
            # fallback: construct trivial sigma points
            sigma_points = np.tile(mu_new[None, :], (2 * self.n_theta + 1, 1))

        innovation = float(y_meas - y_pred)

        # optional reset checks
        was_reset = False
        if self.auto_reset:
            try:
                tr = float(np.trace(P_new))
                if np.isnan(tr) or tr > self.cov_trace_threshold or abs(innovation) > self.innovation_threshold:
                    mu0 = np.zeros(self.n_theta)
                    P0 = np.eye(self.n_theta) * self.reset_P_scale
                    mu_new = mu0
                    P_new = P0
                    was_reset = True
            except Exception:
                mu_new = np.zeros(self.n_theta)
                P_new = np.eye(self.n_theta) * self.reset_P_scale
                was_reset = True

        return mu_new, P_new, innovation, sigma_points, bool(was_reset)


def do_ukf_filterpy(params):
    """Runner that mirrors `do_ukf` but uses FilterPyUKFController."""
    # local imports to avoid hard dependency at module import time
    import jax.numpy as jnp
    from mppi_eece.sim.grid import OccupGrid
    from mppi_eece.jax_mppi.mppi_planners import MPPI_Planner_Occup
    from mppi_eece.jax_mppi.collision_checker import CollisionChecker

    dt = params['dt']
    Nt = params['Nt']
    start = params['start']
    q_ref = params['q_ref']
    obs = params['obs']
    nlmodel = params['nlmodel']
    T = params['T']
    sigma0 = params['sigma0']

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

    ukf = FilterPyUKFController(nlmodel, mppi_planner)

    # initialize
    U_init = np.kron(np.ones((1, Nt)), [0.0, 1.0]).ravel()
    mu = np.array(U_init, dtype=float)
    P = np.eye(ukf.n_theta) * (sigma0 * 1000)

    Q_process = np.eye(ukf.n_theta) * (sigma0 * 3)
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
        mu, P, innovation, sigma_points, was_reset = ukf.step(mu, P, Q_process, R_meas, x_current, q_ref, collision_checker, y_meas=0.0)

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

        # apply first control and simulate
        u_opt = mu[:ukf.n_u]
        x_current = nlmodel.dynamics_jax(x_current, u_opt, dt=dt, params=nlmodel.nominal_params)
        states.append(x_current)

        # convert sigma points into state trajectories
        sigma_np = np.array(sigma_points)
        n_sigma = sigma_np.shape[0]
        state_dim = x_current.shape[0]
        trajs = np.zeros((n_sigma, Nt + 1, state_dim))
        for si in range(n_sigma):
            state = x_current.copy()
            trajs[si, 0, :] = state
            u_seq = sigma_np[si].reshape((-1, ukf.n_u))
            for tt in range(Nt):
                state = nlmodel.dynamics_jax(state, u_seq[tt], dt=dt, params=nlmodel.nominal_params)
                trajs[si, tt + 1, :] = state
        sigma_sequences.append(trajs)

        outputs = mppi_planner.eval_U_seq(mu.reshape((-1, ukf.n_u)), mu, x_current, q_ref, collision_checker)
        cost = outputs[0]
        costs.append(float(cost))

        if np.linalg.norm(x_current[:2] - q_ref[:2]) < 0.5:
            break

    return states, costs, sigma_sequences, cov_norms, cov_traces, innovations, reset_flags, ukf.cov_trace_threshold, ukf.innovation_threshold

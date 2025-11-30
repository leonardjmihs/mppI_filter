import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple


class GaussianSumFilterController:
    """
    Gaussian Sum Filter (GSF) for estimating control sequences (controls-only formulation).

    This is inspired by the UKF-based controller in `UKF_controller.py` but maintains a
    mixture of Gaussian control-sequence estimates. Each component is represented by
    (weight, mu, P). Prediction uses the same shift dynamics as the UKF. Measurement
    update uses a UKF-style measurement update per component and the component weight
    is updated by the measurement likelihood.

    API
    ---
    __init__(system, mppi_planner, n_components=5, alpha=1e-3, beta=2, kappa=0)
    step(components, x_current, q_ref, cost_map, y_meas=0.0)
        - components: list of tuples (weight, mu, P)
        - returns: (mu_est, new_components, innovations, sigma_points_list)

    Notes
    -----
    - `system` should provide `.control_bounds` and `.dynamics_jax` like the Unicycle
      model used elsewhere in the repo.
    - `mppi_planner` should provide an `eval_U_seq(u_seq, original_u, x_current, q_ref, cost_map)`
      method that returns (cost, ...) similar to MPPI planner used in the UKF.
    """

    def __init__(self, system, mppi_planner, n_components: int = 5, alpha=1e-3, beta=2, kappa=0):
        self.system = system
        self.planner = mppi_planner
        self.N = int(mppi_planner.N)
        self.n_u = 2
        self.n_theta = self.N * self.n_u
        self.n_components = max(1, int(n_components))

        # GSF/UKF tuning
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (self.n_theta + kappa) - self.n_theta

        # weights for sigma points (these will be used per component)
        self.n_sigma = 2 * self.n_theta + 1
        self._init_sigma_weights()

    def _init_sigma_weights(self):
        n = self.n_theta
        lam = self.lambda_
        Wm = jnp.zeros(self.n_sigma)
        Wc = jnp.zeros(self.n_sigma)
        Wm = Wm.at[0].set(lam / (n + lam))
        Wc = Wc.at[0].set(lam / (n + lam) + (1 - self.alpha**2 + self.beta))
        for i in range(1, self.n_sigma):
            w = 1.0 / (2.0 * (n + lam))
            Wm = Wm.at[i].set(w)
            Wc = Wc.at[i].set(w)
        self.Wm = Wm
        self.Wc = Wc

    def shift_controls(self, theta: jnp.ndarray) -> jnp.ndarray:
        # theta shape: (n_theta,)
        theta_reshaped = theta.reshape((-1, self.n_u))
        theta_shifted = jnp.roll(theta_reshaped, -1, axis=0)
        theta_shifted = theta_shifted.at[-1].set(theta_reshaped[-1])
        return theta_shifted.ravel()

    def generate_sigma_points(self, mu: jnp.ndarray, P: jnp.ndarray) -> jnp.ndarray:
        n = self.n_theta
        P_reg = P + jnp.eye(n) * 1e-9
        try:
            L = jnp.linalg.cholesky(P_reg)
        except Exception:
            eigvals, eigvecs = jnp.linalg.eigh(P_reg)
            eigvals = jnp.maximum(eigvals, 1e-9)
            L = eigvecs @ jnp.diag(jnp.sqrt(eigvals))
        L_scaled = jnp.sqrt(n + self.lambda_) * L
        sigma_points = jnp.zeros((self.n_sigma, n))
        sigma_points = sigma_points.at[0].set(mu)
        for i in range(n):
            sigma_points = sigma_points.at[i+1].set(mu + L_scaled[:, i])
            sigma_points = sigma_points.at[n+i+1].set(mu - L_scaled[:, i])
        return sigma_points

    def measurement_function(self, theta: jnp.ndarray, x_current: np.ndarray, q_ref: np.ndarray, cost_map) -> float:
        # theta: flattened control sequence (n_theta,)
        u_seq = np.array(theta).reshape((-1, self.n_u))
        outputs = self.planner.eval_U_seq(u_seq, theta, x_current, q_ref, cost_map)
        cost = float(outputs[0])
        return -cost

    def predict_component(self, mu: jnp.ndarray, P: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # generate sigma points
        sigma = self.generate_sigma_points(mu, P)
        # propagate through shift operator
        sigma_pred = jax.vmap(self.shift_controls)(sigma)
        # predicted mean
        mu_pred = jnp.sum(self.Wm[:, None] * sigma_pred, axis=0)
        # predicted covariance
        diff = sigma_pred - mu_pred
        P_pred = jnp.sum(self.Wc[:, None, None] * (diff[:, :, None] @ diff[:, None, :]), axis=0)
        return mu_pred, P_pred, sigma_pred

    def update_component(self, mu_pred: jnp.ndarray, P_pred: jnp.ndarray, sigma_pred: jnp.ndarray,
                         x_current: np.ndarray, q_ref: np.ndarray, cost_map, y_meas: float, R_meas: float):
        # propagate sigma points through measurement
        meas_sigma = jax.vmap(lambda th: self.measurement_function(th, x_current, q_ref, cost_map))(sigma_pred)
        # clip
        meas_sigma = jnp.clip(meas_sigma, -1e10, 0)
        # predicted measurement mean
        y_pred = jnp.sum(self.Wm * meas_sigma)
        # innovation covariance (scalar)
        meas_diff = meas_sigma - y_pred
        S = jnp.sum(self.Wc * meas_diff**2) + R_meas
        # cross-covariance
        state_diff = sigma_pred - mu_pred
        P_xy = jnp.sum(self.Wc[:, None] * state_diff * meas_diff[:, None], axis=0)
        # Kalman gain
        K = P_xy / S
        # innovation
        innovation = y_meas - y_pred
        # update mean & cov
        mu_upd = mu_pred + K * innovation
        P_upd = P_pred - jnp.outer(K, K) * S
        # likelihood for weight update (Gaussian)
        # Avoid zero S
        S_val = float(S) if float(S) > 1e-12 else 1e-12
        likelihood = float((1.0 / jnp.sqrt(2.0 * jnp.pi * S_val)) * jnp.exp(-0.5 * (innovation**2) / S_val))
        return mu_upd, P_upd, innovation, likelihood, meas_sigma

    def step(self, components: List[Tuple[float, jnp.ndarray, jnp.ndarray]], x_current: np.ndarray,
             q_ref: np.ndarray, cost_map, y_meas: float = 0.0, R_meas: float = 1.0):
        """
        Perform one GSF step over all components.

        components: list of (weight, mu, P)
        Returns: (mu_est, new_components, innovations, sigma_pred_list)
        """
        new_components = []
        sigma_pred_list = []
        innovations = []
        weights = []

        # Predict+update per component
        for (w, mu, P) in components:
            mu_pred, P_pred, sigma_pred = self.predict_component(mu, P)
            mu_upd, P_upd, innov, likelihood, meas_sigma = self.update_component(
                mu_pred, P_pred, sigma_pred, x_current, q_ref, cost_map, y_meas, R_meas
            )
            # new weight = prior_weight * likelihood
            new_w = w * (likelihood + 1e-300)
            new_components.append((new_w, mu_upd, P_upd))
            sigma_pred_list.append(sigma_pred)
            innovations.append(float(innov))
            weights.append(new_w)

        # normalize weights
        weights = np.array([float(w) for w in weights])
        if weights.sum() <= 0:
            # fallback to uniform
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights = weights / weights.sum()

        # replace normalized weights into components
        normalized_components = []
        for i, (_, mu_upd, P_upd) in enumerate(new_components):
            normalized_components.append((float(weights[i]), mu_upd, P_upd))

        # compute weighted estimate (mean of mus)
        mus = jnp.stack([comp[1] for comp in normalized_components], axis=0)
        w_js = jnp.array([comp[0] for comp in normalized_components])
        mu_est = jnp.sum(w_js[:, None] * mus, axis=0)

        return mu_est, normalized_components, innovations, sigma_pred_list

    def init_components_random(self, base_mu=None, base_P=None, n_init=None, scale=1.0):
        """
        Helper to initialize components (random perturbations around base_mu)
        Returns list of (weight, mu, P)
        """
        M = n_init if n_init is not None else self.n_components
        base_mu = base_mu if base_mu is not None else jnp.zeros(self.n_theta)
        base_P = base_P if base_P is not None else jnp.eye(self.n_theta) * 1.0
        comps = []
        for i in range(M):
            mu_i = base_mu + scale * jnp.array(np.random.randn(self.n_theta))
            P_i = base_P * (1.0 + 0.1 * i)
            comps.append((1.0 / M, mu_i, P_i))
        return comps


# Small example usage (not executed at import):
# gsf = GaussianSumFilterController(system, mppi_planner, n_components=5)
# comps = gsf.init_components_random()
# mu_est, comps, innovations, sigma_pred_list = gsf.step(comps, x_current, q_ref, cost_map)

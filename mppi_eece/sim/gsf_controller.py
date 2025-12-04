# gsf_controller_expanded.py
"""
Expanded Gaussian-Sum Filter (GSF) controller that grows components
by using mixtures of process and measurement noises.

Call `do_gsf(params)` with a params dict similar to your do_ukf/do_gsf expectations.

Key params (examples):
  - dt, Nt, start, q_ref, obs, nlmodel, T
  - sigma0: initial covariance (float or array)
  - process_noises: list of dicts [{"weight":w, "Q": Q_matrix}, ...]
  - measurement_noises: list of dicts [{"weight":w, "R": scalar_or_matrix}, ...]
  - wmin, gamma, Jmax
  - other optional items referenced in code
"""

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import functools

try:
    from mppi_eece.jax_mppi.mppi_planners import MPPI_Planner_Occup
    from mppi_eece.jax_mppi.collision_checker import CollisionChecker
    from mppi_eece.sim.grid import OccupGrid
except Exception:
    # If not available, set placeholders. User will likely have real modules in their environment.
    MPPI_Planner_Occup = None
    CollisionChecker = None
    OccupGrid = None


class GSF_Controller_Expanded:
    def __init__(self, nlmodel, planner,
                 n_u=2,
                 N=10,
                 Jmax=12,
                 wmin=1e-6,
                 gamma=4.0,
                 ukf_alpha=1e-2, ukf_beta=2.0, ukf_kappa=0.0):
        self.nlmodel = nlmodel
        self.planner = planner
        self.n_u = n_u
        self.N = N
        self.n_theta = self.N * self.n_u

        # mixture management params
        self.wmin = wmin
        self.gamma = gamma
        self.Jmax = Jmax

        # UKF sigma weights
        n = self.n_theta
        self.alpha = ukf_alpha
        self.beta = ukf_beta
        self.kappa = ukf_kappa
        self.lambda_ = self.alpha**2 * (n + self.kappa) - n
        self.n_sigma = 2 * n + 1
        Wm = np.zeros(self.n_sigma)
        Wc = np.zeros(self.n_sigma)
        Wm[0] = self.lambda_ / (n + self.lambda_)
        Wc[0] = self.lambda_ / (n + self.lambda_) + (1 - self.alpha**2 + self.beta)
        for i in range(1, self.n_sigma):
            w = 1.0 / (2.0 * (n + self.lambda_))
            Wm[i] = w
            Wc[i] = w
        self.Wm = Wm
        self.Wc = Wc

    # ---------------- utilities ----------------
    
    def ensure_psd(self, P):
        """Make symmetric and add tiny jitter for numerical stability."""
        P = 0.5 * (P + P.T)
        jitter = 1e-9
        try:
            # try to force via eigh
            vals, vecs = np.linalg.eigh(P)
            vals = np.maximum(vals, jitter)
            return (vecs * vals) @ vecs.T
        except Exception:
            return P + np.eye(P.shape[0]) * jitter

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
        P_reg = P + np.eye(n) * 1e-9
        
        try:
            L = np.linalg.cholesky(P_reg)
        except:
            # Fallback to eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(P_reg)
            eigvals = np.maximum(eigvals, 1e-9)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        L_scaled = np.sqrt(n + self.lambda_) * L

        # Generate sigma points
        sigma_points = np.zeros((self.n_sigma, n))
        # sigma_points = sigma_points.at[0].set(mu)
        # sigma_points[0] = sigma_points.at[0].set(mu)
        sigma_points[0] = mu
        
        for i in range(n):
            # sigma_points = sigma_points.at[i+1].set(mu + L_scaled[:, i])
            # sigma_points = sigma_points.at[n+i+1].set(mu - L_scaled[:, i])
            sigma_points[i+1] = mu + L_scaled[:, i]
            sigma_points[n+i+1] = mu - L_scaled[:, i]

        return sigma_points

    def measurement_function(self, theta, x_current, q_ref, cost_map):
        """Measurement = -cost evaluated by planner for a control sequence theta.

        Use numpy on the host to call planner (non-JAX). Return a Python float.
        """
        u_seq = jnp.array(theta).reshape((-1, self.n_u))
        out = self.planner.eval_U_seq(u_seq, 
                jnp.array(theta), 
                jnp.array(x_current), 
                jnp.array(q_ref), 
                cost_map)
        cost = out[0]
        cost = jnp.clip(cost, 0.0, 1e12)
        return -cost

    @functools.partial(jax.jit, static_argnums=(0,))
    def measure_sigma(self, sigma_pts, x_current, q_ref, cost_map):
        """Evaluate measurement function for all sigma points."""
        meas_sigma = jax.vmap(
            lambda theta: self.measurement_function(theta, x_current, q_ref, cost_map)
        )(sigma_pts)
        return meas_sigma

    # ---------------- predict & update for individual components ----------------
    # def predict_component(self, mu, P, Q):
    #     mu_pred = self.shift_controls(mu)
    #     P_pred = P + Q
    #     P_pred = self.ensure_psd(P_pred)
    #     return mu_pred, P_pred
    def predict_component(self, mu, P, Q):
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

    def update_component_unscented(self, mu_pred, P_pred, sigma_pts, y_meas, R, x_current, q_ref, cost_map):
        """
        Unscented-style scalar measurement update for a component.
        Returns: mu_upd, P_upd, likelihood (scalar), y_pred, S
        """
        # sigma_pts = self.generate_sigma_points(mu_pred, P_pred)
        # Evaluate measurement for each sigma point on host to avoid JAX recompiles
        meas_sigma = self.measure_sigma(sigma_pts, x_current, q_ref, cost_map)
        Wm = np.array(self.Wm, dtype=float)
        Wc = np.array(self.Wc, dtype=float)
        y_pred = float(np.sum(Wm * meas_sigma))
        meandiff = meas_sigma - y_pred
        S = np.sum(self.Wc * (meandiff**2)) + (R if np.ndim(R) == 0 else float(R))
        # compute cross-covariance (n_theta vector)
        state_diff = sigma_pts - mu_pred[None, :]
        P_xy = np.sum((self.Wc[:, None] * state_diff) * meandiff[:, None], axis=0)
        K = P_xy / S
        innovation = y_meas - y_pred
        mu_upd = mu_pred + K * innovation
        P_upd = P_pred - np.outer(K, K) * S
        # P_upd = self.ensure_psd(P_upd)

        P_upd = (P_upd + P_upd.T) / 2.0
        P_upd = P_upd + jnp.eye(self.n_theta) * 1e-9
        # scalar Gaussian likelihood
        denom = np.sqrt(2.0 * np.pi * S)
        expo = -0.5 * ((y_meas - y_pred) ** 2) / S
        likelihood = float(np.exp(expo) / (denom + 1e-30))
        return mu_upd, P_upd, likelihood, float(y_pred), float(S)

    # ---------------- mixture management: prune -> merge -> cap ----------------
    def prune(self, weights, mus, Ps):
        w = np.array(weights, dtype=float)
        keep = np.where(w >= self.wmin)[0]
        if keep.size == 0:
            keep = np.array([int(np.argmax(w))])
        w_n = w[keep]
        w_n = w_n / (np.sum(w_n) + 1e-12)
        mus_n = [mus[i] for i in keep]
        Ps_n = [Ps[i] for i in keep]
        return w_n.tolist(), mus_n, Ps_n

    def merge(self, weights, mus, Ps):
        # Greedy pairwise merging using Mahalanobis distance threshold gamma.
        w = list(weights)
        m = [np.array(mi) for mi in mus]
        P = [np.array(pi) for pi in Ps]
        merged_any = True
        while merged_any and len(w) > 1:
            merged_any = False
            best_pair = None
            best_dist = np.inf
            J = len(w)
            for i in range(J):
                for j in range(i + 1, J):
                    cov_sum = P[i] + P[j]
                    # robust inverse
                    try:
                        inv = np.linalg.inv(cov_sum)
                    except np.linalg.LinAlgError:
                        inv = np.linalg.pinv(cov_sum)
                    diff = m[i] - m[j]
                    d = float(diff.T @ inv @ diff)
                    if d < best_dist:
                        best_dist = d
                        best_pair = (i, j)
            if best_pair is not None and best_dist < self.gamma:
                i, j = best_pair
                wi, wj = w[i], w[j]
                wmerged = wi + wj
                mmerged = (wi * m[i] + wj * m[j]) / (wmerged + 1e-12)
                Pi = P[i]; Pj = P[j]
                term_i = wi * (Pi + np.outer(m[i], m[i]))
                term_j = wj * (Pj + np.outer(m[j], m[j]))
                Pmerged = (term_i + term_j) / (wmerged + 1e-12) - np.outer(mmerged, mmerged)
                # replace i with merged and remove j
                w[i] = wmerged
                m[i] = mmerged
                P[i] = Pmerged
                del w[j]
                del m[j]
                del P[j]
                merged_any = True
            else:
                break
        w = np.array(w)
        if np.sum(w) <= 0:
            w = np.ones_like(w) / float(len(w))
        else:
            w = w / (np.sum(w) + 1e-12)
        return w.tolist(), [mi.tolist() for mi in m], [pi.tolist() for pi in P]

    def cap(self, weights, mus, Ps):
        w = np.array(weights)
        idx = np.argsort(-w)
        keep = idx[: self.Jmax]
        w_new = w[keep]
        w_new = w_new / (np.sum(w_new) + 1e-12)
        mus_new = [mus[i] for i in keep]
        Ps_new = [Ps[i] for i in keep]
        return w_new.tolist(), mus_new, Ps_new

    def manage_mixture(self, weights, mus, Ps):
        w_p, m_p, P_p = self.prune(weights, mus, Ps)
        # w_m, m_m, P_m = self.merge(w_p, m_p, P_p)
        # w_c, m_c, P_c = self.cap(w_m, m_m, P_m)
        w_c, m_c, P_c = self.cap(w_p, m_p, P_p)
        # breakpoint()
        return w_c, m_c, P_c

    def _tree_flatten(self):
        children = (self.wmin, self.gamma, self.alpha, self.beta, self.kappa)
        aux_data = {
            'nlmodel': self.nlmodel, 
            'planner': self.planner,
            'n_u': self.n_u,
            'N': self.N,
            'Jmax': self.Jmax,
        }
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        nl = aux_data.get('nlmodel', None)
        pl = aux_data.get('planner', None)
        n_u = aux_data.get('n_u', 2)
        N = aux_data.get('N', 10)
        J_max = aux_data.get('Jmax', 4)
        # children expected: (wmin, gamma, alpha, beta, kappa)
        # wmin_c, gamma_c, alpha_c, beta_c, kappa_c = children
        wmin, gamma, alpha, beta, kappa = children
        # try:
        #     wmin_c, gamma_c, alpha_c, beta_c, kappa_c = children
        # except Exception:
        #     # fallback: if children missing or malformed, use defaults
        #     wmin_c, gamma_c, alpha_c, beta_c, kappa_c = (1e-6, 4.0, 1e-3, 2.0, 0.0)

        # coerce to python floats to avoid assigning arrays into scalar slots
        return cls(nl, pl, n_u, N, J_max, wmin=wmin, gamma=gamma,
                   ukf_alpha=alpha, ukf_beta=beta, ukf_kappa=kappa)


# ------------------ main driver ------------------
def do_gsf(params):
    """
    params: dict expects (not exhaustive)
      dt, Nt, start, q_ref, obs, nlmodel, T
      sigma0 : initial covariance (array or scalar)
      process_noises : list [{"weight":w, "Q": Q_array}, ...]
      measurement_noises : list [{"weight":w, "R": scalar_or_matrix}, ...]
      wmin, gamma, Jmax
      optionally: planner, grid/collision checker or necessary args to construct them
    Returns:
      states, costs, mixture_records, cov_traces, innovations
    """
    # --- extract core params ---
    dt = params.get("dt", 0.1)
    Nt = params.get("Nt", 10)
    start = np.array(params["start"], dtype=float)
    q_ref = np.array(params["q_ref"], dtype=float)
    obs = params.get("obs", [])
    nlmodel = params["nlmodel"]
    T = params.get("T", 10.0)
    # sigma0 = params.get("sigma0", 1.0)
    sigma0 = params['sigma0']
    sigma0_arr = np.eye(Nt * params.get("n_u", getattr(nlmodel, "n_u", 2))) * float(sigma0) \
        if np.isscalar(sigma0) else np.array(sigma0, dtype=float)

    # default process/measurement mixtures: if not provided, use single Gaussian -> no growth
    process_noises = params.get("process_noises", [{"weight": 1.0, "Q": np.eye(sigma0_arr.shape[0]) * 0.01}])
    measurement_noises = params.get("measurement_noises", [{"weight": 1.0, "R": 1.0}])

    # mixture management params
    wmin = params.get("wmin", 1e-6)
    gamma = params.get("gamma", 4.0)
    Jmax = params.get("Jmax", 4)

    # planner / collision checker: allow user to pass objects; otherwise attempt to construct
    planner = params.get("planner", None)
    cost_map = params.get("cost_map", None)
    if planner is None:
        if MPPI_Planner_Occup is None:
            raise RuntimeError("MPPI_Planner_Occup not found in imports; pass a planner via params['planner'].")
        planner = MPPI_Planner_Occup(
            sigma=sigma0, Q=params.get("Q", None), QT=params.get("QT", None),
            R=params.get("R", None), temperature=params.get("temperature", 1.0),
            system=nlmodel, num_anci=0, n_samples=0, N=Nt, tolerance=0.4
        )
    if cost_map is None:
        # attempt to create basic OccupGrid/collision checker if obs provided
        if OccupGrid is not None and CollisionChecker is not None:
            resolution = params.get("resolution", 0.5)
            origin = np.array(params.get("origin", [-40, -10]))
            wh = np.array(params.get("wh", [50 / resolution, 20 / resolution]), dtype=np.int32)
            boundary = [[origin[0], origin[0] + wh[0] * resolution],
                        [origin[1], origin[1] + wh[1] * resolution]]
            grid = OccupGrid(boundary, resolution)
            grid.find_occupancy_grid(obs, buffer=0.05)
            cost_map = CollisionChecker(np.array(grid.occup_grid), origin, resolution, wh, occup_value=100)
        else:
            cost_map = params.get("cost_map_obj", None)

    # instantiate controller
    n_u = params.get("n_u", getattr(nlmodel, "n_u", 2))
    gsf = GSF_Controller_Expanded(nlmodel, planner, n_u=n_u, N=Nt, wmin=wmin, gamma=gamma, Jmax=Jmax)

    # initial mixture: single component by default (user can pass mixture_init)
    if "mixture_init" in params:
        weights = params["mixture_init"]["weights"]
        mus = [np.array(m, dtype=float) for m in params["mixture_init"]["mus"]]
        Ps = [np.array(P, dtype=float) for P in params["mixture_init"]["Ps"]]
    else:
        # create initial control sequence (zeros) of length Nt*n_u (user can change)
        # U_init = params.get("U_init", np.zeros((Nt * n_u,)))
        U_init = np.kron(np.ones((1, Nt)), [0.0, 1.0]).ravel()
        weights = [1.0]
        mus = [jnp.array(U_init, dtype=float)]
        Ps = [jnp.array(sigma0_arr, dtype=float)]

    # containers
    states = [start.copy()]
    costs = []
    mixture_records = []
    cov_traces_map = []
    innovations_map = []
    cov_norms_map = []
    cov_traces = []
    innovations = []
    mu_states = []
    num_gmm_components = []
    x_current = start.copy()
    t = 0.0
    steps = 0
    max_steps = int(np.ceil(T / dt))

    import cProfile
    import datetime
    import os
    profiler = cProfile.Profile()
    profiler.enable()

    # for step in tqdm(range(max_steps), desc="GSF Controller Steps"):
    for step in range(max_steps):
        # --- PREDICT EXPANSION: for each existing component, branch for each process noise mode ---
        # breakpoint()
        predicted_weights = []
        predicted_mus = []
        predicted_Ps = []
        predicted_sigma_pts = []
        for j, (wj, muj, Pj) in enumerate(zip(weights, mus, Ps)):
            for mode in process_noises:
                alpha_r = float(mode.get("weight", 1.0))
                Q_r = np.array(mode["Q"], dtype=float)
                mu_p, P_p, sigma_pts_p = gsf.predict_component(muj, Pj, Q_r)
                predicted_weights.append(wj * alpha_r)
                predicted_mus.append(mu_p)
                predicted_Ps.append(P_p)
                predicted_sigma_pts.append(sigma_pts_p)

        # --- UPDATE EXPANSION: for each predicted component, branch for each measurement noise mode ---
        updated_weights = []
        updated_mus = []
        updated_Ps = []
        y_preds_local = []
        Ss_local = []
        lambdas_local = []  # likelihoods
        for j_pred, (w_pred, mu_pred, P_pred, sigma_pts_pred) in enumerate(zip(predicted_weights, predicted_mus, predicted_Ps, predicted_sigma_pts)):
            for m_mode in measurement_noises:
                beta_s = float(m_mode.get("weight", 1.0))
                R_s = m_mode["R"]
                mu_u, P_u, Lambda_js, y_pred_j, S_j = gsf.update_component_unscented(
                    mu_pred, P_pred, sigma_pts_pred, y_meas=0.0, R=R_s, x_current=x_current, q_ref=q_ref, cost_map=cost_map
                )
                w_new = w_pred * beta_s * float(Lambda_js)
                updated_weights.append(float(w_new))
                updated_mus.append(mu_u)
                updated_Ps.append(P_u)
                y_preds_local.append(y_pred_j)
                Ss_local.append(S_j)
                lambdas_local.append(float(Lambda_js))

        # normalize weights, avoid all-zero
        w_arr = np.array(updated_weights, dtype=float)
        if np.sum(w_arr) <= 0:
            # avoid degenerate case: reassign proportional weights from predicted (without likelihood)
            w_arr = np.array(predicted_weights, dtype=float)
            if np.sum(w_arr) <= 0:
                w_arr = np.ones_like(w_arr) / float(len(w_arr))
            else:
                w_arr = w_arr / (np.sum(w_arr) + 1e-12)
        else:
            w_arr = w_arr / (np.sum(w_arr) + 1e-12)
        # mixture management: prune -> merge -> cap
        w_list = w_arr.tolist()
        mus_list = [m.tolist() for m in updated_mus]
        Ps_list = [p.tolist() for p in updated_Ps]

        w_list, mus_list, Ps_list = gsf.manage_mixture(w_list, mus_list, Ps_list)
        n_sigmas = gsf.n_sigma

        num_components = int(len(w_list))
        num_gmm_components.append(num_components)
        # set for next iteration
        weights = w_list
        mus = []
        for m in mus_list:
            mu = jnp.clip(np.array(m).reshape(-1, n_u), 
                          nlmodel.control_bounds[0],nlmodel.control_bounds[1]) 
            mus.append(mu.ravel())
                
        # mus = [jnp.clip(np.array(m).reshape((-1, n_u)), nlmodel.control_bounds[0],nlmodel.control_bounds[1]) for m in mus_list]
        
        Ps = [np.array(p) for p in Ps_list]

        # diagnostics
        cov_traces.append([float(np.trace(P)) for P in Ps])
        innovations.append([float(0.0 - yp) for yp in y_preds_local])

        # choose MAP component for control (largest weight)
        idx_map = int(np.argmax(weights))
        theta_map = np.array(mus[idx_map])
        cov_norms_map.append([float(np.linalg.norm(Ps[idx_map], ord=2))])
        cov_traces_map.append([float(np.trace(Ps[idx_map]))])
        innovations_map.append([float(0.0 - y_preds_local[idx_map])])
        # apply control: take first u from theta_map
        u_opt = theta_map[: n_u]
        x_current = nlmodel.dynamics_jax(x_current, u_opt, dt=dt, params=getattr(nlmodel, "nominal_params", None))
        trajs = np.zeros((Jmax*n_sigmas, Nt+1, x_current.shape[0]))
        for ji in range(len(mus)):
            for si in range(n_sigmas):
                mu = mus[ji]
                P = Ps[ji]
                sigma_pts = gsf.generate_sigma_points(mu, P)
                theta_sigma = sigma_pts[si]
                u_seq = np.array(theta_sigma).reshape((-1, 2))
                state = x_current.copy()
                trajs[ji*n_sigmas + si, 0, :] = state
                for tt in range(Nt):
                    # simulate one step using the system dynamics (jax version)
                    state = nlmodel.dynamics_jax(state, u_seq[tt], dt=dt, params=nlmodel.nominal_params)
                    trajs[ji*n_sigmas + si, tt+1, :] = state
            # state = x_current.copy()
            # trajs[si, 0, :] = state
            # mu = mus[si]
            # u_seq = np.array(mu).reshape((-1, 2))
            # for tt in range(Nt):
            #     # simulate one step using the system dynamics (jax version)
            #     state = nlmodel.dynamics_jax(state, u_seq[tt], dt=dt, params=nlmodel.nominal_params)
            #     trajs[si, tt+1, :] = state
        print(f"x_current: {x_current}, q_ref: {q_ref}, mu_new: {u_opt}, weights: {weights}")  # Debug print
        # stop if reached goal
        mu_states.append(trajs)
        states.append(x_current.copy())

        # cost diagnostics via planner using the chosen map theta
        # try:
        #     out = planner.eval_U_seq(theta_map.reshape((-1, n_u)), theta_map, x_current, q_ref, cost_map)
        #     cost = float(out[0])
        # except Exception:
        cost = 0.0
        costs.append(cost)

        mixture_records.append({"weights": weights.copy(), "mus": [m.copy() for m in mus], "Ps": [p.copy() for p in Ps]})

        # steps += 1
        t += dt
        if step == 20:
            profile_dir = os.path.join(os.getcwd(), "profiles")
            os.makedirs(profile_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"test_gsf_{ts}"
            prof_path = os.path.join(str(profile_dir), base + ".prof")
            profiler.dump_stats(prof_path)
            profiler.disable()
            print("Profiling data collected.")
            print(f"Wrote profile .prof to: {os.path.abspath(prof_path)}")
        # optional goal check
        try:
            if np.linalg.norm(x_current[:2] - q_ref[:2]) < params.get("goal_tol", 0.5):
                print(f"GSF reached goal at step {steps}, t={t:.2f}")
                break
        except Exception:
            pass
  
    return states, costs, mu_states, cov_norms_map, cov_traces_map, innovations_map, num_gmm_components
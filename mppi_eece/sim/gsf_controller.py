# gsf_controller.py
"""
Expanded Gaussian-Sum Filter (GSF) controller that grows components
by using mixtures of process and measurement noises.

Uses UKF_Controller for individual component updates with adaptive noise.
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
    from UKF_controller import UKF_Controller  # Import your UKF
except Exception:
    MPPI_Planner_Occup = None
    CollisionChecker = None
    OccupGrid = None
    UKF_Controller = None


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

        # Create a UKF controller instance for component updates
        self.ukf = UKF_Controller(
            system=nlmodel,
            mppi_planner=planner,
            alpha=ukf_alpha,
            beta=ukf_beta,
            kappa=ukf_kappa,
            auto_reset=False  # We'll handle resets manually in GSF
        )

    # ---------------- utilities ----------------
    
    def ensure_psd(self, P):
        """Make symmetric and add tiny jitter for numerical stability."""
        P = 0.5 * (P + P.T)
        jitter = 1e-9
        try:
            vals, vecs = np.linalg.eigh(P)
            vals = np.maximum(vals, jitter)
            return (vecs * vals) @ vecs.T
        except Exception:
            return P + np.eye(P.shape[0]) * jitter

    # ---------------- predict & update using UKF methods ----------------
    
    def predict_component(self, mu, P, Q, m):
        """
        UKF Prediction step using UKF_Controller
        
        Args:
            mu: current mean [n_theta]
            P: current covariance [n_theta, n_theta]
            Q: process noise covariance [n_theta, n_theta]
        Returns:
            mu_pred, P_pred, sigma_points_pred
        """
        if m is None:
            m = jnp.zeros_like(mu)
        mu_jnp = jnp.array(mu)
        P_jnp = jnp.array(P)
        Q_jnp = jnp.array(Q)
        
        # Use UKF's predict method
        mu_pred, P_pred, sigma_pts_pred = self.ukf.predict(mu_jnp, P_jnp, Q_jnp)
        
        # Convert back to numpy for host-side operations
        m_contrib = np.expand_dims(m, axis=-1)
        return np.array(mu_pred+m), np.array(P_pred+m_contrib@m_contrib.T), np.array(sigma_pts_pred)

    def update_component(self, mu_pred, P_pred, sigma_pts, y_meas, R, 
                        x_current, q_ref, cost_map):
        """
        UKF Update step using UKF_Controller
        
        Returns: mu_upd, P_upd, log_likelihood, y_pred, S
        """
        mu_pred_jnp = jnp.array(mu_pred)
        P_pred_jnp = jnp.array(P_pred)
        sigma_pts_jnp = jnp.array(sigma_pts)
        R_jnp = jnp.array(R) if np.ndim(R) > 0 else float(R)
        x_current_jnp = jnp.array(x_current)
        q_ref_jnp = jnp.array(q_ref)
        
        # Use UKF's update method
        mu_upd, P_upd, innovation = self.ukf.update(
            mu_pred_jnp, P_pred_jnp, sigma_pts_jnp,
            y_meas, R_jnp, x_current_jnp, q_ref_jnp, cost_map
        )
        
        # Calculate predicted measurement and innovation covariance
        meas_sigma = jax.vmap(
            lambda theta: self.ukf.measurement_function(theta, x_current_jnp, q_ref_jnp, cost_map)
        )(sigma_pts_jnp)
        meas_sigma = jnp.clip(meas_sigma, -1e10, 0)
        
        Wm = self.ukf.Wm
        Wc = self.ukf.Wc
        
        y_pred = jnp.sum(Wm * meas_sigma)
        meas_diff = meas_sigma - y_pred
        S = jnp.sum(Wc * meas_diff**2) + R_jnp
        
        # Compute log-likelihood (more numerically stable)
        log_likelihood = -0.5 * (jnp.log(2.0 * jnp.pi * S + 1e-10) + (innovation ** 2) / (S + 1e-10))
        
        # Convert back to numpy
        return (np.array(mu_upd), np.array(P_upd), float(log_likelihood), 
                float(y_pred), float(S))

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
        """Greedy pairwise merging using Mahalanobis distance threshold gamma."""
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
        w_m, m_m, P_m = self.merge(w_p, m_p, P_p)  # Re-enabled merge
        w_c, m_c, P_c = self.cap(w_m, m_m, P_m)
        return w_c, m_c, P_c

    def _tree_flatten(self):
        children = (self.wmin, self.gamma, self.ukf.alpha, self.ukf.beta, self.ukf.kappa)
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
        wmin, gamma, alpha, beta, kappa = children
        return cls(nl, pl, n_u, N, J_max, wmin=wmin, gamma=gamma,
                   ukf_alpha=alpha, ukf_beta=beta, ukf_kappa=kappa)


def do_gsf(params):
    """
    params: dict expects (not exhaustive)
      dt, Nt, start, q_ref, obs, nlmodel, T
      sigma0 : initial covariance (array or scalar)
      process_noises : list [{"weight":w, "Q": Q_array}, ...] (optional)
      measurement_noises : list [{"weight":w, "R": scalar_or_matrix}, ...] (optional)
      wmin, gamma, Jmax
      adaptive_R: bool (default True) - use adaptive measurement noise
      R_scale: float (default 0.5) - scale factor for adaptive R
      R_min: float (default 1.0) - minimum measurement noise
    Returns:
      states, costs, mixture_records, cov_traces, innovations
    """
    dt = params.get("dt", 0.1)
    Nt = params.get("Nt", 10)
    start = np.array(params["start"], dtype=float)
    q_ref = np.array(params["q_ref"], dtype=float)
    obs = params.get("obs", [])
    nlmodel = params["nlmodel"]
    T = params.get("T", 10.0)
    sigma0 = params['sigma0']
    sigma0_arr = np.eye(Nt * params.get("n_u", getattr(nlmodel, "n_u", 2))) * float(sigma0) \
        if np.isscalar(sigma0) else np.array(sigma0, dtype=float)

    n_u = params.get("n_u", getattr(nlmodel, "n_u", 2))
    process_noises = params.get("process_noises", [
        {"weight": 0.5, "Q": np.eye(sigma0_arr.shape[0]) * 0.5, 'm': np.zeros((Nt*n_u,))},
        {"weight": 0.5, "Q": np.eye(sigma0_arr.shape[0]) * 0.5, 'm': np.ones((Nt*n_u,))},
        # {"weight": 0.3, "Q": np.eye(sigma0_arr.shape[0]), 'm': np.zeros_like(sigma0_arr)}
    ])
    
    # adaptive_R = params.get("adaptive_R", True)
    adaptive_R =False 
    R_scale = params.get("R_scale", 0.5)
    R_min = params.get("R_min", 1.0)
    
    # Base measurement noise (used if not adaptive)
    measurement_noises_base = params.get("measurement_noises", [{"weight": 1.0, "R": 10.0}])

    # mixture management params
    wmin = params.get("wmin", 1e-6)
    gamma = params.get("gamma", 4.0)
    Jmax = params.get("Jmax", 3)

    # planner / collision checker
    planner = params.get("planner", None)
    cost_map = params.get("cost_map", None)
    if planner is None:
        if MPPI_Planner_Occup is None:
            raise RuntimeError("MPPI_Planner_Occup not found; pass via params['planner'].")
        planner = MPPI_Planner_Occup(
            sigma=sigma0, Q=params.get("Q", None), QT=params.get("QT", None),
            R=params.get("R", None), temperature=params.get("temperature", 1.0),
            system=nlmodel, num_anci=0, n_samples=0, N=Nt, tolerance=0.4
        )
    if cost_map is None:
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

    gsf = GSF_Controller_Expanded(nlmodel, planner, n_u=n_u, N=Nt, wmin=wmin, gamma=gamma, Jmax=Jmax)

    if "mixture_init" in params:
        weights = params["mixture_init"]["weights"]
        mus = [np.array(m, dtype=float) for m in params["mixture_init"]["mus"]]
        Ps = [np.array(P, dtype=float) for P in params["mixture_init"]["Ps"]]
    else:
        U_init = np.kron(np.ones((1, Nt)), [0.0, 1.0]).ravel()
        weights = [1.0]
        mus = [jnp.array(U_init, dtype=float)]
        Ps = [jnp.array(sigma0_arr, dtype=float)]

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
    adaptive_R_history = []
    x_current = start.copy()
    t = 0.0
    max_steps = int(np.ceil(T / dt))

    for step in range(max_steps):
        predicted_weights = []
        predicted_mus = []
        predicted_Ps = []
        predicted_sigma_pts = []
        
        for j, (wj, muj, Pj) in enumerate(zip(weights, mus, Ps)):
            for mode in process_noises:
                alpha_r = float(mode.get("weight", 1.0))
                Q_r = np.array(mode["Q"], dtype=float)
                m = np.array(mode['m'])
                
                mu_p, P_p, sigma_pts_p = gsf.predict_component(muj, Pj, Q_r, m=m)
                
                predicted_weights.append(wj * alpha_r)
                predicted_mus.append(mu_p)
                predicted_Ps.append(P_p)
                predicted_sigma_pts.append(sigma_pts_p)

        if adaptive_R:
            # Evaluate costs for all predicted components to get spread
            y_preds_for_R = []
            for mu_pred in predicted_mus:
                try:
                    y_pred_temp = gsf.ukf.measurement_function(
                        jnp.array(mu_pred), 
                        jnp.array(x_current), 
                        jnp.array(q_ref), 
                        cost_map
                    )
                    y_preds_for_R.append(float(y_pred_temp))
                except Exception:
                    continue
            
            if len(y_preds_for_R) > 1:
                # Convert to costs (negative of measurement)
                costs_pred = np.array([-y for y in y_preds_for_R])
                cost_spread = float(np.std(costs_pred))
                R_adaptive = max(cost_spread * R_scale, R_min)
            else:
                R_adaptive = R_min
            
            # Use adaptive R for all measurement modes
            measurement_noises = [{"weight": 1.0, "R": R_adaptive}]
            adaptive_R_history.append(R_adaptive)
        else:
            measurement_noises = measurement_noises_base
            adaptive_R_history.append(measurement_noises[0]["R"])

        # --- UPDATE EXPANSION ---
        updated_weights = []
        updated_mus = []
        updated_Ps = []
        y_preds_local = []
        Ss_local = []
        log_lambdas_local = []  # log-likelihoods
        
        y_meas = 0.0  # Optimality assumption
        
        for j_pred, (w_pred, mu_pred, P_pred, sigma_pts_pred) in enumerate(
            zip(predicted_weights, predicted_mus, predicted_Ps, predicted_sigma_pts)):
            for m_mode in measurement_noises:
                beta_s = float(m_mode.get("weight", 1.0))
                R_s = m_mode["R"]
                
                # Use GSF's update_component (which uses UKF internally)
                mu_u, P_u, log_Lambda_js, y_pred_j, S_j = gsf.update_component(
                    mu_pred, P_pred, sigma_pts_pred, y_meas, R_s, 
                    x_current, q_ref, cost_map
                )
                
                # Compute log-weight
                log_w_new = np.log(w_pred + 1e-30) + np.log(beta_s) + log_Lambda_js
                
                updated_weights.append(log_w_new)  # Store as log initially
                updated_mus.append(mu_u)
                updated_Ps.append(P_u)
                y_preds_local.append(y_pred_j)
                Ss_local.append(S_j)
                log_lambdas_local.append(log_Lambda_js)

        # Convert from log-space to linear weights
        log_weights = np.array(updated_weights, dtype=float)
        max_log_w = np.max(log_weights)
        w_arr = np.exp(log_weights - max_log_w)
        
        # Normalize
        if np.sum(w_arr) <= 0:
            w_arr = np.ones_like(w_arr) / float(len(w_arr))
        else:
            w_arr = w_arr / (np.sum(w_arr) + 1e-12)
        
        # Mixture management: prune -> merge -> cap
        w_list = w_arr.tolist()
        mus_list = [m.tolist() for m in updated_mus]
        Ps_list = [p.tolist() for p in updated_Ps]

        w_list, mus_list, Ps_list = gsf.manage_mixture(w_list, mus_list, Ps_list)

        num_components = int(len(w_list))
        num_gmm_components.append(num_components)
        
        # Set for next iteration (DON'T clip means, only clip when applying)
        weights = w_list
        mus = [np.array(m) for m in mus_list]
        Ps = [np.array(p) for p in Ps_list]

        # Diagnostics
        cov_traces.append([float(np.trace(P)) for P in Ps])
        innovations.append([float(0.0 - yp) for yp in y_preds_local])

        # Choose MAP component for control
        idx_map = int(np.argmax(weights))
        theta_map = np.array(mus[idx_map])
        
        cov_norms_map.append([float(np.linalg.norm(Ps[idx_map]))])
        cov_traces_map.append([float(np.trace(Ps[idx_map]))])
        innovations_map.append([float(0.0 - y_preds_local[idx_map])])
        
        # Apply control (with clipping)
        u_opt_raw = theta_map[:n_u]
        u_opt = np.clip(u_opt_raw, nlmodel.control_bounds[0], nlmodel.control_bounds[1])
        
        x_current = nlmodel.dynamics_jax(x_current, u_opt, dt=dt, 
                                        params=getattr(nlmodel, "nominal_params", None))
        
        # Generate trajectories for visualization
        n_sigmas = gsf.ukf.n_sigma
        trajs = np.zeros((Jmax * n_sigmas, Nt + 1, x_current.shape[0]))
        
        for ji in range(len(mus)):
            mu = mus[ji]
            P = Ps[ji]
            sigma_pts = gsf.ukf.generate_sigma_points(jnp.array(mu), jnp.array(P))
            sigma_pts = np.array(sigma_pts)
            
            for si in range(n_sigmas):
                theta_sigma = sigma_pts[si]
                # Clip controls when simulating
                u_seq = np.clip(
                    np.array(theta_sigma).reshape((-1, n_u)),
                    nlmodel.control_bounds[0], 
                    nlmodel.control_bounds[1]
                )
                state = x_current.copy()
                trajs[ji * n_sigmas + si, 0, :] = state
                
                for tt in range(Nt):
                    state = nlmodel.dynamics_jax(state, u_seq[tt], dt=dt, 
                                                params=nlmodel.nominal_params)
                    trajs[ji * n_sigmas + si, tt + 1, :] = state
        
        print(f"Step {step}: x={x_current[:2]}, goal={q_ref[:2]}, "
              f"weights={[f'{w:.3f}' for w in weights]}, n_comp={num_components}, "
              f"R_adaptive={adaptive_R_history[-1]:.2f}")
        
        mu_states.append(trajs)
        states.append(x_current.copy())
        
        cost = 0.0
        costs.append(cost)
        
        mixture_records.append({
            "weights": weights.copy(), 
            "mus": [m.copy() for m in mus], 
            "Ps": [p.copy() for p in Ps]
        })
        
        t += dt
        
        try:
            if np.linalg.norm(x_current[:2] - q_ref[:2]) < params.get("goal_tol", 0.5):
                print(f"GSF reached goal at step {step}, t={t:.2f}")
                break
        except Exception:
            pass
    
    return (states, costs, mu_states, cov_norms_map, cov_traces_map, 
            innovations_map, num_gmm_components, adaptive_R_history)
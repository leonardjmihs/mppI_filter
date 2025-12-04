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
from copy import deepcopy
from math import ceil

try:
    from mppi_eece.jax_mppi.mppi_planners import MPPI_Planner_Occup
    from mppi_eece.jax_mppi.collision_checker import CollisionChecker
    from mppi_eece.sim.grid import OccupGrid
except Exception:
    # If not available, set placeholders. User will likely have real modules in their environment.
    MPPI_Planner_Occup = None
    CollisionChecker = None
    OccupGrid = None


def ensure_psd(P, jitter=1e-9):
    P = 0.5 * (P + P.T)
    try:
        vals, vecs = np.linalg.eigh(P)
        vals = np.maximum(vals, jitter)
        return (vecs * vals) @ vecs.T
    except Exception:
        return P + np.eye(P.shape[0]) * jitter


def generate_sigma_points(mu, P, alpha=1e-2, beta=2.0, kappa=0.0):
    """Classic Unscented sigma points (numpy). Returns (sigma_points, Wm, Wc).

    mu: (n,) vector
    P: (n,n) covariance
    """
    n = mu.size
    lam = alpha * alpha * (n + kappa) - n
    n_sigma = 2 * n + 1

    # weights
    Wm = np.zeros(n_sigma)
    Wc = np.zeros(n_sigma)
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1 - alpha * alpha + beta)
    for i in range(1, n_sigma):
        w = 1.0 / (2.0 * (n + lam))
        Wm[i] = w
        Wc[i] = w

    # cholesky-safe
    P_reg = P + np.eye(n) * 1e-9
    try:
        L = np.linalg.cholesky(P_reg)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(P_reg)
        vals = np.maximum(vals, 1e-9)
        L = vecs @ np.diag(np.sqrt(vals))

    scale = np.sqrt(n + lam)
    Ls = L * scale

    sigma = np.zeros((n_sigma, n))
    sigma[0] = mu.copy()
    for i in range(n):
        sigma[1 + i] = mu + Ls[:, i]
        sigma[1 + n + i] = mu - Ls[:, i]

    return sigma, Wm, Wc


# ------------------ mixture management ------------------

def prune_mixture(weights, mus, Ps, wmin=1e-6):
    w = np.array(weights, dtype=float)
    keep_idx = np.where(w >= wmin)[0]
    if keep_idx.size == 0:
        keep_idx = np.array([int(np.argmax(w))])
    w_n = w[keep_idx]
    w_n = w_n / (np.sum(w_n) + 1e-12)
    mus_n = [mus[i] for i in keep_idx]
    Ps_n = [Ps[i] for i in keep_idx]
    return w_n.tolist(), mus_n, Ps_n


def merge_mixture(weights, mus, Ps, gamma=4.0):
    """Greedy pairwise merging using Mahalanobis distance threshold gamma.
    Returns normalized weights and merged components.
    """
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
        if best_pair is not None and best_dist < gamma:
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

    w = np.array(w, dtype=float)
    if np.sum(w) <= 0:
        w = np.ones_like(w) / float(len(w))
    else:
        w = w / (np.sum(w) + 1e-12)
    return w.tolist(), [mi.tolist() for mi in m], [pi.tolist() for pi in P]


def cap_mixture(weights, mus, Ps, Jmax=10):
    w = np.array(weights)
    idx = np.argsort(-w)
    keep = idx[: Jmax]
    w_new = w[keep]
    w_new = w_new / (np.sum(w_new) + 1e-12)
    mus_new = [mus[i] for i in keep]
    Ps_new = [Ps[i] for i in keep]
    return w_new.tolist(), mus_new, Ps_new


def manage_mixture(weights, mus, Ps, wmin=1e-6, gamma=4.0, Jmax=10):
    w_p, m_p, P_p = prune_mixture(weights, mus, Ps, wmin=wmin)
    # w_m, m_m, P_m = merge_mixture(w_p, m_p, P_p, gamma=gamma)
    # w_c, m_c, P_c = cap_mixture(w_m, m_m, P_m, Jmax=Jmax)
    w_c, m_c, P_c = cap_mixture(w_p, m_p, P_p, Jmax=Jmax)
    return w_c, m_c, P_c


# ------------------ controller class ------------------
class GSF_MPPISimple:

    def __init__(self, nlmodel, planner, n_u=2, N=10,
                 wmin=1e-6, gamma=4.0, Jmax=12,
                 ukf_alpha=1e-2, ukf_beta=2.0, ukf_kappa=0.0):
        self.nlmodel = nlmodel
        self.planner = planner
        self.n_u = int(n_u)
        self.N = int(N)
        self.n_theta = self.N * self.n_u

        # mixture params
        self.wmin = float(wmin)
        self.gamma = float(gamma)
        self.Jmax = int(Jmax)

        # UKF params (used in sigma points only)
        self.alpha = ukf_alpha
        self.beta = ukf_beta
        self.kappa = ukf_kappa

    # shift operator for control sequences (Phi)
    def shift_controls(self, theta):
        th = theta.reshape((-1, self.n_u)).copy()
        th[:-1] = th[1:]
        # th[-1] = th[-1]  # duplicate last control
        th[-1] = np.zeros_like(th[-1])  # duplicate last control
        return th.ravel()

    def predict_component(self, mu, P, Q):
        """UKF-style prediction: shift controls and add process noise Q."""
        # sigma points for control sequence
        sigma_pts, Wm, Wc = generate_sigma_points(mu, P, self.alpha, self.beta, self.kappa)
        # propagate sigma points through shift operator (deterministic)
        sigma_pred = np.array([self.shift_controls(sp) for sp in sigma_pts])
        mu_pred = np.sum(Wm[:, None] * sigma_pred, axis=0)
        diff = sigma_pred - mu_pred[None, :]
        P_pred = np.zeros((self.n_theta, self.n_theta))
        for i in range(sigma_pred.shape[0]):
            P_pred += Wc[i] * np.outer(diff[i], diff[i])
        P_pred = P_pred + Q
        P_pred = ensure_psd(P_pred)
        return mu_pred, P_pred, sigma_pred

    def evaluate_cost(self, theta_mean, x_current, q_ref, cost_map=None):
        """Evaluate planner cost for a control sequence mean theta_mean.

        theta_mean: (n_theta,) array shaped into (N, n_u)
        returns scalar cost
        """
        u_seq = np.array(theta_mean).reshape((-1, self.n_u))
        out = self.planner.eval_U_seq(u_seq, theta_mean, x_current, q_ref, cost_map)
        cost = float(out[0])
        return cost

def select_control_from_mixture(weights, mus, selection='weighted_mean'):
    n_u = mus[0].shape[0]
    if selection == 'map':
        idx = int(np.argmax(weights))
        theta_map = np.array(mus[idx])
        return theta_map, idx
    else:
        w = np.array(weights)
        mus_arr = np.array([m for m in mus])
        theta_bar = np.sum(w[:, None] * mus_arr, axis=0)
        return theta_bar,0 


# ------------------ main driver ------------------

def do_gsf_mppi(params):
    """Run the GSF-MPPI hybrid controller.

    Returns states, costs, mixture_records, cov_traces, innovations (empty here)
    """
    dt = params.get('dt', 0.1)
    Nt = int(params.get('Nt', 10))
    sigma0 = params['sigma0']
    start = np.array(params['start'], dtype=float)
    obs = params["obs"]
    q_ref = np.array(params['q_ref'], dtype=float)
    nlmodel = params['nlmodel']
    planner = params.get('planner', None)
    cost_map = params.get('cost_map', None)

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
    # mixture / algorithmic params
    sigma0 = params.get('sigma0', 1.0)
    if np.isscalar(sigma0):
        sigma0_arr = np.eye(Nt * params.get('n_u', getattr(nlmodel, 'n_u', 2))) * float(sigma0)
    else:
        sigma0_arr = np.array(sigma0, dtype=float)

    process_noises = params.get('process_noises', [{'weight': 1.0, 'Q': np.eye(sigma0_arr.shape[0]) * 1e-2}])
    measurement_noises = params.get('measurement_noises', [{'weight': 1.0, 'R': 1.0}])

    wmin = params.get('wmin', 1e-6)
    gamma = params.get('gamma', 4.0)
    Jmax = params.get('Jmax', 10)
    mppi_lambda = params.get('mppi_lambda', 1.0)

    # instantiate controller
    n_u = params.get('n_u', getattr(nlmodel, 'n_u', 2))
    # gsf = GSF_MPPISimple(nlmodel, planner, n_u=n_u, N=Nt, wmin=wmin, gamma=gamma, Jmax=Jmax)

    # initialize mixture
    if 'mixture_init' in params:
        weights = params['mixture_init']['weights']
        mus = [np.array(m, dtype=float) for m in params['mixture_init']['mus']]
        Ps = [np.array(P, dtype=float) for P in params['mixture_init']['Ps']]
    else:
        # default initial control sequence: zeros
        # U_init = np.zeros((Nt * n_u,))
        U_init = np.kron(np.ones((1, Nt)), [0.0, 1.0]).ravel()
        weights = [1.0]
        mus = [U_init.copy()]
        Ps = [sigma0_arr.copy()]

    # records
    states = [start.copy()]
    costs = []
    mixture_records = []
    cov_traces_opt = []
    # innovations_map = []
    cov_norms_opt = []
    # innovations = []
    mu_states = []
    num_gmm_components = []

    x_current = start.copy()
    T = params.get('T', 10.0)
    max_steps = int(ceil(T / dt))
    n_theta = Nt*n_u

    import cProfile
    import datetime
    import os
    profiler = cProfile.Profile()
    profiler.enable()

    for step in range(max_steps):
        # --- Multi-Gaussian MPPI update block (host/NumPy) ---
        # params you can tune
        M = params.get("samples_per_component", 128)     # samples per component
        lambda_mp = params.get("mppi_lambda", 1.0)       # MPPI temperature
        Q_proc = params.get("process_noise_cov", np.eye(n_theta) * 1e-3)
        min_cov_jitter = 1e-6

        new_weights_log = []
        new_mus = []
        new_Ps = []
        costs_mean = []
        sampled_u_seqs = []

        # Precompute clip bounds
        u_low, u_high = nlmodel.control_bounds[0], nlmodel.control_bounds[1]

        for j, (w_j, mu_j, P_j) in enumerate(zip(weights, mus, Ps)):
            # ensure arrays
            mu_j = np.array(mu_j).reshape(-1)           # shape (n_theta,)
            P_j = np.array(P_j)
            # draw samples ~ N(mu_j, P_j)
            try:
                samples = np.random.multivariate_normal(mu_j, P_j, size=M)   # (M, n_theta)
            except np.linalg.LinAlgError:
                # fallback: add jitter to P_j
                Pj_safe = P_j + np.eye(n_theta) * 1e-6
                samples = np.random.multivariate_normal(mu_j, Pj_safe, size=M)

            # reshape each sample to sequence (M, Nt, n_u) for planner
            samples_seq = samples.reshape((M, Nt, n_u))
            sampled_u_seqs.append(samples_seq)

            # clip controls to bounds
            samples_seq = np.clip(samples_seq, u_low, u_high)

            # evaluate costs for each sample (host call to planner)
            costs_m = np.zeros((M,), dtype=float)
            for m in range(M):
                u_seq = samples_seq[m]          # shape (Nt, n_u)
                try:
                    out = planner.eval_U_seq(jnp.array(u_seq), jnp.array(samples[m]), jnp.array(x_current), jnp.array(q_ref), cost_map)
                    costs_m[m] = float(out[0])
                except Exception:
                    costs_m[m] = 1e6

            # normalize costs (optional) — helps with huge cost scales
            # shift to min
            minc = np.min(costs_m)
            costs_shift = costs_m - minc

            # MPPI sample weights (numerically stable)
            log_w_samples = - costs_shift / max(lambda_mp, 1e-12)
            log_w_samples -= np.max(log_w_samples)
            w_samples = np.exp(log_w_samples)
            w_samples /= (np.sum(w_samples) + 1e-12)    # normalized across M

            # importance-weighted mean (MPPI update for this component)
            # samples is shape (M, n_theta)
            mu_updated = np.sum(w_samples[:, None] * samples, axis=0)   # (n_theta,)

            # weighted covariance (biased form) + process noise
            diff = samples - mu_updated[None, :]
            # weighted covariance: sum w * diff^T diff
            cov_w = (w_samples[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0)
            P_updated = cov_w + Q_proc
            # stabilize P
            P_updated = 0.5 * (P_updated + P_updated.T)
            try:
                vals, vecs = np.linalg.eigh(P_updated)
                vals = np.maximum(vals, min_cov_jitter)
                P_updated = (vecs * vals) @ vecs.T
            except Exception:
                P_updated = P_updated + np.eye(n_theta) * (min_cov_jitter)

            # compute mean cost for the updated mean (used for component weight)
            try:
                out_mean = planner.eval_U_seq(jnp.array(mu_updated.reshape((-1, n_u))), jnp.array(mu_updated), jnp.array(x_current), jnp.array(q_ref), cost_map)
                cost_mean_j = float(out_mean[0])
            except Exception:
                cost_mean_j = 1e6

            # store
            new_mus.append(mu_updated.tolist())
            new_Ps.append(P_updated.tolist())
            costs_mean.append(cost_mean_j)

            # component-level log weight (combine prior and MPPI factor)
            # Use log space: log(w_prior) - cost_mean / lambda
            log_prior = np.log(max(w_j, 1e-300))
            logw_unnorm = log_prior - (cost_mean_j / max(lambda_mp, 1e-12))
            new_weights_log.append(float(logw_unnorm))

        # normalize component weights with log-sum-exp
        logw = np.array(new_weights_log, dtype=float)
        mx = np.max(logw)
        w_exp = np.exp(logw - mx)
        w_norm = w_exp / (np.sum(w_exp) + 1e-12)
        weights_after = w_norm.tolist()

        # now run your mixture management (prune/merge/cap) with these updated mus and Ps
        weights_after, mus_after, Ps_after = manage_mixture(weights_after, new_mus, new_Ps, Jmax=Jmax)

        # finalize: set weights/mus/Ps for next loop
        weights = weights_after
        mus = np.array(mus_after)
        Ps = Ps_after

        # trajs = np.zeros((Jmax*M, Nt+1, x_current.shape[0]))
        trajs = np.zeros((Jmax, Nt+1, x_current.shape[0]))

        for ji in range(len(mus)):
                mu = mus[ji]
                u_seq = np.array(mu).reshape((-1, 2))
                state = x_current.copy()
                trajs[ji, 0, :] = state
                for tt in range(Nt):
                    state = nlmodel.dynamics_jax(state, u_seq[tt], dt=dt, params=nlmodel.nominal_params)
                    trajs[ji, tt+1, :] = state
            # state = x_current.copy()
        mu_states.append(trajs)

        mu_opt = select_control_from_mixture(weights, mus, selection='weighted_mean')[0]
        u_opt = np.array(mu_opt).reshape((-1, n_u))[0]  # first control in sequence
        x_current = nlmodel.dynamics_jax(x_current, u_opt, dt=dt, params=getattr(nlmodel, 'nominal_params', None))

        # cost diagnostics for chosen control (optional)
        # try:
        #     cost_chosen = gsf.evaluate_cost(np.array(mus[int(np.argmax(weights))]), x_current, q_ref, cost_map)
        # except Exception:
        #     cost_chosen = float(costs_local[0]) if costs_local else 0.0
        cost_chosen = 0.0

        costs.append(cost_chosen)
        states.append(x_current.copy())
        mixture_records.append({'weights': deepcopy(weights), 'mus': deepcopy(mus), 'Ps': deepcopy(Ps)})
        print(f"x_current: {x_current}, q_ref: {q_ref}, mu_new: {u_opt}, weights: {weights}")  # Debug print

        if step == 10:
            profile_dir = os.path.join(os.getcwd(), "profiles")
            os.makedirs(profile_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"test_gsf_{ts}"
            prof_path = os.path.join(str(profile_dir), base + ".prof")
            profiler.dump_stats(prof_path)
            profiler.disable()
            print("Profiling data collected.")
            print(f"Wrote profile .prof to: {os.path.abspath(prof_path)}")

        # goal check
        try:
            if np.linalg.norm(x_current[:2] - q_ref[:2]) < params.get('goal_tol', 0.5):
                break
        except Exception:
            pass
        
    # return states, costs, mixture_records, cov_traces
    return states, costs, mu_states, cov_norms_opt, cov_traces_opt, None, num_gmm_components
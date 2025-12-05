# gsf_controller.py - CEM-based update
"""
MPPI-style Gaussian-Sum Filter with Cross-Entropy Method updates.
"""

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from mppi_eece.jax_mppi.mppi_planners import MPPI_Planner_Occup
from mppi_eece.jax_mppi.collision_checker import CollisionChecker
from mppi_eece.sim.grid import OccupGrid

class GSF_CEM_Controller:
    def __init__(self, nlmodel, planner,
                 n_u=2,
                 N=10,
                 n_samples=2000,
                 elite_ratio=0.1,  # Use top 10% as elites
                 Jmax=12,
                 wmin=1e-6,
                 gamma=4.0,
                 temperature=1.0):
        self.nlmodel = nlmodel
        self.planner = planner
        self.n_u = n_u
        self.N = N
        self.n_theta = self.N * self.n_u
        self.n_samples = n_samples
        self.elite_ratio = elite_ratio
        self.n_elites = max(int(n_samples * elite_ratio), 1)

        # mixture management params
        self.wmin = wmin
        self.gamma = gamma
        self.Jmax = Jmax
        self.temperature = temperature

    # ... [keep sample_from_mixture and evaluate_samples same as before] ...

    def sample_from_mixture(self, weights, mus, Ps, rng_key, n_samples):
        """Sample control sequences from Gaussian mixture."""
        J = len(weights)
        weights_arr = jnp.array(weights)
        
        # Sample component assignments
        key_comp, key_samples = jax.random.split(rng_key)
        component_ids = jax.random.categorical(
            key_comp,
            logits=jnp.log(weights_arr + 1e-10),
            shape=(n_samples,)
        )
        
        # Sample from assigned components
        # breakpoint()
        mus = jnp.array([mus[component_id] for component_id in component_ids])
        Ps = jnp.array([Ps[component_id] for component_id in component_ids])
        keys = jax.random.split(key_samples, n_samples)
        
        # def sample_from_component(key, comp_id):
        def sample_from_component(key, mu, P):
            # mu = jnp.array(mus[comp_id])
            # P = jnp.array(Ps[comp_id])
            P_reg = P + jnp.eye(self.n_theta) * 1e-6
            
            sample = jax.random.multivariate_normal(key, mu, P_reg)
            sample_reshaped = sample.reshape((-1, self.n_u))
            sample_clipped = jnp.clip(
                sample_reshaped,
                self.nlmodel.control_bounds[0],
                self.nlmodel.control_bounds[1]
            )
            return sample_clipped.ravel()
        
        # samples = jax.vmap(sample_from_component)(keys, component_ids)
        samples = jax.vmap(sample_from_component)(keys, mus, Ps)
        return samples, component_ids

    def evaluate_samples(self, samples, x_current, q_ref, cost_map):
        """Evaluate cost for each sampled control sequence."""
        def eval_single(u_flat):
            u_seq = u_flat.reshape((-1, self.n_u))
            outputs = self.planner.eval_U_seq(
                u_seq, u_flat, x_current, q_ref, cost_map
            )
            return outputs[0]
        
        costs = jax.vmap(eval_single)(samples)
        return costs

    # ============ CEM-BASED MIXTURE UPDATE ============
    
    def update_mixture_cem(self, weights, mus, Ps, samples, costs, component_ids):
        """
        Update mixture using Cross-Entropy Method.
        
        For each component:
        1. Get its samples
        2. Select elite samples (best k%)
        3. Fit Gaussian to elites
        4. Update component weight based on elite ratio
        
        Args:
            weights: [J] mixture weights
            mus: [J x n_theta] means
            Ps: [J x n_theta x n_theta] covariances
            samples: [n_samples, n_theta] sampled sequences
            costs: [n_samples] costs
            component_ids: [n_samples] component assignments
        Returns:
            weights_new, mus_new, Ps_new
        """
        J = len(weights)
        weights_new = []
        mus_new = []
        Ps_new = []
        
        # Convert to numpy
        samples_np = np.array(samples)
        costs_np = np.array(costs)
        component_ids_np = np.array(component_ids)
        
        # Handle infinite costs
        finite_mask = np.isfinite(costs_np)
        if not np.any(finite_mask):
            # All costs infinite - return unchanged
            return weights, mus, Ps
        
        # Global elite threshold for weight updates
        costs_finite = costs_np[finite_mask]
        global_elite_threshold = np.percentile(costs_finite, self.elite_ratio * 100)
        
        for j in range(J):
            # Get samples from this component
            mask = (component_ids_np == j)
            n_samples_j = int(np.sum(mask))
            
            if n_samples_j == 0:
                # No samples - penalize weight, keep distribution
                weights_new.append(weights[j] * 0.01)
                mus_new.append(mus[j])
                Ps_new.append(Ps[j])
                continue
            
            samples_j = samples_np[mask]
            costs_j = costs_np[mask]
            
            # Filter finite costs
            finite_j = np.isfinite(costs_j)
            if not np.any(finite_j):
                # All infinite - heavily penalize
                weights_new.append(weights[j] * 1e-6)
                mus_new.append(mus[j])
                Ps_new.append(Ps[j])
                continue
            
            samples_j_finite = samples_j[finite_j]
            costs_j_finite = costs_j[finite_j]
            
            # === CEM: SELECT ELITE SAMPLES ===
            n_elites_j = max(int(len(costs_j_finite) * self.elite_ratio), 1)
            elite_indices = np.argsort(costs_j_finite)[:n_elites_j]
            elite_samples = samples_j_finite[elite_indices]
            elite_costs = costs_j_finite[elite_indices]
            
            # === CEM: FIT GAUSSIAN TO ELITES ===
            mu_new = np.mean(elite_samples, axis=0)
            
            # Covariance with regularization
            if len(elite_samples) > 1:
                P_new = np.cov(elite_samples.T) + np.eye(self.n_theta) * 1e-4
            else:
                # Single elite - use small covariance
                P_new = np.eye(self.n_theta) * 0.1
            
            # === UPDATE COMPONENT WEIGHT ===
            # Weight proportional to: (prior weight) × (elite ratio) × (quality)
            # Elite ratio: how many of this component's samples were elite
            n_global_elites_from_j = np.sum(costs_j_finite <= global_elite_threshold)
            elite_ratio_j = n_global_elites_from_j / len(costs_j_finite)
            
            # Quality: average cost of elites
            avg_elite_cost = np.mean(elite_costs)
            quality = np.exp(-avg_elite_cost / self.temperature)
            
            # Combined weight update
            weight_new = weights[j] * elite_ratio_j * quality
            
            weights_new.append(float(weight_new))
            mus_new.append(mu_new)
            Ps_new.append(P_new)
        
        # Normalize weights
        total_weight = sum(weights_new)
        if total_weight > 1e-10:
            weights_new = [w / total_weight for w in weights_new]
        else:
            # All components failed - reset to uniform
            weights_new = [1.0 / J] * J
        
        return weights_new, mus_new, Ps_new

    
    def update_mixture_cem_adaptive(self, weights, mus, Ps, samples, costs, component_ids):
        """
        Adaptive CEM: elite ratio adapts per component based on cost variance.
        
        Components with high cost variance → more exploration → larger elite set
        Components with low cost variance → exploitation → smaller elite set
        """
        J = len(weights)
        weights_new = []
        mus_new = []
        Ps_new = []
        
        samples_np = np.array(samples)
        costs_np = np.array(costs)
        component_ids_np = np.array(component_ids)
        
        finite_mask = np.isfinite(costs_np)
        if not np.any(finite_mask):
            return weights, mus, Ps
        
        costs_finite = costs_np[finite_mask]
        global_elite_threshold = np.percentile(costs_finite, self.elite_ratio * 100)
        
        for j in range(J):
            mask = (component_ids_np == j)
            n_samples_j = int(np.sum(mask))
            
            if n_samples_j == 0:
                weights_new.append(weights[j] * 0.01)
                mus_new.append(mus[j])
                Ps_new.append(Ps[j])
                continue
            
            samples_j = samples_np[mask]
            costs_j = costs_np[mask]
            
            finite_j = np.isfinite(costs_j)
            if not np.any(finite_j):
                weights_new.append(weights[j] * 1e-6)
                mus_new.append(mus[j])
                Ps_new.append(Ps[j])
                continue
            
            samples_j_finite = samples_j[finite_j]
            costs_j_finite = costs_j[finite_j]
            
            # === ADAPTIVE ELITE RATIO ===
            cost_std = np.std(costs_j_finite)
            cost_mean = np.mean(costs_j_finite)
            cv = cost_std / (abs(cost_mean) + 1e-6)  # Coefficient of variation
            
            # High variance → more exploration → larger elite ratio
            # Low variance → exploitation → smaller elite ratio
            adaptive_elite_ratio = np.clip(
                self.elite_ratio * (1 + cv),  # Scale by variance
                0.05,  # Min 5%
                0.3    # Max 30%
            )
            
            n_elites_j = max(int(len(costs_j_finite) * adaptive_elite_ratio), 1)
            elite_indices = np.argsort(costs_j_finite)[:n_elites_j]
            elite_samples = samples_j_finite[elite_indices]
            elite_costs = costs_j_finite[elite_indices]
            
            # Fit Gaussian to elites
            mu_new = np.mean(elite_samples, axis=0)
            
            if len(elite_samples) > 1:
                P_new = np.cov(elite_samples.T) + np.eye(self.n_theta) * 1e-4
            else:
                P_new = np.eye(self.n_theta) * 0.1
            
            # Weight update
            n_global_elites_from_j = np.sum(costs_j_finite <= global_elite_threshold)
            elite_ratio_j = n_global_elites_from_j / len(costs_j_finite)
            avg_elite_cost = np.mean(elite_costs)
            quality = np.exp(-avg_elite_cost / self.temperature)
            
            weight_new = weights[j] * elite_ratio_j * quality
            
            weights_new.append(float(weight_new))
            mus_new.append(mu_new)
            Ps_new.append(P_new)
        
        # Normalize
        total_weight = sum(weights_new)
        if total_weight > 1e-10:
            weights_new = [w / total_weight for w in weights_new]
        else:
            weights_new = [1.0 / J] * J
        
        return weights_new, mus_new, Ps_new

    
    def update_single_component_cem(self, mu, P, samples, costs):
        """
        Pure CEM update for single Gaussian (useful for comparison).
        
        Returns: mu_new, P_new
        """
        samples_np = np.array(samples)
        costs_np = np.array(costs)
        
        # Filter finite
        finite_mask = np.isfinite(costs_np)
        if not np.any(finite_mask):
            return mu, P
        
        samples_finite = samples_np[finite_mask]
        costs_finite = costs_np[finite_mask]
        
        # Select elites
        n_elites = max(int(len(costs_finite) * self.elite_ratio), 1)
        elite_indices = np.argsort(costs_finite)[:n_elites]
        elite_samples = samples_finite[elite_indices]
        
        # Fit Gaussian
        mu_new = np.mean(elite_samples, axis=0)
        
        if len(elite_samples) > 1:
            P_new = np.cov(elite_samples.T) + np.eye(self.n_theta) * 1e-4
        else:
            P_new = np.eye(self.n_theta) * 0.1
        
        return mu_new, P_new

    
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
        w_m, m_m, P_m = self.merge(w_p, m_p, P_p)
        w_c, m_c, P_c = self.cap(w_m, m_m, P_m)
        # w_c, m_c, P_c = self.cap(w_p, m_p, P_p)
        return w_c, m_c, P_c

    def add_process_noise(self, weights, mus, Ps, Qs, expl_mus, alphas):
        """Add process noise for exploration."""
        
        mus_pred = []
        Ps_pred = []
        weights_pred = []
        
        for w, mu, P in zip(weights, mus, Ps):
            for alpha, expl_mu, Q in zip(alphas, expl_mus, Qs):
                # Shift operator
                mu_arr = np.array(mu).reshape((-1, self.n_u))
                mu_shifted = np.roll(mu_arr, -1, axis=0)
                mu_shifted[-1] = mu_arr[-1] 
                mus_pred.append(mu_shifted.ravel() + expl_mu)
                weights_pred.append(w*alpha)  
            
                # Add process noise
                P_pred = np.array(P) + Q
                Ps_pred.append(P_pred)

        return weights_pred, mus_pred, Ps_pred


# ============ MAIN DRIVER ============

def do_gsf_cem(params):
    """GSF with CEM updates."""
    # [Same setup as before]
    dt = params.get("dt", 0.1)
    Nt = params.get("Nt", 10)
    start = np.array(params["start"], dtype=float)
    q_ref = np.array(params["q_ref"], dtype=float)
    obs = params.get("obs", [])
    nlmodel = params["nlmodel"]
    T = params.get("T", 10.0)
    n_u = params.get("n_u", 2)
    
    sigma0 = params['sigma0']
    sigma0_arr = np.eye(Nt * n_u) * float(sigma0) if np.isscalar(sigma0) else np.array(sigma0)
    
    n_samples = params.get("n_samples", 1000)
    elite_ratio = params.get("elite_ratio", 0.2) 
    temperature = params.get("temperature", 1.0)
    Jmax = params.get("Jmax", 4)

    process_noises = params.get("process_noises", [
        # {"weight": 0.3333, "Q": np.eye(sigmae[0]) * 0.5, 'm': np.zeros((Nt*n_u,))},
        {"weight": 0.5, "Q": sigma0_arr, 'm': np.kron(np.ones((1, Nt)), [0.0, 0.5]).ravel()},
        {"weight": 0.25, "Q": sigma0_arr, 'm': np.kron(np.ones((1, Nt)), [0.1, 0.0]).ravel()},
        {"weight": 0.25, "Q": sigma0_arr, 'm': np.kron(np.ones((1, Nt)), [-0.1, 0.0]).ravel()},
        # {"weight": 0.3, "Q": np.eye(sigma0_arr.shape[0]), 'm': np.zeros_like(sigma0_arr)}
    ])
    Qs = [process_noise["Q"] for process_noise in process_noises]
    expl_mus = [process_noise['m'] for process_noise in process_noises]
    alphas = [process_noise["weight"] for process_noise in process_noises]
    
    # [Setup planner and cost_map as before...]
    # planner = params["planner"]
    # cost_map = params["cost_map"]
    # if planner is None:
    if MPPI_Planner_Occup is None:
        raise RuntimeError("MPPI_Planner_Occup not found; pass via params['planner'].")
    planner = MPPI_Planner_Occup(
        sigma=sigma0, Q=params.get("Q", None), QT=params.get("QT", None),
        R=params.get("R", None), temperature=params.get("temperature", 1.0),
        system=nlmodel, num_anci=0, n_samples=0, N=Nt, tolerance=0.4
    )
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
    # Initialize controller
    gsf = GSF_CEM_Controller(
        nlmodel, planner, n_u=n_u, N=Nt, n_samples=n_samples,
        elite_ratio=elite_ratio, Jmax=Jmax, temperature=temperature, gamma=10.0, wmin=0.05
    )

    # Initial mixture
    U_init = np.kron(np.ones((1, Nt)), [0.0, 1.0]).ravel()
    weights = [1.0]
    mus = [U_init]
    Ps = [sigma0_arr]
    mu_states = []

    # Containers
    states = [start.copy()]
    x_current = start.copy()
    rng_key = jax.random.PRNGKey(0)
    max_steps = int(np.ceil(T / dt))

    for step in range(max_steps):
        rng_key, subkey = jax.random.split(rng_key)
        
        # 1. Add process noise
        weights, mus, Ps = gsf.add_process_noise(weights, mus, Ps, Qs, expl_mus, alphas)
        
        # 2. Sample from mixture
        samples, component_ids = gsf.sample_from_mixture(weights, mus, Ps, subkey, n_samples)
        
        # 3. Evaluate costs
        costs = gsf.evaluate_samples(samples, x_current, q_ref, cost_map)
        
        # 4. CEM UPDATE
        weights, mus, Ps = gsf.update_mixture_cem(weights, mus, Ps, samples, costs, component_ids)
        # weights, mus, Ps = gsf.update_mixture_cem_adaptive(weights, mus, Ps, samples, costs, component_ids)
        
        # 5. Mixture management
        weights, mus, Ps = gsf.manage_mixture(weights, mus, Ps)
        trajs = np.zeros((len(mus), Nt + 1, len(x_current)))     
        for ji in range(len(mus)):
            mu = mus[ji]
            P = Ps[ji]
            u_seq = np.clip(
                np.array(mu).reshape((-1, n_u)),
                nlmodel.control_bounds[0], 
                nlmodel.control_bounds[1]
            )
            state = x_current.copy()
            trajs[ji, 0, :] = state
                
            for tt in range(Nt):
                state = nlmodel.dynamics_jax(state, u_seq[tt], dt=dt, 
                                            params=nlmodel.nominal_params)
                trajs[ji, tt + 1, :] = state
        mu_states.append(trajs)

        # 6. Extract control
        theta_map = sum(w * np.array(mu) for w, mu in zip(weights, mus))
        # theta_map = np.array(calculate_new_means(0.1, costs, samples, None)[0].ravel())

        u_opt = np.clip(theta_map[:n_u], nlmodel.control_bounds[0], nlmodel.control_bounds[1])
        
        # 7. Simulate
        x_current = nlmodel.dynamics_jax(x_current, u_opt, dt=dt, params=nlmodel.nominal_params)
        states.append(x_current.copy())
        
        print(f"Step {step}: x={x_current[:2]}, weights={[f'{w:.3f}' for w in weights]}, "
              f"n_comp={len(weights)}, min_cost={float(np.nanmin(costs)):.2f}")
        
        if np.linalg.norm(x_current[:2] - q_ref[:2]) < 0.5:
            print(f"Reached goal at step {step}")
            break
    
    return states, weights, mu_states, Ps


def calculate_new_means(temp,  costs, seq, original_seq):
    lowest_ind = jnp.nanargmin(costs)
    temperature = temp 
    exp_cost = jnp.exp(1/temperature*(jnp.nanmin(costs)-costs))
    denom = jnp.nansum(exp_cost) + 1e-7
    # best_u = original_seq + jnp.nansum(exp_cost[..., None, None] * (seq-original_seq), axis=0) / denom
    best_u = jnp.nansum(exp_cost[..., None, None] * (seq), axis=0) / denom

    lowest_u = seq[lowest_ind]
    return best_u, lowest_u, temperature
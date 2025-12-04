import jax
import jax.numpy as jnp
import numpy as np
from mppi_eece.jax_mppi.collision_checker import CollisionChecker 
import functools
from dataclasses import dataclass
from typing import List, Tuple
from mppi_eece.jax_mppi.mppi_planners import MPPI_Planner_Occup

@dataclass
class ControlMode:
    """Single mode in control-space GSF"""
    u_mean: jnp.ndarray  # (N, 2) control sequence
    u_cov: jnp.ndarray   # (N*2, N*2) covariance
    weight: float        # mode probability
    
    def _tree_flatten(self):
        children = (self.u_mean, self.u_cov, self.weight)
        aux_data = {}
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children)

from jax import tree_util
tree_util.register_pytree_node(ControlMode,
                               ControlMode._tree_flatten,
                               ControlMode._tree_unflatten)


class MPPI_GSF_Planner(MPPI_Planner_Occup):
    """MPPI with Gaussian Sum Filter in control space"""
    
    def __init__(self, sigma, Q, QT, R, temperature, system, num_anci, n_samples, N, 
                 tolerance, occup_value=[100], alpha=1,
                 # GSF-specific parameters
                 init_num_modes=3,
                 q_process=0.01,
                 z_desired=0.0,
                 prune_threshold=0.05,
                 merge_threshold=2.0,
                 split_threshold=0.3,
                 min_modes=2,
                 max_modes=10):
        
        super().__init__(sigma, Q, QT, R, temperature, system, num_anci, 
                        n_samples, N, tolerance, occup_value, alpha)
        
        # GSF parameters
        self.init_num_modes = init_num_modes
        self.q_process = q_process  # Process noise for control diffusion
        self.z_desired = z_desired  # Desired cost (typically 0)
        self.prune_threshold = prune_threshold
        self.merge_threshold = merge_threshold
        self.split_threshold = split_threshold
        self.min_modes = min_modes
        self.max_modes = max_modes
        
        # Initialize modes
        self.modes = self._initialize_modes()
    
    def _initialize_modes(self):
        """Initialize control modes"""
        control_dim = 2
        total_dim = self.N * control_dim
        
        modes = []
        for i in range(self.init_num_modes):
            mode = ControlMode(
                u_mean=jnp.zeros((self.N, control_dim)),
                u_cov=jnp.eye(total_dim) * jnp.diag(self.sigma).mean(),
                weight=1.0 / self.init_num_modes
            )
            modes.append(mode)
        
        return modes
    
    def gsf_predict(self, modes):
        """GSF predict step: add process noise to maintain exploration"""
        Q_process = jnp.eye(self.N * 2) * self.q_process
        
        updated_modes = []
        for mode in modes:
            new_mode = ControlMode(
                u_mean=mode.u_mean,
                u_cov=mode.u_cov + Q_process,
                weight=mode.weight
            )
            updated_modes.append(new_mode)
        
        return updated_modes
    
    def sample_from_modes(self, modes, rng_key, n_samples):
        """Sample control sequences from GMM with mode tracking"""
        # Calculate sampling weights (balance probability and uncertainty)
        sampling_weights = jnp.array([
            mode.weight * jnp.sqrt(jnp.linalg.det(mode.u_cov + 1e-6*jnp.eye(self.N*2)))
            for mode in modes
        ])
        sampling_weights = sampling_weights / jnp.sum(sampling_weights)
        
        # Sample mode indices
        key_modes, key_samples = jax.random.split(rng_key)
        mode_indices = jax.random.categorical(
            key_modes, 
            logits=jnp.log(sampling_weights + 1e-10),
            shape=(n_samples,)
        )
        
        # Sample from each mode
        keys = jax.random.split(key_samples, n_samples)

        mode_means = jnp.array([modes[idx].u_mean.flatten() for idx in mode_indices])
        mode_covs = jnp.array([modes[idx].u_cov for idx in mode_indices])

        def sample_single(key, mode_mean, mode_cov):
            # Sample from multivariate normal
            u_flat = jax.random.multivariate_normal(
                key, 
                mode_mean, 
                mode_cov
            )
            u_seq = u_flat.reshape((self.N, 2))
            # Clip to control bounds
            u_seq = jnp.clip(u_seq, self.system.control_bounds[0], self.system.control_bounds[1])
            return u_seq

        u_samples = jax.vmap(sample_single)(keys, mode_means, mode_covs)
        return u_samples, mode_indices

    def gsf_update_ukf(self, modes, state, q_ref, cost_map):
        """
        UKF-style update: use sigma points instead of random samples
        """
        updated_modes = []
        n_dim = self.N * 2
    
        for mode in modes:
            # === Generate Sigma Points ===
            # Use unscented transform
            alpha, beta, kappa = 1e-3, 2.0, 0.0
            lambda_ukf = alpha**2 * (n_dim + kappa) - n_dim
        
            # Cholesky decomposition
            try:
                L = jnp.linalg.cholesky(mode.u_cov)
            except:
                # Fall back if not PSD
                updated_modes.append(mode)
                continue
        
            # Sigma points (2n+1 points)
            sigma_points = jnp.zeros((2*n_dim + 1, self.N, 2))
            sigma_points = sigma_points.at[0].set(mode.u_mean)
        
            sqrt_term = jnp.sqrt(n_dim + lambda_ukf) * L
            for i in range(n_dim):
                sigma_points = sigma_points.at[i+1].set(
                    (mode.u_mean.flatten() + sqrt_term[:, i]).reshape(self.N, 2)
                )
                sigma_points = sigma_points.at[i+1+n_dim].set(
                    (mode.u_mean.flatten() - sqrt_term[:, i]).reshape(self.N, 2)
                )
        
            # Clip sigma points
            sigma_points = jnp.clip(sigma_points, 
                                self.system.control_bounds[0],
                                self.system.control_bounds[1])
        
            # === Evaluate Costs at Sigma Points ===
            costs = jax.vmap(
                lambda u: self.eval_U_seq(u, jnp.zeros_like(u.flatten()), 
                                        state, q_ref, cost_map)[0]
            )(sigma_points)
        
            # === Weights for Sigma Points ===
            w_m_0 = lambda_ukf / (n_dim + lambda_ukf)
            w_c_0 = w_m_0 + (1 - alpha**2 + beta)
            w_i = 1.0 / (2 * (n_dim + lambda_ukf))
        
            weights_mean = jnp.concatenate([
                jnp.array([w_m_0]),
                jnp.ones(2*n_dim) * w_i
            ])
        
            weights_cov = jnp.concatenate([
                jnp.array([w_c_0]),
                jnp.ones(2*n_dim) * w_i
            ])
        
            # === Predicted Measurement ===
            h_mean = jnp.sum(weights_mean * costs)
        
            # === Cross-covariance ===
            u_deviations = sigma_points - mode.u_mean
            h_deviations = costs - h_mean
        
            P_uh = jnp.sum(
                weights_cov[:, None, None, None] * 
                u_deviations[:, :, :, None] * h_deviations[:, None, None, None],
                axis=0
            ).reshape(n_dim)
        
            P_hh = jnp.sum(weights_cov * h_deviations**2) + self.temperature
        
            # === Kalman Update ===
            K = P_uh / P_hh
            innovation = self.z_desired - h_mean
        
            u_new_flat = mode.u_mean.flatten() + K * innovation
            u_new = u_new_flat.reshape((self.N, 2))
        
            P_new = mode.u_cov - jnp.outer(K, K) * P_hh
            P_new = 0.5 * (P_new + P_new.T) + 1e-6 * jnp.eye(n_dim)
        
            # === Weight Update ===
            likelihood = jnp.exp(-0.5 * innovation**2 / P_hh) / jnp.sqrt(2 * jnp.pi * P_hh)
            new_weight = mode.weight * likelihood
        
            updated_modes.append(ControlMode(
                jnp.clip(u_new, self.system.control_bounds[0], self.system.control_bounds[1]),
                P_new,
                new_weight
            ))
    
        # Normalize
        total_weight = sum(m.weight for m in updated_modes)
        if total_weight > 1e-10:
            updated_modes = [
                ControlMode(m.u_mean, m.u_cov, m.weight / total_weight)
                for m in updated_modes
            ]
    
        return updated_modes

    def gsf_update(self, modes, samples, costs, mode_indices):
        """GSF update step: update modes based on cost 'measurements'"""
        
        updated_modes = []
        min_cost = jnp.min(costs)
        for mode_id, mode in enumerate(modes):
            # Get samples from this mode
            mask = (mode_indices == mode_id)
            mode_samples = samples[mask]
            mode_costs = costs[mask]
            
            # Skip if no samples
            if jnp.sum(mask) == 0:
                updated_modes.append(mode)
                continue
            
            # === Weight Update (Bayes rule with z=0 observation) ===
            # p(z=0 | u) ∝ exp(-cost(u) / λ)
            finite_mask = jnp.isfinite(mode_costs)
            if jnp.sum(finite_mask) == 0:
                # No finite costs, heavily penalize this mode
                new_weight = mode.weight * 1e-6
            else:
                likelihoods = jnp.exp((min_cost-mode_costs) / self.temperature)
                avg_likelihood = jnp.nanmean(jnp.where(finite_mask, likelihoods, 0.0))
                new_weight = mode.weight * avg_likelihood
            
            # === Mean Update (MPPI-style weighted average) ===
            if jnp.sum(finite_mask) > 0:
                finite_samples = mode_samples[finite_mask]
                finite_costs = mode_costs[finite_mask]
                
                # MPPI weights
                mppi_weights = jnp.exp(
                    (jnp.nanmin(finite_costs) - finite_costs) / self.temperature
                )
                mppi_weights = mppi_weights / (jnp.sum(mppi_weights) + 1e-10)
                
                # Weighted mean
                new_u_mean = jnp.sum(
                    mppi_weights[:, None, None] * finite_samples, 
                    axis=0
                )
                
                # === Covariance Update (empirical weighted covariance) ===
                deviations = finite_samples - new_u_mean
                deviations_flat = deviations.reshape(-1, self.N * 2)
                
                new_u_cov = jnp.sum(
                    mppi_weights[:, None, None] * 
                    jax.vmap(lambda d: jnp.outer(d, d))(deviations_flat),
                    axis=0
                )
                
                # Add regularization
                new_u_cov = new_u_cov + 1e-4 * jnp.eye(self.N * 2)
            else:
                # Keep current estimates if all costs are infinite
                new_u_mean = mode.u_mean
                new_u_cov = mode.u_cov
            
            updated_mode = ControlMode(
                u_mean=new_u_mean,
                u_cov=new_u_cov,
                weight=new_weight
            )
            updated_modes.append(updated_mode)
        
        # Normalize weights
        total_weight = sum(m.weight for m in updated_modes)
        if total_weight > 1e-10:
            updated_modes = [
                ControlMode(m.u_mean, m.u_cov, m.weight / total_weight)
                for m in updated_modes
            ]
        
        return updated_modes
    
    def manage_modes(self, modes, samples, costs, mode_indices):
        """Mode management: prune, merge, split"""
        
        # === PRUNING ===
        modes = [m for m in modes if m.weight > self.prune_threshold]
        
        if len(modes) == 0:
            # Reinitialize if all pruned
            return self._initialize_modes()
        
        # === SPLITTING ===
        new_modes = []
        modes_to_remove = []
        
        for mode_id, mode in enumerate(modes):
            mask = (mode_indices == mode_id)
            mode_samples = samples[mask]
            mode_costs = costs[mask]
            
            # Need enough finite samples to consider splitting
            finite_mask = jnp.isfinite(mode_costs)
            n_finite = jnp.sum(finite_mask)
            
            if n_finite < 10 or len(modes) >= self.max_modes:
                continue
            
            # Check if cost distribution is bimodal
            finite_costs = mode_costs[finite_mask]
            if self._is_bimodal(finite_costs):
                # Split using k-means on weighted samples
                finite_samples = mode_samples[finite_mask]
                weights = jnp.exp(-finite_costs / self.temperature)
                
                clusters = self._weighted_kmeans(finite_samples, weights, k=2)
                
                if clusters is not None:
                    cluster_samples_1, cluster_samples_2, w1, w2 = clusters
                    
                    # Create two new modes
                    mode_1 = ControlMode(
                        u_mean=jnp.mean(cluster_samples_1, axis=0),
                        u_cov=jnp.cov(cluster_samples_1.reshape(-1, self.N*2).T) + 1e-4*jnp.eye(self.N*2),
                        weight=mode.weight * w1
                    )
                    mode_2 = ControlMode(
                        u_mean=jnp.mean(cluster_samples_2, axis=0),
                        u_cov=jnp.cov(cluster_samples_2.reshape(-1, self.N*2).T) + 1e-4*jnp.eye(self.N*2),
                        weight=mode.weight * w2
                    )
                    
                    new_modes.extend([mode_1, mode_2])
                    modes_to_remove.append(mode)
        
        # Apply splits
        for mode in modes_to_remove:
            modes.remove(mode)
        modes.extend(new_modes)
        
        # === MERGING ===
        i = 0
        while i < len(modes):
            j = i + 1
            while j < len(modes):
                if self._should_merge(modes[i], modes[j]):
                    modes[i] = self._merge_modes(modes[i], modes[j])
                    modes.pop(j)
                else:
                    j += 1
            i += 1
        
        # === EXPANSION ===
        if len(modes) < self.min_modes:
            # Add exploration mode
            exploration_mode = ControlMode(
                u_mean=jnp.zeros((self.N, 2)),
                u_cov=jnp.eye(self.N * 2) * jnp.diag(self.sigma).mean() * 2,
                weight=0.1
            )
            modes.append(exploration_mode)
            
            # Renormalize
            total = sum(m.weight for m in modes)
            modes = [ControlMode(m.u_mean, m.u_cov, m.weight/total) for m in modes]
        
        return modes
    
    def _is_bimodal(self, costs):
        """Check if cost distribution has multiple peaks"""
        if len(costs) < 10:
            return False
        
        # Simple heuristic: high variance and outliers suggest bimodality
        cost_std = jnp.std(costs)
        cost_mean = jnp.mean(costs)
        
        # Check for significant spread
        return cost_std > self.split_threshold * cost_mean
    
    def _weighted_kmeans(self, samples, weights, k=2):
        """Simple weighted k-means clustering"""
        n_samples = samples.shape[0]
        if n_samples < k:
            return None
        
        # Initialize centroids randomly
        indices = jnp.array([0, n_samples // 2])
        centroids = samples[indices]
        
        # Simple 2-means
        for _ in range(10):
            # Assign to closest centroid
            dists_1 = jnp.sum((samples - centroids[0])**2, axis=(1,2))
            dists_2 = jnp.sum((samples - centroids[1])**2, axis=(1,2))
            assignments = (dists_2 < dists_1).astype(jnp.float32)
            
            # Update centroids
            w1 = jnp.sum(weights * (1 - assignments)) + 1e-10
            w2 = jnp.sum(weights * assignments) + 1e-10
            
            centroids_1 = jnp.sum(
                weights[:, None, None] * (1-assignments)[:, None, None] * samples,
                axis=0
            ) / w1
            centroids_2 = jnp.sum(
                weights[:, None, None] * assignments[:, None, None] * samples,
                axis=0
            ) / w2
            
            centroids = jnp.stack([centroids_1, centroids_2])
        
        # Final assignment
        dists_1 = jnp.sum((samples - centroids[0])**2, axis=(1,2))
        dists_2 = jnp.sum((samples - centroids[1])**2, axis=(1,2))
        assignments = (dists_2 < dists_1)
        
        cluster_1 = samples[~assignments]
        cluster_2 = samples[assignments]
        
        if len(cluster_1) == 0 or len(cluster_2) == 0:
            return None
        
        w1_normalized = jnp.sum(weights[~assignments]) / jnp.sum(weights)
        w2_normalized = jnp.sum(weights[assignments]) / jnp.sum(weights)
        
        return cluster_1, cluster_2, w1_normalized, w2_normalized
    
    def _should_merge(self, mode_i, mode_j):
        """Check if two modes should be merged based on Mahalanobis distance"""
        diff = mode_i.u_mean.flatten() - mode_j.u_mean.flatten()
        avg_cov = 0.5 * (mode_i.u_cov + mode_j.u_cov)
        
        # Mahalanobis distance
        dist_sq = diff @ jnp.linalg.solve(avg_cov + 1e-4*jnp.eye(self.N*2), diff)
        dist = jnp.sqrt(dist_sq)
        
        return dist < self.merge_threshold
    
    def _merge_modes(self, mode_i, mode_j):
        """Merge two modes using moment matching"""
        w_total = mode_i.weight + mode_j.weight
        w_i = mode_i.weight / w_total
        w_j = mode_j.weight / w_total
        
        # Merged mean
        u_merged = w_i * mode_i.u_mean + w_j * mode_j.u_mean
        
        # Merged covariance (preserving total uncertainty)
        diff_i = (mode_i.u_mean - u_merged).flatten()
        diff_j = (mode_j.u_mean - u_merged).flatten()
        
        cov_merged = (
            w_i * (mode_i.u_cov + jnp.outer(diff_i, diff_i)) +
            w_j * (mode_j.u_cov + jnp.outer(diff_j, diff_j))
        )
        
        return ControlMode(u_merged, cov_merged, w_total)
    
    def compute_control_from_modes(self, modes):
        """Extract control from mixture - weighted average of first control"""
        u_optimal = sum(
            mode.weight * mode.u_mean
            for mode in modes
        )
        return u_optimal

    def calculate_new_means(self, costs, seq, original_seq):
        lowest_ind = jnp.nanargmin(costs)
        temperature = self.temperature
        exp_cost = jnp.exp(1/temperature*(jnp.nanmin(costs)-costs))
        denom = jnp.nansum(exp_cost) + 1e-7
        best_u = original_seq + jnp.nansum(exp_cost[..., None, None] * (seq-original_seq), axis=0) / denom

        lowest_u = seq[lowest_ind]
        return best_u, lowest_u, temperature
    
 
    # @jax.jit
    def mppi_gsf_step(self, state, U_original, rng_key, q_ref, cost_map, modes):
        """Single MPPI-GSF planning step"""
        
        # === 1. GSF PREDICT ===
        modes_p = self.gsf_predict(modes)
        
        # === 2. SAMPLE FROM MODES ===
        u_samples, mode_indices = self.sample_from_modes(
            modes_p, rng_key, self.n_samples
        )
        
        # === 3. EVALUATE COSTS ===
        costs_and_states = jax.vmap(
            self.eval_U_seq, 
            in_axes=(0, None, None, None, None)
        )(u_samples, U_original, state, q_ref, cost_map)
        
        costs = costs_and_states[0]
        all_state_seq = costs_and_states[2]
        
        # === 4. GSF UPDATE ===
        # modes_u = self.gsf_update(modes_p, u_samples, costs, mode_indices, state, q_ref, cost_map)
        modes_u = self.gsf_update(modes_p, u_samples, costs, mode_indices)
        # modes_u = self.gsf_update_enkf(modes_p, u_samples, costs, mode_indices)
        
        # === 5. MODE MANAGEMENT ===
        modes_n = self.manage_modes(modes_u, u_samples, costs, mode_indices)
        # breakpoint()
        
        # === 6. EXTRACT CONTROL ===
        # best_u = self.compute_control_from_modes(modes)

        # best_u = self.compute_control_from_modes(modes)

        best_u = self.calculate_new_means(costs, u_samples, U_original.reshape((-1, 2)))[0]
        # calculate best u using weihgts from samples

        
        # Prepare outputs (compatible with parent class interface)
        finite_inds = jnp.array(jnp.nonzero(jnp.isfinite(costs), 
                                            size=self.n_samples, 
                                            fill_value=0))
        collision_free = jnp.take(all_state_seq, finite_inds, axis=0).squeeze()
        min_cost = jnp.nanmin(costs)
        
        # Roll forward for next step
        new_u = jnp.roll(best_u, -1, axis=0)
        new_u = new_u.at[-1].set(new_u[-2])
        new_U = new_u.reshape((1, -1)).squeeze(0)
        
        return (
            best_u,
            new_u,
            new_U,
            min_cost,
            collision_free,
            costs,
            all_state_seq,
            modes_n  # Return updated modes
        )
    
    def _tree_flatten(self):
        children, aux_data = super()._tree_flatten()
        # Add GSF-specific parameters to aux_data
        aux_data.update({
            'init_num_modes': self.init_num_modes,
            'q_process': self.q_process,
            'z_desired': self.z_desired,
            'prune_threshold': self.prune_threshold,
            'merge_threshold': self.merge_threshold,
            'split_threshold': self.split_threshold,
            'min_modes': self.min_modes,
            'max_modes': self.max_modes,
        })
        return (children, aux_data)

tree_util.register_pytree_node(MPPI_GSF_Planner,
                               MPPI_GSF_Planner._tree_flatten,
                               MPPI_GSF_Planner._tree_unflatten)
print('GSF Planner registered')

from mppi_eece.jax_mppi.mppi_planners import MPPI_Planner_Occup
from mppi_eece.jax_mppi.collision_checker import CollisionChecker
from mppi_eece.sim.grid import OccupGrid
from mppi_eece.ancillary_controller import AncillaryController
from tqdm import tqdm

def do_gsf_mppi(params):
    rng_key = jax.random.PRNGKey(0)
    dt = params['dt']
    Nt = params['Nt']
    n_samples = params['n_samples']
    start = params['start']
    q_ref = params['q_ref']
    obs = params['obs']
    nlmodel = params['nlmodel']
    T = params['T']
    sigma0 = params['sigma0']
    temperature = params['temperature']
    Q = params['Q']
    R = params['R']
    QT = params['QT']

    ns = 20
    ratio_sim_mppi = 10
    states = [start]
    costs = []
    min_cost = []
    total_costs = 0

    U = np.kron(np.ones((1, Nt)), [0.0, 1.0]).ravel()
    global_U = U.copy()
    global_Us = [global_U.copy()]
    global_us = [global_U.copy().reshape((-1, 2))]
    
    resolution = 0.5
    origin = np.array([-40, -10])
    wh = np.array([50/resolution, 20/resolution], dtype=np.int32)
    boundary = [[origin[0], origin[0]+wh[0]*resolution], [origin[1], origin[1]+wh[1]*resolution]]
    grid = OccupGrid(boundary, resolution)
    grid.find_occupancy_grid(obs, buffer=0.05)

    occupied = grid.find_all_occupied(obs)
    collision_checker = CollisionChecker(jnp.array(grid.occup_grid), origin, resolution, wh, occup_value=100)
    num_anci = 4
    planner_params={'reserve_num':num_anci,
                    'max_raw_path':2,
                    'max_raw_path2':2,
                    'ratio_to_short':2.0,
                    'sample_sz_p':0.0,
                    'occup_value':100,
                    'max_time':0.1}

    ancillary_controller = AncillaryController(mpc_params=params, 
                                               planner_params=planner_params, 
                                               solver_type='acados',
                                               planner_type='topo_prm')
    timestep_reached = -1

    max_modes = 4
    mppi_planner = MPPI_GSF_Planner(sigma=sigma0,
                                      Q=Q,
                                      QT=QT,
                                      R=R,
                                      temperature=temperature,
                                      system=nlmodel, num_anci=num_anci, 
                                      n_samples=n_samples, 
                                      N=Nt,
                                      tolerance=0.4,
                                      occup_value=100,
                                      max_modes=max_modes)
    ancillary_controller.planner.set_occupancy_grid(grid.occup_grid, origin, resolution, wh)
    state = start
    U = np.kron(np.ones((1, Nt)), [0.0, 1.0]).ravel()
    global_U = U.copy()
    sim_state = state.copy()
    cost = 0
    timestep_prog = tqdm(np.arange(0, T, dt))
    iter = 0

    sampled_states = []
    for t in timestep_prog:
        rng_key, subkey = jax.random.split(rng_key)
        if t == 0:
            U_modes = [ControlMode(
                u_mean=jnp.array(global_U).reshape((Nt, 2)),
                u_cov=jnp.eye(Nt*2) * np.diag(sigma0).mean(),
                weight=1.0 ) ]

        # if len(U_modes)< max_modes:
        #     all_planner_paths, all_mpc_paths, all_control_sequences = ancillary_controller.plan_multi(
        #                         start=sim_state, q_ref=q_ref, occupied=occupied, collision_checker=collision_checker)

        #     for control_seq in all_control_sequences:
        #         if len(U_modes)>= max_modes:
        #         u_sol = control_seq
        #         U_modes.append(ControlMode(
        #             u_mean=jnp.array(u_sol).reshape((Nt, 2)),
        #             u_cov=jnp.eye(Nt*2) * np.diag(sigma0).mean(),
        #             weight=1.0 / max_modes))

        # renormalize weights
        total_weight = sum(mode.weight for mode in U_modes)
        U_modes = [ControlMode(mode.u_mean, mode.u_cov, mode.weight / total_weight) for mode in U_modes]
        outputs = mppi_planner.mppi_gsf_step(sim_state, global_U,  subkey, q_ref, collision_checker, U_modes)
        best_u = outputs[0]
        new_u = outputs[1]
        global_U = outputs[2]
        min_cost = outputs[3]
        collision_free = outputs[4]
        all_costs = outputs[5]
        all_state_seq = outputs[6]
        U_modes = outputs[7]

        print(f"{[U_mode.u_mean[0] for U_mode in U_modes]}")


        # cost, _ = ancillary_controller.eval_trajectory(outputs[1], sim_state, q_ref, dt=dt)
        cost = 0.0

        for i in range(ratio_sim_mppi):
            sim_state = nlmodel.dynamics_jax(sim_state, best_u[0], dt=dt/ratio_sim_mppi, params=nlmodel.nominal_params)
            dist = np.linalg.norm((sim_state - q_ref)[0:2])
            if (dist<0.5) and timestep_reached==-1:
                timestep_reached = t / dt
        sim_state = np.array(sim_state)

        sampled_states.append(all_state_seq) 
        states.append(sim_state)
        global_us.append(best_u)
        global_Us.append(global_U.copy())
        costs.append(cost)
        total_costs += cost
        iter += 1

    return (states,
            costs, 
            sampled_states,
            timestep_reached, 
            global_us,)

## Utility
def get_mode_means(modes):
    """Extract means from list of ControlMode"""
    return [mode.u_mean for mode in modes]

def get_mode_covariances(modes):
    """Extract covariances from list of ControlMode"""
    return [mode.u_cov for mode in modes]

def get_mode_weights(modes):
    """Extract weights from list of ControlMode"""
    return [mode.weight for mode in modes]
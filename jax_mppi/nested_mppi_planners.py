import jax
import jax.numpy as jnp
import numpy as np
from jax_mppi import reachability
import functools

class MPPI_Planner_Occup:
    def __init__(self,sigma, Q, QT, R, temperature, system, num_anci, n_samples, n_mini, N, N_mini, N_safe,  max_sz, tolerance, footprint=[[0.0,0.0]], occup_value=[100], alpha=1, heuristic_weight=0.0):
        self.system = system
        self.num_anci = num_anci
        self.n_samples = n_samples
        self.n_mini = n_mini
        self.N = N
        self.N_mini = N_mini
        self.N_safe = N_safe
        self.max_sz = max_sz
        self.sigma = sigma
        self.Q = Q
        self.QT = QT
        self.R = R
        self.temperature = temperature
        self.tolerance = tolerance
        self.footprint = np.array(footprint)
        self.occup_value = occup_value
        self.alpha = alpha
        self.heuristic_weight= heuristic_weight
        # print(f" self.system: {self.system}")
        # print(f"self.num_anci: {self.num_anci}")
        # print(f"self.n_samples: {self.n_samples}")
        # print(f"self.n_mini: {self.n_mini}")
        # print(f"self.N: {self.N}")
        # print(f"self.N_mini: {self.N_mini}")
        # print(f"self.N_safe: {self.N_safe}")
        # print(f"self.Q: {self.Q}")
        # print(f"self.QT: {self.QT}")
        # print(f"self.R: {self.R}")
        # print(f"self.temperature:{ self.temperature}")

    def eval_U_seq(self, u_seq, original_u, state, q_ref, safe_zones, cost_map, origin, resolution, wh):
        cost_and_term, (state_seq, min_sz_dist) = jax.lax.scan(self.single_sample_running_cost, 
                                            (1.0, state, q_ref, safe_zones, cost_map, origin, resolution, wh), 
                                            u_seq)
        cost = cost_and_term[0]
        terminal_state = cost_and_term[1]
        terminal_cost = jnp.dot((terminal_state - q_ref), jnp.dot(self.QT, (terminal_state - q_ref)))
        cost += terminal_cost 
        cost += u_seq.ravel().T @ jnp.diag(1.0 / jnp.diag(self.sigma)) @ (u_seq.ravel().T - original_u)* (1-self.alpha)*(self.temperature)
        
        return (cost, terminal_state, state_seq, jnp.sum(min_sz_dist))

    def single_sample_running_cost(self, carry, params):
        u = params
        cost, sim_state, q_ref, safe_zones, cost_map, origin, resolution, wh = carry
        new_state = self.system.jax_dynamics(sim_state, u, 0, self.system.dt, self.system.nominal_params)
        dist = sim_state - q_ref
        dx = jnp.dot(new_state - sim_state, jnp.dot(self.Q, new_state - sim_state))
        new_cost = cost + jnp.dot(dist, jnp.dot(self.Q, dist)) * (1.0 + jax.lax.cond(dx == 0.0, lambda _: 0.0, lambda x: 1.0 / x, dx))

        ind = jnp.floor((sim_state[:2] - origin) / resolution)
        ind1 = jnp.maximum(ind, np.array([0, 0]))
        ind1 = jnp.minimum(ind1, wh).astype(jnp.int32)
        barrier_value = cost_map[ind1[0], ind1[1]] >= 10
        new_cost += jax.lax.cond(barrier_value, lambda _: jnp.inf, lambda _: 0.0, operand=None)

        def _min_dist_fn(_):
            centres = safe_zones[:, :2]
            radii   = safe_zones[:, 2]
            dists   = jnp.linalg.norm(new_state[:2] - centres, axis=1) - radii
            return jnp.min(jnp.maximum(0.0, dists))

        n_sz = safe_zones.shape[0]
        min_dist = jax.lax.cond(
            n_sz > 0,
            _min_dist_fn,
            lambda _: jnp.array(0.0, dtype=new_state.dtype),
            operand=None,
        )
        new_cost += (min_dist ** 2) * self.heuristic_weight

        return (new_cost, new_state, q_ref, safe_zones, cost_map, origin, resolution, wh), (sim_state, min_dist)

    def mini_mppi(self, state_seq, rng_key, cost_map, origin, resolution, wh, safe_zones):
        """
        Given a state seq, does mini rollouts mapped across seqeunces 
        and determines if all states in sequence is safe
        """
        if self.N_mini > 0 and self.n_mini > 0:
            safe_flag, safe_state_seq, safe_u_seqs = jax.vmap(reachability.get_reachability_costmap, 
                            in_axes=(None, None, None, None, None, None, 0, None, None, None, None, None, None)) \
            (self.system, 
             cost_map,
             origin, 
             resolution, 
             wh, 
             self.tolerance, 
             state_seq, 
             safe_zones, 
             self.footprint, 
             rng_key, 
             self.n_mini, 
             self.N_mini,
             self.occup_value
             )
            return jnp.all(safe_flag), safe_state_seq.squeeze(), safe_flag[0], safe_u_seqs
        else:
            # return True, jnp.array([state_seq[0]]), True, np.zeros((1,self.system.N_DIMS, 1))
            # return True, jnp.expand_dims([state_seq[0]], 1), True, np.zeros((1,self.system.N_DIMS, 1))
            # return True, jnp.expand_dims(state_seq[0],(0,1)), True, np.zeros((1,self.system.N_DIMS, 1))
            # --- vanilla MPPI W - Block ---
            # return True, jnp.zeros((1, 0, self.system.N_DIMS)), True, jnp.zeros((1, self.system.N_DIMS, 0))
            return (
                jnp.array(True),                               # mini_cost (scalar bool)
                jnp.zeros((1, self.system.N_DIMS)),            # safe_state_seq per item: shape (T=1, D)
                jnp.array(True),                               # current_safe (scalar bool)
                jnp.zeros((1, self.system.N_DIMS, 1)),         # safe_u_seqs placeholder
            )

    def mini_mppi_ais(self, state_seq, rng_key, cost_map, origin, resolution, wh, safe_zones):
        """
        Given a state seq, does mini rollouts mapped across seqeunces 
        and determines if all states in sequence is safe
        """
        if self.N_mini > 0 and self.n_mini > 0:
            ais = 3
            safe_flag, safe_state_seq, state_seqs_all, state_seqs_means, safe_u_seqs = jax.vmap(reachability.get_reachability_mppi_costmap, 
                in_axes=(None, None, None, None, None, 0, None, None, None, None, None, None, None,  None, None, None)) \
                (self.system,  
                 cost_map,  
                origin, 
                resolution, 
                wh,         
                state_seq, 
                safe_zones, 
                self.footprint, 
                rng_key, 
                self.n_mini, 
                self.N_mini,  
                self.sigma[:self.N_mini*2, :self.N_mini*2]*2, 
                jnp.kron(jnp.ones((self.N_mini,)), jnp.array([0,self.system.control_bounds[1][1]])), 
                # self.temperature, 
                self.temperature/1000, 
                ais,
                self.occup_value
                ) 
            current_safe_state_seq = safe_state_seq.squeeze()
            # current_safe_state_seq = jax.lax.cond(safe_flag[0] < self.tolerance, lambda x: current_safe_state_seq, lambda x: jnp.ones_like(current_safe_state_seq)*0.0, 0)
            # return jnp.all(safe_flag<self.tolerance, axis=0), safe_state_seq.squeeze()[0], safe_flag[0]<self.tolerance, safe_u_seqs
            return jnp.all(safe_flag<self.tolerance, axis=0), current_safe_state_seq, safe_flag[0]<self.tolerance, safe_u_seqs
        else:
            # jax.debug.breakpoint()
            # return True, jnp.expand_dims(state_seq[0],(0,1)), True, np.zeros((1,self.system.N_DIMS, 1))
            # --- vanilla MPPI C - Block ---
            # return True, jnp.zeros((0, self.system.N_DIMS)), True, jnp.zeros((1, self.system.N_DIMS, 0))
            return (
                jnp.array(True),                                 # mini_cost (scalar bool)
                jnp.zeros((1, self.system.N_DIMS)),             # all_con_seq per item (time>=1 so [:,0,:] is valid)
                jnp.array(True),                                 # current_safe (scalar bool)
                jnp.zeros((1, self.system.N_DIMS, 1)),          # safe_u_seqs (placeholder)
            )

    def single_u_seq(self, rng_subkey, U, sigma):
        # noise_scaled = jax.random.multivariate_normal(rng_subkey, jnp.zeros(N*2), sigma,  method='svd')
        noise_scaled = jax.random.normal(rng_subkey, shape=(self.N*2,)) * jnp.diagonal(sigma)
        u_seq = (U + noise_scaled).reshape((-1,2))
        u_seq = jnp.clip(u_seq, self.system.control_bounds[0], self.system.control_bounds[1])
        return u_seq 

    # @jax.jit
    def ess(self, costs, temperature):
        w_i = jnp.exp(1/temperature*(jnp.nanmin(costs)-costs))
        ess = jnp.nansum(w_i)**2 / jnp.nansum(w_i **2)
        return ess


    def calculate_new_means(self, costs, seq, original_seq):
        lowest_ind = jnp.nanargmin(costs)
        temperature = self.temperature
        exp_cost = jnp.exp(1/temperature*(jnp.nanmin(costs)-costs))
        denom =  jnp.nansum(exp_cost) + 1e-7
        best_u =  original_seq + jnp.nansum(exp_cost[..., None, None] * (seq-original_seq), axis=0) / denom

        lowest_u = seq[lowest_ind]
        return best_u, lowest_u, temperature
    

    # @jax.jit
    # def mppi_mmodal(self, state, U_original, U_total, rng_key, q_ref, safe_zones, cost_map, origin, resolution, wh):
    #     num_modes  = self.num_anci + 1
    #     N          = self.N
    #     n_samples  = self.n_samples
    #     N_safe     = self.N_safe
    #     each_n     = n_samples // num_modes

    #     rng_keys_flat = jax.random.split(rng_key, n_samples)
    #     U_rep         = jnp.repeat(U_total, each_n, axis=0)

    #     u_flat        = jax.vmap(self.single_u_seq, in_axes=(0, 0, None))(rng_keys_flat, U_rep, self.sigma)

    #     costs_and_states = jax.vmap(self.eval_U_seq, in_axes=(0, None, None, None, None, None, None, None, None))(
    #         u_flat, U_original, state, q_ref, safe_zones, cost_map, origin, resolution, wh
    #     )
    #     costs         = costs_and_states[0]
    #     all_state_seq = costs_and_states[2]

    #     finite_inds    = jnp.array(jnp.nonzero(jnp.isfinite(costs), size=n_samples, fill_value=0))
    #     collision_free = jnp.take(all_state_seq, finite_inds, axis=0).squeeze()

    #     contingency_off = (self.N_safe <= 0) or (self.N_mini <= 0) or (self.n_mini <= 0)

    #     if contingency_off:
    #         batch        = collision_free.shape[0]
    #         mini_costs   = jnp.ones((batch,), dtype=jnp.bool_)
    #         current_safe = jnp.zeros((batch,), dtype=jnp.bool_)
    #         all_con_seq  = jnp.zeros((batch, 0, collision_free.shape[-1]))
    #         safe_u_seqs  = jnp.zeros((batch, 0, self.system.N_CONTROLS))

    #         safe_inds      = finite_inds.squeeze()
    #         number_safe    = jnp.sum(mini_costs)
    #         min_cost       = jnp.nanmin(jnp.take(costs, safe_inds, fill_value=jnp.inf))
    #         safe_state_seq = jnp.take(all_state_seq, safe_inds, axis=0)
    #         con_seq_one    = jnp.zeros((self.system.N_DIMS,))
    #     else:
    #     # if True: # Don't know why we had the contingency_off check?
    #         mini_fun = functools.partial(
    #             self.mini_mppi_ais,
    #             cost_map=cost_map,
    #             origin=origin,
    #             resolution=resolution,
    #             wh=wh,
    #             safe_zones=safe_zones,
    #         )
    #         mini_costs, all_con_seq, current_safe, safe_u_seqs = jax.vmap(mini_fun, in_axes=(0, 0))(
    #             collision_free[:, :N_safe, :],
    #             rng_keys_flat,
    #         )

    #         number_safe = jnp.sum(mini_costs)

    #         safe_inds = jnp.take(
    #             finite_inds,
    #             jnp.array(jnp.nonzero(mini_costs, size=n_samples, fill_value=0))
    #         ).squeeze()
    #         min_cost = jnp.nanmin(jnp.take(costs, safe_inds, fill_value=jnp.inf))

    #         not_safe_inds = jnp.take(
    #             finite_inds,
    #             jnp.array(jnp.nonzero(1 - mini_costs, size=n_samples, fill_value=0))
    #         ).squeeze()
    #         costs = costs.at[not_safe_inds].set(jnp.inf)

    #         safe_state_seq = jnp.take(all_state_seq, safe_inds, axis=0)
    #         con_seq_one    = all_con_seq[jnp.nonzero(mini_costs, size=n_samples, fill_value=0), 0, :][0]

    #     def do_AIS(costs, seqs):
    #         idxs = jnp.argsort(costs)[:10]
    #         elite = seqs[idxs]
    #         mu = jnp.mean(elite, axis=0)
    #         cov = jnp.cov(elite, rowvar=False) + jnp.eye(N * dim_u) * 1e-8
    #         return mu, cov

    #     best_u, lowest_u, temperature = self.calculate_new_means(costs, u_flat, U_original.reshape((-1, 2)))

    #     for i in range(ais):
    #         bests = jnp.argsort(costs)
    #         mu, cov = do_AIS(costs, u_flat.reshape((n_samples, -1)))
    #         U_i = self.calculate_new_means(costs, u_flat, U_i.reshape((N, -1))).reshape(-1)
    #         sigma_i = cov
    #         elite_idxs = bests[:m_elite]
    #         elite_states = states[elite_idxs]
    #         state_seqs_all = state_seqs_all.at[i].set(elite_states)
    #         state_seqs_means = state_seqs_means.at[i].set(state_rollouts(U_i.reshape((N_mini, dim_u))))
    #     best_u = lowest_u
    #     new_u  = jnp.roll(best_u, -1, axis=0)
    #     new_u  = new_u.at[-1].set(new_u[-2])
    #     new_U  = new_u.reshape((1, -1)).squeeze(0)

    #     return (
    #         best_u,
    #         new_u,
    #         new_U,
    #         min_cost,
    #         safe_state_seq,
    #         con_seq_one,
    #         number_safe,
    #         jnp.any(current_safe),
    #         collision_free,
    #         temperature,
    #         safe_u_seqs.squeeze(),
    #         all_con_seq,
    #         costs,
    #         all_state_seq,
    #     )
    
    # @jax.jit
    @functools.partial(jax.jit, static_argnames='ais_iters')
    def mppi_mmodal(self, state, U_original, U_total, rng_key, q_ref, safe_zones, cost_map, origin, resolution, wh, ais_iters=0):
        num_modes  = self.num_anci + 1
        N          = self.N
        dim_u      = self.system.N_CONTROLS
        # n_samples  = jnp.round(self.n_samples / (ais_iters+1))
        n_samples  = self.n_samples // (ais_iters+1)
        N_safe     = self.N_safe
        each_n     = n_samples // num_modes
        # ais_iters  = 3                  # number of AIS refinement steps
        # m_elite    = max(10, n_samples // 10)
        m_elite    = min(10, n_samples // 10)

        rng_keys_flat = jax.random.split(rng_key, each_n*num_modes)
        U_rep         = jnp.repeat(U_total, each_n, axis=0)

        u_flat        = jax.vmap(self.single_u_seq, in_axes=(0, 0, None))(rng_keys_flat, U_rep, self.sigma)
        costs_and_states = jax.vmap(self.eval_U_seq, in_axes=(0, None, None, None, None, None, None, None, None))(
            u_flat, U_original, state, q_ref, safe_zones, cost_map, origin, resolution, wh
        )
        costs         = costs_and_states[0]
        all_state_seq = costs_and_states[2]

        finite_inds    = jnp.array(jnp.nonzero(jnp.isfinite(costs), size=n_samples, fill_value=0))
        collision_free = jnp.take(all_state_seq, finite_inds, axis=0).squeeze()
        mu_final, lowest_u, temperature = self.calculate_new_means(costs, u_flat, U_original.reshape((-1, 2)))
        u_final = [u_flat]
        states_final = [all_state_seq]
        costs_final = [costs]
        # === Initialize proposal distribution ===
        # mu = U_original.reshape(-1)
        mu = mu_final
        cov = self.sigma

        def single_ais_iter(carry, rng_subkeys):
            mu, cov = carry
            rngs = jax.random.split(rng_subkeys, n_samples)
            u_samples        = jax.vmap(self.single_u_seq, in_axes=(0, None, None))(rngs, mu.flatten(), cov)
            # noises = jax.vmap(lambda r: jax.random.multivariate_normal(r, jnp.zeros_like(mu), cov))(rngs)
            # u_samples = mu + noises
            # u_samples = u_samples.reshape((n_samples, N, dim_u))
            # u_samples = jnp.clip(u_samples, self.system.control_bounds[0], self.system.control_bounds[1])

            # --- Evaluate rollouts ---
            costs_and_states = jax.vmap(self.eval_U_seq, in_axes=(0, None, None, None, None, None, None, None, None))(
                u_samples, U_original, state, q_ref, safe_zones, cost_map, origin, resolution, wh
            )
            costs = costs_and_states[0]
            all_state_seq = costs_and_states[2]

            # --- Safety filtering ---
            finite_inds = jnp.nonzero(jnp.isfinite(costs), size=n_samples, fill_value=0)[0]
            collision_free = jnp.take(all_state_seq, finite_inds, axis=0).squeeze()

            # Optional mini safety check
            def safe_check(x):
                mini_fun = functools.partial(
                    self.mini_mppi_ais,
                    cost_map=cost_map,
                    origin=origin,
                    resolution=resolution,
                    wh=wh,
                    safe_zones=safe_zones,
                )
                mini_costs, _, _, _ = jax.vmap(mini_fun, in_axes=(0, 0))(
                    collision_free[:, :N_safe, :],
                    rng_keys_flat[:collision_free.shape[0]],
                )
                return mini_costs
            # safe_flags = jax.lax.cond(N_safe > 0, safe_check, lambda _: jnp.ones((collision_free.shape[0],), dtype=jnp.bool_), None)

            # safe_flags = jnp.ones((collision_free.shape[0],), dtype=jnp.bool_)
            # safe_mask = safe_flags.astype(bool)
            # costs = jnp.where(safe_mask, costs, jnp.inf)

            # costs = costs.at[jnp.logical_not(safe_mask)].set(jnp.inf)
            # --- Compute importance weights ---
            # T = self.temperature
            # min_cost = jnp.nanmin(costs)
            # weights = jnp.exp(-(costs - min_cost) / T)
            # weights = weights / (jnp.sum(weights) + 1e-8)

            temperature = self.temperature
            weights = jnp.exp(1/temperature*(jnp.nanmin(costs)-costs))

            # --- Update AIS proposal ---
            elite_idx = jnp.argsort(costs)[:m_elite]
            elite = u_samples[elite_idx].reshape((m_elite, -1))
            mu_new = jnp.mean(elite, axis=0).reshape((-1, dim_u))
            cov_new = jnp.cov(elite, rowvar=False) + jnp.eye(N * dim_u) * 1e-6
            # cov_new = cov 

            # Alternative: weighted mean/covariance
            # weights_elite = weights[elite_idx]
            # mu_new = jnp.sum(weights[:,None, None] * u_samples, axis=0)
            # cov_new = jnp.cov(u_samples.reshape((n_samples, -1)), rowvar=False) + jnp.eye(N * dim_u) * 1e-6
            return (mu_new, cov_new), (u_samples, costs, all_state_seq)

        if ais_iters > 0:
            # === Run AIS refinement iterations ===
            (mu_final, cov_final), (u_final, costs_final, states_final) = jax.lax.scan(
                single_ais_iter,
                (mu, cov),
                rng_keys_flat[:ais_iters, :],
            )
            # for i in range(ais_iters):
            #     (mu, cov), (u_i, costs_i, states_i) = single_ais_iter((mu, cov), rng_keys_flat[i])
            #     u_final.append(u_i)
            #     costs_final.append(costs_i)
            #     states_final.append(states_i)
            # mu_final = mu
            # breakpoint()

        # --- Choose best sequence ---
        # best_idx = jnp.nanargmin(costs_final[-1])
        # best_u = u_final[-1][best_idx]
        # breakpoint()
        best_u = mu_final

        best_u = best_u.reshape((N, dim_u))


        # --- Roll control forward ---
        new_u = jnp.roll(best_u, -1, axis=0)
        new_u = new_u.at[-1].set(new_u[-2])
        new_U = new_u.reshape((1, -1)).squeeze(0)

        return (
            best_u,
            new_u,
            new_U,
            jnp.nanmin(costs_final[-1]),
            states_final[-1],
            jnp.zeros((self.system.N_DIMS,)),  # con_seq_one placeholder
            jnp.sum(jnp.isfinite(costs_final[-1])),
            jnp.any(jnp.isfinite(costs_final[-1])),
            states_final[-1],
            self.temperature,
            jnp.zeros((1, self.system.N_DIMS, 1)),  # safe_u_seqs placeholder
            states_final[-1],
            costs_final[-1],
            states_final[-1],
        )
    # def mppi_mmodal(self, state, U_original, U_total, rng_key, q_ref, safe_zones, cost_map, origin, resolution, wh, ais_iters=0):
    #     """
    #     Corrected MPPI with optional AIS refinement.
    #     Assumptions:
    #     - self.sigma: scalar standard deviation for control noise (if you store covariance differently, adapt cov construction).
    #     - self.single_u_seq(rng, U_template, sigma) -> (N,dim_u) sequence (used for initial multimodal sampling).
    #     - self.eval_U_seq(u_seq, U_original, state, ...) returns tuple where [0] are costs and [2] are all_state_seq (works as in original).
    #     """

    #     num_modes = self.num_anci + 1
    #     N = self.N
    #     dim_u = self.system.N_CONTROLS
    #     n_samples = self.n_samples // (ais_iters + 1)
    #     N_safe = self.N_safe
    #     each_n = n_samples // num_modes
    #     m_elite = max(10, n_samples // 10)

    #     # --- RNG split: one key for initial multimodal sampling, and ais_iters keys for AIS loops ---
    #     # create ais_iters + 1 keys (first used for initial sampling)
    #     total_keys = ais_iters + 1 if ais_iters >= 0 else 1
    #     rng_keys = jax.random.split(rng_key, total_keys + 1)  # +1 so we always have spare
    #     rng_init = rng_keys[0]
    #     ais_keys = rng_keys[1:1 + ais_iters] if ais_iters > 0 else []

    #     # --- Initial multimodal sampling (as original) ---
    #     rng_keys_flat = jax.random.split(rng_init, each_n * num_modes)
    #     U_rep = jnp.repeat(U_total, each_n, axis=0)  # shape (each_n * num_modes, N, dim_u)
    #     # Use existing single_u_seq for initial generation (matches original code pattern)
    #     u_flat = jax.vmap(self.single_u_seq, in_axes=(0, 0, None))(rng_keys_flat, U_rep, self.sigma)  # (n_samples, N, dim_u)
    #     costs_and_states = jax.vmap(self.eval_U_seq, in_axes=(0, None, None, None, None, None, None, None, None))(
    #         u_flat, U_original, state, q_ref, safe_zones, cost_map, origin, resolution, wh
    #     )
    #     costs = costs_and_states[0]            # shape (n_samples,)
    #     all_state_seq = costs_and_states[2]    # shape (n_samples, T, state_dim) maybe

    #     # fallback if all costs inf: keep U_original as fallback plan
    #     all_valid = jnp.any(jnp.isfinite(costs))
    #     # compute initial mu (flattened) as weighted or best sample; here use mean of best m_elite finite samples
    #     finite_idx = jnp.nonzero(jnp.isfinite(costs), size=n_samples, fill_value=-1)[0]
    #     finite_mask = jnp.isfinite(costs)
    #     # safe fallback: if none finite, set mu = U_original
    #     def compute_initial_mu():
    #         # choose the best m_elite finite samples (if less than m_elite available, use what's available)
    #         valid_costs = jnp.where(finite_mask, costs, jnp.inf)
    #         k = jnp.minimum(m_elite, jnp.sum(finite_mask))
    #         k = jnp.maximum(k, 1)
    #         elite_idx = jnp.argsort(valid_costs)[:k]
    #         elite = u_flat[elite_idx].reshape((k, -1))  # (k, N*dim_u)
    #         mu_flat = jnp.mean(elite, axis=0)
    #         return mu_flat

    #     mu_flat_init = jax.lax.cond(all_valid, compute_initial_mu, lambda: U_original.reshape(-1))
    #     mu = mu_flat_init.reshape((N, dim_u))  # proposal mean as (N, dim_u)

    #     # initialize covariance as diagonal from scalar sigma (assume self.sigma is std)
    #     D = N * dim_u
    #     sigma_scalar = jnp.asarray(self.sigma) if jnp.ndim(self.sigma) == 0 else jnp.sqrt(jnp.maximum(0.0, jnp.asarray(self.sigma)))
    #     cov_matrix = jnp.eye(D) * (sigma_scalar ** 2 + 1e-6)

    #     # store history for returns
    #     u_final_list = [u_flat]
    #     costs_final_list = [costs]
    #     states_final_list = [all_state_seq]

    #     # --- single AIS iteration function ---
    #     def single_ais_iter(carry, rng_key_iter):
    #         mu_flat, cov_mat = carry  # mu_flat: (D,), cov_mat: (D,D)
    #         # sample n_samples vectors from multivariate normal with mean mu_flat and cov_mat
    #         rngs = jax.random.split(rng_key_iter, n_samples)
    #         # vmapped sampling: returns (n_samples, D)
    #         sample_flat = jax.vmap(lambda r: jax.random.multivariate_normal(r, mu_flat, cov_mat))(rngs)
    #         u_samples = sample_flat.reshape((n_samples, N, dim_u))

    #         # clip to control bounds if defined
    #         try:
    #             lb, ub = self.system.control_bounds
    #             u_samples = jnp.clip(u_samples, lb, ub)
    #         except Exception:
    #             # if no bounds or different format, ignore
    #             pass

    #         # evaluate rollouts
    #         costs_and_states = jax.vmap(self.eval_U_seq, in_axes=(0, None, None, None, None, None, None, None, None))(
    #             u_samples, U_original, state, q_ref, safe_zones, cost_map, origin, resolution, wh
    #         )
    #         costs_iter = costs_and_states[0]
    #         states_iter = costs_and_states[2]

    #         # safety filtering: keep only finite cost samples for elite selection
    #         finite_mask_iter = jnp.isfinite(costs_iter)
    #         n_finite = jnp.sum(finite_mask_iter)

    #         # set costs of unsafe samples to +inf so they won't be selected as elite
    #         costs_for_selection = jnp.where(finite_mask_iter, costs_iter, jnp.inf)

    #         # compute importance weights robustly (avoid overflow): use min-cost baseline
    #         valid_min = jnp.where(n_finite > 0, jnp.nanmin(jnp.where(finite_mask_iter, costs_iter, jnp.nan)), jnp.nan)
    #         T = self.temperature
    #         # For samples that are inf we set weight zero
    #         weights = jnp.where(finite_mask_iter, jnp.exp((valid_min - costs_iter) / (T + 1e-12)), 0.0)
    #         weights = weights / (jnp.sum(weights) + 1e-12)

    #         # elite selection: pick best up to m_elite among finite samples
    #         k = jnp.minimum(m_elite, n_finite)
    #         k = jnp.maximum(k, 1)
    #         elite_idx = jnp.argsort(costs_for_selection)[:k]

    #         elite_flat = sample_flat[elite_idx]  # (k, D)
    #         # update mu and cov from elite (unweighted mean/cov)
    #         mu_new_flat = jnp.mean(elite_flat, axis=0)
    #         # compute covariance: (X - mu).T @ (X - mu) / (k-1)
    #         Xc = elite_flat - mu_new_flat
    #         cov_new = (Xc.T @ Xc) / jnp.maximum((k - 1), 1.0)
    #         cov_new = cov_new + jnp.eye(D) * 1e-6  # regularize

    #         return (mu_new_flat, cov_new), (u_samples, costs_iter, states_iter, weights)

    #     # --- Run AIS iterations ---
    #     if ais_iters > 0:
    #         # prepare initial carry for scan (mu as flat)
    #         carry0 = (mu.reshape(-1), cov_matrix)
    #         # use ais_keys for iterations (if not enough keys, split rng further)
    #         if len(ais_keys) < ais_iters:
    #             more_keys = jax.random.split(rng_key, ais_iters)
    #             ais_keys = more_keys[:ais_iters]

    #         # run scan
    #         (mu_final_flat, cov_final), scan_out = jax.lax.scan(
    #             single_ais_iter,
    #             carry0,
    #             jnp.array(ais_keys)
    #         )
    #         # scan_out is tuple of sequences: we extract last iteration's outputs
    #         u_all_iters, costs_all_iters, states_all_iters, weights_all_iters = scan_out
    #         # append last iter to history lists
    #         u_final_list.append(u_all_iters[-1])
    #         costs_final_list.append(costs_all_iters[-1])
    #         states_final_list.append(states_all_iters[-1])
    #         mu = mu_final_flat.reshape((N, dim_u))
    #         cov_matrix = cov_final
    #     else:
    #         # no AIS; keep initial mu/cov
    #         mu = mu.reshape((N, dim_u))
    #         cov_matrix = cov_matrix

    #     # --- Choose best sequence from last evaluated costs ---
    #     final_costs = costs_final_list[-1]
    #     final_us = u_final_list[-1]

    #     # if all costs are inf, fallback to U_original
    #     any_finite = jnp.any(jnp.isfinite(final_costs))
    #     best_idx = jnp.nanargmin(jnp.where(jnp.isfinite(final_costs), final_costs, jnp.inf))
    #     best_u = jax.lax.cond(
    #         any_finite,
    #         lambda idx: final_us[idx],
    #         lambda idx: U_original.reshape((N, dim_u)),
    #         best_idx
    #     )

    #     best_u = best_u.reshape((N, dim_u))

    #     # --- Roll control forward ---
    #     new_u = jnp.roll(best_u, -1, axis=0)
    #     # keep the last control as a repeat of second-last (safe fallback)
    #     new_u = new_u.at[-1].set(new_u[-2])
    #     new_U = new_u.reshape((1, -1)).squeeze(0)

    #     # return (matching original long tuple)
    #     return (
    #         best_u,
    #         new_u,
    #         new_U,
    #         jnp.nanmin(final_costs) if any_finite else jnp.inf,
    #         states_final_list[-1],
    #         jnp.zeros((self.system.N_DIMS,)),  # con_seq_one placeholder
    #         jnp.sum(jnp.isfinite(final_costs)),
    #         jnp.any(jnp.isfinite(final_costs)),
    #         states_final_list[-1],
    #         self.temperature,
    #         jnp.zeros((1, self.system.N_DIMS, 1)),  # safe_u_seqs placeholder
    #         states_final_list[-1],
    #         final_costs,
    #         states_final_list[-1],
    #     )
    # # --- Vanilla MPPI W - Block ----

    @jax.jit
    def mppi_mmodal_no_ais(self, state, U_original, U_anci, rng_key, q_ref, safe_zones, cost_map, origin, resolution, wh):
        num_anci = self.num_anci
        n_samples = self.n_samples
        N = self.N
        N_safe = self.N_safe

        # each_controller_n = n_samples
        each_controller_n = int(n_samples/(num_anci+1))
        original_seqs = U_original.reshape((-1,2))

        rng_keys = jax.random.split(rng_key, (num_anci+1)*(each_controller_n)).reshape((num_anci+1, each_controller_n,2))
        generate_u = jax.vmap(self.single_u_seq, (0,None, None))
        mini_guy = functools.partial(self.mini_mppi, cost_map=cost_map, origin=origin, resolution=resolution, wh=wh, safe_zones=safe_zones)
        sigma = self.sigma
        number_safe = 0

        U_total = jnp.vstack((U_original, U_anci)).reshape((num_anci+1, N*2))
        u_seqs = jax.vmap(generate_u, in_axes=(0,0, None))(rng_keys, U_total, sigma).reshape((-1,N,self.system.N_CONTROLS))
        all_costs_and_state = jax.vmap(self.eval_U_seq,(0, None, None, None, None, None,None,None,None)) \
                                (u_seqs, U_original, state, q_ref, safe_zones, cost_map, origin, resolution, wh)
        costs = all_costs_and_state[0]
        all_state_seq = all_costs_and_state[2]
        min_sz_dist = all_costs_and_state[3]
        finite_cost_ind = jnp.array(jnp.nonzero(jnp.isfinite(costs), size=n_samples, fill_value=0))
        collision_free_state_seq = jnp.take(all_state_seq, finite_cost_ind,axis=0).squeeze()
        mini_costs, all_con_state_seq, current_safe, safe_u_seqs = jax.vmap(mini_guy, in_axes=[0,0])(collision_free_state_seq[:, :N_safe,:], rng_keys.reshape((-1,2)))
        # print(safe_flags)
        # breakpoint()
        con_state_seq = all_con_state_seq[jnp.nonzero(mini_costs, size=n_samples, fill_value=100000), 0, :][0]
        number_safe += jnp.sum(mini_costs)
        safe_ind = jnp.take(finite_cost_ind,jnp.array(jnp.nonzero(mini_costs, size=n_samples, fill_value=1000009))).squeeze()
        min_cost = jnp.nanmin(jnp.take(costs, safe_ind, fill_value=np.inf))
        not_safe_ind = jnp.take(finite_cost_ind,jnp.array(jnp.nonzero(1-mini_costs, size=n_samples, fill_value=1000000))).squeeze()

        # uncomment to consider safety condition
        costs = costs.at[not_safe_ind].set(np.inf)
        # costs = jax.lax.cond(jnp.any(current_safe), lambda x: costs.at[not_safe_ind].set(np.inf), lambda x: min_sz_dist, 0)
        # costs = jax.lax.cond(jnp.any(jnp.isfinite(costs)), lambda x: costs, lambda x: min_sz_dist, 0)

        # costs =  costs.at[not_safe_ind].set(np.inf)
        # breakpoint()
        safe_state_seq = jnp.take(all_state_seq, safe_ind, axis=0)
        best_u, lowest_u, temperature = self.calculate_new_means(costs, u_seqs, original_seqs)
        new_u = jnp.roll(best_u, shift=-1, axis=0)
        new_u = new_u.at[-1].set(jnp.zeros_like(new_u[-2]))
        new_U = new_u.reshape((1,-1)).squeeze(0)
        return best_u, new_u, new_U, min_cost, safe_state_seq, con_state_seq, number_safe, jnp.any(current_safe), collision_free_state_seq, temperature, safe_u_seqs.squeeze(), all_con_state_seq, costs # u_applied, new control history, all control seqs, # # turn 

    def _tree_flatten(self):
        children = ( self.sigma, self.Q, self.QT, self.R, self.temperature,)
        aux_data = {'system': self.system, 
                    'num_anci': self.num_anci,
                    'n_samples': self.n_samples,
                    'n_mini':self.n_mini,
                    'N': self.N,
                    'N_mini': self.N_mini,
                    'N_safe': self.N_safe,
                    'max_sz': self.max_sz,
                    'tolerance': self.tolerance,
                    'footprint': self.footprint,
                    'occup_value': self.occup_value,
                    'alpha': self.alpha,
                    'heuristic_weight': self.heuristic_weight,
                    }
        return (children, aux_data)
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


from jax import tree_util 
tree_util.register_pytree_node(MPPI_Planner_Occup,
                               MPPI_Planner_Occup._tree_flatten,
                               MPPI_Planner_Occup._tree_unflatten)
print('Planners registered')

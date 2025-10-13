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

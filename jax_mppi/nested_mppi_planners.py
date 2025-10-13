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
    

    @jax.jit
    def mppi_mmodal(self, state, U_original, U_total, rng_key, q_ref, safe_zones, cost_map, origin, resolution, wh):
        num_modes  = self.num_anci + 1
        N          = self.N
        n_samples  = self.n_samples
        N_safe     = self.N_safe
        each_n     = n_samples // num_modes

        rng_keys_flat = jax.random.split(rng_key, n_samples)
        U_rep         = jnp.repeat(U_total, each_n, axis=0)

        u_flat        = jax.vmap(self.single_u_seq, in_axes=(0, 0, None))(rng_keys_flat, U_rep, self.sigma)

        costs_and_states = jax.vmap(self.eval_U_seq, in_axes=(0, None, None, None, None, None, None, None, None))(
            u_flat, U_original, state, q_ref, safe_zones, cost_map, origin, resolution, wh
        )
        costs         = costs_and_states[0]
        all_state_seq = costs_and_states[2]

        finite_inds    = jnp.array(jnp.nonzero(jnp.isfinite(costs), size=n_samples, fill_value=0))
        collision_free = jnp.take(all_state_seq, finite_inds, axis=0).squeeze()

        contingency_off = (self.N_safe <= 0) or (self.N_mini <= 0) or (self.n_mini <= 0)

        if contingency_off:
            batch        = collision_free.shape[0]
            mini_costs   = jnp.ones((batch,), dtype=jnp.bool_)
            current_safe = jnp.zeros((batch,), dtype=jnp.bool_)
            all_con_seq  = jnp.zeros((batch, 0, collision_free.shape[-1]))
            safe_u_seqs  = jnp.zeros((batch, 0, self.system.N_CONTROLS))

            safe_inds      = finite_inds.squeeze()
            number_safe    = jnp.sum(mini_costs)
            min_cost       = jnp.nanmin(jnp.take(costs, safe_inds, fill_value=jnp.inf))
            safe_state_seq = jnp.take(all_state_seq, safe_inds, axis=0)
            con_seq_one    = jnp.zeros((self.system.N_DIMS,))
        else:
        # if True: # Don't know why we had the contingency_off check?
            mini_fun = functools.partial(
                self.mini_mppi_ais,
                cost_map=cost_map,
                origin=origin,
                resolution=resolution,
                wh=wh,
                safe_zones=safe_zones,
            )
            mini_costs, all_con_seq, current_safe, safe_u_seqs = jax.vmap(mini_fun, in_axes=(0, 0))(
                collision_free[:, :N_safe, :],
                rng_keys_flat,
            )

            number_safe = jnp.sum(mini_costs)

            safe_inds = jnp.take(
                finite_inds,
                jnp.array(jnp.nonzero(mini_costs, size=n_samples, fill_value=0))
            ).squeeze()
            min_cost = jnp.nanmin(jnp.take(costs, safe_inds, fill_value=jnp.inf))

            not_safe_inds = jnp.take(
                finite_inds,
                jnp.array(jnp.nonzero(1 - mini_costs, size=n_samples, fill_value=0))
            ).squeeze()
            costs = costs.at[not_safe_inds].set(jnp.inf)

            safe_state_seq = jnp.take(all_state_seq, safe_inds, axis=0)
            con_seq_one    = all_con_seq[jnp.nonzero(mini_costs, size=n_samples, fill_value=0), 0, :][0]

        def do_AIS(costs, seqs):
            idxs = jnp.argsort(costs)[:10]
            elite = seqs[idxs]
            mu = jnp.mean(elite, axis=0)
            cov = jnp.cov(elite, rowvar=False) + jnp.eye(N * dim_u) * 1e-8
            return mu, cov

        best_u, lowest_u, temperature = self.calculate_new_means(costs, u_flat, U_original.reshape((-1, 2)))

        for i in range(ais):
            bests = jnp.argsort(costs)
            mu, cov = do_AIS(costs, u_flat.reshape((n_samples, -1)))
            U_i = self.calculate_new_means(costs, u_flat, U_i.reshape((N, -1))).reshape(-1)
            sigma_i = cov
            elite_idxs = bests[:m_elite]
            elite_states = states[elite_idxs]
            state_seqs_all = state_seqs_all.at[i].set(elite_states)
            state_seqs_means = state_seqs_means.at[i].set(state_rollouts(U_i.reshape((N_mini, dim_u))))
        best_u = lowest_u
        new_u  = jnp.roll(best_u, -1, axis=0)
        new_u  = new_u.at[-1].set(new_u[-2])
        new_U  = new_u.reshape((1, -1)).squeeze(0)

        return (
            best_u,
            new_u,
            new_U,
            min_cost,
            safe_state_seq,
            con_seq_one,
            number_safe,
            jnp.any(current_safe),
            collision_free,
            temperature,
            safe_u_seqs.squeeze(),
            all_con_seq,
            costs,
            all_state_seq,
        )
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

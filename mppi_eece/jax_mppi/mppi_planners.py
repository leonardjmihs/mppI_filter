import jax
import jax.numpy as jnp
import numpy as np
from mppi_eece.jax_mppi.collision_checker import CollisionChecker 
import functools

class MPPI_Planner_Occup:
    def __init__(self, sigma, Q, QT, R, temperature, system, num_anci, n_samples, N, tolerance, occup_value=[100], alpha=1):
        self.system = system
        self.num_anci = num_anci
        self.n_samples = n_samples
        self.N = N
        self.sigma = sigma
        self.Q = Q
        self.QT = QT
        self.R = R
        self.temperature = temperature
        self.tolerance = tolerance
        self.occup_value = occup_value
        self.alpha = alpha

    def eval_U_seq(self, u_seq: jnp.ndarray, 
            original_u: jnp.ndarray, 
            state: jnp.ndarray, 
            q_ref: jnp.ndarray, 
            cost_map: CollisionChecker):
        
        cost_and_term, (state_seq, min_sz_dist) = jax.lax.scan(self.single_sample_running_cost, 
                                            (1.0, state, q_ref, cost_map), 
                                            u_seq)
        cost = cost_and_term[0]
        terminal_state = cost_and_term[1]
        terminal_cost = jnp.dot((terminal_state - q_ref), jnp.dot(self.QT, (terminal_state - q_ref)))
        cost += terminal_cost 
        cost += u_seq.ravel().T @ jnp.diag(1.0 / jnp.diag(self.sigma)) @ (u_seq.ravel().T - original_u) * (1-self.alpha) * (self.temperature)
        
        return (cost, terminal_state, state_seq, jnp.sum(min_sz_dist))

    def single_sample_running_cost(self, carry, params):
        '''
        Input:
            carry: 
            (
                cost:           float, 
                sim_state:      jnp.ndarray, 
                q_ref:          jnp.ndarray, 
                cost_map:       CollisionChecker, 
            )
            param: 
                u:              jnp.ndarray
        Output:
            new_carry: 
            (
                cost:           float, 
                sim_state:      jnp.ndarray, 
                q_ref:          jnp.ndarray, 
                cost_map:       CollisionChecker,
            )
            res:
            (
                sim_state:      jnp.ndarray,
                min_dist:       float (unused)
            )
        '''
        u = params
        cost, sim_state, q_ref, cost_map = carry
        new_state = self.system.dynamics_jax(sim_state, u, self.system.dt, self.system.nominal_params)
        dist = sim_state - q_ref
        # dx = jnp.dot(new_state - sim_state, jnp.dot(self.Q, new_state - sim_state))
        new_cost = cost + jnp.dot(dist, jnp.dot(self.Q, dist))

        val = cost_map.get_value(new_state[:2])
        occupied_now = jnp.logical_or(jnp.isinf(val), (val >= cost_map.occup_value))
        new_cost += jax.lax.cond(occupied_now, lambda _: jnp.inf, lambda _: 0.0, operand=None)

        path_blocked = cost_map.dda_path_check(sim_state[:2], new_state[:2])
        new_cost += jax.lax.cond(path_blocked, lambda _: 10000.0, lambda _: 0.0, operand=None)
        min_dist = 0.0

        return (new_cost, new_state, q_ref, cost_map), (sim_state, min_dist)

    def single_u_seq(self, rng_subkey, U, sigma):
        noise_scaled = jax.random.normal(rng_subkey, shape=(self.N*2,)) * jnp.diagonal(sigma)
        u_seq = (U + noise_scaled).reshape((-1, 2))
        u_seq = jnp.clip(u_seq, self.system.control_bounds[0], self.system.control_bounds[1])
        return u_seq 

    def ess(self, costs, temperature):
        w_i = jnp.exp(1/temperature*(jnp.nanmin(costs)-costs))
        ess = jnp.nansum(w_i)**2 / jnp.nansum(w_i**2)
        return ess

    def calculate_new_means(self, costs, seq, original_seq):
        lowest_ind = jnp.nanargmin(costs)
        temperature = self.temperature
        exp_cost = jnp.exp(1/temperature*(jnp.nanmin(costs)-costs))
        denom = jnp.nansum(exp_cost) + 1e-7
        best_u = original_seq + jnp.nansum(exp_cost[..., None, None] * (seq-original_seq), axis=0) / denom

        lowest_u = seq[lowest_ind]
        return best_u, lowest_u, temperature

    @jax.jit
    def mppi_mmodal(self, state, U_original, U_total, rng_key, q_ref, cost_map):
        num_modes = self.num_anci + 1
        N = self.N
        n_samples = self.n_samples
        each_n = n_samples // num_modes

        U_rep = jnp.repeat(U_total, each_n, axis=0)
        rng_keys_flat = jax.random.split(rng_key, each_n*num_modes)
        u_flat = jax.vmap(self.single_u_seq, in_axes=(0, 0, None))(rng_keys_flat, U_rep, self.sigma)

        costs_and_states = jax.vmap(self.eval_U_seq, in_axes=(0, None, None, None, None))(
            u_flat, U_original, state, q_ref, cost_map)
        costs = costs_and_states[0]
        all_state_seq = costs_and_states[2]
        finite_inds = jnp.array(jnp.nonzero(jnp.isfinite(costs), size=each_n*num_modes, fill_value=0))
        collision_free = jnp.take(all_state_seq, finite_inds, axis=0).squeeze()
        batch = collision_free.shape[0]
        safe_inds = finite_inds.squeeze()
        min_cost = jnp.nanmin(jnp.take(costs, safe_inds, fill_value=jnp.inf))
      
        best_u, lowest_u, temperature = self.calculate_new_means(costs, u_flat, U_original.reshape((-1, 2)))
        best_u = jax.lax.cond(
            jnp.isinf(min_cost),
            lambda x: best_u,
            lambda x: lowest_u,
            0,
        )
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
        )

    def _tree_flatten(self):
        children = (self.sigma, self.Q, self.QT, self.R, self.temperature,)
        aux_data = {
            'system': self.system, 
            'num_anci': self.num_anci,
            'n_samples': self.n_samples,
            'N': self.N,
            'tolerance': self.tolerance,
            'occup_value': self.occup_value,
            'alpha': self.alpha,
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
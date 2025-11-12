import jax
import jax.numpy as jnp
import numpy as np
import functools


def state_rollouts(system, start, u_seq):
    '''
        Get Reachability thorugh Uniform Sampling on Control bounds
        K_mini: Safety Sampling number
        N_mini: Safety Horizion timesteps
    '''
    def mini_cost(carry, params):
        u = params
        sim_state = carry
        new_state = system.jax_dynamics(sim_state, u, 0, system.dt, system.nominal_params)
        return (new_state), (sim_state)

    term_state, state_hist = jax.lax.scan(mini_cost, start, u_seq)
    return state_hist

    
@functools.partial(jax.jit, static_argnames=('K_mini','N_mini','ais','occup_value'))
def get_reachability_mppi_costmap(
    system, cost_map, origin, resolution, wh,
    start, goals, footprint, rng_key,
    K_mini, N_mini, sigma, U_init,
    temperature=1.0, ais=3, occup_value=100
):
    m_elite = int((K_mini // ais) * 0.1)
    p = 10
    dim_u = system.control_bounds[0].shape[0]

    def mini_step(sim_state, u):
        new_state = system.dynamics_jax(sim_state, u, system.dt, system.nominal_params)
        pos = footprint + sim_state[:2]
        ind = jnp.floor((pos - origin) / resolution)
        ind = jnp.clip(ind, 0, wh).astype(jnp.int32)
        breakpoint()
        barrier = cost_map[ind[:, 0], ind[:, 1]]
        cbf_ok = jnp.all(barrier <= occup_value)
        centres = goals[:, :2]
        radii = goals[:, 2]
        dvecs = sim_state[:2] - centres
        dists = jnp.linalg.norm(dvecs, axis=1) - radii
        safe = jnp.min(jnp.maximum(dists, 0.0))
        return new_state, (sim_state, safe, cbf_ok, dists)

    def eval_seq(u_seq):
        _, aux = jax.lax.scan(lambda s, u: mini_step(s, u), start, u_seq)
        sim_states, safes, cbfs, dists = aux
        rollout_cost = jnp.min(dists)
        cost = jax.lax.cond(jnp.all(cbfs), lambda x: x, lambda x: jnp.inf + x, safes)
        rollout_cost = jax.lax.cond(jnp.all(cbfs), lambda x: x, lambda x: jnp.inf + x, rollout_cost)
        return cost, sim_states, rollout_cost

    def single_u_seq_uniform(rng):
        alpha, beta = 0.7, 0.7
        return (
            jax.random.beta(rng, alpha, beta, (N_mini, dim_u))
            * (system.control_bounds[1] - system.control_bounds[0])
            + system.control_bounds[0]
        )

    def single_u_seq_normal(rng, U, sigma_mat):
        noise = jax.random.normal(rng, (N_mini * dim_u,)) * jnp.diag(sigma_mat)
        u = (U + noise).reshape((N_mini, dim_u))
        return jnp.clip(u, system.control_bounds[0], system.control_bounds[1])

    def state_rollouts(U_seq):
        def step(s, u):
            ns = system.dynamics_jax(s, u, system.dt, system.nominal_params)
            return ns, ns
        _, states = jax.lax.scan(step, start, U_seq)
        return states

    def do_AIS(costs, seqs):
        idxs = jnp.argsort(costs)[:m_elite]
        elite = seqs[idxs]
        mu = jnp.mean(elite, axis=0)
        cov = jnp.cov(elite, rowvar=False) + jnp.eye(N_mini * dim_u) * 1e-8
        return mu, cov

    def calc_means(costs, seqs, orig):
        w = jnp.exp((jnp.min(costs) - costs) / temperature)
        w = w / (jnp.sum(w) + 1e-7)
        return orig + jnp.tensordot(w, seqs - orig, axes=1)

    keys = jax.random.split(rng_key, K_mini // ais)
    min_cost = jnp.inf
    U_i = U_init
    sigma_i = sigma
    state_seqs_all = jnp.zeros((ais, m_elite, N_mini, system.N_DIMS))
    state_seqs_means = jnp.zeros((ais, N_mini, system.N_DIMS))
    safe_state = jnp.zeros((N_mini, system.N_DIMS))
    safe_u = jnp.zeros((N_mini, dim_u))

    for i in range(ais):
        if i == 0:
            u_seqs = jax.vmap(single_u_seq_uniform)(keys)
        else:
            u_seqs = jax.vmap(single_u_seq_normal, in_axes=(0, None, None))(keys, U_i, sigma_i)
        costs, states, roll_costs = jax.vmap(eval_seq)(u_seqs)
        bests = jnp.argsort(roll_costs)
        idx = bests[0]
        best_cost = roll_costs[idx]
        safe_state = jax.lax.select(best_cost < min_cost, states[idx], safe_state)
        safe_u = jax.lax.select(best_cost < min_cost, u_seqs[idx], safe_u)
        min_cost = jnp.minimum(min_cost, best_cost)
        mu, cov = do_AIS(roll_costs, u_seqs.reshape((-1, N_mini * dim_u)))
        U_i = calc_means(roll_costs, u_seqs, U_i.reshape((N_mini, dim_u))).reshape(-1)
        sigma_i = cov
        elite_idxs = bests[:m_elite]
        elite_states = states[elite_idxs]
        state_seqs_all = state_seqs_all.at[i].set(elite_states)
        state_seqs_means = state_seqs_means.at[i].set(state_rollouts(U_i.reshape((N_mini, dim_u))))

    return min_cost, safe_state, state_seqs_all, state_seqs_means, safe_u


def get_reachability_controls(system, environment, start, goals, ns, rng_key, K_mini, N_mini):
    '''
        Get Reachability thorugh Uniform Sampling on Control bounds
        K_mini: Safety Sampling number
        N_mini: Safety Horizion timesteps
    '''
    def mini_cost(carry, params):
        dcbf_alpha=0.01
        u = params
        sim_state = carry
        new_state = system.jax_dynamics(sim_state, u, 0, system.dt, system.nominal_params)
    
        barrier_value = jax.vmap(system.h_x, in_axes=(None,0))(sim_state, environment)
        next_barrier_value =  jax.vmap(system.h_x, in_axes=(None,0))(new_state, environment)
        cbf_cost = -next_barrier_value + dcbf_alpha * barrier_value
        cbf_cost = jnp.max(cbf_cost) <= 0
        norm_dist = jnp.linalg.norm(sim_state[0:2] - goals[:,0:2], axis=1)
        safe = jnp.min(norm_dist)

        return (new_state), (sim_state, safe, cbf_cost)
    
    def cond_fun(carry):
        i, safe, u_seq = carry 
        break_condition = (safe<0.5)
        return ~break_condition & (i < N_mini)
    def body_fun(carry):
        i, safe, u_seq = carry


    def mini_rollout(state, rng_key):
        # Given a state and rollout and determines if rollout is safe
        u_seq = jax.random.uniform(rng_key, (N_mini,2), minval=system.control_bounds[0], maxval=system.control_bounds[1])
        term_state, state_and_safe = jax.lax.scan(mini_cost, state, u_seq)
        state_hist = state_and_safe[0]
        safe_hist = state_and_safe[1]
        cbf_hist = state_and_safe[2]
        safe = jnp.any(safe_hist<0.5)
        cbf_safe = jnp.all(cbf_hist)
        return (jnp.logical_and(safe,cbf_safe), state_hist, u_seq)
    rng_keys = jax.random.split(rng_key, K_mini)
    safe_controls, state_hist, u_seq = jax.vmap(mini_rollout, in_axes=(None, 0))(start.copy(), rng_keys)

    safe_ind = jnp.array(jnp.nonzero(safe_controls, size=1))
    safe_u_seq = jnp.take(u_seq, safe_ind, axis=0) 
    return jnp.any(safe_controls), safe_u_seq

def get_reachable_set(system, environment, start, rng_key, K_mini, N_mini):
    def mini_cost(carry, params):
        dcbf_alpha=0.01
        u = params
        sim_state = carry
        new_state = system.jax_dynamics(sim_state, u, 0, system.dt, system.nominal_params)
    
        barrier_value = jax.vmap(system.h_x, in_axes=(None,0))(sim_state, environment)
        next_barrier_value =  jax.vmap(system.h_x, in_axes=(None,0))(new_state, environment)
        cbf_cost = -next_barrier_value + dcbf_alpha * barrier_value
        cbf_cost = jnp.max(cbf_cost) <= 0

        return (new_state), (sim_state, cbf_cost)
    def mini_rollout(state, rng_key):
            # Given a state and rollout and determines if rollout is safe
            u_seq = jax.random.uniform(rng_key, (N_mini,2), minval=system.control_bounds[0], maxval=system.control_bounds[1])
            term_state, state_and_safe = jax.lax.scan(mini_cost, state, u_seq)
            state_hist = state_and_safe[0]
            cbf_hist = state_and_safe[1]
            cbf_safe = jnp.all(cbf_hist)
            return (cbf_safe, state_hist)

    rng_keys = jax.random.split(rng_key, K_mini)
    safe_controls, state_hist = jax.vmap(mini_rollout, in_axes=(None, 0))(start.copy(), rng_keys)
    safe_ind = jnp.array(jnp.nonzero(safe_controls))
    safe_state_seq = jnp.array(jnp.take(state_hist, safe_ind, axis=0))
    return safe_state_seq.squeeze()
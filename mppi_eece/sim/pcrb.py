import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from functools import partial


class PCRBController:
    """
    Posterior Cramér-Rao Bound (PCRB) computation for control-by-filtering.
    
    State equations:
        θ_k = [u_k, u_{k+1}, ..., u_{k+T}]  (control sequence)
        θ_{k+1} = E(θ_k) + w_k,  w_k ~ N(0, Q)
        y_k = -C(θ_k) + v_k,  v_k ~ N(0, σ²)
    
    where E(•) is the shift operator and C(•) is the trajectory cost.
    """
    
    def __init__(
        self,
        Nt: int,
        nu: int,
        Q: np.ndarray,
        sigma_r: float,
        n_samples: int = 100
    ):
        """
        Args:
            Nt: Horizon length
            nu: Control dimension (e.g., 2 for [steering, velocity])
            Q: Process noise covariance (Nt*nu × Nt*nu)
            sigma_r: Measurement noise std
            n_samples: Number of Monte Carlo samples for gradient expectation
        """
        self.Nt = Nt
        self.nu = nu
        self.dim = Nt * nu
        self.Q = jnp.array(Q)
        self.Q_inv = jnp.linalg.inv(self.Q)
        self.sigma_r = sigma_r
        self.sigma_r_sq = sigma_r ** 2
        self.n_samples = n_samples
        
        # Build shift operator matrix F
        self.F = self._build_shift_matrix()
        self.F_T = self.F.T
        
        # Precompute D11 (constant for linear dynamics)
        self.D11 = self.F_T @ self.Q_inv @ self.F
        
        # Precompute D12 (constant for linear dynamics)
        self.D12 = -self.F_T @ self.Q_inv
        
    def _build_shift_matrix(self) -> jnp.ndarray:
        """
        Build the shift operator matrix F for θ_{k+1} = F θ_k
        
        F shifts controls forward and zeros out the last control:
        [u_k, u_{k+1}, ..., u_{k+T}] -> [u_{k+1}, u_{k+2}, ..., u_{k+T}, 0]
        """
        # For control sequences, F is block-structured
        F = jnp.zeros((self.dim, self.dim))
        
        # Shift each control block forward
        for i in range(self.Nt - 1):
            start_i = i * self.nu
            end_i = (i + 1) * self.nu
            start_j = (i + 1) * self.nu
            end_j = (i + 2) * self.nu
            F = F.at[start_i:end_i, start_j:end_j].set(jnp.eye(self.nu))
        
        return F
    
    def compute_cost_gradient_finite_diff(
        self,
        theta: jnp.ndarray,
        x0: jnp.ndarray,
        x_ref: jnp.ndarray,
        system,
        planner,
        epsilon: float = 1e-4
    ) -> jnp.ndarray:
        """
        Compute gradient of trajectory cost C(θ) w.r.t. control sequence θ
        using finite differences (for non-differentiable costs).
        
        Args:
            theta: Control sequence [Nt*nu]
            x0: Initial state
            x_ref: Reference state
            system: System with dynamics_jax method
            planner: Planner with eval_U_seq method
            epsilon: Finite difference step size
            
        Returns:
            g: Cost gradient [Nt*nu]
        """
        theta_np = np.array(theta)
        U = theta_np.reshape((self.Nt, self.nu))
        
        # Compute baseline cost using planner.eval_U_seq
        cost_baseline, _ = planner.eval_U_seq(U, x0, x_ref)
        
        # Compute finite difference gradient
        gradient = np.zeros_like(theta_np)
        
        for i in range(len(theta_np)):
            # Perturb control sequence
            theta_perturbed = theta_np.copy()
            theta_perturbed[i] += epsilon
            U_perturbed = theta_perturbed.reshape((self.Nt, self.nu))
            
            # Evaluate perturbed cost
            cost_perturbed, _ = planner.eval_U_seq(U_perturbed, x0, x_ref)
            
            # Finite difference approximation
            gradient[i] = (cost_perturbed - cost_baseline) / epsilon
        
        return jnp.array(gradient)
    
    def compute_D22_monte_carlo_finite_diff(
        self,
        theta_mean: jnp.ndarray,
        x0: jnp.ndarray,
        x_ref: jnp.ndarray,
        system,
        planner,
        key: jax.random.PRNGKey,
        epsilon: float = 1e-4
    ) -> jnp.ndarray:
        """
        Compute D22_k = (1/σ²) E[g_k g_k^T] via Monte Carlo sampling
        using finite differences for non-differentiable costs.
        
        Samples from p(θ_k | y_{1:k}) ≈ N(θ_mean, P_k) and computes
        gradient outer products using finite differences.
        
        Note: P_k is obtained from the instance variable self.P_k_current
        which should be set before calling this method.
        """
        # Sample control sequences from current distribution
        theta_mean_np = np.array(theta_mean)
        
        # Use current covariance from instance variable
        P_k_np = np.array(self.P_k_current) if hasattr(self, 'P_k_current') else np.eye(self.dim) * 0.1
        
        # Ensure P_k is positive definite
        eigvals = np.linalg.eigvalsh(P_k_np)
        if np.min(eigvals) < 1e-8:
            P_k_np = P_k_np + np.eye(len(P_k_np)) * 1e-6
        
        samples = np.random.multivariate_normal(
            theta_mean_np, P_k_np, size=self.n_samples
        )
        
        gradient_outer_products = []
        
        for sample in samples:
            # Compute gradient using finite differences
            g = self.compute_cost_gradient_finite_diff(
                jnp.array(sample), x0, x_ref, system, planner, epsilon
            )
            gradient_outer_products.append(np.outer(g, g))
        
        # Average over samples
        E_gg_T = np.mean(gradient_outer_products, axis=0)
        
        D22 = jnp.array(E_gg_T) / self.sigma_r_sq
        return D22
    
    def set_current_covariance(self, P_k: np.ndarray):
        """Set the current covariance for sampling."""
        self.P_k_current = P_k
    
    @partial(jax.jit, static_argnums=(0,))
    def pcrb_update(
        self,
        J_k: jnp.ndarray,
        D22_k: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Recursive PCRB update:
        J_{k+1} = D22_k - D12_k (J_k + D11_k)^{-1} (D12_k)^T
        
        Args:
            J_k: Current Fisher Information Matrix [dim × dim]
            D22_k: Measurement information matrix [dim × dim]
            
        Returns:
            J_{k+1}: Updated Fisher Information Matrix [dim × dim]
        """
        # J_k + D11_k
        sum_term = J_k + self.D11
        
        # Solve linear system instead of explicit inverse for stability
        # (J_k + D11_k)^{-1} (D12_k)^T
        try:
            solve_term = jnp.linalg.solve(sum_term, self.D12.T)
        except:
            # Fallback to pseudo-inverse if singular
            solve_term = jnp.linalg.pinv(sum_term) @ self.D12.T
        
        # D12_k (J_k + D11_k)^{-1} (D12_k)^T
        correction = self.D12 @ solve_term
        
        # J_{k+1} = D22_k - correction
        J_next = D22_k - correction
        
        return J_next
    
    def compute_pcrb_bound(self, J: jnp.ndarray) -> jnp.ndarray:
        """
        Compute PCRB from Fisher Information Matrix.
        PCRB = trace(J^{-1})
        
        Returns lower bound on estimation error variance.
        """
        try:
            J_inv = jnp.linalg.inv(J)
        except:
            J_inv = jnp.linalg.pinv(J)
        
        return jnp.trace(J_inv)


def visualize_pcrb_evolution(
    pcrb_history: np.ndarray,
    ukf_trace_history: np.ndarray,
    timesteps: np.ndarray,
    save_path: str = None
):
    """
    Plot PCRB vs UKF covariance trace over time.
    
    Args:
        pcrb_history: PCRB values at each timestep
        ukf_trace_history: trace(P_k) from UKF at each timestep
        timesteps: Time array
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: PCRB vs UKF covariance trace
    ax[0].plot(timesteps, pcrb_history, 'b-', linewidth=2, label='PCRB (Lower Bound)')
    ax[0].plot(timesteps, ukf_trace_history, 'r--', linewidth=2, label='UKF trace(P)')
    ax[0].fill_between(timesteps, pcrb_history, ukf_trace_history, 
                        alpha=0.3, color='gray', label='Estimation Gap')
    ax[0].set_ylabel('Estimation Error Variance', fontsize=12)
    ax[0].set_title('PCRB vs UKF Covariance Trace', fontsize=14, fontweight='bold')
    ax[0].legend(fontsize=10)
    ax[0].grid(True, alpha=0.3)
    ax[0].set_yscale('log')
    
    # Plot 2: Efficiency ratio
    efficiency = pcrb_history / (ukf_trace_history + 1e-10)
    ax[1].plot(timesteps, efficiency, 'g-', linewidth=2)
    ax[1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect Efficiency')
    ax[1].set_ylabel('Efficiency (PCRB / trace(P))', fontsize=12)
    ax[1].set_xlabel('Time [s]', fontsize=12)
    ax[1].set_title('Filter Efficiency', fontsize=14, fontweight='bold')
    ax[1].legend(fontsize=10)
    ax[1].grid(True, alpha=0.3)
    ax[1].set_ylim([0, 1.2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_fim_eigenvalues(
    J_history: list,
    timesteps: np.ndarray,
    save_path: str = None
):
    """
    Plot evolution of Fisher Information Matrix eigenvalues.
    
    Shows information gain in different directions of control space.
    """
    eigenvalues_history = []
    for J in J_history:
        eigvals = np.linalg.eigvalsh(J)
        eigenvalues_history.append(eigvals)
    
    eigenvalues_history = np.array(eigenvalues_history)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot top 5 eigenvalues
    n_plot = min(5, eigenvalues_history.shape[1])
    for i in range(n_plot):
        ax.plot(timesteps, eigenvalues_history[:, -(i+1)], 
                label=f'λ_{i+1} (largest)', linewidth=2)
    
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title('Fisher Information Matrix Eigenvalues', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# Example usage functions
def example_dynamics(x, u):
    """Example unicycle dynamics."""
    dt = 0.2
    theta = x[2]
    v = u[1]
    omega = u[0]
    
    x_next = jnp.array([
        x[0] + v * jnp.cos(theta) * dt,
        x[1] + v * jnp.sin(theta) * dt,
        x[2] + omega * dt
    ])
    return x_next


def example_cost(x, u, x_ref):
    """Example quadratic cost."""
    Q = jnp.diag(jnp.array([1.0, 1.0, 0.0]))
    R = jnp.diag(jnp.array([0.1, 0.1]))
    
    state_error = x - x_ref
    cost = state_error.T @ Q @ state_error + u.T @ R @ u
    return cost
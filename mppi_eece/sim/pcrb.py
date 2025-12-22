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
        n_samples: int = 1000
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
        self.D12 = self.F_T @ self.Q_inv
        
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
    
    def compute_D22_vmap(
        self,
        mu: jnp.ndarray,
        P: jnp.ndarray,
        x_current: jnp.ndarray,
        q_ref: jnp.ndarray,
        mppi_planner,
        collision_checker,
        epsilon: float = 1e-4
    ) -> Tuple[jnp.ndarray, list]:
        """
        Compute D22 = (1/σ²) E[g g^T] via Monte Carlo sampling with JAX vmap.
        
        Args:
            mu: Mean control sequence [Nt*nu]
            P: Covariance matrix [Nt*nu × Nt*nu]
            x_current: Current state
            q_ref: Reference state
            mppi_planner: MPPI planner for cost evaluation
            collision_checker: Collision checker
            epsilon: Finite difference step size
            
        Returns:
            D22: Measurement information matrix [Nt*nu × Nt*nu]
            grad_norms: List of gradient norms for diagnostics
        """
        # Ensure positive definite covariance
        mu_np = np.array(mu).flatten()
        P_np = np.array(P)
        
        eigvals = np.linalg.eigvalsh(P_np)
        if np.min(eigvals) < 1e-8:
            P_np = P_np + np.eye(len(P_np)) * 1e-6
        
        # Sample control sequences
        samples = np.random.multivariate_normal(mu_np, P_np, size=self.n_samples)
        
        # Define gradient computation for a single sample using JAX vmap
        def compute_gradient_for_sample(sample):
            """Compute gradient for a single sampled control sequence."""
            # Compute baseline cost
            U_baseline = sample.reshape((self.Nt, self.nu))
            outputs = mppi_planner.eval_U_seq(
                U_baseline, sample, x_current, q_ref, collision_checker
            )
            cost_baseline = outputs[0]
            cost_baseline = jnp.clip(cost_baseline, 0, 1e10)
            
            # Compute gradient via finite differences (vmapped over dimensions)
            def compute_partial_derivative(i):
                sample_perturbed = sample.at[i].add(epsilon)
                U_perturbed = sample_perturbed.reshape((self.Nt, self.nu))
                
                outputs = mppi_planner.eval_U_seq(
                    U_perturbed, sample_perturbed, x_current, q_ref, collision_checker
                )
                cost_perturbed = outputs[0]
                
                return (cost_perturbed - cost_baseline) / epsilon
            
            # Vmap over all control dimensions
            gradient = jax.vmap(compute_partial_derivative)(jnp.arange(len(sample)))
            return gradient, jnp.outer(gradient, gradient), jnp.linalg.norm(gradient)
        
        # Vmap over all samples
        gradients, outer_products, norms = jax.vmap(compute_gradient_for_sample)(samples)
        
        # Convert to numpy and compute mean
        gradient_outer_products = [np.array(op) for op in outer_products]
        grad_norms = [float(n) for n in norms]
        
        # Compute D22 = (1/σ²) E[g g^T]
        E_gg_T = np.mean(gradient_outer_products, axis=0)
        D22 = jnp.array(E_gg_T) / self.sigma_r_sq
        
        return D22, grad_norms
    
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
        tmp = J_k + jnp.linalg.inv(self.Q)
        middle = jnp.linalg.inv(tmp)
        J_next = D22_k + self.F.T @ middle @ self.F

        # # J_k + D11_k
        # sum_term = J_k + self.D11
        
        # # Solve linear system instead of explicit inverse for stability
        # # (J_k + D11_k)^{-1} (D12_k)^T
        # try:
        #     solve_term = jnp.linalg.solve(sum_term, self.D12.T)
        # except:
        #     # Fallback to pseudo-inverse if singular
        #     solve_term = jnp.linalg.pinv(sum_term) @ self.D12.T
        
        # # D12_k (J_k + D11_k)^{-1} (D12_k)^T
        # correction = self.D12 @ solve_term
        
        # # J_{k+1} = D22_k - correction
        # J_next = D22_k - correction

        # # Project onto PSD cone (hacky solution as this seems to go negative sometimes)
        # eigvals, eigvecs = jnp.linalg.eigh(J_next)
        # eigvals = jnp.maximum(eigvals, 1e-5)  # Enforce positive eigenvalues
        # J_next = eigvecs @ jnp.diag(eigvals) @ eigvecs.T
        return J_next
    
    def pcrb_step(
        self,
        J_k: jnp.ndarray,
        mu: jnp.ndarray,
        P: jnp.ndarray, # only used for sampling for gradient calculation 
        x_current: jnp.ndarray,
        q_ref: jnp.ndarray,
        mppi_planner,
        collision_checker,
        epsilon: float = 1e-4
    ) -> Tuple[jnp.ndarray, float, list]:
        """
        Complete PCRB step: compute D22 and update Fisher Information Matrix.
        
        Args:
            J_k: Current Fisher Information Matrix [dim × dim]
            mu: Mean control sequence [Nt*nu]
            P: Covariance matrix [Nt*nu × Nt*nu]
            x_current: Current state
            q_ref: Reference state
            mppi_planner: MPPI planner for cost evaluation
            collision_checker: Collision checker
            epsilon: Finite difference step size
            
        Returns:
            J_next: Updated Fisher Information Matrix [dim × dim]
            pcrb_bound: PCRB bound (trace of J_next^{-1})
            grad_norms: List of gradient norms for diagnostics
        """
        # Set current covariance for potential other uses
        
        # Compute D22 with gradient computation
        D22, grad_norms = self.compute_D22_vmap(
            mu, P, x_current, q_ref, mppi_planner, collision_checker, epsilon
        )
        
        # Update Fisher Information Matrix
        J_next = self.pcrb_update(J_k, D22)
        
        # Compute PCRB bound
        pcrb_bound = float(self.compute_pcrb_bound(J_next))
        # if pcrb_bound < 0:
        #     # pcrb_bound = float('inf')
        #     breakpoint()
        
        return J_next, pcrb_bound, grad_norms
    
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


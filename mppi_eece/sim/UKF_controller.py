import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mppi_eece.jax_mppi import plot_utils
from tqdm import tqdm
import time
import functools
from mppi_eece.systems import Unicycle
from mppi_eece.jax_mppi.grid import OccupGrid
from mppi_eece.jax_mppi.topo_prm import TopoPRM
from mppi_eece.jax_mppi.ca_mpc import *
import cProfile
import datetime
import casadi as ca
import os
import json
import copy 
from mppi_eece.jax_mppi.mppi_planners import MPPI_Planner_Occup
from mppi_eece.jax_mppi.collision_checker import CollisionChecker
from mppi_eece.jax_mppi.do_mpc import find_mpc, gen_and_save_mpc_results, do_mpc

class UKF_Controller:
    def __init__(self, system, mppi_planner, alpha=1e-3, beta=2, kappa=0):
        """
        UKF for control sequence estimation (controls-only formulation)
        
        Args:
            system: Your Unicycle system
            mppi_planner: Your MPPI_Planner_Occup instance (for cost evaluation)
            alpha, beta, kappa: UKF tuning parameters
        """
        self.system = system
        self.planner = mppi_planner
        self.N = mppi_planner.N  # Horizon length
        self.n_u = 2  # control dimension per timestep
        self.n_theta = self.N * self.n_u  # total control sequence dimension
        
        # UKF parameters
        self.n_sigma = 2 * self.n_theta + 1
        self.lambda_ = alpha**2 * (self.n_theta + kappa) - self.n_theta
        
        # Weights
        self.Wm = jnp.zeros(self.n_sigma)
        self.Wc = jnp.zeros(self.n_sigma)
        
        self.Wm = self.Wm.at[0].set(self.lambda_ / (self.n_theta + self.lambda_))
        self.Wc = self.Wc.at[0].set(self.lambda_ / (self.n_theta + self.lambda_) + 
                                     (1 - alpha**2 + beta))
        
        for i in range(1, self.n_sigma):
            w = 1 / (2 * (self.n_theta + self.lambda_))
            self.Wm = self.Wm.at[i].set(w)
            self.Wc = self.Wc.at[i].set(w)
    
    def shift_controls(self, theta):
        """
        Shift operator: Phi(theta)
        Moves controls forward, duplicates last control
        
        Args:
            theta: [2N] control sequence
        Returns:
            theta_new: [2N] shifted control sequence
        """
        theta_reshaped = theta.reshape((-1, self.n_u))
        theta_shifted = jnp.roll(theta_reshaped, -1, axis=0)
        # Keep last control
        theta_shifted = theta_shifted.at[-1].set(theta_reshaped[-1])
        return theta_shifted.ravel()
    
    def generate_sigma_points(self, mu, P):
        """
        Generate sigma points using Cholesky decomposition
        
        Args:
            mu: mean [n_theta]
            P: covariance [n_theta, n_theta]
        Returns:
            sigma_points: [n_sigma, n_theta]
        """
        n = self.n_theta
        
        # Regularization for numerical stability
        P_reg = P + jnp.eye(n) * 1e-9
        
        try:
            L = jnp.linalg.cholesky(P_reg)
        except:
            # Fallback to eigendecomposition
            eigvals, eigvecs = jnp.linalg.eigh(P_reg)
            eigvals = jnp.maximum(eigvals, 1e-9)
            L = eigvecs @ jnp.diag(jnp.sqrt(eigvals))
        
        L_scaled = jnp.sqrt(n + self.lambda_) * L
        
        # Generate sigma points
        sigma_points = jnp.zeros((self.n_sigma, n))
        sigma_points = sigma_points.at[0].set(mu)
        
        for i in range(n):
            sigma_points = sigma_points.at[i+1].set(mu + L_scaled[:, i])
            sigma_points = sigma_points.at[n+i+1].set(mu - L_scaled[:, i])
        
        return sigma_points
    
    def measurement_function(self, theta, x_current, q_ref, cost_map):
        """
        Measurement: h(theta) = -C(theta | x_current)
        
        Args:
            theta: [n_theta] control sequence
            x_current: [3] current observed system state
            q_ref: [3] reference state
            cost_map: collision checker
        Returns:
            y: scalar (negative cost)
        """
        u_seq = theta.reshape((-1, self.n_u))
        
        # Evaluate trajectory cost starting from known x_current
        outputs = self.planner.eval_U_seq(
            u_seq, 
            theta,  # original_u for regularization
            x_current,  # known current state
            q_ref, 
            cost_map,
        )
        cost = outputs[0] 
        return -cost
    
    # @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, mu, P, Q):
        """
        UKF Prediction step
        
        Args:
            mu: current mean [n_theta]
            P: current covariance [n_theta, n_theta]
            Q: process noise covariance [n_theta, n_theta]
        Returns:
            mu_pred, P_pred, sigma_points_pred
        """
        # Generate sigma points
        sigma_points = self.generate_sigma_points(mu, P)
        
        # Propagate through shift dynamics
        sigma_points_pred = jax.vmap(self.shift_controls)(sigma_points)
        
        # Compute predicted mean
        mu_pred = jnp.sum(self.Wm[:, None] * sigma_points_pred, axis=0)
        
        # Compute predicted covariance
        diff = sigma_points_pred - mu_pred
        P_pred = jnp.sum(self.Wc[:, None, None] * 
                        (diff[:, :, None] @ diff[:, None, :]), 
                        axis=0) + Q
        if np.any(np.isnan(mu_pred)):
            breakpoint()
        
        return mu_pred, P_pred, sigma_points_pred
    
    def update(self, mu_pred, P_pred, sigma_points_pred, 
               y_meas, R, x_current, q_ref, cost_map):
        """
        UKF Update step
        
        Args:
            mu_pred: predicted mean
            P_pred: predicted covariance
            sigma_points_pred: predicted sigma points
            y_meas: measurement (0 for Option B)
            R: measurement noise variance (scalar)
            x_current: current observed system state
            q_ref: reference state
            cost_map: collision checker
        Returns:
            mu_updated, P_updated, innovation
        """
        # Propagate sigma points through measurement function
        meas_sigma = jax.vmap(
            lambda theta: self.measurement_function(theta, x_current, q_ref, cost_map)
        )(sigma_points_pred)
        
        # Clip infinite costs
        meas_sigma = jnp.clip(meas_sigma, -1e10, 0)
        
        # Predicted measurement mean
        y_pred = jnp.sum(self.Wm * meas_sigma)
        
        # Innovation covariance
        meas_diff = meas_sigma - y_pred
        S = jnp.sum(self.Wc * meas_diff**2) + R
        
        # Cross-covariance
        state_diff = sigma_points_pred - mu_pred
        P_xy = jnp.sum(self.Wc[:, None] * state_diff * meas_diff[:, None], 
                      axis=0)
        
        # Kalman gain
        K = P_xy / S
        
        # Update
        innovation = y_meas - y_pred
        mu_updated = mu_pred + K * innovation
        P_updated = P_pred - K[:, None] * K[None, :] * S
        
        return mu_updated, P_updated, innovation
    
    def step(self, mu, P, Q, R, x_current, q_ref, cost_map, y_meas=0.0):
        """
        Full UKF prediction + update step
        
        Args:
            mu: current control sequence estimate
            P: current covariance
            Q: process noise
            R: measurement noise
            x_current: current observed system state
            q_ref: reference state
            cost_map: collision checker
            y_meas: measurement (default 0 for Option B)
        Returns:
            mu_new, P_new, innovation
        """
        # Predict
        mu_pred, P_pred, sigma_pred = self.predict(mu, P, Q)

        # Update
        mu_new, P_new, innov = self.update(
            mu_pred, P_pred, sigma_pred,
            y_meas, R, x_current, q_ref, cost_map
        )

        # Clip controls to bounds
        theta_new = mu_new.reshape((-1, self.n_u))
        theta_new = jnp.clip(theta_new, 
                            self.system.control_bounds[0],
                            self.system.control_bounds[1])
        mu_new = theta_new.ravel()
        
        return mu_new, P_new, innov

def do_ukf(params):
    dt = params['dt']
    Nt = params['Nt']
    start = params['start']
    q_ref = params['q_ref']
    obs = params['obs']
    nlmodel = params['nlmodel']
    T = params['T']
    sigma0 = params['sigma0']
    Q = params['Q']
    QT = params['QT']
    
    # Setup collision checker
    resolution = 0.5
    origin = np.array([-40, -10])
    wh = np.array([50/resolution, 20/resolution], dtype=np.int32)
    boundary = [[origin[0], origin[0]+wh[0]*resolution], 
                [origin[1], origin[1]+wh[1]*resolution]]
    

    
    grid = OccupGrid(boundary, resolution)
    grid.find_occupancy_grid(obs, buffer=0.05)
    
    collision_checker = CollisionChecker(
        jnp.array(grid.occup_grid), origin, resolution, wh, occup_value=100
    )
    
    # Create MPPI planner (for cost evaluation only)
    mppi_planner = MPPI_Planner_Occup(
        sigma=sigma0, Q=Q, QT=QT, R=params['R'],
        temperature=params['temperature'],
        system=nlmodel, num_anci=0, 
        n_samples=0,  # not sampling
        N=Nt, tolerance=0.4, occup_value=100
    )
    
    
    # Create UKF controller
    ukf = UKF_Controller(nlmodel, mppi_planner)
    
    # Initialize
    U_init = np.kron(np.ones((1, Nt)), [0.0, 1.0]).ravel()
    mu = jnp.array(U_init)
    P = sigma0*1000  # Control uncertainty only
    
    # Process and measurement noise
    Q_process = sigma0 * 0.1  # Control drift
    R_meas = 100  # Measurement noise variance
    
    # Simulate
    states = [start]
    costs = []
    control_sequences = []
    x_current = start.copy()
    
    for t in tqdm(np.arange(0, T, dt)):
        # UKF step - note x_current is observed, not estimated!
        mu, P, innovation = ukf.step(
            mu, P, Q_process, R_meas, x_current, q_ref, collision_checker, y_meas=0.0
        )
        
        # Extract first control
        u_opt = mu[:ukf.n_u]
        
        # Simulate system
        x_current = nlmodel.dynamics(x_current, u_opt, 0, 
                                     dt=dt, params=nlmodel.nominal_params)
        print(u_opt) 
        states.append(x_current)
        control_sequences.append(mu.reshape((-1, 2)))
        
        # Evaluate cost
        outputs = mppi_planner.eval_U_seq(
            mu.reshape((-1, 2)), mu, x_current, q_ref, collision_checker
        )
        cost = outputs[0]
        costs.append(float(cost))
        
        # Check if reached goal
        if np.linalg.norm(x_current[:2] - q_ref[:2]) < 0.5:
            print(f"Reached goal at t={t:.2f}")
            break
    
    return states, costs, control_sequences
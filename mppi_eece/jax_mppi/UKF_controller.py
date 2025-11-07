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
    def __init__(self, system, mppi_planner, n_sigma_points='auto'):
        """
        UKF for control sequence estimation
        
        Args:
            system: Your Unicycle system
            mppi_planner: Your MPPI_Planner_Occup instance
            n_sigma_points: 'auto' uses standard 2n+1, or specify custom
        """
        self.system = system
        self.planner = mppi_planner
        self.N = mppi_planner.N  # Horizon length
        
        # State dimension: [x, y, theta] + controls [u0, v0, u1, v1, ...]
        self.n_x = 3  # system state dimension
        self.n_u = 2  # control dimension per timestep
        self.n_theta = self.N * self.n_u  # total control sequence dimension
        self.n_z = self.n_x + self.n_theta  # augmented state dimension
        
        # UKF parameters (van der Merwe scaling)
        if n_sigma_points == 'auto':
            self.n_sigma = 2 * self.n_z + 1
        else:
            self.n_sigma = n_sigma_points
            
        alpha = 1e-3  # spread of sigma points (1e-4 to 1)
        kappa = 0  # secondary scaling (usually 0 or 3-n)
        beta = 2  # prior knowledge (2 is optimal for Gaussian)
        
        self.lambda_ = alpha**2 * (self.n_z + kappa) - self.n_z
        
        # Weights for mean and covariance
        self.Wm = jnp.zeros(self.n_sigma)
        self.Wc = jnp.zeros(self.n_sigma)
        
        self.Wm = self.Wm.at[0].set(self.lambda_ / (self.n_z + self.lambda_))
        self.Wc = self.Wc.at[0].set(self.lambda_ / (self.n_z + self.lambda_) + 
                                     (1 - alpha**2 + beta))
        
        for i in range(1, self.n_sigma):
            self.Wm = self.Wm.at[i].set(1 / (2 * (self.n_z + self.lambda_)))
            self.Wc = self.Wc.at[i].set(1 / (2 * (self.n_z + self.lambda_)))
    
    def generate_sigma_points(self, mu, P):
        """
        Generate sigma points using Cholesky decomposition
        
        Args:
            mu: mean [n_z]
            P: covariance [n_z, n_z]
        Returns:
            sigma_points: [n_sigma, n_z]
        """
        n = self.n_z
        
        # Ensure P is positive definite by adding small regularization
        P_reg = P + jnp.eye(n) * 1e-9
        
        try:
            # Cholesky decomposition: P = L @ L.T
            L = jnp.linalg.cholesky(P_reg)
        except:
            # If Cholesky fails, use eigendecomposition
            eigvals, eigvecs = jnp.linalg.eigh(P_reg)
            eigvals = jnp.maximum(eigvals, 1e-9)
            L = eigvecs @ jnp.diag(jnp.sqrt(eigvals))
        
        # Scale
        L_scaled = jnp.sqrt(n + self.lambda_) * L
        
        # Generate sigma points
        sigma_points = jnp.zeros((self.n_sigma, n))
        sigma_points = sigma_points.at[0].set(mu)
        
        for i in range(n):
            sigma_points = sigma_points.at[i+1].set(mu + L_scaled[:, i])
            sigma_points = sigma_points.at[n+i+1].set(mu - L_scaled[:, i])
        
        return sigma_points
    
    def dynamics_function(self, z):
        """
        Augmented dynamics: [x, theta] -> [x_new, theta_new]
        
        Args:
            z: [n_z] = [x (3), theta (N*2)]
        Returns:
            z_new: [n_z]
        """
        x = z[:self.n_x]  # system state [x, y, theta]
        theta = z[self.n_x:]  # control sequence
        
        # Apply first control
        u0 = theta[:self.n_u]
        x_new = self.system.jax_dynamics(x, u0, 0, 
                                         self.system.dt, 
                                         self.system.nominal_params)
        
        # Shift controls (this is your Phi operator)
        theta_new = jnp.roll(theta, -self.n_u)
        # Keep last control or set to nominal
        theta_new = theta_new.at[-self.n_u:].set(theta[-self.n_u:])
        
        return jnp.concatenate([x_new, theta_new])
    
    def measurement_function(self, z, q_ref, cost_map):
        """
        Measurement: h(z) = -C(theta | x)
        
        Args:
            z: [n_z] = [x, theta]
            q_ref: reference state
            cost_map: collision checker
        Returns:
            y: scalar (negative cost)
        """
        x = z[:self.n_x]
        theta = z[self.n_x:]
        u_seq = theta.reshape((-1, self.n_u))
        
        # Evaluate trajectory cost (use the UKF version with finite penalties)
        outputs = self.planner.eval_U_seq(
            u_seq, 
            theta,  # original_u
            x, 
            q_ref, 
            cost_map,
        )
        cost = outputs[0]
        
        return -cost  # negative cost as measurement
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, mu, P, Q):
        """
        UKF Prediction step
        
        Args:
            mu: current mean [n_z]
            P: current covariance [n_z, n_z]
            Q: process noise covariance [n_z, n_z]
        Returns:
            mu_pred, P_pred
        """
        # Generate sigma points
        sigma_points = self.generate_sigma_points(mu, P)
        
        # Propagate through dynamics
        sigma_points_pred = jax.vmap(self.dynamics_function)(sigma_points)
        
        # Compute predicted mean
        mu_pred = jnp.sum(self.Wm[:, None] * sigma_points_pred, axis=0)
        
        # Compute predicted covariance
        diff = sigma_points_pred - mu_pred
        P_pred = jnp.sum(self.Wc[:, None, None] * 
                        (diff[:, :, None] @ diff[:, None, :]), 
                        axis=0) + Q
        
        return mu_pred, P_pred, sigma_points_pred
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, mu_pred, P_pred, sigma_points_pred, 
               y_meas, R, q_ref, cost_map):
        """
        UKF Update step
        
        Args:
            mu_pred: predicted mean
            P_pred: predicted covariance
            sigma_points_pred: predicted sigma points
            y_meas: measurement (0 for Option B)
            R: measurement noise variance (scalar)
            q_ref: reference state
            cost_map: collision checker
        Returns:
            mu_updated, P_updated
        """
        # Propagate sigma points through measurement function
        meas_sigma = jax.vmap(
            lambda z: self.measurement_function(z, q_ref, cost_map)
        )(sigma_points_pred)
        
        # Handle infinite costs by clipping
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
    
    def step(self, mu, P, Q, R, q_ref, cost_map, y_meas=0.0):
        """
        Full UKF prediction + update step
        
        Args:
            mu: current state estimate
            P: current covariance
            Q: process noise
            R: measurement noise
            q_ref: reference
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
            y_meas, R, q_ref, cost_map
        )
        
        # Clip controls to bounds
        x_new = mu_new[:self.n_x]
        theta_new = mu_new[self.n_x:].reshape((-1, self.n_u))
        theta_new = jnp.clip(theta_new, 
                            self.system.control_bounds[0],
                            self.system.control_bounds[1])
        mu_new = jnp.concatenate([x_new, theta_new.ravel()])
        
        return mu_new, P_new, innov
    

def do_ukf(params):
    """
    Run UKF controller (analogous to do_mppi)
    """
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
    
    # Setup (same as MPPI)
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
    
    # Create MPPI planner (for cost evaluation)
    mppi_planner = MPPI_Planner_Occup(
        sigma=sigma0, Q=Q, QT=QT, R=params['R'],
        temperature=params['temperature'],
        system=nlmodel, num_anci=0, 
        n_samples=0,  # not sampling
        N=Nt, tolerance=0.4, occup_value=100
    )
    
    # Create UKF controller
    ukf = UKF_Controller(nlmodel, mppi_planner)
    
    # Initialize augmented state: [x, y, theta, u0, v0, u1, v1, ...]
    U_init = np.kron(np.ones((1, Nt)), [0.0, 1.0]).ravel()
    mu = jnp.concatenate([start, U_init])
    
    # Initialize covariance
    P_x = jnp.diag(jnp.array([0.1, 0.1, 0.1]))  # state uncertainty
    P_theta = sigma0  # control uncertainty
    P = jax.scipy.linalg.block_diag(P_x, P_theta)
    
    # Process noise
    Q_x = jnp.diag(jnp.array([0.01, 0.01, 0.01]))  # state process noise
    Q_theta = sigma0 * 0.1  # control drift
    Q_process = jax.scipy.linalg.block_diag(Q_x, Q_theta)
    
    # Measurement noise
    R_meas = 1.0  # measurement noise variance (sigma^2)
    
    # Simulate
    states = [start]
    costs = []
    sim_state = start.copy()
    
    for t in tqdm(np.arange(0, T, dt)):
        # UKF step
        mu, P, innovation = ukf.step(
            mu, P, Q_process, R_meas, q_ref, collision_checker, y_meas=0.0
        )
        
        # Extract control
        u_opt = mu[ukf.n_x:ukf.n_x+ukf.n_u]
        
        # Simulate system
        for i in range(10):
            sim_state = nlmodel.dynamics(sim_state, u_opt, 0, 
                                        dt=dt/10, params=nlmodel.nominal_params)
        
        # Update state part of mu with actual state
        mu = mu.at[:ukf.n_x].set(sim_state)
        
        states.append(sim_state)
        
        # Evaluate cost
        theta_current = mu[ukf.n_x:].reshape((-1, 2))
        outputs = mppi_planner.eval_U_seq(
            theta_current, mu[ukf.n_x:], sim_state, q_ref, collision_checker
        )
        cost = outputs[0]
        costs.append(cost)
    
    return states, costs
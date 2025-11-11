import numpy as np
from typing import Tuple
import jax.numpy as jnp
import numpy as np
import casadi as ca
from casadi import MX, vertcat
from acados_template import AcadosModel
from .base import System
import functools
import jax

class Unicycle(System):
    N_DIMS = 3
    N_CONTROLS = 2
    
    def __init__(self, nominal_params, dt):
        super().__init__(nominal_params, dt)

        self.lb = nominal_params['lb']
        self.ub = nominal_params['ub']

    @property
    def n_dims(self) -> int:
        return Unicycle.N_DIMS

    @property
    def n_controls(self) -> int:
        return Unicycle.N_CONTROLS

    @property
    def state_bounds(self) -> Tuple[jnp.array, jnp.array]:
        lb = jnp.array([-50, -50, -jnp.inf])
        ub = jnp.array([ 50,  50,  jnp.inf])
        return (lb, ub)

    @property
    def control_bounds(self) -> Tuple[jnp.array, jnp.array]:
        return (self.lb, self.ub)


    @functools.partial(jax.jit, static_argnums=(0,))
    def dynamics_jax(self, state, control, dt=None, params=None):
        """
        Simplified kinematic unicycle car dynamics with JAX-compatible code.
        """
        if dt is None:
           dt = self.dt 
        if params is None:
            params = self.nominal_params
        x, y, theta = state
        control = jnp.clip(control, self.control_bounds[0], self.control_bounds[1])
        delta, v = control


        dx = v * jnp.cos(theta)
        dy = v * jnp.sin(theta)
        dtheta = delta 
        next_state = jnp.clip(state + jnp.array([dx, dy, dtheta])*dt, self.state_bounds[0], self.state_bounds[1])
        return next_state

    def create_casadi_model(self, params=None):
        """
        Simplified kinematic unicycle car dynamics with JAX-compatible code.
        """
        if params is None:
            params = self.nominal_params
        state = ca.MX.sym('state', 3,1)
        control = ca.MX.sym('control',2,1)
        # dxdt = ca.SX.sym('dxdt',3,1)
        dxdt = ca.MX.zeros(3,1)
        x = state[0]
        y = state[1]
        theta = state[2]
        # delta = control[0]
        # v = control[1]
        delta = ca.fmax(ca.fmin(control[0], 
                                self.control_bounds[1][0]), 
                                self.control_bounds[0][0])
        v = ca.fmax(ca.fmin(control[1], 
                                self.control_bounds[1][1]), 
                                self.control_bounds[0][1])
        dxdt[0] = v * ca.cos(theta)
        dxdt[1] = v * ca.sin(theta)
        dxdt[2] = delta

        return dxdt, state, control

    def create_acados_model(self, params=None):
        model = AcadosModel()
        model.name = 'unicycle' 
        X = MX.sym("X")
        Y = MX.sym("Y")
        theta = MX.sym("theta")
        v = MX.sym("v_x")
        r = MX.sym("r")  # r = yaw rate, often called omega as well
        model.u = vertcat(
            r,
            v,
        )
        model.x = vertcat(
            X,
            Y,
            theta,
        )
        model.p = vertcat([])
        model.x0 = np.array([0, 0, 0])
        model.xdot = System.auto_xdot(model.x)
        # right hand side for differential equations
        model.f_expl_expr = vertcat(
            v * ca.cos(theta),  # = X_dot
            v * ca.sin(theta),  # = Y_dot
            r,  # = phi_dot
        )
        model.f_impl_expr = model.xdot - model.f_expl_expr
        return model

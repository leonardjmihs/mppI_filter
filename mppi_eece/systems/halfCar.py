import numpy as np
from typing import Tuple, Optional, List, Dict
from .nonlinear_system import NonlinerSystem
import jax.numpy as jnp
import jax
import numpy as np
import casadi as ca
from casadi import sin, cos, MX, vertcat, atan, sign
from acados_template import AcadosModel
import functools
from .base import System

class HalfCar(System):
    """
    Half-car model for vehicle dynamics.
    """
    N_DIMS = 3
    N_CONTROLS = 2

    def __init__(self, params):
        self.params = params
    @property
    def n_dims(self) -> int:
        return HalfCar.N_DIMS

    @property
    def n_controls(self) -> int:
        return HalfCar.N_CONTROLS

    @property
    def state_bounds(self) -> Tuple[jnp.array, jnp.array]:
        lb = np.array([-50, -50, -jnp.inf])
        ub = np.array([ 50,  50,  jnp.inf])
        return (lb, ub)

    @property
    def control_bounds(self) -> Tuple[jnp.array, jnp.array]:
        lb = np.array([-jnp.pi/3, 0.0])
        ub = np.array([jnp.pi/3, 4.0])
        return (lb, ub)

    def dynamics_jax(self, state, control, dt=None, params=None):
        """JAX-compatible dynamics function."""
        if dt is None:
           dt = self.dt 
        if params is None:
            params = self.nominal_params
        L = params["L"]
        x, y, theta = state
        control = jnp.clip(control, self.control_bounds[0], self.control_bounds[1])
        delta, v = control
        dx = v * jnp.cos(theta)
        dy = v * jnp.sin(theta)
        dtheta = v*jnp.tan(delta)/L
        next_state = jnp.clip(state + jnp.array([dx, dy, dtheta])*dt, self.state_bounds[0], self.state_bounds[1])
        return next_state

    def create_casadi_model(self, params=None):
        """Returns CasADi symbolic dynamics."""
        if params is None:
            params = self.nominal_params
        L = params["L"]
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
        dxdt[2] = v*ca.tan(delta)/L
        # return ca.Function('ode', [state, control], [dxdt]), state, control
        return dxdt, state, control

    def create_acados_model(self, params=None):
        model = AcadosModel()
        model.name = 'halfcar'
        # CasADi Model
        # states
        if constants is None:
            constants = {'L': 2.0} 
        L = constants["L"]
        X = MX.sym("X")
        Y = MX.sym("Y")
        theta = MX.sym("theta")
        v = MX.sym("v_x")
        delta = MX.sym("delta")  # r = yaw rate, often called omega as well
        model.u = vertcat(
            delta,
            v,
        )
        model.x = vertcat(
            X,
            Y,
            theta,
        )
        model.p = vertcat([])
        model.x0 = np.array([0, 0, 0])
        model.xdot = auto_xdot(model.x)
        # right hand side for differential equations
        model.f_expl_expr = vertcat(
            v * ca.cos(theta),  # = X_dot
            v * ca.sin(theta),  # = Y_dot
            v*ca.tan(delta)/L,  # = phi_dot
        )
        model.f_impl_expr = model.xdot - model.f_expl_expr
        return model
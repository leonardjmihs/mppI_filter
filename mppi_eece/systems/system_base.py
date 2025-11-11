import casadi as ca

class System:
    """
    Abstract base class for dynamical systems.
    Should be JAX compatible and provide CasADi/ACADOS conversion methods.
    """
    def __init__(self, params, dt):
        self.params = params # params for system
        self.dt = dt # default timestep

    @staticmethod
    def auto_xdot(model_x):
        """
            For a vector of scalar casadi state variables, generate a corresponding vector of {name}_dot casadi symbols.
        """
        xdot = ca.vertcat([])
        for i in range(model_x.size()[0]):
            xdot = ca.vertcat(xdot, ca.MX.sym(model_x[i].name() + "_dot"))
        return xdot

    @staticmethod
    def get_symbol(symbol_vector, name):
        """
            Find a casadi symbol by its name in a vertcat (vertical concatenation) vector of symbols.
            This method can also find vector-shaped symbols that span across a part of the full symbol vector.
        """
        idx = System.get_symbol_idx(symbol_vector, name)
        if idx is None:
            return None
        elif isinstance(idx, tuple):
            return symbol_vector[idx[0]:idx[1]]
        else:
            return symbol_vector[idx]


    @staticmethod
    def get_symbol_idx(symbol_vector, name):
        """
            Find the index of a casadi symbol by its name in a vertcat (vertical concatenation) of symbols.
            If the symbol is a vector instead of a scalar, this method returns the index range of the symbol.
        """
        v_len = symbol_vector.size()[0]
        slice_start = 0
        for i in range(v_len):
            info = symbol_vector[slice_start:i + 1].info()
            if "slice" not in info:
                if symbol_vector[slice_start:i + 1].name() == name:
                    return (slice_start, i + 1) if i != slice_start else i
                else:
                    slice_start = i + 1
        return None
 
    @property
    def n_dims(self) -> int:
        """Number of state dimensions."""
        return NotImplementedError 

    @property
    def n_controls(self) -> int:
        """Number of control inputs."""
        return NotImplementedError 

    @property
    def state_bounds(self):
        """State bounds."""
        return NotImplementedError 

    @property
    def control_bounds(self):
        """Control input bounds."""
        return NotImplementedError 

    def dynamics_jax(self, x, u, dt, params):
        """JAX-compatible dynamics function."""
        raise NotImplementedError

    def create_casadi_model(self, params):
        """Returns CasADi symbolic dynamics."""
        raise NotImplementedError

    def create_acados_model(self, params):
        """Returns ACADOS-compatible dynamics representation."""
        raise NotImplementedError

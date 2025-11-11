class System:
    """
    Abstract base class for dynamical systems.
    Should be JAX compatible and provide CasADi/ACADOS conversion methods.
    """
    def __init__(self, params):
        self.params = params

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

    def dynamics_jax(self):
        """JAX-compatible dynamics function."""
        raise NotImplementedError

    def dynamics_casadi(self):
        """Returns CasADi symbolic dynamics."""
        raise NotImplementedError

    def create_acados_model(self):
        """Returns ACADOS-compatible dynamics representation."""
        raise NotImplementedError

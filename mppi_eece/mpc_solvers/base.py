class MPCSolver:
    """
    Abstract base class for MPC solvers (CasADi, ACADOS, etc).
    """
    def setup(self, system):
        """Setup solver with given system."""
        raise NotImplementedError

    def solve(self, system, x0, ref, constraints=None, obstacles=None):
        """Solve MPC problem for system from x0 to ref."""
        raise NotImplementedError

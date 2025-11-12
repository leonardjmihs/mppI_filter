from dataclasses import dataclass, field
import numpy as np

@dataclass
class MPCSolverParams:
    Nt: int = 30
    dt: float = 0.2
    T: float = 12.0
    Q: np.ndarray = field(default_factory=lambda: np.eye(3))
    R: np.ndarray = field(default_factory=lambda: np.eye(2))
    QT: np.ndarray = field(default_factory=lambda: np.eye(3))
    num_sides: int = 9
    nlmodel: object = None

class MPCSolver:
    """
    base class for MPC solvers (CasADi, ACADOS, etc).
    """

    def get_solver(self, system):
        """Setup solver with given system."""
        raise NotImplementedError

    def _create_solver(self, system, params):
        """Setup solver with given system."""
        raise NotImplementedError

    def solve(self, system, x0, ref, constraints=None, obstacles=None):
        """Solve MPC problem for system from x0 to ref."""
        raise NotImplementedError

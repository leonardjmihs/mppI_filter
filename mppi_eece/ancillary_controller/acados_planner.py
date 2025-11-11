
from mppi_eece.controllers.ancillary_controller import AncillaryController
from mppi_eece.solvers.acados_solver import AcadosSolver

class AnciAcadosController(AncillaryController):
    """
    Ancillary controller using ACADOS solver.
    """
    def __init__(self, params):
        super().__init__(
            system=params['nlmodel'],
            planner=None,  # You can pass a planner instance if needed
            mpc_solver=AcadosSolver(params)
        )
        self.params = params
        self.Nt = params['Nt']
        self.T = params['T']
        self.Q = params['Q']
        self.R = params['R']
        self.QT = params['QT']
        self.start = params['start']
        self.q_ref = params['q_ref']
        self.obs = params['obs']
        self.num_sides = params.get('num_sides', 9)
        self.resolution = params.get('resolution', 0.5)

class AncillaryController:
    """
    Orchestrates planning, decomposition, and MPC solving using System, Planner, and MPCSolver abstractions.
    """
    def __init__(self, system, planner, mpc_solver):
        self.system = system
        self.planner = planner
        self.mpc_solver = mpc_solver

    def run(self, x0, goal, obstacles=None, constraints=None):
        """
        1. Plan path(s) from x0 to goal using planner (may return multiple plans).
        2. (Optional) Decompose environment if needed.
        3. Solve MPC using mpc_solver and each planned path.
        Returns: dict with keys 'plans', 'mpc_results', etc.
        """
        plans = self.planner.plan(x0, goal, obstacles=obstacles)
        # Decomposition step can be added here if needed
        mpc_results = []
        if plans is None:
            return {'plans': None, 'mpc_results': None}
        # If planner returns a list of plans, process each
        if isinstance(plans, list) and len(plans) > 0 and isinstance(plans[0], (list, np.ndarray)):
            for plan in plans:
                mpc_result = self.mpc_solver.solve(self.system, x0, goal, constraints=constraints, obstacles=obstacles)
                mpc_results.append({'plan': plan, 'mpc_result': mpc_result})
            return {'plans': plans, 'mpc_results': mpc_results}
        else:
            # Single plan case
            mpc_result = self.mpc_solver.solve(self.system, x0, goal, constraints=constraints, obstacles=obstacles)
            return {'plans': plans, 'mpc_results': [mpc_result]}

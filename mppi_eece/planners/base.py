class Planner:
    """
    Abstract base class for planners (TopoPRM, RRTStar, etc).
    """
    def plan(self, start, goal, obstacles=None):
        """Plan a path from start to goal."""
        raise NotImplementedError

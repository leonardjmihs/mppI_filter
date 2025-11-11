from .base import Planner
# ...existing imports from topo_prm/planner.py and helpers...

import time
import numpy as np
from planners.base import Planner
from planners.sampler import EllipsoidSampler
from planners.path_processor import PathProcessor
# If you migrate graph.py, nodes.py, search.py, import them from planners as well
from planners.graph import GraphBuilder
from planners.search import PathSearcher
from planners.nodes import Node
from controllers.ancillary_controller import AncillaryController  # if needed

class TopoPRMPlanner(Planner):
    """
    Concrete planner using Topological PRM.
    Inherits from Planner and uses internal helper modules (sampler, graph, etc).
    """
    def __init__(self, collision_checker, sample_inflate=[5.0, 5.0, 0.0], origin=[0.0, 0.0], resolution=0.1, wh=[10, 10],
                 max_raw_path=10, max_raw_path2=5, max_sample_num=200, reserve_num=5, ratio_to_short=10.0, sample_sz_p=0.5,
                 footprint=[[0.0, 0.0]], occup_value=[100], max_time=0.05):
        self.collision_checker = collision_checker
        self.sample_inflate = sample_inflate
        self.origin = np.array(origin)
        self.resolution = resolution
        self.wh = np.array(wh)
        self.occup_value = occup_value
        self.max_raw_path = max_raw_path
        self.max_raw_path2 = max_raw_path2
        self.max_sample_num = max_sample_num
        self.reserve_num = reserve_num
        self.ratio_to_short = ratio_to_short
        self.sample_sz_p = sample_sz_p
        self.footprint = np.array(footprint)
        self.max_time = max_time

        self.sampler = EllipsoidSampler(sample_inflate=tuple(sample_inflate), sample_sz_p=sample_sz_p)
        self.graph_builder = GraphBuilder(collision_checker=self.collision_checker, ray_resolution=0.1, topo_resolution=0.1,
                                         max_sample_num=max_sample_num, max_time=max_time)
        self.path_searcher = PathSearcher(max_raw_path=max_raw_path, max_raw_path2=max_raw_path2)
        self.path_processor = PathProcessor(reserve_num=reserve_num, ratio_to_short=ratio_to_short)
        self.graph = []
        self.safe_zones = None

    def plan(self, start, goal, obstacles=None):
        # This is a simplified version of findTopoPaths, adapted to the Planner API
        self.sampler.configure(start, goal)
        if self.safe_zones is not None:
            self.sampler.safe_zones = self.safe_zones
        self.graph = []
        self.graph, samples = self.graph_builder.create_graph(self.sampler, start, goal, self.graph)
        raw_paths = self.path_searcher.search_paths(self.graph)
        if raw_paths is None:
            return None
        shortcut_paths = self.path_processor.shortcut_paths(raw_paths, self.graph_builder._line_visible)
        filtered_paths = self.path_processor.prune_equivalent(shortcut_paths, self.graph_builder.topo_resolution, self.graph_builder._line_visible)
        selected_paths = self.path_processor.select_short_paths(filtered_paths, self.graph_builder._line_visible)
        return selected_paths

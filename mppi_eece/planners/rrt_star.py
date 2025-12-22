import numpy as np
import math
import time
from scipy.spatial import cKDTree
from .planner_base import Planner
from .path_processor import PathProcessor
from .graph import GraphBuilder
from .sampler import Sampler, UniformSampler

class Node:
    def __init__(self, pos, parent=None):
        self.pos = np.array(pos)
        self.parent = parent
        self.cost = 0.0

class RRTStarPlanner(Planner):
    """
    Concrete planner using RRT*.
    Inherits from Planner and wraps RRT* logic.
    """
    def __init__(self, collision_checker, max_iter=200, expand_dist=1.0,
                 goal_sample_rate=0.1, connect_circle_dist=20.0, footprint=[[0.0, 0.0]], sampler=None):
        self.collision_checker = collision_checker
        self.max_iter = max_iter
        self.expand_dist = expand_dist
        self.goal_sample_rate = goal_sample_rate
        self.connect_circle_dist = connect_circle_dist
        self.footprint = np.array(footprint)
        self.node_list = []
        # Use provided sampler or default to UniformSampler
        if sampler is None:
            self.sampler = UniformSampler()
        else:
            self.sampler = sampler
                
        self.graph_builder = None

        self.path_processor = PathProcessor()

    def set_collision_checker(self, collision_checker):
        self.collision_checker = collision_checker
        if self.graph_builder is None:
            self.graph_builder = GraphBuilder(
                collision_checker=collision_checker,
                ray_resolution=0.1,
                topo_resolution=0.1
            )
        else:
            self.graph_builder.collision_checker = collision_checker

    def plan(self, start, goal,reset=True):
        self.start_node = Node(start)
        self.end_node = Node(goal)
        self.node_list = [self.start_node]

        # Configure the sampler for the current start/goal
        self.sampler.configure(start, goal)

        kdtree = None
        rebuild_every = 5
        last_rebuild = 0
        for i in range(self.max_iter):
            # Use sampler to generate a random sample
            sample = self.sampler.get_sample()
            # With probability goal_sample_rate, sample the goal directly
            if np.random.uniform() < self.goal_sample_rate:
                rnd_node = self.end_node
            else:
                rnd_node = Node(sample)

            if kdtree is None or (i - last_rebuild) >= rebuild_every:
                positions = np.array([n.pos for n in self.node_list]) if len(self.node_list) > 0 else np.empty((0, 3))
                if len(positions) > 0:
                    kdtree = cKDTree(positions[:, :2])
                else:
                    kdtree = None
                last_rebuild = i

            if kdtree is not None:
                _, nearest_ind = kdtree.query(rnd_node.pos[:2], k=1)
                nearest_node = self.node_list[int(nearest_ind)]
            else:
                nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
                nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dist)

            if not self.collision_checker.find_occupied(new_node.pos):
                near_inds = self.find_near_nodes(new_node, kdtree=kdtree)
                new_node = self.choose_parent(new_node, near_inds)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)
        # try to connect the end node
        if self.end_node not in self.node_list:
            last_index = self.get_nearest_node_index(self.node_list, self.end_node)
            last_node = self.node_list[last_index]
            if self.calc_dist_to_goal(last_node) <= self.expand_dist:
                final_node = self.steer(last_node, self.end_node)
                if not self.collision_checker.find_occupied(final_node.pos):
                    self.end_node.parent = last_node
                    self.end_node.cost = last_node.cost + self.calc_dist_to_goal(last_node)
                    self.node_list.append(self.end_node)

        path = self.generate_final_path(self.end_node)
        if path is None:
            return None
        return [path]

    # get_random_node is now deprecated; sampling is handled by self.sampler

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(to_node.pos)
        d, theta = self.calc_distance_and_angle(from_node, new_node)
        new_node.pos[0] = from_node.pos[0] + min(extend_length, d) * math.cos(theta)
        new_node.pos[1] = from_node.pos[1] + min(extend_length, d) * math.sin(theta)
        new_node.cost = from_node.cost + min(extend_length, d)
        new_node.parent = from_node
        return new_node

    def generate_final_path(self, goal_node):
        if goal_node.parent is None:
            return None
        path = []
        node = goal_node
        while node.parent is not None:
            path.append([node.pos[0], node.pos[1]])
            node = node.parent
        path.append([node.pos[0], node.pos[1]])

        ### NOTE: shortcut_paths expects and returns a list of paths                                                
        final_path = self.path_processor.shortcut_paths([path[::-1]], 
                                                        self.graph_builder._line_visible)
        return final_path[0]

    def calc_dist_to_goal(self, from_node):
        return np.linalg.norm(from_node.pos - self.end_node.pos)

    def find_near_nodes(self, new_node, kdtree=None):
        nnode = len(self.node_list) + 1
        if nnode <= 1:
            return []
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        if kdtree is not None:
            inds = kdtree.query_ball_point(new_node.pos[:2], r)
            return [int(i) for i in inds]
        dist_list = [np.linalg.norm(node.pos - new_node.pos) for node in self.node_list]
        near_inds = [i for i, d in enumerate(dist_list) if d <= r]
        return near_inds

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None
        cost_pairs = []
        for i in near_inds:
            near_node = self.node_list[i]
            c = self.calc_new_cost(near_node, new_node)
            cost_pairs.append((c, i))
        cost_pairs.sort(key=lambda x: x[0])
        starts = np.array([self.node_list[i].pos[:2] for _, i in cost_pairs])
        ends = np.tile(new_node.pos[:2], (len(cost_pairs), 1))
        if len(starts) == 0:
            return None
        free_mask = self.collision_checker.is_line_free_batch(starts, ends, num_samples=10, early_exit=False)
        for (c, idx), free in zip(cost_pairs, free_mask):
            if free:
                new_node = self.steer(self.node_list[idx], new_node)
                new_node.parent = self.node_list[idx]
                new_node.cost = c
                return new_node
        return None

    def rewire(self, new_node, near_inds):
        candidates = []
        for i in near_inds:
            near_node = self.node_list[i]
            new_cost = self.calc_new_cost(new_node, near_node)
            if new_cost < near_node.cost:
                candidates.append((i, new_cost))
        if not candidates:
            return
        starts = np.tile(new_node.pos[:2], (len(candidates), 1))
        ends = np.array([self.node_list[i].pos[:2] for i, _ in candidates])
        free_mask = self.collision_checker.is_line_free_batch(starts, ends, num_samples=10, early_exit=False)
        for (i_cost, (_, new_cost)), free in zip(enumerate(candidates), free_mask):
            idx, _ = candidates[i_cost]
            if free:
                near_node = self.node_list[idx]
                near_node.parent = new_node
                near_node.cost = new_cost

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d
        
    def discretize_path(self, path, pt_num):
        # breakpoint()
        len_list = [0.0]
        for i in range(len(path)-1):
            length = np.linalg.norm(path[i+1] - path[i])
            len_list.append(length+len_list[i])
        len_total = len_list[-1]
        dl = len_total / (pt_num-1.0)
        cur_l = 0.0
        dis_path = []
        path_idx = []
        for i in range(pt_num) :
            cur_l = i * dl
            idx = -1
            for j in range(len(len_list)-1):
                if (cur_l >= len_list[j] - 1e-4) and cur_l <= len_list[j+1] + 1e-4:
                    idx = j
                    break
            l = (cur_l - len_list[idx]) / (len_list[idx+1] - len_list[idx])
            # breakpoint()
            inter_pt = (1-l) * path[idx] + l * path[idx+1]
            dis_path.append(inter_pt)
            path_idx.append(idx)

        # path_idx.append(len(path))
        return dis_path, path_idx
    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [np.linalg.norm(node.pos - rnd_node.pos) for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.pos[0] - from_node.pos[0]
        dy = to_node.pos[1] - from_node.pos[1]
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

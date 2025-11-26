"""
Graph construction and visibility checking for TopoPRM.

This module handles building the roadmap graph by determining node visibility,
checking collision-free connections, and managing graph topology.
"""

import cProfile
import os
import pstats
import subprocess
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import numpy.typing as npt

from .nodes import Node, NODE_TYPE


class GraphBuilder:
    """
    Builds and manages the topological roadmap graph.
    
    Handles visibility checking between nodes, determining when to create new
    guard or connector nodes, and pruning isolated graph components.
    
    Attributes:
        ray_resolution: Step size for ray casting collision checks
        topo_resolution: Resolution for topological equivalence checking
        max_sample_num: Maximum number of samples to generate
        max_time: Maximum time allowed for graph construction
    """
    
    def __init__(
        self,
        collision_checker: Any,
        ray_resolution: float = 0.2,
        topo_resolution: float = 0.2,
        max_sample_num: int = 200,
        max_time: float = 0.05,
        enable_profiling: bool = False,
        profiling_output_dir: str = "/tmp"
    ) -> None:
        """
        Initialize the graph builder.
        
        Args:
            collision_checker: Object with find_occupied(pt) method
            ray_resolution: Step size for line-of-sight checks (meters)
            topo_resolution: Resolution for discretizing paths (meters)
            max_sample_num: Maximum samples to generate
            max_time: Maximum construction time (seconds)
            enable_profiling: If True, enable cProfile profiling for flamegraph generation
            profiling_output_dir: Directory to save profiling outputs (default: /tmp)
            
        Raises:
            ValueError: If parameters are out of valid ranges
            TypeError: If collision_checker is invalid
        """
        # Validate collision checker
        if not hasattr(collision_checker, 'find_occupied'):
            raise TypeError("collision_checker must have find_occupied() method")
        if not callable(collision_checker.find_occupied):
            raise TypeError("collision_checker.find_occupied must be callable")
        
        # Validate resolutions
        if ray_resolution <= 0:
            raise ValueError(f"ray_resolution must be positive, got {ray_resolution}")
        if topo_resolution <= 0:
            raise ValueError(f"topo_resolution must be positive, got {topo_resolution}")
        
        # Validate sample number
        if max_sample_num <= 0:
            raise ValueError(f"max_sample_num must be positive, got {max_sample_num}")
        
        # Validate time
        if max_time <= 0:
            raise ValueError(f"max_time must be positive, got {max_time}")
        
        self.collision_checker = collision_checker
        self.ray_resolution = ray_resolution
        self.topo_resolution = topo_resolution
        self.max_sample_num = max_sample_num
        self.max_time = max_time
        
        # Timing diagnostics
        self.timing_stats: Dict[str, float] = {}
        self.enable_timing: bool = True
        
        # Profiling support
        self.enable_profiling = enable_profiling
        self.profiling_output_dir = profiling_output_dir
        self.profiler: Optional[cProfile.Profile] = None
        if self.enable_profiling:
            self.profiler = cProfile.Profile()
    def check_if_in_graph(self, pt, graph):
        print(graph)
        for node in graph:
            if np.allclose(node.pos, pt):
                return True
        return False

    def create_graph(
        self,
        sampler: Any,
        start: npt.NDArray[np.floating],
        end: npt.NDArray[np.floating],
        graph: Optional[List[Node]] = None
    ) -> Tuple[List[Node], List[npt.NDArray[np.floating]]]:
        """
        Create the topological roadmap graph by sampling and connecting nodes.
        
        Iteratively samples points and determines whether to create guard or
        connector nodes based on visibility to existing guards.
        
        Args:
            sampler: EllipsoidSampler instance configured for start-goal
            start: Start position [x, y, z]
            end: Goal position [x, y, z]
            graph: Existing graph to extend (default: create new)
            
        Returns:
            tuple: (graph, samples) where graph is list of nodes and samples
                   is list of all collision-free sample points
                   
        Raises:
            ValueError: If start/end are invalid
            TypeError: If sampler is invalid
        """
        # Validate sampler
        if not hasattr(sampler, 'get_sample'):
            raise TypeError("sampler must have get_sample() method")
        
        # Reset timing stats for this construction call
        if self.enable_timing:
            self.timing_stats = {}
            t_total_start = time.perf_counter()
        
        # Start profiling if enabled
        if self.enable_profiling and self.profiler is not None:
            self.profiler.enable()
        
        # Validate start/end
        start = np.array(start).reshape((3, 1))
        end = np.array(end).reshape((3, 1))
        
        if not np.all(np.isfinite(start)):
            raise ValueError(f"Start must be finite, got {start.ravel()}")
        if not np.all(np.isfinite(end)):
            raise ValueError(f"End must be finite, got {end.ravel()}")
        
        # Initialize with start and goal as guards
        if self.enable_timing:
            t_init_start = time.perf_counter()
        
        if graph is None or len(graph) == 0:
            graph = []
            start = np.array(start).reshape((3, 1))
            end = np.array(end).reshape((3, 1))
            graph.append(Node(start.ravel(), NODE_TYPE.GUARD, 0))
            graph.append(Node(end.ravel(), NODE_TYPE.GUARD, 1))
        
        node_id = len(graph)
        sample_num = 0
        samples = []
        start_time = time.time()
        
        if self.enable_timing:
            self.timing_stats['initialization'] = time.perf_counter() - t_init_start
            # Counters for detailed stats
            collision_checks: int = 0
            visibility_checks: int = 0
            guard_creations: int = 0
            connector_creations: int = 0
            topo_checks: int = 0
            
            t_sampling: float = 0.0
            t_collision: float = 0.0
            t_visibility: float = 0.0
            t_connector_logic: float = 0.0
            t_pruning: float = 0.0
        else:
            # Initialize variables even when timing is disabled to avoid unbound errors
            collision_checks = 0
            visibility_checks = 0
            guard_creations = 0
            connector_creations = 0
            topo_checks = 0
            t_sampling = 0.0
            t_collision = 0.0
            t_visibility = 0.0
            t_connector_logic = 0.0
            t_pruning = 0.0
        
        # Sample and build graph

        while sample_num < self.max_sample_num and (time.time() - start_time) < self.max_time:
            if self.enable_timing:
                t_sample_start = time.perf_counter()
            
            pt = sampler.get_sample()
            sample_num += 1
            
            if self.enable_timing:
                t_sampling += time.perf_counter() - t_sample_start
                t_coll_start = time.perf_counter()
            
            # Skip if in collision
            if self.collision_checker.find_occupied(pt):
                if self.enable_timing:
                    t_collision += time.perf_counter() - t_coll_start
                    collision_checks += 1
                continue
            if self.check_if_in_graph(pt,graph):
                continue
            
            if self.enable_timing:
                t_collision += time.perf_counter() - t_coll_start
                collision_checks += 1
                t_vis_start = time.perf_counter()
            
            # Determine node type based on visible guards
            visible_guards = self._find_visible_guards(pt, graph)
            
            if self.enable_timing:
                t_visibility += time.perf_counter() - t_vis_start
                visibility_checks += 1
            
            if len(visible_guards) == 0:
                # No guards visible → new region boundary (GUARD)
                guard = Node(pt, NODE_TYPE.GUARD, node_id)
                node_id += 1
                graph.append(guard)
                samples.append(pt)
                
                if self.enable_timing:
                    guard_creations += 1
                
            elif len(visible_guards) >= 2:
                # Multiple guards visible → potential connector
                samples.append(pt)
                
                if self.enable_timing:
                    t_conn_start = time.perf_counter()
                
                # Sanity check: guards should be different nodes
                assert visible_guards[0].id != visible_guards[1].id, \
                    f"Found same guard twice: {visible_guards[0].id} Guard 1: {visible_guards[0].pos}, Guard 2: {visible_guards[1].pos}"
                
                need_conn = self._need_connection(visible_guards[0], visible_guards[1], pt)
                
                if self.enable_timing:
                    t_connector_logic += time.perf_counter() - t_conn_start
                    topo_checks += 1
                
                if not need_conn:
                    continue
                
                # Create connector and link to guards
                connector = Node(pt, NODE_TYPE.CONNECTOR, node_id)
                node_id += 1
                
                visible_guards[0].neighbors.append(connector)
                visible_guards[1].neighbors.append(connector)
                connector.neighbors.append(visible_guards[0])
                connector.neighbors.append(visible_guards[1])
                
                # Verify connections
                assert connector in visible_guards[0].neighbors, "Connection failed"
                assert connector in visible_guards[1].neighbors, "Connection failed"
                
                graph.append(connector)
                
                if self.enable_timing:
                    connector_creations += 1
        
        # Remove isolated nodes
        if self.enable_timing:
            t_prune_start = time.perf_counter()
        
        graph = self._prune_graph(graph)

        if self.enable_timing:
            t_pruning = time.perf_counter() - t_prune_start
        
        # Sanity check: should still have start and goal
        assert len(graph) >= 2, f"Graph pruned too aggressively, only {len(graph)} nodes remain"
        assert graph[0].id == 0, f"Start node missing after pruning"
        assert graph[1].id == 1, f"Goal node missing after pruning"
        
        # Stop profiling and save results if enabled
        if self.enable_profiling and self.profiler is not None:
            self.profiler.disable()
            self._save_profiling_results()
        
        # Store timing statistics
        if self.enable_timing:
            self.timing_stats['total_time'] = time.perf_counter() - t_total_start
            self.timing_stats['sampling'] = t_sampling
            self.timing_stats['collision_checking'] = t_collision
            self.timing_stats['visibility_checking'] = t_visibility
            self.timing_stats['connector_logic'] = t_connector_logic
            self.timing_stats['graph_pruning'] = t_pruning
            
            # Counters
            self.timing_stats['num_samples'] = sample_num
            self.timing_stats['num_collision_checks'] = collision_checks
            self.timing_stats['num_visibility_checks'] = visibility_checks
            self.timing_stats['num_guards_created'] = guard_creations
            self.timing_stats['num_connectors_created'] = connector_creations
            self.timing_stats['num_topo_checks'] = topo_checks
            self.timing_stats['num_final_nodes'] = len(graph)
            self.timing_stats['num_collision_free_samples'] = len(samples)
            
            # Derived metrics
            if sample_num > 0:
                self.timing_stats['collision_free_rate'] = len(samples) / sample_num
                if collision_checks > 0:
                    self.timing_stats['avg_collision_check_time'] = t_collision / collision_checks
            if visibility_checks > 0:
                self.timing_stats['avg_visibility_check_time'] = t_visibility / visibility_checks
            if topo_checks > 0:
                self.timing_stats['avg_topo_check_time'] = t_connector_logic / topo_checks

        self.check_graph_has_unique_nodes(graph)
        return graph, samples
    
    def _find_visible_guards(
        self, 
        pt: npt.NDArray[np.floating], 
        graph: List[Node]
    ) -> List[Node]:
        """
        Find guard nodes visible from a point.
        
        Uses vectorized collision checking for performance while maintaining
        deterministic guard selection order (first 2 visible guards in graph order).
        
        Args:
            pt: Point to check visibility from [x, y, z]
            graph: Current graph
            
        Returns:
            List of visible guard nodes (max 2, in graph order)
        """
        # Extract guard nodes (preserve graph order)
        guard_nodes = [node for node in graph if node.type == NODE_TYPE.GUARD]
        
        if not guard_nodes:
            return []
        
        # Check if any guard is at the same position (skip self)
        for node in guard_nodes:
            if np.allclose(node.pos, pt):
                return []
        
        # Vectorized visibility check: check all guards at once
        guard_positions = np.array([node.pos for node in guard_nodes])
        pt_repeated = np.tile(pt, (len(guard_nodes), 1))
        
        # Calculate number of samples needed (based on max distance)
        distances = np.linalg.norm(guard_positions - pt, axis=1)
        max_dist = np.max(distances)
        num_samples = max(2, int(np.floor(max_dist / self.ray_resolution)))
        
        # Batch check visibility to all guards
        visible = self.collision_checker.is_line_free_batch(
            pt_repeated, 
            guard_positions,
            num_samples=num_samples,
            early_exit=True  # Can exit early for each line individually
        )
        
        # Return visible guards in original graph order (max 2 for compatibility)
        # This maintains deterministic behavior matching the original sequential code
        visible_guards = [guard_nodes[i] for i in range(len(guard_nodes)) if visible[i]]
        return visible_guards[:2]
    
    def _need_connection(
        self, 
        node1: Node, 
        node2: Node, 
        pt: npt.NDArray[np.floating]
    ) -> bool:
        """
        Check if a new connector at pt provides a topologically distinct path.
        
        Compares against existing connectors between node1 and node2. If an
        equivalent path already exists, no new connector is needed.
        
        Args:
            node1: First guard node
            node2: Second guard node
            pt: Proposed connector position
            
        Returns:
            True if new connector is needed, False otherwise
        """
        from .path_processor import PathProcessor  # Avoid circular import
        
        path_new = [node1.pos, pt, node2.pos]
        
        # Check all common neighbors
        for n1_neighbor in node1.neighbors:
            for n2_neighbor in node2.neighbors:
                if n1_neighbor.id == n2_neighbor.id:
                    path_existing = [node1.pos, n1_neighbor.pos, node2.pos]
                    
                    # Check topological equivalence
                    if PathProcessor.same_topo_path(
                        path_new, path_existing, self.topo_resolution, self._line_visible
                    ):
                        # Update existing connector if new path is shorter
                        if PathProcessor.path_length(path_new) < PathProcessor.path_length(path_existing):
                            n1_neighbor.pos = pt
                        return False
        
        return True
    
    def _line_visible(
        self,
        pos1: npt.NDArray[np.floating],
        pos2: npt.NDArray[np.floating],
        check_endpoints: bool = True
    ) -> bool:
        """
        Check if line segment between two points is collision-free.
        
        Uses vectorized collision checking for improved performance.
        
        Args:
            pos1: Start position [x, y, z]
            pos2: End position [x, y, z]
            check_endpoints: Whether to check endpoints for collision
            
        Returns:
            True if line is collision-free, False otherwise
        """
        dist = np.linalg.norm(pos2 - pos1)
        steps = int(np.floor(dist / self.ray_resolution))
        
        if steps < 2:
            steps = 2
        
        # Generate all ray sample points at once (vectorized)
        ray = np.linspace(pos1, pos2, steps)
        
        if not check_endpoints:
            ray = ray[1:-1]
        
        # Batch collision check all points at once (vectorized)
        if len(ray) == 0:
            return True
        
        occupied = self.collision_checker.find_occupied_batch(ray, mark_out_of_bounds=True)
        
        # Return False if any point is occupied
        return not np.any(occupied)
    
    def _line_visible_batch(
        self,
        pos1_array: npt.NDArray[np.floating],
        pos2_array: npt.NDArray[np.floating],
        check_endpoints: bool = True
    ) -> npt.NDArray[np.bool_]:
        """
        Check if multiple line segments are collision-free (vectorized).
        
        This is a batch version of _line_visible for improved performance when
        checking many lines at once.
        
        Args:
            pos1_array: Start positions array, shape (N, 3)
            pos2_array: End positions array, shape (N, 3)
            check_endpoints: Whether to check endpoints for collision
            
        Returns:
            Boolean array of shape (N,) where True indicates line is free
            
        Example:
            >>> starts = np.array([[0,0,0], [1,1,0]])
            >>> ends = np.array([[5,5,0], [6,6,0]])
            >>> visible = builder._line_visible_batch(starts, ends)
        """
        if pos1_array.shape != pos2_array.shape:
            raise ValueError(f"pos1_array and pos2_array must have same shape")
        
        n_lines = len(pos1_array)
        
        # Calculate distances for each line
        dists = np.linalg.norm(pos2_array - pos1_array, axis=1)
        
        # Calculate steps needed for each line
        steps_array = np.floor(dists / self.ray_resolution).astype(int)
        steps_array = np.maximum(steps_array, 2)
        
        # Use max steps for uniform sampling across all lines
        max_steps = np.max(steps_array)
        
        # Generate sample points for all lines (vectorized)
        # Shape: (n_lines, max_steps, 3)
        alphas = np.linspace(0, 1, max_steps)
        sample_pts = (
            pos1_array[:, None, :] + 
            alphas[None, :, None] * (pos2_array - pos1_array)[:, None, :]
        )
        
        # Remove endpoints if requested
        if not check_endpoints:
            if max_steps > 2:
                sample_pts = sample_pts[:, 1:-1, :]
            else:
                # If only 2 steps and not checking endpoints, nothing to check
                return np.ones(n_lines, dtype=bool)
        
        # Flatten to check all points at once
        # Shape: (n_lines * steps, 3)
        sample_pts_flat = sample_pts.reshape(-1, sample_pts.shape[-1])
        
        # Batch collision check
        occupied_flat = self.collision_checker.find_occupied_batch(
            sample_pts_flat, mark_out_of_bounds=True
        )
        
        # Reshape back to (n_lines, steps)
        occupied = occupied_flat.reshape(n_lines, -1)
        
        # Line is free if no point along it is occupied
        line_free = ~np.any(occupied, axis=1)
        
        return line_free

    def _prune_graph(self, graph: List[Node]) -> List[Node]:
        """
        Remove isolated nodes with insufficient connections.
        
        Nodes beyond start/goal with 1 or fewer neighbors are removed
        along with their connections.
        
        Args:
            graph: Graph to prune
            
        Returns:
            Pruned graph
        """
        if len(graph) <= 2:
            return graph
        
        # Find nodes to remove
        to_remove = []
        for node in graph:
            # Never remove start (0) or goal (1)
            if node.id <= 1:
                continue
            
            # Remove if insufficiently connected
            if len(node.neighbors) <= 1:
                to_remove.append(node)
        
        # Remove nodes and their connections
        for node in to_remove:
            # Remove from neighbors' lists
            for other in graph:
                if node in other.neighbors:
                    other.neighbors.remove(node)
            # Remove from graph
            if node in graph:
                graph.remove(node)
        
        return graph
    
    def update_start_goal(
        self,
        graph: List[Node],
        new_start: npt.NDArray[np.floating],
        new_goal: npt.NDArray[np.floating]
    ) -> List[Node]:
        """
        Update graph with new start and goal positions.
        
        Replaces the start (node 0) and goal (node 1) positions and
        recomputes their connections.
        
        Args:
            graph: Existing graph
            new_start: New start position [x, y, z]
            new_goal: New goal position [x, y, z]
            
        Returns:
            Updated graph
        """        
        # print(f"Printing Original Graph")
        # for node in graph:
        #     print(f"ID: {node.id}, pos: {node.pos} type: {node.type}")
        
        if len(graph) == 0:
            return []
        
        # Remove old connections
        # breakpoint()
        for neighbor in graph[0].neighbors[:]:
            neighbor.neighbors.remove(graph[0])
        for neighbor in graph[1].neighbors[:]:
            neighbor.neighbors.remove(graph[1])
        
        # Remove and re-add start/goal
        # prev_start = graph.pop(0)
        # prev_goal = graph.pop(0)

        
        # Renumber remaining nodes
        # for node in graph:
        #     node.id += 2
        
        # Insert new start/goal
        updated_graph = []
        updated_graph.insert(0, Node(new_start.ravel(), NODE_TYPE.GUARD, 0))
        updated_graph.insert(1, Node(new_goal.ravel(), NODE_TYPE.GUARD, 1))
        
        # Reconnect connectors to new start
        for i,node in enumerate(graph[2:]):
            id = i +2
            for neighbor in node.neighbors:
                neighbor.id = id
            # if self.collision_checker.find_occupied(node.pos):
            if False:
                continue
            else:
                '''
                !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                THIS IS a temporrary fix for a bug that occurs where
                two guards get the same node id
                !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                '''
                node.id = id
                if node.type == NODE_TYPE.CONNECTOR and len(node.neighbors) < 2:
                    if self._line_visible(node.pos, graph[0].pos):
                        node.neighbors.append(graph[0])
                        graph[0].neighbors.append(node)
                updated_graph.append(node)  
        # print(f"Printing Updated graph")
        # for node in updated_graph:
        #     print(f"    ID: {node.id}, pos: {node.pos} type: {node.type}")
        
        return updated_graph

    @staticmethod
    def check_graph_has_unique_nodes(graph: List[Node]) -> None:
        """
        Check that the graph has unique node IDs.

        Raises:
            RuntimeError: If duplicate node IDs are found
        """
        node_ids = [node.id for node in graph]
        duplicate_ids = set([x for x in node_ids if node_ids.count(x) > 1])
        if duplicate_ids:
            raise RuntimeError(f"Graph has duplicate node IDs: {duplicate_ids}")

    def _save_profiling_results(self) -> None:
        """
        Save profiling results and generate human-readable hotspot report.
        
        Creates:
        - profile.prof: Raw cProfile data for advanced tools
        - profile_hotspots.txt: Human-readable report identifying performance bottlenecks
        """
        if self.profiler is None:
            return
        
        # Ensure output directory exists
        os.makedirs(self.profiling_output_dir, exist_ok=True)
        
        # Save profile data
        prof_path = os.path.join(self.profiling_output_dir, "profile.prof")
        self.profiler.dump_stats(prof_path)
        
        # Generate human-readable hotspot report
        report_path = os.path.join(self.profiling_output_dir, "profile_hotspots.txt")
        
        try:
            import io
            
            # Create stats object
            stats = pstats.Stats(prof_path)
            
            # Generate comprehensive text report
            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("GraphBuilder Profiling Report - Performance Hotspots\n")
                f.write("=" * 80 + "\n\n")
                
                # Section 1: Top functions by cumulative time
                f.write("TOP 20 FUNCTIONS BY CUMULATIVE TIME (including callees)\n")
                f.write("-" * 80 + "\n")
                f.write("These functions consume the most total time (self + all called functions)\n\n")
                
                stream = io.StringIO()
                stats_copy = pstats.Stats(prof_path, stream=stream)
                stats_copy.strip_dirs()
                stats_copy.sort_stats('cumulative')
                stats_copy.print_stats(20)
                f.write(stream.getvalue())
                
                # Section 2: Top functions by self time
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("TOP 20 FUNCTIONS BY SELF TIME (excluding callees)\n")
                f.write("-" * 80 + "\n")
                f.write("These functions have the most time spent in their own code\n\n")
                
                stream = io.StringIO()
                stats_copy = pstats.Stats(prof_path, stream=stream)
                stats_copy.strip_dirs()
                stats_copy.sort_stats('time')
                stats_copy.print_stats(20)
                f.write(stream.getvalue())
                
                # Section 3: Focus on GraphBuilder methods
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("GRAPHBUILDER METHODS (visibility & connector logic)\n")
                f.write("-" * 80 + "\n\n")
                
                stream = io.StringIO()
                stats_copy = pstats.Stats(prof_path, stream=stream)
                stats_copy.strip_dirs()
                stats_copy.sort_stats('cumulative')
                # Filter to graph.py methods
                stats_copy.print_stats('graph.py:')
                f.write(stream.getvalue())
                
                # Section 4: Call relationships for key methods
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("CALLERS OF KEY HOTSPOT FUNCTIONS\n")
                f.write("-" * 80 + "\n")
                f.write("Shows what calls the expensive functions\n\n")
                
                # Identify top functions to analyze callers
                stats_copy = pstats.Stats(prof_path)
                stats_copy.strip_dirs()
                stats_copy.sort_stats('cumulative')
                
                # Get top function names
                stream = io.StringIO()
                stats_analysis = pstats.Stats(prof_path, stream=stream)
                stats_analysis.strip_dirs()
                stats_analysis.sort_stats('cumulative')
                
                # Print callers for specific important methods
                important_methods = [
                    '_find_visible_guards',
                    '_line_visible', 
                    '_need_connection',
                    'find_occupied',
                    'same_topo_path'
                ]
                
                for method in important_methods:
                    stream = io.StringIO()
                    stats_copy = pstats.Stats(prof_path, stream=stream)
                    stats_copy.strip_dirs()
                    try:
                        stats_copy.print_callers(method)
                        output = stream.getvalue()
                        if output.strip():
                            f.write(f"\nCallers of {method}:\n")
                            f.write(output)
                    except:
                        pass  # Method not found
                
                # Summary and recommendations
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("ANALYSIS SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                
                # Parse stats to give specific recommendations
                stats_dict = stats.stats if hasattr(stats, 'stats') else {}  # type: ignore
                
                # Calculate total time in key functions
                visibility_time = 0
                collision_time = 0
                topo_time = 0
                
                for (filename, lineno, funcname), (cc, nc, tt, ct, callers) in stats_dict.items():
                    if '_find_visible_guards' in funcname or '_line_visible' in funcname:
                        visibility_time += ct
                    if 'find_occupied' in funcname:
                        collision_time += ct
                    if '_need_connection' in funcname or 'same_topo_path' in funcname:
                        topo_time += ct
                
                f.write(f"Time in visibility checking: {visibility_time*1000:.2f} ms\n")
                f.write(f"Time in collision checking: {collision_time*1000:.2f} ms\n")
                f.write(f"Time in topology checks: {topo_time*1000:.2f} ms\n\n")
                
                # Recommendations
                f.write("OPTIMIZATION RECOMMENDATIONS:\n")
                f.write("-" * 80 + "\n\n")
                
                total_time = max(visibility_time + collision_time + topo_time, 0.001)
                
                if visibility_time / total_time > 0.4:
                    f.write("⚠ VISIBILITY CHECKING is a major bottleneck (>40% of time)\n")
                    f.write("  - Consider increasing ray_resolution to reduce ray density\n")
                    f.write("  - Use spatial indexing (KD-tree) for guard node lookups\n")
                    f.write("  - Cache line-of-sight results for frequently checked pairs\n\n")
                
                if collision_time / total_time > 0.3:
                    f.write("⚠ COLLISION CHECKING is expensive (>30% of time)\n")
                    f.write("  - Optimize collision_checker.find_occupied() implementation\n")
                    f.write("  - Use spatial hashing or octree for obstacle lookups\n")
                    f.write("  - Consider vectorizing collision checks\n\n")
                
                if topo_time / total_time > 0.2:
                    f.write("⚠ TOPOLOGICAL CHECKS are significant (>20% of time)\n")
                    f.write("  - Increase topo_resolution to coarsen path comparisons\n")
                    f.write("  - Reduce max_sample_num to limit connector evaluations\n")
                    f.write("  - Cache topological signatures for paths\n\n")
                
                f.write("\n" + "=" * 80 + "\n")
            
            print(f"\n[GraphBuilder] Profiling complete!")
            print(f"  Binary data:  {prof_path}")
            print(f"  Hotspot report: {report_path}")
            print(f"\nView the report:")
            print(f"  cat {report_path}")
            print(f"  # or")
            print(f"  less {report_path}")
            
        except Exception as e:
            print(f"[GraphBuilder] Warning: Could not generate report: {e}")
            print(f"  Raw profile data saved at: {prof_path}")
            print(f"  View with: python -m pstats {prof_path}")
    
    def get_timing_stats(self) -> Dict[str, float]:
        """
        Get raw timing statistics from the last graph construction.
        
        Returns:
            Dictionary with timing information in seconds and counters
        """
        return self.timing_stats.copy()
    
    def get_timing_report(self, detailed: bool = True) -> str:
        """
        Get a formatted timing report from the last graph construction.
        
        Args:
            detailed: If True, include all timing breakdowns; if False, summary only
            
        Returns:
            Formatted string report of timing statistics
        """
        if not self.timing_stats:
            return "No timing data available. Run create_graph() first."
        
        lines = []
        lines.append("=" * 70)
        lines.append("GraphBuilder Timing Report")
        lines.append("=" * 70)
        
        # Total time
        total = self.timing_stats.get('total_time', 0)
        lines.append(f"\nTotal Graph Construction Time: {total*1000:.2f} ms ({total:.4f} s)")
        
        if detailed and total > 0:
            lines.append("\nTime Breakdown:")
            lines.append("-" * 70)
            
            # Main phases with percentages
            phases = [
                ('initialization', 'Initialization'),
                ('sampling', 'Random Sampling'),
                ('collision_checking', 'Collision Checking'),
                ('visibility_checking', 'Visibility Checking'),
                ('connector_logic', 'Connector Logic & Topo Checks'),
                ('graph_pruning', 'Graph Pruning'),
            ]
            
            for key, label in phases:
                if key in self.timing_stats:
                    t = self.timing_stats[key]
                    pct = (t / total * 100) if total > 0 else 0
                    lines.append(f"  {label:.<40} {t*1000:>8.2f} ms ({pct:>5.1f}%)")
            
            # Graph statistics
            lines.append("\nGraph Statistics:")
            lines.append("-" * 70)
            
            stats = [
                ('num_samples', 'Total Samples Generated'),
                ('num_collision_free_samples', 'Collision-Free Samples'),
                ('num_collision_checks', 'Collision Checks Performed'),
                ('num_visibility_checks', 'Visibility Checks Performed'),
                ('num_topo_checks', 'Topological Checks Performed'),
                ('num_guards_created', 'Guard Nodes Created'),
                ('num_connectors_created', 'Connector Nodes Created'),
                ('num_final_nodes', 'Final Graph Nodes'),
            ]
            
            for key, label in stats:
                if key in self.timing_stats:
                    val = self.timing_stats[key]
                    if isinstance(val, float):
                        lines.append(f"  {label:.<40} {val:>8.1f}")
                    else:
                        lines.append(f"  {label:.<40} {val:>8}")
            
            # Efficiency metrics
            lines.append("\nEfficiency Metrics:")
            lines.append("-" * 70)
            
            collision_free_rate = self.timing_stats.get('collision_free_rate', 0)
            lines.append(f"  Collision-Free Rate..................... {collision_free_rate*100:>6.1f}%")
            
            num_samples = self.timing_stats.get('num_samples', 1)
            num_nodes = self.timing_stats.get('num_final_nodes', 0)
            if num_samples > 0:
                node_efficiency = ((num_nodes - 2) / num_samples * 100)  # Exclude start/goal
                lines.append(f"  Sample-to-Node Efficiency............... {node_efficiency:>6.1f}%")
            
            # Average times
            if 'avg_collision_check_time' in self.timing_stats:
                avg_coll = self.timing_stats['avg_collision_check_time'] * 1e6  # Convert to microseconds
                lines.append(f"  Avg Collision Check Time................ {avg_coll:>6.1f} μs")
            
            if 'avg_visibility_check_time' in self.timing_stats:
                avg_vis = self.timing_stats['avg_visibility_check_time'] * 1e6
                lines.append(f"  Avg Visibility Check Time............... {avg_vis:>6.1f} μs")
            
            if 'avg_topo_check_time' in self.timing_stats:
                avg_topo = self.timing_stats['avg_topo_check_time'] * 1000  # Convert to ms
                lines.append(f"  Avg Topological Check Time.............. {avg_topo:>6.2f} ms")
            
            # Performance analysis
            lines.append("\nPerformance Analysis:")
            lines.append("-" * 70)
            
            # Identify bottleneck
            phase_times = {
                k: v for k, v in self.timing_stats.items()
                if k in ['sampling', 'collision_checking', 'visibility_checking',
                        'connector_logic', 'graph_pruning']
            }
            
            if phase_times:
                slowest_phase = max(phase_times.items(), key=lambda x: x[1])
                slowest_pct = (slowest_phase[1] / total * 100) if total > 0 else 0
                
                phase_names = {
                    'sampling': 'Random Sampling',
                    'collision_checking': 'Collision Checking',
                    'visibility_checking': 'Visibility Checking',
                    'connector_logic': 'Connector Logic',
                    'graph_pruning': 'Graph Pruning'
                }
                
                lines.append(f"  Bottleneck: {phase_names.get(slowest_phase[0], slowest_phase[0])}")
                lines.append(f"  Time: {slowest_phase[1]*1000:.2f} ms ({slowest_pct:.1f}% of total)")
                
                # Suggestions
                suggestions = {
                    'sampling': [
                        "- Sampler is slow; check if safe zone sampling is being used efficiently"
                    ],
                    'collision_checking': [
                        "- Collision checking is the bottleneck",
                        "- Consider simplifying the collision checker implementation",
                        "- Check if occupancy grid lookups can be optimized"
                    ],
                    'visibility_checking': [
                        "- Visibility checks dominate (ray casting)",
                        "- Increase ray_resolution to reduce ray cast density",
                        "- Consider spatial indexing for guard node lookups"
                    ],
                    'connector_logic': [
                        "- Topological checks are expensive",
                        "- Increase topo_resolution to coarsen comparisons",
                        "- Many connectors being evaluated indicates dense graph"
                    ],
                    'graph_pruning': [
                        "- Pruning is slow (unusual)",
                        "- Check for excessively large graphs with many disconnected nodes"
                    ]
                }
                
                if slowest_phase[0] in suggestions:
                    lines.append("\n  Optimization Suggestions:")
                    for suggestion in suggestions[slowest_phase[0]]:
                        lines.append(f"  {suggestion}")
                
                # Collision-free rate analysis
                if collision_free_rate < 0.3:
                    lines.append("\n  ⚠ Low collision-free rate (<30%):")
                    lines.append("    - Many samples hitting obstacles")
                    lines.append("    - Consider adjusting sample_inflate to focus on free space")
                    lines.append("    - Use safe zones if available")
                elif collision_free_rate > 0.8:
                    lines.append("\n  ✓ High collision-free rate (>80%):")
                    lines.append("    - Excellent sampling in free space")
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def print_timing_report(self, detailed: bool = True) -> None:
        """
        Print formatted timing report to console.
        
        Args:
            detailed: If True, include all timing breakdowns; if False, summary only
        """
        print(self.get_timing_report(detailed=detailed))
    
    def enable_timing_diagnostics(self, enable: bool = True) -> None:
        """
        Enable or disable timing diagnostics.
        
        Args:
            enable: If True, collect timing data; if False, skip timing overhead
        """
        self.enable_timing = enable
        if not enable:
            self.timing_stats = {}


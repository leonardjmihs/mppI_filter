"""
Main TopoPRM planner interface.

This module provides the high-level TopoPRM planner that orchestrates sampling,
graph construction, path search, and path optimization to find multiple
topologically distinct paths.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from .sampler import EllipsoidSampler
from .graph import GraphBuilder
from .search import PathSearcher
from .path_processor import PathProcessor
from .nodes import Node
from ..collision_checker import CollisionChecker


class TopoPRM:
    """
    Topological Probabilistic Roadmap planner.
    
    Finds multiple topologically distinct paths between start and goal by building
    a roadmap graph that captures the topology of the free space.
    
    This class maintains the same interface as the original monolithic TopoPRM
    for backward compatibility while using a modular architecture internally.
    
    Args:
        occup_grid: Occupancy grid for collision checking
        sample_inflate: Ellipsoid inflation [x, y, z] for sampling region
        origin: Grid origin [x, y]
        resolution: Grid resolution (meters/cell)
        wh: Grid width and height [w, h] in cells
        max_raw_path: Maximum raw paths to find during search
        max_raw_path2: Maximum paths after first filtering
        max_sample_num: Maximum number of samples to generate
        reserve_num: Maximum number of final paths to return
        ratio_to_short: Maximum ratio to shortest path for filtering
        sample_sz_p: Probability of sampling from safe zones
        footprint: Robot footprint for collision checking
        occup_value: Occupancy values considered as obstacles
        max_time: Maximum planning time in seconds
        
    Example:
        >>> planner = TopoPRM(occup_grid, resolution=0.4)
        >>> paths, samples = planner.findTopoPaths(start, goal)
    """
    
    def __init__(
        self,
        occup_grid: Optional[npt.NDArray[np.integer]],
        sample_inflate: Union[List[float], Tuple[float, ...]] = [5.0, 5.0, 0.0],
        origin: Union[List[float], Tuple[float, ...]] = [0.0, 0.0],
        resolution: float = 0.1,
        wh: Union[List[int], Tuple[int, ...]] = [10, 10],
        max_raw_path: int = 10,
        max_raw_path2: int = 5,
        max_sample_num: int = 200,
        reserve_num: int = 5,
        ratio_to_short: float = 10.0,
        sample_sz_p: float = 0.5,
        footprint: List[List[float]] = [[0.0, 0.0]],
        occup_value: List[int] = [100],
        max_time: float = 0.05
    ) -> None:
        """Initialize TopoPRM planner with configuration parameters."""
        # Validate resolution
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")
        
        # Validate grid dimensions
        if not isinstance(wh, (list, tuple, np.ndarray)) or len(wh) != 2:
            raise ValueError(f"wh must be 2-element array [width, height], got {wh}")
        if any(x <= 0 for x in wh):
            raise ValueError(f"wh values must be positive, got {wh}")
        
        # Validate origin
        if not isinstance(origin, (list, tuple, np.ndarray)) or len(origin) != 2:
            raise ValueError(f"origin must be 2-element array [x, y], got {origin}")
        
        # Validate path limits
        if max_raw_path <= 0:
            raise ValueError(f"max_raw_path must be positive, got {max_raw_path}")
        if max_raw_path2 <= 0:
            raise ValueError(f"max_raw_path2 must be positive, got {max_raw_path2}")
        if max_raw_path2 > max_raw_path:
            raise ValueError(f"max_raw_path2 ({max_raw_path2}) cannot exceed max_raw_path ({max_raw_path})")
        
        # Validate sample parameters
        if max_sample_num <= 0:
            raise ValueError(f"max_sample_num must be positive, got {max_sample_num}")
        if not 0.0 <= sample_sz_p <= 1.0:
            raise ValueError(f"sample_sz_p must be in [0, 1], got {sample_sz_p}")
        
        # Validate reserve and ratio
        if reserve_num <= 0:
            raise ValueError(f"reserve_num must be positive, got {reserve_num}")
        if ratio_to_short <= 1.0:
            raise ValueError(f"ratio_to_short must be > 1.0, got {ratio_to_short}")
        
        # Validate time limit
        if max_time <= 0:
            raise ValueError(f"max_time must be positive, got {max_time}")
        
        # Store configuration using private attributes
        # We'll use properties to intercept changes and update collision checker
        self._resolution = resolution
        self._wh = np.array(wh)
        self._origin = np.array(origin)
        self._occup_grid = occup_grid
        self._occup_value = occup_value
        
        # Initialize collision checker with current grid parameters
        # This is decoupled from the planner class
        self.collision_checker = CollisionChecker(
            occup_grid=self._occup_grid,
            origin=self._origin,
            resolution=self._resolution,
            wh=self._wh,
            occup_value=self._occup_value
        )
        
        # Initialize components
        self.sampler = EllipsoidSampler(
            sample_inflate=tuple(sample_inflate),
            sample_sz_p=sample_sz_p
        )
        
        self.graph_builder = GraphBuilder(
            collision_checker=self.collision_checker,
            ray_resolution=0.1,
            topo_resolution=0.1,
            max_sample_num=max_sample_num,
            max_time=max_time
        )
        
        self.path_searcher = PathSearcher(
            max_raw_path=max_raw_path,
            max_raw_path2=max_raw_path2
        )
        
        self.path_processor = PathProcessor(
            reserve_num=reserve_num,
            ratio_to_short=ratio_to_short
        )
        
        # State
        self.graph: List[Node] = []
        self.safe_zones: Optional[np.ndarray] = None
        
        # Timing diagnostics
        self.timing_stats: Dict[str, float] = {}
        self.enable_timing: bool = True
        
        # Legacy attributes for backward compatibility
        self.sample_inflate = sample_inflate
        self.max_sample_num = max_sample_num
        self.ray_resolution = 0.2
        self.topo_resolution = 0.2
        self.reserve_num = reserve_num
        self.ratio_to_short = ratio_to_short
        self.max_raw_path = max_raw_path
        self.max_raw_path2 = max_raw_path2
        self.max_time = max_time
        self.sample_sz_p = sample_sz_p
        self.footprint = np.array(footprint)
    
    def findTopoPaths(
        self,
        start: Union[npt.NDArray[np.floating], List[float]],
        end: Union[npt.NDArray[np.floating], List[float]],
        start_pts: List[Any] = [],
        end_pts: List[Any] = [],
        reset: bool = True
    ) -> Tuple[Optional[List[List[npt.NDArray[np.floating]]]], List[npt.NDArray[np.floating]]]:
        """
        Find topologically distinct paths from start to end.
        
        This is the main entry point for the planner. It creates a roadmap graph,
        searches for multiple paths, shortcuts them, and returns topologically
        distinct paths.

        Args:
            start: Start position [x, y, theta]
            end: Goal position [x, y, theta]
            start_pts: Optional start waypoints (unused, for compatibility)
            end_pts: Optional end waypoints (unused, for compatibility)
            reset: If True, rebuild graph from scratch; if False, reuse existing

        Returns:
            tuple: (paths, samples) where paths is list of path arrays and samples
                   is list of all collision-free sample points tested
                   
        Raises:
            ValueError: If start/end are invalid
            RuntimeError: If occupancy grid is not set
            
        Note:
            Timing statistics are stored in self.timing_stats after execution.
            Use get_timing_report() to get a formatted report.
        """
        # Reset timing stats for this planning call
        if self.enable_timing:
            self.timing_stats = {}
            t_total_start = time.perf_counter()
        
        # Validate occupancy grid is set
        if self.occup_grid is None:
            raise RuntimeError("Occupancy grid not set! Set planner.occup_grid before planning.")
        
        # Validate and prepare start/end
        if self.enable_timing:
            t_validation_start = time.perf_counter()
        try:
            start = np.array(start).reshape((3, 1))
            end = np.array(end).reshape((3, 1))
        except (ValueError, TypeError) as e:
            raise ValueError(f"start and end must be convertible to 3D arrays: {e}")
        
        if not np.all(np.isfinite(start)):
            raise ValueError(f"start must contain finite values, got {start.ravel()}")
        if not np.all(np.isfinite(end)):
            raise ValueError(f"end must contain finite values, got {end.ravel()}")
        
        start[2] = 0.0  # Ignore theta for planning
        end[2] = 0.0
        
        if self.enable_timing:
            self.timing_stats['validation'] = time.perf_counter() - t_validation_start
        
        # Configure sampler
        if self.enable_timing:
            t_config_start = time.perf_counter()
        self.sampler.configure(start, end)
        if self.safe_zones is not None:
            self.sampler.safe_zones = self.safe_zones
        
        if self.enable_timing:
            self.timing_stats['sampler_config'] = time.perf_counter() - t_config_start
        
        # Build or update graph
        if self.enable_timing:
            t_graph_start = time.perf_counter()
        if reset:
            self.graph = []
            self.graph, samples = self.graph_builder.create_graph(
                self.sampler, start, end, self.graph
            )
        else:
            # Update existing graph with new start/goal
            if self.enable_timing:
                t_update_start = time.perf_counter()
            self.graph = self.graph_builder.update_start_goal(
                self.graph, start, end
            )
            if self.enable_timing:
                self.timing_stats['graph_update'] = time.perf_counter() - t_update_start
            # Add more samples
            self.graph, samples = self.graph_builder.create_graph(
                self.sampler, start, end, self.graph
            )
        
        if self.enable_timing:
            self.timing_stats['graph_construction'] = time.perf_counter() - t_graph_start
            self.timing_stats['num_nodes'] = len(self.graph)
            self.timing_stats['num_samples'] = len(samples)
        
        # Search for paths
        if self.enable_timing:
            t_search_start = time.perf_counter()
        raw_paths = self.path_searcher.search_paths(self.graph)
        
        if self.enable_timing:
            self.timing_stats['path_search'] = time.perf_counter() - t_search_start
            self.timing_stats['num_raw_paths'] = len(raw_paths) if raw_paths is not None else 0
        
        if raw_paths is None:
            if self.enable_timing:
                self.timing_stats['total_time'] = time.perf_counter() - t_total_start
            return None, samples
        
        # Optimize paths
        if self.enable_timing:
            t_shortcut_start = time.perf_counter()
        shortcut_paths = self.path_processor.shortcut_paths(
            raw_paths, self.graph_builder._line_visible
        )
        
        if self.enable_timing:
            self.timing_stats['path_shortcut'] = time.perf_counter() - t_shortcut_start
        
        # Remove topologically equivalent paths
        if self.enable_timing:
            t_prune_start = time.perf_counter()
        filtered_paths = self.path_processor.prune_equivalent(
            shortcut_paths,
            self.graph_builder.topo_resolution,
            self.graph_builder._line_visible
        )
        
        if self.enable_timing:
            self.timing_stats['path_pruning'] = time.perf_counter() - t_prune_start
            self.timing_stats['num_filtered_paths'] = len(filtered_paths) if filtered_paths is not None else 0
        
        # Select best paths
        if self.enable_timing:
            t_select_start = time.perf_counter()
        selected_paths = self.path_processor.select_short_paths(
            filtered_paths, self.graph_builder._line_visible
        )
        
        if self.enable_timing:
            self.timing_stats['path_selection'] = time.perf_counter() - t_select_start
            self.timing_stats['num_final_paths'] = len(selected_paths) if selected_paths is not None else 0
            self.timing_stats['total_time'] = time.perf_counter() - t_total_start
        
        return selected_paths, samples
    
    def get_timing_stats(self) -> Dict[str, float]:
        """
        Get raw timing statistics from the last planning call.
        
        Returns:
            Dictionary with timing information in seconds and counters
            
        Example:
            >>> planner.findTopoPaths(start, goal)
            >>> stats = planner.get_timing_stats()
            >>> print(f"Total time: {stats['total_time']:.4f}s")
        """
        return self.timing_stats.copy()
    
    def get_timing_report(self, detailed: bool = True) -> str:
        """
        Get a formatted timing report from the last planning call.
        
        Args:
            detailed: If True, include all timing breakdowns; if False, summary only
            
        Returns:
            Formatted string report of timing statistics
            
        Example:
            >>> planner.findTopoPaths(start, goal)
            >>> print(planner.get_timing_report())
        """
        if not self.timing_stats:
            return "No timing data available. Run findTopoPaths() first."
        
        lines = []
        lines.append("=" * 70)
        lines.append("TopoPRM Timing Report")
        lines.append("=" * 70)
        
        # Total time (most important)
        total = self.timing_stats.get('total_time', 0)
        lines.append(f"\nTotal Planning Time: {total*1000:.2f} ms ({total:.4f} s)")
        
        if detailed and total > 0:
            lines.append("\nTime Breakdown:")
            lines.append("-" * 70)
            
            # Main phases with percentages
            phases = [
                ('validation', 'Input Validation'),
                ('sampler_config', 'Sampler Configuration'),
                ('graph_construction', 'Graph Construction'),
                ('graph_update', 'Graph Update (incremental)'),
                ('path_search', 'Path Search (DFS)'),
                ('path_shortcut', 'Path Shortcutting'),
                ('path_pruning', 'Topological Pruning'),
                ('path_selection', 'Path Selection & Optimization'),
            ]
            
            for key, label in phases:
                if key in self.timing_stats:
                    t = self.timing_stats[key]
                    pct = (t / total * 100) if total > 0 else 0
                    lines.append(f"  {label:.<40} {t*1000:>8.2f} ms ({pct:>5.1f}%)")
            
            # Counters
            lines.append("\nGraph Statistics:")
            lines.append("-" * 70)
            
            counters = [
                ('num_samples', 'Samples Generated'),
                ('num_nodes', 'Graph Nodes'),
                ('num_raw_paths', 'Raw Paths Found'),
                ('num_filtered_paths', 'After Topological Filtering'),
                ('num_final_paths', 'Final Paths Returned'),
            ]
            
            for key, label in counters:
                if key in self.timing_stats:
                    val = self.timing_stats[key]
                    if isinstance(val, float):
                        lines.append(f"  {label:.<40} {val:>8.1f}")
                    else:
                        lines.append(f"  {label:.<40} {val:>8}")
            
            # Performance insights
            lines.append("\nPerformance Analysis:")
            lines.append("-" * 70)
            
            # Identify bottlenecks
            phase_times = {k: v for k, v in self.timing_stats.items() 
                          if k in ['graph_construction', 'path_search', 'path_shortcut', 
                                  'path_pruning', 'path_selection']}
            
            if phase_times:
                slowest_phase = max(phase_times.items(), key=lambda x: x[1])
                slowest_pct = (slowest_phase[1] / total * 100) if total > 0 else 0
                
                phase_names = {
                    'graph_construction': 'Graph Construction',
                    'path_search': 'Path Search',
                    'path_shortcut': 'Path Shortcutting',
                    'path_pruning': 'Topological Pruning',
                    'path_selection': 'Path Selection'
                }
                
                lines.append(f"  Bottleneck: {phase_names.get(slowest_phase[0], slowest_phase[0])}")
                lines.append(f"  Time: {slowest_phase[1]*1000:.2f} ms ({slowest_pct:.1f}% of total)")
                
                # Suggestions based on bottleneck
                suggestions = {
                    'graph_construction': [
                        "- Reduce max_sample_num to decrease sampling iterations",
                        "- Increase ray_resolution to reduce collision check density",
                        "- Consider using cached/reused graphs (reset=False)"
                    ],
                    'path_search': [
                        "- Reduce max_raw_path to limit DFS exploration",
                        "- Simplify graph by reducing max_sample_num",
                        "- Check for overly connected graph (too many edges)"
                    ],
                    'path_shortcut': [
                        "- This is often necessary; consider if fewer raw paths needed",
                        "- Check discretization resolution in path_processor"
                    ],
                    'path_pruning': [
                        "- Increase topo_resolution to speed up equivalence checks",
                        "- Reduce number of paths to compare (lower max_raw_path2)"
                    ],
                    'path_selection': [
                        "- Reduce reserve_num to select fewer final paths",
                        "- This phase includes final shortcutting optimization"
                    ]
                }
                
                if slowest_phase[0] in suggestions:
                    lines.append("\n  Optimization Suggestions:")
                    for suggestion in suggestions[slowest_phase[0]]:
                        lines.append(f"  {suggestion}")
                
                # Sampling efficiency
                num_samples = self.timing_stats.get('num_samples', 0)
                num_nodes = self.timing_stats.get('num_nodes', 2)
                if num_samples > 0:
                    efficiency = (num_nodes - 2) / num_samples * 100  # Exclude start/goal
                    lines.append(f"\n  Sample Efficiency: {efficiency:.1f}% ({num_nodes-2}/{num_samples} samples became nodes)")
                    if efficiency < 20:
                        lines.append("    → Low efficiency: Many samples in collision or redundant")
                    elif efficiency > 60:
                        lines.append("    → High efficiency: Good sampling strategy")
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def print_timing_report(self, detailed: bool = True) -> None:
        """
        Print formatted timing report to console.
        
        Args:
            detailed: If True, include all timing breakdowns; if False, summary only
            
        Example:
            >>> planner.findTopoPaths(start, goal)
            >>> planner.print_timing_report()
        """
        print(self.get_timing_report(detailed=detailed))
    
    def reset_timing_stats(self) -> None:
        """Clear timing statistics."""
        self.timing_stats = {}
    
    def enable_timing_diagnostics(self, enable: bool = True) -> None:
        """
        Enable or disable timing diagnostics.
        
        Args:
            enable: If True, collect timing data; if False, skip timing overhead
            
        Note:
            Timing overhead is minimal (~microseconds) but can be disabled
            for maximum performance in production.
        """
        self.enable_timing = enable
        if not enable:
            self.timing_stats = {}
    
    # Properties to intercept attribute access and keep collision checker in sync
    @property
    def occup_grid(self) -> Optional[npt.NDArray[np.integer]]:
        """Get the occupancy grid."""
        return self._occup_grid
    
    @occup_grid.setter
    def occup_grid(self, value: Optional[npt.NDArray[np.integer]]) -> None:
        """Set the occupancy grid and update collision checker."""
        self._occup_grid = value
        self.collision_checker.occup_grid = value
    
    @property
    def origin(self) -> npt.NDArray[np.floating]:
        """Get the grid origin."""
        return self._origin
    
    @origin.setter
    def origin(self, value: Union[List[float], npt.NDArray[np.floating]]) -> None:
        """Set the grid origin and update collision checker."""
        self._origin = np.array(value)
        self.collision_checker.origin = self._origin
    
    @property
    def resolution(self) -> float:
        """Get the grid resolution."""
        return self._resolution
    
    @resolution.setter
    def resolution(self, value: float) -> None:
        """Set the grid resolution and update collision checker."""
        self._resolution = value
        self.collision_checker.resolution = value
    
    @property
    def wh(self) -> npt.NDArray[np.integer]:
        """Get the grid width and height."""
        return self._wh
    
    @wh.setter
    def wh(self, value: Union[List[int], npt.NDArray[np.integer]]) -> None:
        """Set the grid width and height and update collision checker."""
        self._wh = np.array(value)
        self.collision_checker.wh = self._wh
    
    @property
    def occup_value(self) -> Union[List[int], int]:
        """Get the occupancy threshold value(s)."""
        return self._occup_value
    
    @occup_value.setter
    def occup_value(self, value: Union[List[int], int]) -> None:
        """Set the occupancy threshold and update collision checker."""
        self._occup_value = value
        # Update collision checker's occup_value
        if isinstance(value, list):
            self.collision_checker.occup_value = value[0] if value else 100
        else:
            self.collision_checker.occup_value = value

    def set_occupancy_grid(
        self,
        occup_grid: npt.NDArray[np.integer],
        origin: Optional[Union[List[float], Tuple[float, ...]]] = None,
        resolution: Optional[float] = None,
        wh: Optional[Union[List[int], Tuple[int, ...]]] = None
    ) -> None:
        """
        Update the occupancy grid and optionally its parameters.
        
        This method updates both the planner's grid and the collision checker's grid.
        
        Args:
            occup_grid: New occupancy grid
            origin: New origin (optional, keeps current if None)
            resolution: New resolution (optional, keeps current if None)
            wh: New grid dimensions (optional, keeps current if None)
            
        Example:
            >>> planner.set_occupancy_grid(new_grid, origin=[0, 0])
        """
        # Update planner's grid parameters
        self.occup_grid = occup_grid
        if origin is not None:
            self.origin = np.array(origin)
        if resolution is not None:
            self.resolution = resolution
        if wh is not None:
            self.wh = np.array(wh)
        
        # Update collision checker
        self.collision_checker.update_grid(
            occup_grid=self.occup_grid,
            origin=self.origin,
            resolution=self.resolution,
            wh=self.wh
        )

    def cutToSafe(self, path):
        temp_path, idx = self.path_processor.discretize_path(path, 10)
        for i, pt in enumerate(temp_path):
            if self.find_occupied(self.collision_checker.occup_grid, pt, mark_out_of_bounds=True):
                break
        new_path = path[:idx[i]+1]
        new_path.append(temp_path[i])
        return new_path

    def cutToMax(self, path, max_len):
        new_path = self.path_processor.cutToMax(path, max_len)
        return new_path

    def find_occupied(
        self, 
        occup: Any, 
        pt: Union[npt.NDArray[np.floating], List[float]], 
        mark_out_of_bounds: bool = False
    ) -> bool:
        """
        Legacy method for backward compatibility.
        
        Check if a point is in collision with obstacles.
        
        Args:
            occup: Occupancy grid (ignored, uses self.occup_grid)
            pt: Point to check [x, y, z]
            mark_out_of_bounds: If True, treat out-of-bounds as occupied
            
        Returns:
            True if point is occupied, False otherwise
        """
        return self.collision_checker.find_occupied(pt, mark_out_of_bounds)

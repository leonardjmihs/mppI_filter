"""
Path processing utilities for TopoPRM.

This module provides path optimization, discretization, topological equivalence
checking, and path selection based on length and diversity.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt


class PathProcessor:
    """
    Processes and optimizes paths from the roadmap graph.
    
    Handles path shortcutting, discretization, topological equivalence checking,
    and selection of diverse high-quality paths.
    
    Attributes:
        reserve_num: Maximum number of final paths to return
        ratio_to_short: Maximum ratio to shortest path for filtering
    """
    
    def __init__(self, reserve_num: int = 5, ratio_to_short: float = 10.0) -> None:
        """
        Initialize the path processor.
        
        Args:
            reserve_num: Maximum paths to return after selection
            ratio_to_short: Filter out paths longer than ratio * shortest_path
            
        Raises:
            ValueError: If parameters are invalid
        """
        if reserve_num <= 0:
            raise ValueError(f"reserve_num must be positive, got {reserve_num}")
        if ratio_to_short <= 1.0:
            raise ValueError(f"ratio_to_short must be > 1.0, got {ratio_to_short}")
        
        self.reserve_num = reserve_num
        self.ratio_to_short = ratio_to_short
    
    def shortcut_paths(
        self,
        paths: Optional[List[List[npt.NDArray[np.floating]]]],
        line_visible: Callable[..., bool]
    ) -> Optional[List[List[npt.NDArray[np.floating]]]]:
        """
        Shortcut all paths by removing unnecessary waypoints.
        
        Args:
            paths: List of paths to shortcut
            line_visible: Function to check line-of-sight between points
            
        Returns:
            List of shortcut paths, or None if input is None
        """
        if paths is None:
            return None
        
        shortcut_paths = []
        for path in paths:
            # Discretize then apply exponential shortcut
            discretized, _ = self.discretize_path(path, 20)
            shortcut = self._exp_shortcut(discretized, line_visible)
            shortcut_paths.append(shortcut)
        
        return shortcut_paths
    
    def select_short_paths(
        self,
        paths: Optional[List[List[npt.NDArray[np.floating]]]],
        line_visible: Callable[..., bool]
    ) -> Optional[List[List[npt.NDArray[np.floating]]]]:
        """
        Select diverse short paths based on length criteria.
        
        Selects up to reserve_num paths, filtering by ratio to shortest path.
        Also applies final shortcutting to selected paths.
        
        Args:
            paths: Candidate paths
            line_visible: Function for line-of-sight checking
            
        Returns:
            Selected and optimized paths, or None if input is None
        """
        if paths is None or len(paths) == 0:
            return None
        
        min_length = float('inf')
        selected_paths = []
        
        # Select paths by length
        for i in range(self.reserve_num):
            if len(paths) == 0:
                break
            
            # Find shortest remaining path
            shortest_idx = self._find_shortest_path(paths)
            
            if i == 0:
                # First path sets the baseline
                selected_paths.append(paths[shortest_idx])
                min_length = self.path_length(paths[shortest_idx])
                paths.pop(shortest_idx)
            else:
                # Check length ratio
                ratio = self.path_length(paths[shortest_idx]) / min_length
                if ratio < self.ratio_to_short:
                    selected_paths.append(paths[shortest_idx])
                    paths.pop(shortest_idx)
                else:
                    break
        
        # Apply final optimization to selected paths
        for i, path in enumerate(selected_paths):
            discretized, _ = self.discretize_path(path, 20)
            selected_paths[i] = self._lazy_shortcut(discretized, 10, line_visible)
        
        return selected_paths
    
    def prune_equivalent(
        self,
        paths: Optional[List[List[npt.NDArray[np.floating]]]],
        topo_resolution: float,
        line_visible: Callable[..., bool]
    ) -> Optional[List[List[npt.NDArray[np.floating]]]]:
        """
        Remove topologically equivalent paths.
        
        Args:
            paths: Paths to filter
            topo_resolution: Resolution for equivalence checking
            line_visible: Function for line-of-sight checking
            
        Returns:
            Filtered list of topologically distinct paths
        """
        if paths is None or len(paths) == 0:
            return None if paths is None else []
        
        unique_paths = []
        unique_indices = [0]  # First path is always unique
        
        for i in range(1, len(paths)):
            is_unique = True
            
            # Compare against existing unique paths
            for j in unique_indices:
                if self.same_topo_path(
                    paths[i], paths[j], topo_resolution, line_visible
                ):
                    is_unique = False
                    break
            
            if is_unique:
                unique_indices.append(i)
        
        # Collect unique paths
        for idx in unique_indices:
            unique_paths.append(paths[idx])
        
        return unique_paths
    
    @staticmethod
    def same_topo_path(
        path1: List[npt.NDArray[np.floating]],
        path2: List[npt.NDArray[np.floating]],
        thresh: float,
        line_visible: Callable[..., bool]
    ) -> bool:
        """
        Check if two paths are topologically equivalent (homotopic).
        
        Discretizes both paths and checks if corresponding waypoints have
        clear line-of-sight. If all pairs are visible to each other, the
        paths are homotopic.
        
        Args:
            path1: First path
            path2: Second path
            thresh: Discretization resolution
            line_visible: Function(pos1, pos2, check_endpoints) -> bool
            
        Returns:
            True if paths are topologically equivalent
            
        Raises:
            ValueError: If paths or threshold are invalid
            TypeError: If line_visible is not callable
        """
        if not isinstance(path1, list) or len(path1) < 2:
            raise ValueError(f"path1 must be list with >= 2 waypoints, got length {len(path1) if isinstance(path1, list) else 'N/A'}")
        if not isinstance(path2, list) or len(path2) < 2:
            raise ValueError(f"path2 must be list with >= 2 waypoints, got length {len(path2) if isinstance(path2, list) else 'N/A'}")
        if thresh <= 0:
            raise ValueError(f"thresh must be positive, got {thresh}")
        if not callable(line_visible):
            raise TypeError("line_visible must be callable")
        
        len1 = PathProcessor.path_length(path1)
        len2 = PathProcessor.path_length(path2)
        max_len = max(max(len1, len2), 2.0)
        
        # Discretize both paths to same number of points
        pt_num = int(np.ceil(max_len / thresh))
        pts1, _ = PathProcessor.discretize_path(path1, pt_num)
        pts2, _ = PathProcessor.discretize_path(path2, pt_num)
        
        # Check for NaN (shouldn't happen)
        if np.any(np.isnan(pts1)) or np.any(np.isnan(pts2)):
            return False
        
        # Check visibility between corresponding points
        # Try to use vectorized batch checking if available (via __self__ for bound methods)
        if hasattr(line_visible, '__self__') and hasattr(line_visible.__self__, '_line_visible_batch'):
            # Use vectorized batch checking for better performance
            pts1_array = np.array(pts1)
            pts2_array = np.array(pts2)
            visible = line_visible.__self__._line_visible_batch(
                pts1_array, pts2_array, check_endpoints=False
            )
            # Return True only if ALL lines are visible
            return bool(np.all(visible))
        else:
            # Fall back to sequential checking
            for i in range(pt_num):
                if not line_visible(pts1[i], pts2[i], check_endpoints=False):
                    return False
            return True
    
    @staticmethod
    def path_length(path: List[npt.NDArray[np.floating]]) -> float:
        """
        Compute total Euclidean length of a path.
        
        Args:
            path: List of waypoints
            
        Returns:
            Total path length
        """
        if len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(path) - 1):
            length += float(np.linalg.norm(path[i + 1] - path[i]))
        
        return length

    @staticmethod
    def cutToMax(path, max_len):
        np_path = np.array(path)
        lengths = np.cumsum(np.linalg.norm(np_path[:-1]-np_path[1:], axis=1))
        if lengths[-1] < max_len:
            return path
        else:
            last_exist = np.argmax(lengths>max_len)
            backtrack = lengths[last_exist] - max_len
            new_end = np_path[last_exist+1]-backtrack*(np_path[last_exist+1]-np_path[last_exist])/ \
                                        np.linalg.norm(np_path[last_exist+1]-np_path[last_exist])
            new_path = path[:last_exist+1]
            new_path.append(new_end)
            return new_path

    @staticmethod
    def discretize_path(
        path: List[npt.NDArray[np.floating]], 
        pt_num: int
    ) -> Tuple[List[npt.NDArray[np.floating]], List[int]]:
        """
        Discretize a path into a fixed number of evenly-spaced waypoints.
        
        Args:
            path: Original path waypoints
            pt_num: Number of points in discretized path
            
        Returns:
            tuple: (discretized_path, segment_indices) where segment_indices[i]
                   is the original segment index for discretized point i
                   
        Raises:
            ValueError: If path or pt_num are invalid
        """
        # if not isinstance(path, list) or len(path) == 0:
        #     raise ValueError(f"path must be non-empty list, got {type(path).__name__} with length {len(path) if isinstance(path, list) else 'N/A'}")
        
        if pt_num <= 0:
            raise ValueError(f"pt_num must be positive, got {pt_num}")
        
        if len(path) < 2:
            return path, [0] * len(path)
        
        # Compute cumulative lengths
        path = np.array(path)
        lengths = [0.0]
        for i in range(len(path) - 1):
            seg_len = float(np.linalg.norm(path[i + 1] - path[i]))
            lengths.append(lengths[-1] + seg_len)
        
        total_length = lengths[-1]
        
        if total_length < 1e-6:
            return path, list(range(len(path)))
        
        # Interpolate evenly-spaced points
        dl = total_length / (pt_num - 1.0)
        discretized = []
        indices = []
        
        for i in range(pt_num):
            target_length = i * dl
            
            # Find segment containing target_length
            seg_idx = 0
            for j in range(len(lengths) - 1):
                if lengths[j] - 1e-4 <= target_length <= lengths[j + 1] + 1e-4:
                    seg_idx = j
                    break
            
            # Interpolate within segment
            if lengths[seg_idx + 1] - lengths[seg_idx] > 1e-6:
                ratio = (target_length - lengths[seg_idx]) / (lengths[seg_idx + 1] - lengths[seg_idx])
            else:
                ratio = 0.0
            
            interp_pt = (1 - ratio) * path[seg_idx] + ratio * path[seg_idx + 1]
            discretized.append(interp_pt)
            indices.append(seg_idx)
        
        return discretized, indices
    
    def _find_shortest_path(self, paths: List[List[npt.NDArray[np.floating]]]) -> int:
        """Find index of shortest path."""
        min_length = float('inf')
        shortest_idx = -1
        
        for i, path in enumerate(paths):
            length = self.path_length(path)
            if length < min_length:
                shortest_idx = i
                min_length = length
        
        return shortest_idx

    
    def _lazy_shortcut(
        self,
        path: List[npt.NDArray[np.floating]],
        iterations: int,
        line_visible: Callable[..., bool]
    ) -> List[npt.NDArray[np.floating]]:
        """
        Lazy shortcut algorithm: randomly try to connect distant waypoints.
        
        Args:
            path: Path to shortcut
            iterations: Number of random attempts
            line_visible: Function to check line-of-sight
            
        Returns:
            Shortcut path
        """
        for _ in range(iterations):
            if len(path) <= 2:
                break
            
            # Random indices
            idx1 = int(np.random.rand() * (len(path) - 1))
            idx2 = int(np.random.rand() * (len(path) - 1))
            
            if idx2 < idx1:
                idx1, idx2 = idx2, idx1
            
            # Ensure gap between indices
            if idx2 - idx1 < 2:
                continue
            
            # Try to shortcut
            if line_visible(path[idx1], path[idx2]):
                del path[idx1 + 1:idx2]
        
        return path
    
    def _exp_shortcut(
        self, 
        path: List[npt.NDArray[np.floating]], 
        line_visible: Callable[..., bool]
    ) -> List[npt.NDArray[np.floating]]:
        """
        Exponential shortcut: greedily skip ahead as far as possible.
        
        Args:
            path: Path to shortcut
            line_visible: Function to check line-of-sight
            
        Returns:
            Shortcut path
        """
        if len(path) <= 2:
            return path
        
        # Discretize first
        discretized, _ = self.discretize_path(path, 20)
        
        new_path = [discretized[0]]
        
        # Greedily skip forward
        for i in range(2, len(discretized)):
            if not line_visible(discretized[i], new_path[-1]):
                # Can't skip to i, so add i-1
                new_path.append(discretized[i - 1])
        
        new_path.append(discretized[-1])
        
        return new_path

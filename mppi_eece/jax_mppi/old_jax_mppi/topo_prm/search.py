"""
Path search algorithms for TopoPRM.

This module implements depth-first search to find multiple paths through
the roadmap graph.
"""

from typing import List, Optional

import numpy as np
import numpy.typing as npt

from .nodes import Node


class PathSearcher:
    """
    Searches for multiple paths in the topological roadmap graph.
    
    Uses depth-first search to enumerate paths from start to goal, grouping
    results by path length to prioritize shorter paths.
    
    Attributes:
        max_raw_path: Maximum number of paths to find during DFS
        max_raw_path2: Maximum number of paths to return after filtering
    """
    
    def __init__(self, max_raw_path: int = 10, max_raw_path2: int = 5) -> None:
        """
        Initialize the path searcher.
        
        Args:
            max_raw_path: Stop DFS after finding this many paths
            max_raw_path2: Maximum paths to return after length filtering
            
        Raises:
            ValueError: If parameters are invalid
        """
        if max_raw_path <= 0:
            raise ValueError(f"max_raw_path must be positive, got {max_raw_path}")
        if max_raw_path2 <= 0:
            raise ValueError(f"max_raw_path2 must be positive, got {max_raw_path2}")
        if max_raw_path2 > max_raw_path:
            raise ValueError(f"max_raw_path2 ({max_raw_path2}) cannot exceed max_raw_path ({max_raw_path})")
        
        self.max_raw_path = max_raw_path
        self.max_raw_path2 = max_raw_path2
    
    def search_paths(
        self, 
        graph: List[Node]
    ) -> Optional[List[List[npt.NDArray[np.floating]]]]:
        """
        Search for all paths from start (node 0) to goal (node 1).
        
        Performs DFS to find multiple paths, then filters by path length
        to prioritize shorter routes.
        
        Args:
            graph: Roadmap graph with start at index 0, goal at index 1
            
        Returns:
            List of paths (each path is list of positions), or None if no path found
            
        Raises:
            ValueError: If graph is invalid
        """
        if not isinstance(graph, list):
            raise TypeError(f"graph must be a list, got {type(graph).__name__}")
        
        if len(graph) < 2:
            return None
        
        # Validate start and goal nodes exist
        if graph[0].id != 0:
            raise ValueError(f"First node must have id=0 (start), got {graph[0].id}")
        if graph[1].id != 1:
            raise ValueError(f"Second node must have id=1 (goal), got {graph[1].id}")
        
        # Run DFS from start node
        visited = [graph[0]]
        raw_paths = []
        raw_paths = self._depth_first_search(visited, raw_paths)
        
        if not raw_paths:
            return None
        
        # Group paths by number of nodes
        min_nodes = min(len(path) for path in raw_paths)
        max_nodes = max(len(path) for path in raw_paths)
        
        # path_list[n] contains indices of paths with n nodes
        path_list = [[] for _ in range(max_nodes + 1)]
        
        for i, path in enumerate(raw_paths):
            path_list[len(path)].append(i)
        
        # Select paths starting from shortest
        filtered_paths = []
        for num_nodes in range(min_nodes, max_nodes + 1):
            for path_idx in path_list[num_nodes]:
                filtered_paths.append(raw_paths[path_idx])
                if len(filtered_paths) >= self.max_raw_path2:
                    return filtered_paths
        
        return filtered_paths
    
    def _depth_first_search(
        self,
        visited: List[Node],
        raw_paths: List[List[npt.NDArray[np.floating]]]
    ) -> List[List[npt.NDArray[np.floating]]]:
        """
        Recursively search for paths using depth-first search.
        
        Explores the graph by visiting neighbors, backtracking when necessary.
        Stops when max_raw_path paths are found.
        
        Args:
            visited: Stack of currently visited nodes
            raw_paths: Accumulator for found paths
            
        Returns:
            Updated raw_paths with newly discovered paths
        """
        current = visited[-1]
        
        # Sanity check
        assert len(visited) > 0, "Visited stack should not be empty"
        
        # Check if any neighbor is the goal (node id 1)
        for neighbor in current.neighbors:
            if neighbor.id == 1:
                # Found a path to goal
                path = [node.pos for node in visited]
                path.append(neighbor.pos)
                
                # Sanity check: path should have at least 2 points
                assert len(path) >= 2, f"Path too short: {len(path)} points"
                
                raw_paths.append(path)
                
                if len(raw_paths) >= self.max_raw_path:
                    return raw_paths
                break
        
        # Explore other neighbors
        for neighbor in current.neighbors:
            # Skip goal (already handled) and visited nodes
            if neighbor.id == 1:
                continue
            
            if any(neighbor.id == node.id for node in visited):
                continue
            
            # Recurse on unvisited neighbor
            visited.append(neighbor)
            self._depth_first_search(visited, raw_paths)
            
            if len(raw_paths) >= self.max_raw_path:
                return raw_paths
            
            visited.pop()
        
        return raw_paths

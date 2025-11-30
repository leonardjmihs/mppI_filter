"""
Collision checking for TopoPRM planner.

This module provides collision detection functionality using occupancy grids.
It is decoupled from the main planner to allow for easier testing and reuse.
"""

from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
import jax
import jax.numpy as jnp


class CollisionChecker:
    """
    Occupancy grid-based collision checker.

    Provides collision detection for points in 2D/3D space using a discrete
    occupancy grid representation.

    Args:
        occup_grid: Occupancy grid where higher values indicate obstacles
        origin: Grid origin [x, y] in world coordinates
        resolution: Grid resolution (meters/cell)
        wh: Grid width and height [w, h] in cells
        occup_value: List of occupancy values considered as obstacles

    Example:
        >>> checker = CollisionChecker(grid, origin=[0, 0], resolution=0.1, wh=[100, 100])
        >>> is_occupied = checker.find_occupied([1.5, 2.3, 0.0])
    """

    def __init__(
        self,
        occup_grid: Optional[npt.NDArray[np.integer]],
        origin: Union[List[float], npt.NDArray[np.floating]],
        resolution: float,
        wh: Union[List[int], npt.NDArray[np.integer]],
        occup_value: Union[List[int], int] = 100
    ) -> None:
        """
        Initialize collision checker.

        Args:
            occup_grid: Occupancy grid (can be None initially)
            origin: Grid origin [x, y]
            resolution: Grid resolution in meters/cell
            wh: Grid width and height [w, h] in cells
            occup_value: Threshold value(s) for occupied cells
        """
        self.occup_grid = occup_grid
        self.origin = origin
        self.resolution = resolution
        self.wh = wh
        
        # Handle occup_value as list or single value
        if isinstance(occup_value, list):
            self.occup_value = occup_value[0] if occup_value else 100
        else:
            self.occup_value = occup_value

    def update_grid(
        self,
        occup_grid: npt.NDArray[np.integer],
        origin: Optional[Union[List[float], npt.NDArray[np.floating]]] = None,
        resolution: Optional[float] = None,
        wh: Optional[Union[List[int], npt.NDArray[np.integer]]] = None
    ) -> None:
        """
        Update the occupancy grid and optionally its parameters.

        Args:
            occup_grid: New occupancy grid
            origin: New origin (optional, keeps current if None)
            resolution: New resolution (optional, keeps current if None)
            wh: New grid dimensions (optional, keeps current if None)
        """
        self.occup_grid = occup_grid
        if origin is not None:
            self.origin = np.array(origin)
        if resolution is not None:
            self.resolution = resolution
        if wh is not None:
            self.wh = np.array(wh)

    # Helper for coordinate queries
    def grid_index(self, pos: jnp.ndarray) -> jnp.ndarray:
        ind = jnp.floor((pos - self.origin) / self.resolution)
        ind = jnp.clip(ind, 0, jnp.array(self.wh) - 1)
        return ind.astype(jnp.int32)

    def out_of_bounds(self, pos: jnp.ndarray) -> bool:
        """
        Returns True if pos lies outside the costmap boundaries.
        """
        ind = self.grid_index(pos)
        # valid if 0 <= ind < wh in both axes
        valid = jnp.logical_and(jnp.all(ind >= 0), jnp.all(ind < jnp.array(self.wh)))
        return jnp.logical_not(jnp.all(valid))

    def get_all_nonzero(self):
        occupied = np.argwhere(self.occup_grid > 0)
        occupied = occupied *self.resolution + self.origin
        return occupied

    def get_all_value(self, value):
        occupied = np.argwhere(self.occup_grid == value)
        occupied = occupied *self.resolution + self.origin
        return occupied
    @jax.jit
    def get_value(self, pos: jnp.ndarray) -> float:
        """
        Returns occupancy value at pos (no clipping).
        If pos is out of bounds, returns +inf.
        """
        ind = self.grid_index(pos)
        out = self.out_of_bounds(pos)
        # Use clipping only to safely index, but will ignore value if out=True
        ind_safe = jnp.clip(ind, 0, jnp.array(self.wh)).astype(jnp.int32)
        val = self.occup_grid[ind_safe[0], ind_safe[1]].astype(jnp.float32)
        val = jax.lax.cond(out, lambda x: jnp.inf, lambda x: x, val)
        return val
    @jax.jit
    def find_occupied_jax(self, pos: jnp.ndarray) -> float:
        """
        Returns occupancy value at pos (no clipping).`
        If pos is out of bounds, returns +inf.
        """
        ind = self.grid_index(pos)
        out = self.out_of_bounds(pos)
        # Use clipping only to safely index, but will ignore value if out=True
        ind_safe = jnp.clip(ind, 0, jnp.array(self.wh)).astype(jnp.int32)
        val = self.occup_grid[ind_safe[0], ind_safe[1]]
        val = jax.lax.cond(out, lambda x: jnp.inf, lambda x: x, val)
        return val

    def find_occupied(
        self,
        pt: Union[npt.NDArray[np.floating], List[float]],
        mark_out_of_bounds: bool = False
    ) -> bool:
        """
        Check if a point is in collision with obstacles.

        Args:
            pt: Point to check [x, y, z] or [x, y]
            mark_out_of_bounds: If True, treat out-of-bounds as occupied

        Returns:
            True if point is occupied or out of bounds (when mark_out_of_bounds=True),
            False otherwise
        """
        # If no grid, return based on out-of-bounds flag
        if self.occup_grid is None:
            return mark_out_of_bounds

        # Extract x, y coordinates (ignore z if present)
        footprint = np.array([pt[:2]])
        
        # Convert world coordinates to grid indices
        inds = np.floor((footprint - self.origin) / self.resolution).astype(np.int32)

        # Check bounds
        if (np.any(inds < 0) or
            np.any(inds[:, 0] >= self.wh[0]) or
            np.any(inds[:, 1] >= self.wh[1])):
            return mark_out_of_bounds

        # Check occupancy
        try:
            if np.any(self.occup_grid[inds[:, 0], inds[:, 1]] >= self.occup_value):
                return True
        except IndexError:
            # Fallback in case of indexing errors
            return True

        return False

    def is_line_free(
        self,
        pt1: Union[npt.NDArray[np.floating], List[float]],
        pt2: Union[npt.NDArray[np.floating], List[float]],
        num_samples: int = 10
    ) -> bool:
        """
        Check if a straight line between two points is collision-free.

        Args:
            pt1: Start point [x, y, z] or [x, y]
            pt2: End point [x, y, z] or [x, y]
            num_samples: Number of points to sample along the line

        Returns:
            True if line is free of collisions, False otherwise
        """
        pt1 = np.array(pt1)
        pt2 = np.array(pt2)
        
        # Sample points along the line
        for alpha in np.linspace(0, 1, num_samples):
            pt = pt1 + alpha * (pt2 - pt1)
            if self.find_occupied(pt, mark_out_of_bounds=True):
                return False
        
        return True
    @jax.jit
    def dda_path_check(
        self,
        start: jnp.ndarray, 
        end: jnp.ndarray,
        steps=None,
    ) -> bool:
        start_xy = jnp.asarray(start, dtype=jnp.float32).reshape(-1)[:2]
        end_xy   = jnp.asarray(end,   dtype=jnp.float32).reshape(-1)[:2]
        delta = end_xy - start_xy


        # if steps is None:
        steps = jnp.ceil(jnp.max(jnp.abs(delta)) / (self.resolution))+1
        steps = jnp.maximum(steps, 1.0).astype(jnp.int32)

        step_vec = delta / steps.astype(delta.dtype)

        def body_fn(i, carry):
            blocked_flag, pos = carry
            pos_next = pos + step_vec

            val = self.get_value(pos_next)

            hit = jnp.logical_or(jnp.isinf(val), (val >= self.occup_value))

            return (jnp.logical_or(blocked_flag, hit), pos_next)

        init_carry = (jnp.array(False), start_xy)
        blocked, _ = jax.lax.fori_loop(0, steps, body_fn, init_carry)
        # blocked = False
        # for i in range(steps):
        #     pos_next = start_xy + step_vec * (i+1)
        #     val = self.get_value(pos_next)
        #     hit = jnp.logical_or(jnp.isinf(val), (val >= self.occup_value))
        #     blocked = jnp.logical_or(blocked, hit)
        #     if blocked:
        #         break
        return blocked
    

    def find_occupied_batch(
        self,
        pts: npt.NDArray[np.floating],
        mark_out_of_bounds: bool = False
    ) -> npt.NDArray[np.bool_]:
        """
        Check if multiple points are in collision with obstacles (vectorized).

        This is a parallelized version that checks many points at once using
        numpy vectorization for improved performance.

        Args:
            pts: Array of points to check, shape (N, 2) or (N, 3)
                 Each row is [x, y] or [x, y, z]
            mark_out_of_bounds: If True, treat out-of-bounds as occupied

        Returns:
            Boolean array of shape (N,) where True indicates occupied

        Example:
            >>> pts = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            >>> occupied = checker.find_occupied_batch(pts)
            >>> print(occupied)  # [False, True, False]
        """
        pts = np.asarray(pts)
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise ValueError(f"pts must be shape (N, 2) or (N, 3), got {pts.shape}")
        
        # If no grid, return all based on out-of-bounds flag
        if self.occup_grid is None:
            return np.full(len(pts), mark_out_of_bounds, dtype=bool)
        
        # Extract x, y coordinates (ignore z if present)
        footprint = pts[:, :2]
        
        # Convert world coordinates to grid indices (vectorized)
        inds = np.floor((footprint - self.origin) / self.resolution).astype(np.int32)
        
        # Initialize result array
        occupied = np.zeros(len(pts), dtype=bool)
        
        # Check bounds (vectorized)
        out_of_bounds = (
            (inds[:, 0] < 0) |
            (inds[:, 1] < 0) |
            (inds[:, 0] >= self.wh[0]) |
            (inds[:, 1] >= self.wh[1])
        )
        
        if mark_out_of_bounds:
            occupied[out_of_bounds] = True
        
        # Check occupancy for in-bounds points (vectorized)
        in_bounds = ~out_of_bounds
        if np.any(in_bounds):
            try:
                in_bounds_inds = inds[in_bounds]
                occupied[in_bounds] = (
                    self.occup_grid[in_bounds_inds[:, 0], in_bounds_inds[:, 1]] >= self.occup_value
                )
            except IndexError:
                # If indexing fails, mark those points as occupied
                occupied[in_bounds] = True
        
        return occupied

    def is_line_free_batch(
        self,
        pts1: npt.NDArray[np.floating],
        pts2: npt.NDArray[np.floating],
        num_samples: int = 10,
        early_exit: bool = True
    ) -> npt.NDArray[np.bool_]:
        """
        Check if multiple straight lines are collision-free (parallelized).

        This method checks multiple lines in parallel. When early_exit=True,
        each line stops checking once a collision is found.

        Args:
            pts1: Start points array, shape (N, 2) or (N, 3)
            pts2: End points array, shape (N, 2) or (N, 3)
            num_samples: Number of points to sample along each line
            early_exit: If True, stop checking a line once collision found

        Returns:
            Boolean array of shape (N,) where True indicates line is free

        Example:
            >>> starts = np.array([[0.0, 0.0], [1.0, 1.0]])
            >>> ends = np.array([[5.0, 5.0], [6.0, 6.0]])
            >>> free = checker.is_line_free_batch(starts, ends)
            >>> print(free)  # [True, False]
        """
        pts1 = np.asarray(pts1)
        pts2 = np.asarray(pts2)
        
        if pts1.shape != pts2.shape:
            raise ValueError(f"pts1 and pts2 must have same shape, got {pts1.shape} and {pts2.shape}")
        if pts1.ndim != 2 or pts1.shape[1] < 2:
            raise ValueError(f"pts must be shape (N, 2) or (N, 3), got {pts1.shape}")
        
        n_lines = len(pts1)
        alphas = np.linspace(0, 1, num_samples)
        
        if early_exit:
            # Early exit version: check each line sequentially but points along line in parallel
            line_free = np.ones(n_lines, dtype=bool)
            
            for i in range(n_lines):
                # Generate all sample points for this line (vectorized)
                # Shape: (num_samples, 2 or 3)
                sample_pts = pts1[i][None, :] + alphas[:, None] * (pts2[i] - pts1[i])[None, :]
                
                # Check all points at once
                occupied = self.find_occupied_batch(sample_pts, mark_out_of_bounds=True)
                
                # If any point is occupied, line is not free
                if np.any(occupied):
                    line_free[i] = False
            
            return line_free
        else:
            # Fully vectorized version: check all lines and all points at once
            # Shape: (n_lines, num_samples, 2 or 3)
            sample_pts = (
                pts1[:, None, :] + 
                alphas[None, :, None] * (pts2 - pts1)[:, None, :]
            )
            
            # Reshape to (n_lines * num_samples, 2 or 3) for batch checking
            sample_pts_flat = sample_pts.reshape(-1, sample_pts.shape[-1])
            
            # Check all points at once
            occupied_flat = self.find_occupied_batch(sample_pts_flat, mark_out_of_bounds=True)
            
            # Reshape back to (n_lines, num_samples)
            occupied = occupied_flat.reshape(n_lines, num_samples)
            
            # Line is free if no point along it is occupied
            line_free = ~np.any(occupied, axis=1)
            
            return line_free

    def is_line_free_early_exit(
        self,
        pt1: Union[npt.NDArray[np.floating], List[float]],
        pt2: Union[npt.NDArray[np.floating], List[float]],
        num_samples: int = 10
    ) -> bool:
        """
        Check if a line is free with vectorized early exit.

        This is a faster version of is_line_free that checks all sample points
        in parallel but returns immediately upon finding a collision.

        Args:
            pt1: Start point [x, y, z] or [x, y]
            pt2: End point [x, y, z] or [x, y]
            num_samples: Number of points to sample along the line

        Returns:
            True if line is free of collisions, False otherwise
        """
        pt1 = np.array(pt1)
        pt2 = np.array(pt2)
        
        # Generate all sample points at once (vectorized)
        alphas = np.linspace(0, 1, num_samples)
        sample_pts = pt1[None, :] + alphas[:, None] * (pt2 - pt1)[None, :]
        
        # Check all points in one batch
        occupied = self.find_occupied_batch(sample_pts, mark_out_of_bounds=True)
        
        # Return False if any point is occupied
        return not np.any(occupied)

    # Tell JAX which parts are static vs dynamic
    def tree_flatten(self):
        # dynamic leaves go in first tuple, static in second
        dynamic = (self.occup_grid, self.origin)
        wh = tuple(self.wh) if isinstance(self.wh, np.ndarray) else self.wh
        static = (self.resolution, wh)
        return dynamic, static
    
    @classmethod
    def tree_unflatten(cls, static, dynamic):
        resolution, wh = static
        occup_grid, origin = dynamic
        return cls(occup_grid, origin, resolution, wh)

from jax import tree_util 
tree_util.register_pytree_node(CollisionChecker,
                               CollisionChecker.tree_flatten,
                               CollisionChecker.tree_unflatten)


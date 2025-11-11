"""
Sampling strategy for TopoPRM roadmap construction.

This module provides a flexible sampling framework with:
- Abstract Sampler base class for uniform API
- EllipsoidSampler for geometric sampling
- MixedSampler for composing multiple samplers with weights
- Extensible architecture for custom sampling strategies

All samplers are designed primarily for 2D environments (x-y plane),
with z=0 as the default. The code generalizes to 3D if needed, but
2D is the primary use case and default configuration.
"""

import random
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List, Dict, Any

import numpy as np
import numpy.typing as npt


class Sampler(ABC):
    """
    Abstract base class for all sampling strategies.
    
    Defines the interface that all samplers must implement to be compatible
    with the TopoPRM planner and composable sampling strategies.
    
    Subclasses must implement:
        - configure(start, end): Set up the sampling region
        - get_sample(): Generate one random sample
    
    Example:
        >>> class MySampler(Sampler):
        ...     def configure(self, start, end):
        ...         self.start = start
        ...         self.goal = end
        ...     
        ...     def get_sample(self):
        ...         # Your custom sampling logic
        ...         return np.random.uniform(self.start, self.goal)
    """
    
    @abstractmethod
    def configure(
        self, 
        start: Union[npt.NDArray[np.floating], List[float]], 
        end: Union[npt.NDArray[np.floating], List[float]]
    ) -> None:
        """
        Configure the sampler based on start and goal positions.
        
        This method is called before sampling begins and should set up
        any internal state needed to generate samples in the relevant region.
        
        Args:
            start: Start position [x, y, z]
            end: Goal position [x, y, z]
            
        Raises:
            ValueError: If start/end are invalid
        """
        pass
    
    @abstractmethod
    def get_sample(self) -> npt.NDArray[np.floating]:
        """
        Generate a random sample point.
        
        Returns:
            3D position [x, y, z]
            
        Raises:
            RuntimeError: If configure() has not been called
        """
        pass


class EllipsoidSampler(Sampler):
    """
    Samples random points in an ellipsoid region between start and goal.
    
    **Designed for 2D environments (x-y plane) with z=0 by default.**
    
    The ellipsoid is aligned with the start-goal direction and can be inflated
    to control the sampling region size. Optionally supports biased sampling
    from predefined safe zones.
    
    For 2D planning:
    - Set sample_inflate=[along_path, perpendicular, 0.0]
    - Samples will have z=0
    
    Attributes:
        sample_inflate: Inflation factors [along_path, perpendicular, vertical]
                        Default: (5.0, 5.0, 0.0) for 2D
        safe_zones: Optional array of safe zone positions for biased sampling
        sample_sz_p: Probability of sampling from safe zones (0.0 to 1.0)
    
    Example (2D):
        >>> sampler = EllipsoidSampler()  # Uses 2D defaults
        >>> sampler.configure(start, goal)
        >>> point = sampler.get_sample()  # point[2] will be ~0
    """
    
    def __init__(
        self,
        sample_inflate: Union[Tuple[float, ...], List[float]] = (5.0, 5.0, 0.0),
        safe_zones: Optional[npt.NDArray[np.floating]] = None,
        sample_sz_p: float = 0.0
    ) -> None:
        """
        Initialize the ellipsoid sampler.
        
        Args:
            sample_inflate: Inflation [along_path, perpendicular, vertical] in meters
                           Default (5.0, 5.0, 0.0) is optimized for 2D environments
            safe_zones: Optional Nx3 array of safe zone positions
            sample_sz_p: Probability of sampling from safe zones [0, 1]
            
        Raises:
            ValueError: If parameters are out of valid ranges
            TypeError: If safe_zones is wrong type
        """
        # Validate sample_inflate
        if len(sample_inflate) != 3:
            raise ValueError(f"sample_inflate must have 3 elements, got {len(sample_inflate)}")
        if any(x < 0 for x in sample_inflate):
            raise ValueError(f"sample_inflate values must be non-negative, got {sample_inflate}")
        
        # Validate safe_zones
        if safe_zones is not None:
            safe_zones = np.asarray(safe_zones)
            if safe_zones.ndim != 2 or safe_zones.shape[1] != 3:
                raise ValueError(f"safe_zones must be Nx3 array, got shape {safe_zones.shape}")
            if not np.all(np.isfinite(safe_zones)):
                raise ValueError("safe_zones must contain only finite values")
        
        # Validate probability
        if not 0.0 <= sample_sz_p <= 1.0:
            raise ValueError(f"sample_sz_p must be in [0, 1], got {sample_sz_p}")
        
        self.sample_inflate = sample_inflate
        self.safe_zones = safe_zones
        self.sample_sz_p = sample_sz_p
        
        # Configuration set by configure()
        self.sample_r = np.array([0.0, 0.0, 0.0])
        self.translation = np.zeros((3, 1))
        self.R = np.eye(3)
        
    def configure(
        self, 
        start: Union[npt.NDArray[np.floating], List[float]], 
        end: Union[npt.NDArray[np.floating], List[float]]
    ) -> None:
        """
        Configure the ellipsoid based on start and goal positions.
        
        Sets up the ellipsoid radius, center translation, and rotation matrix
        to align with the start-goal direction.
        
        Args:
            start: Start position [x, y, z]
            end: Goal position [x, y, z]
            
        Raises:
            ValueError: If start/end are invalid or identical
        """
        start = np.array(start).reshape((3, 1))
        end = np.array(end).reshape((3, 1))
        
        # Validate inputs
        if not np.all(np.isfinite(start)):
            raise ValueError(f"Start position must be finite, got {start.ravel()}")
        if not np.all(np.isfinite(end)):
            raise ValueError(f"End position must be finite, got {end.ravel()}")
        
        # Check if start and end are too close
        dist = np.linalg.norm(end - start)
        if dist < 1e-6:
            raise ValueError(f"Start and end are too close (distance={dist}), cannot configure ellipsoid")
        
        # Ellipsoid radii: major axis along start-goal + inflation
        self.sample_r = np.array([
            0.5 * dist + self.sample_inflate[0],
            self.sample_inflate[1],
            self.sample_inflate[2]
        ])
        
        # Center at midpoint
        self.translation = 0.5 * (start + end)
        
        # Rotation matrix to align x-axis with start-goal direction
        self.R = self._compute_rotation_matrix(start, end)
    
    def _compute_rotation_matrix(
        self, 
        start: npt.NDArray[np.floating], 
        end: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """
        Compute rotation matrix aligning x-axis with start-goal vector.
        
        Args:
            start: Start position [x, y, z]
            end: Goal position [x, y, z]
            
        Returns:
            3x3 rotation matrix
        """
        # X-axis: normalized start-goal direction
        xtf = (end - self.translation).T
        xtf = xtf / np.linalg.norm(xtf)
        
        # Y-axis: perpendicular to x and z
        ytf = np.cross(xtf, [[0, 0, -1]])
        ytf = ytf / np.linalg.norm(ytf)
        
        # Z-axis: perpendicular to x and y
        ztf = np.cross(xtf, ytf)
        
        return np.vstack((xtf, ytf, ztf)).T
    
    def get_sample(self) -> npt.NDArray[np.floating]:
        """
        Generate a random sample point.
        
        With probability sample_sz_p, samples from safe zones if available.
        Otherwise, samples uniformly in the ellipsoid region.
        
        Returns:
            3D position [x, y, z]
        """
        # Possibly sample from safe zones
        if self.safe_zones is not None and np.random.uniform() < self.sample_sz_p:
            return self._sample_safe_zone()
        
        # Sample from ellipsoid
        return self._sample_ellipsoid()
    
    def _sample_ellipsoid(self) -> npt.NDArray[np.floating]:
        """
        Sample uniformly in the ellipsoid region.
        
        Returns:
            3D position in world frame
        """
        # Sample in unit box [-1, 1]^3
        pt = np.array([
            [np.random.uniform(-1.0, 1.0) * self.sample_r[0]],
            [np.random.uniform(-1.0, 1.0) * self.sample_r[1]],
            [np.random.uniform(-1.0, 1.0) * self.sample_r[2]]
        ])
        
        # Transform to world frame
        pt = self.R @ pt + self.translation
        return pt.ravel()
    
    def _sample_safe_zone(self) -> npt.NDArray[np.floating]:
        """
        Sample randomly from the safe zones.
        
        Returns:
            3D position from safe zones
        
        """
        if self.safe_zones is None or len(self.safe_zones) == 0:
            # raise ValueError("safe_zones is None, cannot sample")
            return self._sample_ellipsoid()
        # print(f"SAFE ZONES: {self.safe_zones}") 
        idx = np.random.randint(0, len(self.safe_zones))
        pt = np.hstack([self.safe_zones[idx].copy(), 0.0])
        return pt
    
    @staticmethod
    def sample_unit_nball() -> npt.NDArray[np.floating]:
        """
        Sample uniformly in a 2D unit circle.
        
        Returns:
            3D point [x, y, 0] where x^2 + y^2 < 1
        """
        while True:
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            if x**2 + y**2 < 1:
                return np.array([[x], [y], [0.0]])


class UniformSampler(Sampler):
    """
    Samples uniformly in an axis-aligned bounding box.
    
    **Designed for 2D environments (x-y plane) with z=0 by default.**
    
    The bounding box is defined by the start and goal positions, with
    optional inflation in each dimension.
    
    For 2D planning:
    - Set inflate=[x, y, 0.0]
    - Samples will have z≈0
    
    Attributes:
        inflate: Inflation factors [x, y, z] added to bounding box
                 Default: (5.0, 5.0, 0.0) for 2D
    
    Example (2D):
        >>> sampler = UniformSampler()  # Uses 2D defaults
        >>> sampler.configure(start, goal)
        >>> point = sampler.get_sample()  # point[2] will be ~0
    """
    
    def __init__(
        self,
        inflate: Union[Tuple[float, ...], List[float]] = (5.0, 5.0, 0.0)
    ) -> None:
        """
        Initialize the uniform sampler.
        
        Args:
            inflate: Inflation [x, y, z] added to bounding box in meters
                    Default (5.0, 5.0, 0.0) is optimized for 2D environments
            
        Raises:
            ValueError: If inflate has wrong size or negative values
        """
        if len(inflate) != 3:
            raise ValueError(f"inflate must have 3 elements, got {len(inflate)}")
        if any(x < 0 for x in inflate):
            raise ValueError(f"inflate values must be non-negative, got {inflate}")
        
        self.inflate = np.array(inflate)
        self.bounds_min = np.zeros(3)
        self.bounds_max = np.zeros(3)
        self._configured = False
    
    def configure(
        self, 
        start: Union[npt.NDArray[np.floating], List[float]], 
        end: Union[npt.NDArray[np.floating], List[float]]
    ) -> None:
        """
        Configure bounding box based on start and goal.
        
        Args:
            start: Start position [x, y, z]
            end: Goal position [x, y, z]
            
        Raises:
            ValueError: If start/end are invalid
        """
        start = np.array(start).ravel()
        end = np.array(end).ravel()
        
        if not np.all(np.isfinite(start)):
            raise ValueError(f"Start position must be finite, got {start}")
        if not np.all(np.isfinite(end)):
            raise ValueError(f"End position must be finite, got {end}")
        
        # Bounding box encompasses start and goal with inflation
        self.bounds_min = np.minimum(start, end) - self.inflate
        self.bounds_max = np.maximum(start, end) + self.inflate
        self._configured = True
    
    def get_sample(self) -> npt.NDArray[np.floating]:
        """
        Sample uniformly in the bounding box.
        
        Returns:
            3D position [x, y, z]
            
        Raises:
            RuntimeError: If configure() not called
        """
        if not self._configured:
            raise RuntimeError("Sampler not configured. Call configure(start, end) first.")
        
        return np.random.uniform(self.bounds_min, self.bounds_max)


class HyperplaneIntersectionSampler(Sampler):
    """
    Samples uniformly within the intersection of a hyperplane and an oriented bounding box.
    
    **Designed for 2D environments - primarily samples on vertical/horizontal planes in x-y space.**
    
    The hyperplane is defined by a normal vector and a point (or offset).
    The oriented bounding box (OBB) is defined by four corner points.
    
    Common 2D use cases:
    - Vertical walls: normal=[1, 0, 0], corners with constant x, z=0
    - Horizontal boundaries: normal=[0, 1, 0], corners with constant y, z=0
    
    This is useful for:
    - Sampling along walls or boundaries in 2D navigation
    - Generating samples in constrained manifolds
    - Creating layered sampling strategies
    
    Attributes:
        normal: Normal vector defining the hyperplane
        point_on_plane: A point that the hyperplane passes through
        obb_corners: Four corners defining the oriented bounding box
    
    Example (2D vertical wall):
        >>> # Sample on a vertical wall at x=-10 (common 2D scenario)
        >>> normal = np.array([1.0, 0.0, 0.0])  # Normal pointing in +x direction
        >>> point = np.array([-10.0, 0.0, 0.0])  # Plane at x=-10
        >>> corners = np.array([
        ...     [-10, -5, 0],  # y from -5 to 5, z=0 (2D)
        ...     [-10, 5, 0],
        ...     [-10, 5, 0],
        ...     [-10, -5, 0]
        ... ])
        >>> sampler = HyperplaneIntersectionSampler(normal, point, corners)
        >>> sampler.configure(start, goal)
        >>> sample = sampler.get_sample()  # Will have x≈-10, z≈0
    """
    
    def __init__(
        self,
        normal: Union[npt.NDArray[np.floating], List[float]],
        point_on_plane: Union[npt.NDArray[np.floating], List[float]],
        obb_corners: Union[npt.NDArray[np.floating], List[List[float]]],
        max_rejection_samples: int = 1000
    ) -> None:
        """
        Initialize the hyperplane intersection sampler.
        
        Args:
            normal: Normal vector [nx, ny, nz] defining hyperplane orientation
            point_on_plane: Point [x, y, z] that the hyperplane passes through
            obb_corners: 4x3 array of corner points defining the OBB
            max_rejection_samples: Maximum rejection sampling attempts
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate and normalize normal
        self.normal = np.array(normal).ravel()
        if self.normal.shape[0] != 3:
            raise ValueError(f"Normal must be 3D vector, got shape {self.normal.shape}")
        normal_mag = np.linalg.norm(self.normal)
        if normal_mag < 1e-9:
            raise ValueError(f"Normal vector magnitude too small: {normal_mag}")
        self.normal = self.normal / normal_mag  # Normalize
        
        # Validate point on plane
        self.point_on_plane = np.array(point_on_plane).ravel()
        if self.point_on_plane.shape[0] != 3:
            raise ValueError(f"Point must be 3D, got shape {self.point_on_plane.shape}")
        if not np.all(np.isfinite(self.point_on_plane)):
            raise ValueError("Point on plane must have finite values")
        
        # Validate OBB corners
        self.obb_corners = np.array(obb_corners)
        if self.obb_corners.shape != (4, 3):
            raise ValueError(f"OBB corners must be 4x3 array, got shape {self.obb_corners.shape}")
        if not np.all(np.isfinite(self.obb_corners)):
            raise ValueError("OBB corners must have finite values")
        
        # Validate max samples
        if max_rejection_samples <= 0:
            raise ValueError(f"max_rejection_samples must be positive, got {max_rejection_samples}")
        self.max_rejection_samples = max_rejection_samples
        
        # Compute plane offset: d in ax + by + cz + d = 0
        self.plane_d = -np.dot(self.normal, self.point_on_plane)
        
        # Build local 2D coordinate system on the plane
        self._build_plane_coords()
        
        # Project OBB corners onto the plane
        self._project_obb_to_plane()
        
        self._configured = False
    
    def _build_plane_coords(self) -> None:
        """Build orthonormal basis for the plane (2D coordinate system)."""
        # Find two orthogonal vectors in the plane
        # Start with an arbitrary vector not parallel to normal
        if abs(self.normal[0]) < abs(self.normal[1]):
            arbitrary = np.array([1.0, 0.0, 0.0])
        else:
            arbitrary = np.array([0.0, 1.0, 0.0])
        
        # First basis vector (u): perpendicular to normal
        self.u = np.cross(self.normal, arbitrary)
        self.u = self.u / np.linalg.norm(self.u)
        
        # Second basis vector (v): perpendicular to both normal and u
        self.v = np.cross(self.normal, self.u)
        self.v = self.v / np.linalg.norm(self.v)
    
    def _project_obb_to_plane(self) -> None:
        """Project OBB corners onto the plane's 2D coordinate system."""
        # Project each corner onto the plane
        projected_3d = []
        for corner in self.obb_corners:
            # Distance from corner to plane
            dist = np.dot(self.normal, corner) + self.plane_d
            # Project onto plane
            projected = corner - dist * self.normal
            projected_3d.append(projected)
        
        # Convert to 2D coordinates in plane's basis
        self.obb_2d = []
        for p in projected_3d:
            # Express in (u, v) basis relative to point_on_plane
            rel = p - self.point_on_plane
            u_coord = np.dot(rel, self.u)
            v_coord = np.dot(rel, self.v)
            self.obb_2d.append([u_coord, v_coord])
        
        self.obb_2d = np.array(self.obb_2d)
        
        # Compute bounding box in 2D
        self.bbox_min = np.min(self.obb_2d, axis=0)
        self.bbox_max = np.max(self.obb_2d, axis=0)
    
    def _point_in_obb_2d(self, point_2d: npt.NDArray[np.floating]) -> bool:
        """
        Check if a 2D point is inside the projected OBB using cross products.
        
        Args:
            point_2d: Point in 2D plane coordinates [u, v]
            
        Returns:
            True if point is inside the OBB quadrilateral
        """
        # Use cross product test for convex quadrilateral
        # Point is inside if it's on the same side of all edges
        def cross_2d(v1, v2):
            return v1[0] * v2[1] - v1[1] * v2[0]
        
        # Check all four edges
        for i in range(4):
            p1 = self.obb_2d[i]
            p2 = self.obb_2d[(i + 1) % 4]
            
            edge = p2 - p1
            to_point = point_2d - p1
            
            # Cross product determines which side of edge the point is on
            cross = cross_2d(edge, to_point)
            
            # For first edge, determine the expected sign
            if i == 0:
                expected_sign = np.sign(cross) if abs(cross) > 1e-9 else 1.0
            else:
                # Subsequent edges should have same sign (inside)
                if abs(cross) > 1e-9 and np.sign(cross) != expected_sign:
                    return False
        
        return True
    
    def configure(
        self, 
        start: Union[npt.NDArray[np.floating], List[float]], 
        end: Union[npt.NDArray[np.floating], List[float]]
    ) -> None:
        """
        Configure the sampler (optional for this sampler type).
        
        The hyperplane sampler is typically pre-configured with geometry,
        but this method is required by the Sampler interface.
        
        Args:
            start: Start position [x, y, z] (unused)
            end: Goal position [x, y, z] (unused)
        """
        # Validate that start/end are valid (interface requirement)
        start = np.array(start).ravel()
        end = np.array(end).ravel()
        
        if not np.all(np.isfinite(start)):
            raise ValueError(f"Start position must be finite, got {start}")
        if not np.all(np.isfinite(end)):
            raise ValueError(f"End position must be finite, got {end}")
        
        # Store for potential future use
        self.start = start
        self.goal = end
        self._configured = True
    
    def get_sample(self) -> npt.NDArray[np.floating]:
        """
        Generate a random sample in the hyperplane-OBB intersection.
        
        Uses rejection sampling: samples in 2D bounding box and rejects
        if outside the OBB quadrilateral.
        
        Returns:
            3D position [x, y, z] on the hyperplane within the OBB
            
        Raises:
            RuntimeError: If configure() not called or rejection sampling fails
        """
        if not self._configured:
            raise RuntimeError("Sampler not configured. Call configure(start, end) first.")
        
        # Rejection sampling in 2D plane coordinates
        for attempt in range(self.max_rejection_samples):
            # Sample in 2D bounding box
            u = np.random.uniform(self.bbox_min[0], self.bbox_max[0])
            v = np.random.uniform(self.bbox_min[1], self.bbox_max[1])
            point_2d = np.array([u, v])
            
            # Check if inside OBB
            if self._point_in_obb_2d(point_2d):
                # Convert back to 3D
                point_3d = (self.point_on_plane + 
                           u * self.u + 
                           v * self.v)
                return point_3d
        
        # Fallback: if rejection fails, return center of OBB
        center_2d = np.mean(self.obb_2d, axis=0)
        center_3d = (self.point_on_plane + 
                    center_2d[0] * self.u + 
                    center_2d[1] * self.v)
        
        print(f"[HyperplaneIntersectionSampler] Warning: Rejection sampling failed "
              f"after {self.max_rejection_samples} attempts, returning OBB center")
        
        return center_3d
    
    def get_obb_area(self) -> float:
        """
        Compute the area of the OBB quadrilateral.
        
        Returns:
            Area in square meters
        """
        # Shoelace formula for quadrilateral area
        area = 0.0
        for i in range(4):
            j = (i + 1) % 4
            area += self.obb_2d[i][0] * self.obb_2d[j][1]
            area -= self.obb_2d[j][0] * self.obb_2d[i][1]
        return abs(area) / 2.0


class MixedSampler(Sampler):
    """
    Composable sampler that mixes multiple sampling strategies with weights.
    
    Allows flexible combination of different sampling approaches, e.g.:
    - 70% ellipsoid sampling + 30% uniform sampling
    - 50% learned sampling + 30% ellipsoid + 20% safe zones
    
    The weights are automatically normalized to sum to 1.0.
    
    Attributes:
        samplers: List of (weight, sampler) tuples
        cumulative_weights: Cumulative distribution for sampling
    
    Example:
        >>> ellipsoid = EllipsoidSampler(sample_inflate=[5.0, 5.0, 0.0])
        >>> uniform = UniformSampler(inflate=[3.0, 3.0, 0.5])
        >>> mixed = MixedSampler([
        ...     (0.7, ellipsoid),  # 70% ellipsoid
        ...     (0.3, uniform)     # 30% uniform
        ... ])
        >>> mixed.configure(start, goal)
        >>> point = mixed.get_sample()
    """
    
    def __init__(
        self,
        samplers: List[Tuple[float, Sampler]]
    ) -> None:
        """
        Initialize the mixed sampler.
        
        Args:
            samplers: List of (weight, sampler) tuples where weight > 0
            
        Raises:
            ValueError: If samplers list is empty or weights are invalid
            TypeError: If samplers don't inherit from Sampler
        """
        if not samplers:
            raise ValueError("samplers list cannot be empty")
        
        # Validate and extract weights
        weights = []
        sampler_objs = []
        
        for i, (weight, sampler) in enumerate(samplers):
            if not isinstance(sampler, Sampler):
                raise TypeError(
                    f"Sampler at index {i} must inherit from Sampler, "
                    f"got {type(sampler).__name__}"
                )
            if weight <= 0:
                raise ValueError(f"Weight at index {i} must be positive, got {weight}")
            
            weights.append(weight)
            sampler_objs.append(sampler)
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # print a warning if normalization is needed, so we know that weights
        # were not normalized
        if not np.isclose(total_weight, 1.0):
            print(
                f"[MixedSampler] Warning: Weights normalized from total "
                f"{total_weight:.4f} to 1.0"
            )
        
        # Create cumulative distribution for sampling
        self.cumulative_weights = np.cumsum(normalized_weights)
        self.samplers = sampler_objs
        self.weights = normalized_weights  # Store for introspection
        self._configured = False
    
    def configure(
        self, 
        start: Union[npt.NDArray[np.floating], List[float]], 
        end: Union[npt.NDArray[np.floating], List[float]]
    ) -> None:
        """
        Configure all constituent samplers.
        
        Calls configure() on each sampler with the same start/goal.
        
        Args:
            start: Start position [x, y, z]
            end: Goal position [x, y, z]
            
        Raises:
            ValueError: If any constituent sampler rejects configuration
        """
        for sampler in self.samplers:
            sampler.configure(start, end)
        self._configured = True
    
    def get_sample(self) -> npt.NDArray[np.floating]:
        """
        Sample from one of the constituent samplers based on weights.
        
        Selects a sampler randomly according to the normalized weights,
        then calls get_sample() on that sampler.
        
        Returns:
            3D position [x, y, z]
            
        Raises:
            RuntimeError: If configure() not called
        """
        if not self._configured:
            raise RuntimeError("Sampler not configured. Call configure(start, end) first.")
        
        # Select sampler based on cumulative weights
        r = np.random.uniform(0, 1)
        idx = np.searchsorted(self.cumulative_weights, r)
        
        # Sample from selected sampler
        return self.samplers[idx].get_sample()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get information about the sampler composition.
        
        Returns:
            Dictionary with sampler types and weights
        """
        return {
            'num_samplers': len(self.samplers),
            'samplers': [
                {
                    'type': type(s).__name__,
                    'weight': w,
                    'percentage': f"{w*100:.1f}%"
                }
                for s, w in zip(self.samplers, self.weights)
            ]
        }


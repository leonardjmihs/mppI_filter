"""
Topological Probabilistic Roadmap (TopoPRM) planner package.

This package provides a modular implementation of the TopoPRM algorithm for
finding multiple topologically distinct paths in 2D/3D environments.

Main components:
- nodes: Graph node data structures and types
- sampler: Flexible sampling framework with composable strategies
- graph: Roadmap graph construction and visibility checking
- search: Path search algorithms (DFS)
- path_processor: Path optimization and topological equivalence checking
- planner: Main TopoPRM planner interface

Usage:
    from branch_mppi.jax_mppi.topo_prm import TopoPRM
    
    planner = TopoPRM(occup_grid, resolution=0.4)
    paths, samples = planner.findTopoPaths(start, goal)

Advanced Usage (Custom Sampling):
    from branch_mppi.jax_mppi.topo_prm import (
        TopoPRM, EllipsoidSampler, UniformSampler, MixedSampler
    )
    
    # Mix 70% ellipsoid + 30% uniform sampling
    ellipsoid = EllipsoidSampler(sample_inflate=[5.0, 5.0, 0.0])
    uniform = UniformSampler(inflate=[3.0, 3.0, 0.5])
    mixed = MixedSampler([(0.7, ellipsoid), (0.3, uniform)])
    
    planner = TopoPRM(occup_grid, resolution=0.4)
    planner.sampler = mixed
    paths, samples = planner.findTopoPaths(start, goal)
"""

from .nodes import NODE_TYPE, NODE_STATE, Node
from .planner import TopoPRM
from .sampler import (
    Sampler, 
    EllipsoidSampler, 
    UniformSampler, 
    HyperplaneIntersectionSampler,
    MixedSampler
)

__all__ = [
    'TopoPRM', 
    'NODE_TYPE', 
    'NODE_STATE', 
    'Node',
    'Sampler',
    'EllipsoidSampler',
    'UniformSampler',
    'HyperplaneIntersectionSampler',
    'MixedSampler',
]

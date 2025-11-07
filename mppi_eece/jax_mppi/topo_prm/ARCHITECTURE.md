# TopoPRM Architecture Documentation

**Version:** 2.0 (Refactored Modular Design)  
**Last Updated:** October 2025  
**Maintainer:** Branch MPPI Team

---

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [Architecture](#architecture)
4. [Module Reference](#module-reference)
5. [Data Flow](#data-flow)
6. [Extension Points](#extension-points)
7. [Performance Considerations](#performance-considerations)
8. [Testing Strategy](#testing-strategy)
9. [Migration Guide](#migration-guide)

---

## Overview

### What is TopoPRM?

**Topological Probabilistic Roadmap (TopoPRM)** is a motion planning algorithm that finds **multiple topologically distinct paths** between a start and goal position in environments with obstacles. Unlike traditional path planners that return a single optimal path, TopoPRM discovers diverse routing options that navigate around obstacles in fundamentally different ways.

### Key Capabilities

- **Multiple Path Discovery**: Returns N diverse paths (e.g., "go left around obstacle" vs "go right")
- **Topological Diversity**: Paths are guaranteed to be topologically distinct (not homotopic)
- **Collision-Free Guarantees**: All returned paths avoid obstacles
- **Configurable Performance**: Tunable parameters for speed vs. path quality trade-offs
- **Incremental Planning**: Can reuse previous roadmaps for faster replanning

### Use Cases

- **Autonomous Navigation**: Multiple route options for dynamic environments
- **Risk-Aware Planning**: Evaluate alternatives with different risk profiles
- **Multi-Agent Coordination**: Assign topologically distinct paths to avoid conflicts
- **Contingency Planning**: Pre-compute backup routes

---

## Design Philosophy

### Architectural Principles

The TopoPRM codebase was refactored (v2.0) from a monolithic ~2000-line file into a modular package structure following these principles:

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Testability**: Components are independently testable with clear interfaces
3. **Extensibility**: Plugin points for custom sampling, collision checking, and path scoring
4. **Backward Compatibility**: Legacy API preserved via re-exports
5. **Performance Transparency**: Built-in profiling and timing diagnostics
6. **Type Safety**: Comprehensive type hints for IDE support and static analysis

### Design Patterns

- **Strategy Pattern**: Interchangeable sampling strategies (ellipsoid, uniform, learned)
- **Builder Pattern**: Incremental graph construction with fluent API
- **Dependency Injection**: Collision checker injected into components
- **Template Method**: Path processing pipeline with customizable steps
- **Observer Pattern**: Timing/profiling hooks for diagnostics

---

## Architecture

### Package Structure

```
topo_prm/
├── __init__.py              # Public API exports
├── nodes.py                 # Node data structures
├── sampler.py               # Sampling strategies
├── graph.py                 # Graph construction
├── search.py                # Path search algorithms
├── path_processor.py        # Path optimization
├── planner.py               # Main orchestrator
└── ARCHITECTURE.md          # This file
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         TopoPRM                              │
│                    (Main Orchestrator)                       │
└────────┬────────────────────────────────────────────────────┘
         │
         ├─────► EllipsoidSampler
         │       └─ Generates collision-free samples
         │
         ├─────► GraphBuilder
         │       ├─ CollisionChecker (injected)
         │       ├─ Visibility testing
         │       └─ Node creation (GUARD/CONNECTOR)
         │
         ├─────► PathSearcher
         │       └─ DFS to find all paths
         │
         └─────► PathProcessor
                 ├─ Path shortcutting
                 ├─ Topological equivalence
                 └─ Path selection/ranking
```

### Dependency Graph

```
planner.py
  ├── nodes.py (Node, NODE_TYPE, NODE_STATE)
  ├── sampler.py (EllipsoidSampler)
  ├── graph.py (GraphBuilder)
  │   └── nodes.py
  ├── search.py (PathSearcher)
  │   └── nodes.py
  └── path_processor.py (PathProcessor)

__init__.py
  └── planner.py (re-exports)
```

**Zero Circular Dependencies**: Clean layered architecture

---

## Module Reference

### 1. `nodes.py` - Graph Data Structures

**Purpose**: Define the node representation for the roadmap graph.

#### Core Classes

##### `NODE_TYPE` (Enum)
```python
class NODE_TYPE(Enum):
    GUARD = 1      # Boundary of visibility region
    CONNECTOR = 2  # Bridge between regions
```

- **GUARD**: Created when a sample sees zero existing guard nodes → new region discovered
- **CONNECTOR**: Created when a sample sees 2+ guard nodes → bridges regions

##### `NODE_STATE` (Enum)
```python
class NODE_STATE(Enum):
    NEW = 1    # Not visited during search
    CLOSE = 2  # Fully processed
    OPEN = 3   # In search frontier
```

##### `Node` (Class)
```python
class Node:
    pos: np.ndarray      # [x, y, z] position
    type: NODE_TYPE      # GUARD or CONNECTOR
    state: NODE_STATE    # Search state
    id: int              # Unique identifier
    sz: bool             # From safe zone?
    neighbors: List[Node]  # Adjacent nodes
```

**Design Notes**:
- Nodes store direct neighbor references (not just IDs) for fast graph traversal
- `sz` flag enables prioritizing safe zones during sampling/search
- Mutable state pattern for DFS (state changes during search)

#### Extension Points

To add new node attributes (e.g., cost, probability):

```python
class Node:
    # Existing attributes...
    cost: float = 0.0  # Add custom attributes
    
    def __init__(self, ...):
        # Existing init...
        self.cost = 0.0
```

---

### 2. `sampler.py` - Sampling Strategies

**Purpose**: Generate random collision-free samples in the configuration space.

#### Core Class: `EllipsoidSampler`

**Algorithm**: Samples uniformly in an axis-aligned ellipsoid between start and goal.

```python
class EllipsoidSampler:
    sample_inflate: Tuple[float, float, float]  # [along_path, perp, vertical]
    safe_zones: Optional[np.ndarray]            # Nx3 array
    sample_sz_p: float                          # P(sample from safe zone)
```

**Key Methods**:

1. **`configure(start, end)`**: Set up ellipsoid geometry
   - Computes rotation matrix to align with start→goal direction
   - Calculates ellipsoid radii based on distance + inflation

2. **`get_sample()`**: Generate one random sample
   - With probability `sample_sz_p`: sample near a random safe zone
   - Otherwise: sample uniformly in ellipsoid

**Mathematical Foundation**:

Ellipsoid equation: `(R^T * (x - center))^T * diag(1/r²) * (R^T * (x - center)) ≤ 1`

Where:
- `R`: Rotation matrix aligning x-axis with start→goal
- `r`: Radii `[dist/2 + inflate[0], inflate[1], inflate[2]]`
- `center`: Midpoint of start and goal

#### Extension: Custom Sampling Strategies

To implement a new sampler:

```python
class MySampler:
    def configure(self, start, end):
        """Set up sampling region."""
        self.start = start
        self.goal = end
    
    def get_sample(self) -> np.ndarray:
        """Return a 3D sample point."""
        # Your strategy here
        return np.array([x, y, z])
```

Replace in `TopoPRM.__init__`:
```python
self.sampler = MySampler(...)
```

**Ideas for Custom Samplers**:
- Learned sampling from neural network
- Gradient-based sampling toward goal
- Obstacle-aware sampling (sample near obstacle boundaries)
- Hybrid: Combine multiple strategies

---

### 3. `graph.py` - Graph Construction

**Purpose**: Build the roadmap by determining node visibility and creating edges.

#### Core Class: `GraphBuilder`

**Responsibilities**:
1. Sample N points in free space
2. Determine visibility between samples and existing nodes
3. Create GUARD or CONNECTOR nodes based on visibility
4. Build edges between visible nodes
5. Prune disconnected graph components

**Key Algorithm: Visibility-Based Node Classification**

For each new collision-free sample `s`:

```
visible_guards = [g for g in graph if g.type == GUARD and line_visible(s, g)]

if len(visible_guards) == 0:
    → Create GUARD node at s (discovered new region)
    → Connect to all visible nodes

elif len(visible_guards) >= 2:
    → Create CONNECTOR node at s (bridges regions)
    → Connect to all visible guards + other visible connectors

else:  # len(visible_guards) == 1
    → Discard sample (inside existing region, not useful)
```

**Key Methods**:

1. **`create_graph(sampler, start, goal, existing_graph)`**
   - Main entry point
   - Generates samples and builds graph
   - Returns `(graph, samples)` tuple

2. **`_check_line_visible(p1, p2)`**
   - Ray-casting collision check between two points
   - Uses `ray_resolution` step size
   - Returns `True` if collision-free

3. **`_is_topo_same(path1, path2)`**
   - Check if two paths are topologically equivalent
   - Discretizes paths and compares voxel signatures
   - Uses `topo_resolution` for discretization

4. **`_prune_graph(graph)`**
   - Removes nodes not connected to both start and goal
   - Ensures all returned paths are valid

#### Performance Tuning Parameters

| Parameter         | Effect                       | Typical Range | Trade-off                  |
| ----------------- | ---------------------------- | ------------- | -------------------------- |
| `max_sample_num`  | More samples = more nodes    | 50-500        | Quality vs. Speed          |
| `ray_resolution`  | Collision check density      | 0.1-0.5m      | Accuracy vs. Speed         |
| `topo_resolution` | Path equivalence granularity | 0.1-0.5m      | Diversity vs. Speed        |
| `max_time`        | Hard time limit              | 0.05-1.0s     | Completeness vs. Real-time |

#### Extension: Custom Visibility Checks

To add custom visibility logic (e.g., kinematic constraints):

```python
class MyGraphBuilder(GraphBuilder):
    def _check_line_visible(self, p1, p2):
        # Standard collision check
        if not super()._check_line_visible(p1, p2):
            return False
        
        # Add custom constraint (e.g., max curvature)
        if self._exceeds_curvature_limit(p1, p2):
            return False
        
        return True
```

---

### 4. `search.py` - Path Search

**Purpose**: Find all paths from start to goal using depth-first search.

#### Core Class: `PathSearcher`

**Algorithm**: Recursive DFS with path enumeration

```python
def _depth_first_search(visited: List[Node], paths: List) -> List:
    current = visited[-1]
    
    if current.id == 1:  # Reached goal
        paths.append(copy(visited))
        return paths
    
    if len(paths) >= max_raw_path:  # Budget exhausted
        return paths
    
    for neighbor in current.neighbors:
        if neighbor not in visited:
            paths = _depth_first_search(visited + [neighbor], paths)
    
    return paths
```

**Key Features**:
- Exhaustive search up to `max_raw_path` limit
- Avoids cycles by checking visited nodes
- Returns paths sorted by length (shortest first)

**Key Methods**:

1. **`search_paths(graph)`**
   - Entry point for path search
   - Returns list of node sequences (start to goal)
   - Groups paths by length and limits output

#### Performance Characteristics

- **Time Complexity**: O(b^d) where b = branching factor, d = graph depth
- **Space Complexity**: O(d * N) for N paths of depth d
- **Bottleneck**: Dense graphs with high connectivity

**Optimization Strategies**:
1. Reduce `max_raw_path` if search is slow
2. Simplify graph (fewer nodes via `max_sample_num`)
3. Add heuristic pruning (see Extension below)

#### Extension: A* or Best-First Search

To replace DFS with A*:

```python
class AStarSearcher(PathSearcher):
    def search_paths(self, graph):
        # Priority queue by f = g + h
        frontier = PriorityQueue()
        frontier.put((0, graph[0], [graph[0]]))
        
        paths = []
        while not frontier.empty() and len(paths) < self.max_raw_path:
            cost, node, path = frontier.get()
            
            if node.id == 1:  # Goal
                paths.append(path)
                continue
            
            for neighbor in node.neighbors:
                if neighbor not in path:
                    g = cost + np.linalg.norm(neighbor.pos - node.pos)
                    h = np.linalg.norm(graph[1].pos - neighbor.pos)
                    frontier.put((g + h, neighbor, path + [neighbor]))
        
        return [[n.pos for n in path] for path in paths]
```

---

### 5. `path_processor.py` - Path Optimization

**Purpose**: Shortcut, filter, and select high-quality diverse paths.

#### Core Class: `PathProcessor`

**Pipeline**:
1. **Shortcut** raw paths (remove redundant waypoints)
2. **Prune** topologically equivalent paths
3. **Select** up to `reserve_num` diverse short paths

**Key Methods**:

1. **`shortcut_paths(paths, line_visible)`**
   - Removes waypoints where direct line-of-sight exists
   - Uses exponential shortcutting algorithm
   - Reduces path complexity while maintaining collision-free property

2. **`prune_topo_equivalent(paths, is_topo_same)`**
   - Removes duplicates using topological equivalence check
   - Keeps first occurrence of each unique topology
   - Critical for ensuring diversity

3. **`select_short_paths(paths, line_visible)`**
   - Filters paths by length ratio to shortest path
   - Selects up to `reserve_num` paths
   - Applies final shortcutting pass

4. **`discretize_path(path, N)`**
   - Uniformly resamples path to N waypoints
   - Used for topological comparison and smoothing
   - Returns (discretized_path, original_indices)

#### Shortcutting Algorithm

**Exponential Shortcut** (faster than iterative):
```python
def _exp_shortcut(path, line_visible):
    i = 0
    shortcut = [path[0]]
    
    while i < len(path) - 1:
        # Try increasingly distant waypoints
        for offset in [2^k for k in range(10)]:
            j = min(i + offset, len(path) - 1)
            if line_visible(path[i], path[j]):
                shortcut.append(path[j])
                i = j
                break
    
    return shortcut
```

**Complexity**: O(N log N) vs O(N²) for naive approach

#### Extension: Custom Path Scoring

To rank paths by custom metrics (e.g., clearance, smoothness):

```python
class CustomPathProcessor(PathProcessor):
    def select_short_paths(self, paths, line_visible):
        # Score each path
        scored = [(self._score_path(p), p) for p in paths]
        scored.sort(key=lambda x: x[0], reverse=True)  # Higher = better
        
        # Select top N
        selected = [p for _, p in scored[:self.reserve_num]]
        
        # Apply shortcutting
        return [self._exp_shortcut(p, line_visible) for p in selected]
    
    def _score_path(self, path):
        # Example: score by minimum clearance to obstacles
        clearances = [self._compute_clearance(wp) for wp in path]
        return min(clearances)
```

---

### 6. `planner.py` - Main Orchestrator

**Purpose**: High-level API that coordinates all components.

#### Core Class: `TopoPRM`

**Responsibilities**:
1. Initialize and configure all subcomponents
2. Manage occupancy grid state
3. Orchestrate planning pipeline
4. Collect timing diagnostics
5. Provide backward-compatible interface

**Key Method: `findTopoPaths(start, goal, reset=True)`**

**Planning Pipeline**:

```python
def findTopoPaths(start, goal, reset=True):
    # 1. Configure sampler
    sampler.configure(start, goal)
    
    # 2. Build graph (or reuse if reset=False)
    if reset:
        graph, samples = graph_builder.create_graph(sampler, start, goal)
    
    # 3. Search for paths
    raw_paths = path_searcher.search_paths(graph)
    
    # 4. Shortcut paths
    shortcut_paths = path_processor.shortcut_paths(raw_paths, line_visible)
    
    # 5. Prune topologically equivalent
    unique_paths = path_processor.prune_topo_equivalent(shortcut_paths)
    
    # 6. Select best N paths
    final_paths = path_processor.select_short_paths(unique_paths, line_visible)
    
    return final_paths, samples
```

**Timing Diagnostics**:

The planner collects detailed timing for each phase:
```python
planner.findTopoPaths(start, goal)
planner.print_timing_report(detailed=True)
```

Output:
```
======================================================================
TopoPRM Timing Report
======================================================================

Total Planning Time: 123.45 ms (0.1235 s)

Time Breakdown:
----------------------------------------------------------------------
  Graph Construction........................... 98.23 ms ( 79.6%)
  Path Search (DFS)............................  12.45 ms ( 10.1%)
  Path Shortcutting............................   8.12 ms (  6.6%)
  ...

Performance Analysis:
----------------------------------------------------------------------
  Bottleneck: Graph Construction
  Optimization Suggestions:
    - Reduce max_sample_num to decrease sampling iterations
    - Consider using cached/reused graphs (reset=False)
```

#### Extension: Incremental Replanning

To reuse graphs across multiple queries:

```python
# Initial planning
planner = TopoPRM(...)
paths1, _ = planner.findTopoPaths(start1, goal1, reset=True)

# Replan with same obstacles (faster!)
paths2, _ = planner.findTopoPaths(start2, goal2, reset=False)
```

**Benefits**:
- Skip expensive graph construction
- ~10-50x faster for similar start/goal
- Useful in dynamic replanning scenarios

**Caveats**:
- Graph centered on original start/goal ellipsoid
- If new query is far from original, coverage may be poor
- Monitor `num_final_paths` in timing report

---

## Data Flow

### High-Level Flow Diagram

```
┌──────────┐
│ Start/   │
│ Goal     │
└────┬─────┘
     │
     ▼
┌────────────────┐
│ Configure      │
│ Sampler        │◄─── safe_zones (optional)
└────┬───────────┘
     │
     ▼
┌────────────────┐
│ Sample N       │
│ Points         │◄─── occup_grid (collision check)
└────┬───────────┘
     │
     ▼
┌────────────────┐
│ Build Graph    │
│ (GUARD/CONN)   │
└────┬───────────┘
     │
     ▼
┌────────────────┐
│ DFS Search     │
│ All Paths      │
└────┬───────────┘
     │
     ▼
┌────────────────┐
│ Shortcut +     │
│ Prune Topo     │
└────┬───────────┘
     │
     ▼
┌────────────────┐
│ Select Best    │
│ N Paths        │
└────┬───────────┘
     │
     ▼
┌──────────┐
│ Final    │
│ Paths    │
└──────────┘
```

### Data Structures at Each Stage

| Stage             | Input          | Output         | Data Structure           |
| ----------------- | -------------- | -------------- | ------------------------ |
| Configure Sampler | start, goal    | sampler state  | Rotation matrix R, radii |
| Sampling          | sampler        | samples        | List[np.ndarray(3,)]     |
| Graph Building    | samples        | graph          | List[Node]               |
| Path Search       | graph          | raw_paths      | List[List[Node]]         |
| Shortcutting      | raw_paths      | shortcut_paths | List[List[np.ndarray]]   |
| Topo Pruning      | shortcut_paths | unique_paths   | List[List[np.ndarray]]   |
| Selection         | unique_paths   | final_paths    | List[List[np.ndarray]]   |

---

## Extension Points

### 1. Custom Collision Checkers

**Use Case**: Non-grid representations (point clouds, meshes, learned occupancy)

```python
class MyCollisionChecker:
    def find_occupied(self, pt, mark_out_of_bounds=False):
        """Check if point [x, y, z] is in collision."""
        # Your implementation
        return is_collision

# Inject into planner
planner.collision_checker = MyCollisionChecker()
```

### 2. Custom Node Attributes

**Use Case**: Cost-aware planning, probabilistic roadmaps

```python
# In nodes.py
class Node:
    # Add new attributes
    cost: float = 0.0
    probability: float = 1.0

# Use in graph building
def create_graph(...):
    node = Node(pos, type, id)
    node.cost = compute_cost(pos)  # Your logic
    node.probability = collision_probability(pos)
```

### 3. Parallel Sampling

**Use Case**: Large `max_sample_num` (>1000)

```python
from multiprocessing import Pool

class ParallelSampler(EllipsoidSampler):
    def get_samples_batch(self, N):
        with Pool() as pool:
            return pool.map(self.get_sample, range(N))
```

### 4. ROS 2 Integration

**Use Case**: Online replanning with sensor updates

```python
class ROS2TopoPRM(Node):  # ROS node
    def __init__(self):
        super().__init__('topo_prm_planner')
        self.planner = TopoPRM(...)
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/costmap', self.update_grid)
    
    def update_grid(self, msg):
        # Convert ROS grid to numpy
        self.planner.occup_grid = self.ros_to_numpy(msg)
        
        # Replan
        paths, _ = self.planner.findTopoPaths(
            self.start, self.goal, reset=False)  # Incremental!
```

---

## Performance Considerations

### Profiling Integration

**Built-in cProfile Support** (added in test suite):

```python
# In tests/topo_prm/test_topo_prm.py
tester.run_test("corridor", profile=True, profile_dir="./profiles")
```

Generates:
- `test_topo_prm_corridor_20251028_143052.prof` (binary, for visualization)
- `test_topo_prm_corridor_20251028_143052.perf` (text summary)

**Visualize with SnakeViz**:
```bash
pip install snakeviz
snakeviz profiles/*.prof
```

**Flamegraph Generation**:
```bash
# Convert to flamegraph format
python -c "import pstats; p=pstats.Stats('file.prof'); p.sort_stats('cumulative'); p.print_stats()" > stats.txt
# Use flamegraph.pl or similar
```

### Bottleneck Analysis

**Typical Bottlenecks** (in order of frequency):

1. **Graph Construction (60-80% of time)**
   - Dominated by collision checks during visibility testing
   - **Fix**: Reduce `max_sample_num`, increase `ray_resolution`

2. **Path Search (10-20%)**
   - Exponential in graph connectivity
   - **Fix**: Reduce `max_raw_path`, simplify graph

3. **Topological Pruning (5-10%)**
   - Pairwise path comparisons
   - **Fix**: Increase `topo_resolution`, reduce `max_raw_path2`

### Complexity Analysis

| Operation      | Time Complexity | Space Complexity | Notes                                        |
| -------------- | --------------- | ---------------- | -------------------------------------------- |
| Sampling       | O(N)            | O(N)             | N = `max_sample_num`                         |
| Graph Building | O(N² · R/r)     | O(N²)            | R = max distance, r = `ray_resolution`       |
| DFS Search     | O(b^d · P)      | O(d · P)         | b = branching, d = depth, P = `max_raw_path` |
| Shortcutting   | O(P · W log W)  | O(P · W)         | W = waypoints per path                       |
| Topo Pruning   | O(P² · W/t)     | O(P · W/t)       | t = `topo_resolution`                        |

**Overall**: O(N² · R/r) dominated by graph construction

### Recommended Parameter Sets

**Real-Time** (< 100ms):
```python
TopoPRM(
    max_sample_num=50,
    max_raw_path=5,
    max_raw_path2=3,
    reserve_num=2,
    ray_resolution=0.3,
    max_time=0.05
)
```

**Balanced** (100-500ms):
```python
TopoPRM(
    max_sample_num=200,
    max_raw_path=20,
    max_raw_path2=10,
    reserve_num=5,
    ray_resolution=0.2,
    max_time=0.2
)
```

**High Quality** (500-2000ms):
```python
TopoPRM(
    max_sample_num=500,
    max_raw_path=50,
    max_raw_path2=20,
    reserve_num=10,
    ray_resolution=0.1,
    max_time=1.0
)
```

---

## Testing Strategy

### Test Suite Structure

```
tests/topo_prm/
├── test_topo_prm.py              # Integration tests
├── test_topo_prm_validation.py   # Correctness tests
├── test_topo_prm_refactoring.py  # Regression tests
└── test_quick_verify.py          # Smoke tests
```

### Test Categories

**1. Unit Tests** (test individual modules):
```python
def test_ellipsoid_sampler():
    sampler = EllipsoidSampler(...)
    sampler.configure(start, goal)
    sample = sampler.get_sample()
    assert is_inside_ellipsoid(sample)
```

**2. Integration Tests** (test full pipeline):
```python
def test_simple_scenario():
    planner = TopoPRM(...)
    paths, _ = planner.findTopoPaths(start, goal)
    assert paths is not None
    assert len(paths) > 0
```

**3. Correctness Tests** (verify guarantees):
```python
def test_paths_are_collision_free():
    paths, _ = planner.findTopoPaths(start, goal)
    for path in paths:
        assert all(not is_collision(wp) for wp in path)

def test_paths_are_topologically_distinct():
    paths, _ = planner.findTopoPaths(start, goal)
    for i, p1 in enumerate(paths):
        for p2 in paths[i+1:]:
            assert not are_topo_equivalent(p1, p2)
```

**4. Performance Tests** (profiling-enabled):
```python
def test_planning_time():
    result = tester.run_test("dense", profile=True)
    assert result['metrics']['planning_time'] < 2.0
```

### Running Tests

**Run all tests**:
```bash
pytest tests/topo_prm/
```

**Run with profiling**:
```bash
python tests/topo_prm/test_topo_prm.py  # Uses profile=True in main()
```

**Run specific test**:
```bash
pytest tests/topo_prm/test_topo_prm.py::TestTopoPRM::test_simple_scenario
```

## New Features

**1. Timing Diagnostics**:
```python
paths, _ = planner.findTopoPaths(start, goal)
planner.print_timing_report(detailed=True)
```

**2. Incremental Replanning**:
```python
# First query: build graph
paths1, _ = planner.findTopoPaths(start1, goal1, reset=True)

# Subsequent queries: reuse graph
paths2, _ = planner.findTopoPaths(start2, goal2, reset=False)
```

**3. Profiling**:
```python
# In test code
tester.run_test("scenario", profile=True, profile_dir="./profiles")
```

## Frequently Asked Questions

### Q: Why are some paths missing obvious routes?

**A**: Probabilistic sampling may miss narrow passages. Solutions:
1. Increase `max_sample_num`
2. Add safe zones in critical areas: `planner.safe_zones = np.array([[x, y, z], ...])`
3. Increase `sample_inflate` to widen sampling ellipsoid

### Q: Planning is too slow. How to speed up?

**A**: See [Performance Considerations](#performance-considerations). Quick wins:
1. Reduce `max_sample_num` (50-100 for real-time)
2. Increase `ray_resolution` (0.3-0.5)

### Q: All paths look the same. How to increase diversity?

**A**: 
1. Reduce `topo_resolution` (more strict equivalence checking)
2. Increase `max_raw_path` (explore more graph paths)

### Q: How to debug "No paths found"?

**A**: Check these in order:
1. Verify start/goal are collision-free: `planner.find_occupied(start)`
2. Increase `max_sample_num` (may need more samples to connect)
3. Check ellipsoid covers feasible region (increase `sample_inflate`)
4. Visualize samples: `plt.scatter(samples[:, 0], samples[:, 1])`

### Q: Can I use this with kinodynamic constraints?

**A**: Not directly. TopoPRM is geometric. For kinodynamic:
1. Use TopoPRM to find topological paths
2. Post-process with trajectory optimization
3. Or: Extend visibility checks to include kinematic feasibility

---

## References

### Academic Background

- **Original TopoPRM Paper**: [Citation needed - add if available]
- **PRM Algorithm**: Kavraki et al., "Probabilistic Roadmaps for Path Planning in High-Dimensional Configuration Spaces", 1996
- **Topological Path Planning**: Bhattacharya et al., "Search-based Path Planning with Homotopy Class Constraints", 2010

### Related Modules

- `grid.py`: Occupancy grid representation
- `pydecomp`: Geometric decomposition utilities (external dependency)
- `jax_mppi`: Model predictive path integral control (parent package)

### External Tools

- **SnakeViz**: cProfile visualization (`pip install snakeviz`)
- **pytest**: Testing framework (`pip install pytest`)
- **matplotlib**: Path visualization (`pip install matplotlib`)

---

## Changelog

### v2.0 (October 2025) - Modular Refactor
- Split monolithic file into 7 focused modules
- Added comprehensive type hints
- Introduced timing diagnostics and profiling support
- Improved documentation and testing
- **Breaking Changes**: None (backward compatible)

### v1.0 (Prior) - Monolithic Implementation
- Single-file implementation (~2000 lines)
- Core TopoPRM algorithm

---

## Contributing

### Code Style

- **Type Hints**: All public methods must have type annotations
- **Docstrings**: Google-style docstrings required
- **Testing**: New features require unit + integration tests
- **Performance**: Profile before/after optimization changes

### Pull Request Checklist

- [ ] Code follows existing architecture patterns
- [ ] Type hints added for new functions
- [ ] Docstrings updated
- [ ] Tests added/updated
- [ ] Performance impact profiled (if relevant)
- [ ] ARCHITECTURE.md updated (if public API changes)

### Contact

For questions or contributions, contact the Branch MPPI team.

---

**End of Documentation**

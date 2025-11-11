"""
Node data structures for the TopoPRM graph.

This module defines the node types and data structures used in the
topological roadmap graph representation.
"""

from enum import Enum
from typing import List, Union

import numpy as np
import numpy.typing as npt


class NODE_TYPE(Enum):
    """Type of node in the topological roadmap.
    
    Attributes:
        GUARD: Boundary of a visibility region. Created when a sample sees
               zero existing guard nodes, indicating a new region.
        CONNECTOR: Bridges between visibility regions. Created when a sample
                   sees two or more guard nodes, connecting distinct regions.
    """
    GUARD = 1
    CONNECTOR = 2


class NODE_STATE(Enum):
    """State of node during graph search.
    
    Attributes:
        NEW: Node has not yet been visited during search
        CLOSE: Node has been fully processed
        OPEN: Node is in the search frontier
    """
    NEW = 1
    CLOSE = 2
    OPEN = 3


class Node:
    """
    A node in the topological roadmap graph.
    
    Nodes represent either boundaries between visibility regions (GUARD nodes)
    or connections between regions (CONNECTOR nodes). The graph is built by
    sampling the configuration space and creating nodes based on visibility
    relationships.
    
    Attributes:
        pos (np.ndarray): 3D position [x, y, z] of the node
        type (NODE_TYPE): Either GUARD or CONNECTOR
        state (NODE_STATE): Current state during search (NEW, CLOSE, or OPEN)
        id (int): Unique identifier for this node
        sz (bool): Whether this node originated from a safe zone
        neighbors (List[Node]): List of nodes directly connected to this node
        
    Example:
        >>> guard = Node(np.array([1.0, 2.0, 0.0]), NODE_TYPE.GUARD, id=0)
        >>> connector = Node(np.array([3.0, 4.0, 0.0]), NODE_TYPE.CONNECTOR, id=1)
        >>> guard.neighbors.append(connector)
    """
    
    def __init__(
        self, 
        pos: Union[npt.NDArray[np.floating], List[float]], 
        type: NODE_TYPE, 
        id: int, 
        sz: bool = False
    ) -> None:
        """
        Initialize a roadmap node.
        
        Args:
            pos: 3D position array [x, y, z]
            type: NODE_TYPE.GUARD or NODE_TYPE.CONNECTOR
            id: Unique integer identifier
            sz: True if node is from a safe zone (default: False)
            
        Raises:
            ValueError: If position is invalid or type is not NODE_TYPE
            TypeError: If id is not an integer
        """
        # Validate position
        pos_array = np.array(pos)
        if pos_array.shape != (3,):
            raise ValueError(f"Position must be 3D array, got shape {pos_array.shape}")
        if not np.all(np.isfinite(pos_array)):
            raise ValueError(f"Position must have finite values, got {pos_array}")
        
        # Validate type
        if not isinstance(type, NODE_TYPE):
            raise TypeError(f"Type must be NODE_TYPE enum, got {type.__class__.__name__}")
        
        # Validate id
        if not isinstance(id, (int, np.integer)):
            raise TypeError(f"ID must be integer, got {id.__class__.__name__}")
        if id < 0:
            raise ValueError(f"ID must be non-negative, got {id}")
        
        # Validate sz flag
        if not isinstance(sz, (bool, np.bool_)):
            raise TypeError(f"sz must be boolean, got {sz.__class__.__name__}")
        
        self.pos = pos_array
        self.type = type
        self.state = NODE_STATE.NEW
        self.id = int(id)
        self.sz = bool(sz)
        self.neighbors: List['Node'] = []
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f'Node(id={self.id}, type={self.type.name}, state={self.state.name}, pos={self.pos})'

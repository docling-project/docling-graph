"""
Defines shared Pydantic models for graph conversion, such as the Edge model.
"""

from typing import TypeVar, Generic, Optional
from pydantic import BaseModel, Field

# T is a type variable that can be any Pydantic BaseModel
T = TypeVar('T', bound=BaseModel)

class Edge(BaseModel, Generic[T]):
    """
    A generic Pydantic model to represent a rich edge in the graph.
    
    Instead of:
        my_field: MyModel
    
    You can use:
        my_field: Edge[MyModel] = Edge(target=MyModel(...), label="CUSTOM_LABEL")
    
    The GraphConverter will automatically unpack this and use the attributes
    (label, weight, etc.) as properties for the graph edge.
    """
    # The 'target' is the actual Pydantic model instance you are linking to.
    target: T
    
    # 'label' is the most common edge property.
    label: str = Field(default="related_to")
    
    # You can add any other properties you want on your edges.
    weight: Optional[float] = None
    context: Optional[str] = None


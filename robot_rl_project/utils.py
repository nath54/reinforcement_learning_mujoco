"""
Utility classes and helper functions.
"""
import random
import xml.etree.ElementTree as ET


class Vec3:
    """3D vector representation."""
    
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x: float = x
        self.y: float = y
        self.z: float = z


class ValType:
    """Custom Value or Interval Type for randomization."""
    
    def __init__(self, from_value: float | tuple[float, float]) -> None:
        self.value: float | tuple[float, float] = from_value
    
    def get_value(self) -> float:
        """Get a random value from the interval or the fixed value."""
        return random.uniform(*self.value) if isinstance(self.value, tuple) else self.value
    
    def get_max_value(self) -> float:
        """Get the maximum value from the interval or the fixed value."""
        return max(*self.value) if isinstance(self.value, tuple) else self.value


def create_geom(name: str, geom_type: str, pos: Vec3, size: Vec3) -> ET.Element:
    """Helper function to create geometry XML elements."""
    new_geom = ET.Element('geom')
    new_geom.set('name', name)
    new_geom.set('type', geom_type)
    new_geom.set('size', f'{size.x} {size.y} {size.z}')
    new_geom.set('pos', f'{pos.x} {pos.y} {pos.z}')
    return new_geom

"""
Corridor generation and management.
"""
from typing import Any, Optional, cast
import xml.etree.ElementTree as ET
from math import floor, sin, pi

from utils import Vec3, ValType, create_geom


class Corridor:
    """Manages corridor generation and extraction from XML."""
    
    def __init__(self) -> None:
        self.xml_file_path: str = "corridor_3x100.xml"
    
    def extract_corridor_from_xml(self) -> dict[str, Any]:
        """
        Extract corridor components from existing XML file.
        This teaches students how to parse and reuse XML components.
        """
        print(f"Extracting corridor components from {self.xml_file_path}...")
        
        # Parse the existing XML file
        tree: ET.ElementTree[ET.Element] = ET.parse(self.xml_file_path)
        root: ET.Element = tree.getroot()
        
        # Extract useful components
        components: dict[str, Optional[Any]] = {
            'compiler': None,
            'option': None,
            'default': None,
            'asset': None,
            'body': None
        }
        
        comps_body: list[ET.Element] = []
        
        # Add global scene ground
        floor_geom = ET.Element('geom')
        floor_geom.set('name', 'global_floor')
        floor_geom.set('type', 'plane')
        floor_geom.set('size', '200 200 0.1')
        floor_geom.set('pos', '0 0 0.12')
        comps_body.append(floor_geom)
        
        # Find and extract each component
        for child in root:
            if child.tag == 'compiler':
                components['compiler'] = child
            elif child.tag == 'option':
                components['option'] = child
            elif child.tag == 'default':
                components['default'] = child
            elif child.tag == 'asset':
                components['asset'] = child
            elif child.tag == 'worldbody':
                for body in child:
                    comps_body.append(body)
                components["body"] = comps_body
            elif child.tag == 'actuator':
                components['actuators'] = child
        
        return components
    
    def generate_corridor(
        self,
        corridor_length: ValType = ValType(100.0),
        corridor_width: ValType = ValType(3.0),
        obstacles_mode: str = "none",
        obstacles_mode_param: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Generate a corridor with obstacles based on parameters."""
        
        # Generate useful components
        components: dict[str, Optional[Any]] = {
            'compiler': None,
            'option': None,
            'default': None,
            'asset': None,
            'body': []
        }
        
        components_body: list[ET.Element] = []
        
        # Generate a corridor based on parameters
        obstacles_mode_param_: dict[str, Any] = {} if obstacles_mode_param is None else obstacles_mode_param
        
        corridor_length_: float = corridor_length.get_value()
        corridor_width_: float = corridor_width.get_value()
        
        wall_width: float = 0.3
        wall_height: float = 10.0
        
        # Add global scene ground
        components_body.append(
            create_geom(
                name="global_floor",
                geom_type="plane",
                size=Vec3(x=corridor_length_ * 10.0, y=corridor_length_ * 10.0, z=1.0),
                pos=Vec3(x=0, y=0, z=0)
            )
        )
        
        # Add the two walls
        components_body.append(
            create_geom(
                name="wall_1",
                geom_type="box",
                size=Vec3(x=corridor_length_, y=wall_width, z=wall_height),
                pos=Vec3(x=corridor_length_, y=-corridor_width_, z=wall_height)
            )
        )
        components_body.append(
            create_geom(
                name="wall_2",
                geom_type="box",
                size=Vec3(x=corridor_length_, y=wall_width, z=wall_height),
                pos=Vec3(x=corridor_length_, y=+corridor_width_, z=wall_height)
            )
        )
        
        # Add the obstacles
        if obstacles_mode == "sinusoidal":
            obs_size_x: ValType = cast(ValType, obstacles_mode_param_.get("obstacle_size_x", 0.2))
            obs_size_y: ValType = cast(ValType, obstacles_mode_param_.get("obstacle_size_y", 0.2))
            obs_size_z: ValType = cast(ValType, obstacles_mode_param_.get("obstacle_size_z", 0.2))
            obs_sep: ValType = cast(ValType, obstacles_mode_param_.get("obstacle_sep", 0.2))
            
            obs_length: float = obs_size_x.get_max_value() + obs_sep.get_max_value()
            nb_obstacles: int = max(floor(corridor_length_ / obs_length) + 1, 1)
            current_obs_x: float = 2 + obs_sep.get_value()
            
            for obs_idx in range(nb_obstacles):
                current_obs_size_x: float = obs_size_x.get_value()
                current_obs_size_y: float = obs_size_y.get_value()
                current_obs_size_z: float = obs_size_z.get_value()
                
                current_obs_y: float = sin(float(obs_idx) * 0.16 * pi) * (corridor_width_ - 2 * current_obs_size_y)
                current_obs_z: float = 0
                
                components_body.append(
                    create_geom(
                        name=f"obstacle_{obs_idx}",
                        geom_type="box",
                        size=Vec3(x=current_obs_size_x, y=current_obs_size_y, z=current_obs_size_z),
                        pos=Vec3(x=current_obs_x, y=current_obs_y, z=current_obs_z)
                    )
                )
                
                current_obs_x += current_obs_size_x + obs_sep.get_max_value()
        
        elif obstacles_mode == "double_sinusoidal":
            pass
        
        elif obstacles_mode == "random":
            pass
        
        components['body'] = components_body
        
        return components

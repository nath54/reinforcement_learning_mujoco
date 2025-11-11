"""
Robot class for managing robot XML generation and visualization.
"""
from typing import Any, Optional
import xml.etree.ElementTree as ET
import math


class Robot:
    """Manages robot components and sensor box generation."""
    
    def __init__(
        self,
        sensor_boxes_enabled: bool = True,
        sensor_box_size: float = 0.5,
        sensor_layers: int = 2,
        boxes_per_layer: int = 8
    ) -> None:
        self.xml_file_path: str = "four_wheels_robot.xml"
        
        # Sensor configuration
        self.sensor_boxes_enabled: bool = sensor_boxes_enabled
        self.sensor_box_size: float = sensor_box_size
        self.sensor_layers: int = sensor_layers
        self.boxes_per_layer: int = boxes_per_layer
        
        # Parse the existing XML file
        self.tree: ET.ElementTree[ET.Element] = ET.parse(self.xml_file_path)
        self.root: ET.Element = self.tree.getroot()
    
    def create_sensor_boxes(self, robot_body: ET.Element) -> list[ET.Element]:
        """
        Create non-colliding sensor boxes around the robot for vision detection.
        Returns a list of sensor site elements.
        """
        if not self.sensor_boxes_enabled:
            return []
        
        sensor_sites: list[ET.Element] = []
        
        # Calculate robot chassis approximate radius
        chassis_radius: float = 0.5
        
        for layer in range(self.sensor_layers):
            # Distance from robot center increases with each layer
            distance: float = chassis_radius + (layer + 1) * self.sensor_box_size
            
            for box_idx in range(self.boxes_per_layer):
                # Distribute boxes evenly around circle
                angle: float = (2 * math.pi * box_idx) / self.boxes_per_layer
                
                # Calculate position
                x_pos: float = distance * math.cos(angle)
                y_pos: float = distance * math.sin(angle)
                z_pos: float = 0  # Same height as robot center
                
                # Create sensor site (not a body with geom)
                sensor_site: ET.Element = ET.Element('site')
                sensor_name = f'sensor_L{layer}_B{box_idx}'
                sensor_site.set('name', sensor_name)
                sensor_site.set('type', 'box')
                sensor_site.set('pos', f'{x_pos:.3f} {y_pos:.3f} {z_pos:.3f}')
                
                # Set site size (half-lengths for box sites)
                half_size: float = self.sensor_box_size / 2
                sensor_site.set('size', f'{half_size:.3f} {half_size:.3f} {half_size:.3f}')
                
                # Make sensor visible for debugging
                sensor_site.set('rgba', '0 1 0 0.3')  # Green with transparency
                
                # Sites don't collide by default, so no contype/conaffinity needed
                sensor_sites.append(sensor_site)
        
        return sensor_sites
    
    def extract_robot_from_xml(self) -> dict[str, Any]:
        """
        Extract robot components from existing XML file.
        This teaches students how to parse and reuse XML components.
        """
        print(f"Extracting robot components from {self.xml_file_path}...")
        
        # Parse the existing XML file
        tree: ET.ElementTree[ET.Element] = ET.parse(self.xml_file_path)
        root: ET.Element = tree.getroot()
        
        # Extract useful components
        components: dict[str, Optional[Any]] = {
            'compiler': None,
            'option': None,
            'default': None,
            'asset': None,
            'robot_body': None,
            'actuators': None,
            'sensor_boxes': []
        }
        
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
                # Extract the robot body from worldbody
                for body in child:
                    if body.get('name') == 'robot':
                        components['robot_body'] = body
            elif child.tag == 'actuator':
                components['actuators'] = child
        
        # Add sensor boxes to robot body
        if self.sensor_boxes_enabled:
            sensor_sites = self.create_sensor_boxes(body)
            for sensor_site in sensor_sites:
                body.append(sensor_site)  # Add sites directly to robot body
            components['sensor_sites'] = sensor_sites
            print(f"  Added {len(sensor_sites)} sensor sites ({self.sensor_layers} layers × {self.boxes_per_layer} boxes)")
        
        return components
    
    def create_enhanced_materials(self) -> dict[str, ET.Element]:
        """
        Create enhanced materials with textures and visual appeal.
        Returns a list of material elements.
        """
        # Enhanced robot materials
        robot_materials: list[dict[str, str]] = [
            {
                'name': 'mat_chassis_violet',
                'texture': 'tex_chassis',
                'rgba': '0.39 0.13 0.63 1',
                'shininess': '0.3',
                'specular': '0.5'
            },
            {
                'name': 'mat_wheel_black',
                'texture': 'tex_wheel_radius',
                'rgba': '0.1 0.1 0.1 1',
                'shininess': '0.6',
                'specular': '0.3'
            }
        ]
        
        # Initialize Materials Containers
        materials: dict[str, ET.Element] = {}
        
        # Create material elements
        for mat_config in robot_materials:
            material: ET.Element = ET.Element('material')
            mat_name: str = mat_config['name']
            material.set('name', mat_name)
            material.set('rgba', mat_config['rgba'])
            
            if 'texture' in mat_config:
                material.set('texture', mat_config['texture'])
            if 'shininess' in mat_config:
                material.set('shininess', mat_config['shininess'])
            if 'specular' in mat_config:
                material.set('specular', mat_config['specular'])
            if 'texrepeat' in mat_config:
                material.set('texrepeat', mat_config['texrepeat'])
            
            materials[mat_name] = material
        
        return materials
    
    def create_textures(self) -> dict[str, ET.Element]:
        """
        Create texture elements for enhanced visuals.
        Returns a list of texture elements.
        """
        textures: dict[str, ET.Element] = {}
        
        # Wheel radius texture to show rotation
        wheel_texture = ET.Element('texture')
        wheel_texture.set('name', 'tex_wheel_radius')
        wheel_texture.set('type', '2d')
        wheel_texture.set('builtin', 'checker')
        wheel_texture.set('rgb1', '0.8 0.2 0.2')
        wheel_texture.set('rgb2', '0.1 0.1 0.1')
        wheel_texture.set('width', '32')
        wheel_texture.set('height', '32')
        wheel_texture.set('mark', 'random')
        wheel_texture.set('markrgb', '0.8 0.8 0.2')
        textures["tex_wheel_radius"] = wheel_texture
        
        wheel_texture = ET.Element('texture')
        wheel_texture.set('name', 'tex_chassis')
        wheel_texture.set('type', '2d')
        wheel_texture.set('builtin', 'gradient')
        wheel_texture.set('rgb1', '0.39 0.13 0.63')
        wheel_texture.set('rgb2', '0 0 0')
        wheel_texture.set('width', '32')
        wheel_texture.set('height', '32')
        textures["tex_chassis"] = wheel_texture
        
        return textures
    
    def enhance_robot_visuals(self, robot_body: Optional[ET.Element]) -> None:
        """
        Enhance robot body with better materials and visual features.
        Also adjusts wheel positions to be outside the robot body.
        Modifies the robot body XML element in place.
        """
        if robot_body is None:
            return
        
        print("Enhancing robot visuals...")
        
        # Find and update chassis material
        for geom in robot_body.iter('geom'):
            if 'chassis' in geom.get('name', ''):
                geom.set('material', 'mat_chassis_violet')
                print("  Updated chassis color")
        
        # Find and update wheel materials
        wheel_count: int = 0
        for body in robot_body.iter('body'):
            body_name: str = body.get('name', '')
            if body_name.startswith('wheel_'):
                # Update wheel material
                for geom in body.iter('geom'):
                    if geom.get('name', '').startswith('geom_'):
                        geom.set('material', 'mat_wheel_black')
                        wheel_count += 1
                
                # Log the wheel position
                current_pos = body.get('pos', '0 0 0')
                print(f"  {body_name} at position: {current_pos}")
        
        print(f"  Updated {wheel_count} wheels with textured black material")
        print("  Wheel positions defined in XML (no runtime modification)")

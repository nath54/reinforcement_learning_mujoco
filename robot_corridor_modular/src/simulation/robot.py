import xml.etree.ElementTree as ET
from typing import Any, Optional, Dict
from src.core.types import RobotConfig

class Robot:
    def __init__(self, config: RobotConfig) -> None:
        self.xml_file_path: str = config.xml_path
        # Use simple parsing initially
        try:
            self.tree: ET.ElementTree = ET.parse(self.xml_file_path)
            self.root: ET.Element = self.tree.getroot()
        except FileNotFoundError:
             # Fallback or error handling if running from different dir
             print(f"Warning: Robot XML {self.xml_file_path} not found.")
             self.root = ET.Element('mujoco') # Dummy

    def extract_robot_from_xml(self) -> Dict[str, Optional[Any]]:
        print(f"Extracting robot components from {self.xml_file_path}...")
        components: Dict[str, Optional[Any]] = {
            'compiler': None, 'option': None, 'default': None,
            'asset': None, 'robot_body': None, 'actuators': None
        }
        
        for child in self.root:
            if child.tag == 'compiler': components['compiler'] = child
            elif child.tag == 'option': components['option'] = child
            elif child.tag == 'default': components['default'] = child
            elif child.tag == 'asset': components['asset'] = child
            elif child.tag == 'worldbody':
                for body in child:
                    if body.get('name') == 'robot':
                        components['robot_body'] = body
            elif child.tag == 'actuator': components['actuators'] = child
        return components

    def create_enhanced_materials(self) -> Dict[str, ET.Element]:
        robot_materials = [
            {'name': 'mat_chassis_violet', 'rgba': '0.39 0.13 0.63 1', 'shininess': '0.3', 'specular': '0.5', 'texture': 'tex_chassis'},
            {'name': 'mat_wheel_black', 'rgba': '0.1 0.1 0.1 1', 'shininess': '0.6', 'specular': '0.3', 'texture': 'tex_wheel_radius'}
        ]
        materials = {}
        for mat_config in robot_materials:
            material = ET.Element('material', **mat_config)
            materials[mat_config['name']] = material
        return materials

    def create_textures(self) -> Dict[str, ET.Element]:
        textures = {}
        # Wheel texture
        textures["tex_wheel_radius"] = ET.Element('texture', name='tex_wheel_radius', type='2d', builtin='checker', rgb1='0.8 0.2 0.2', rgb2='0.1 0.1 0.1', width='32', height='32', mark='random', markrgb='0.8 0.8 0.2')
        # Chassis texture
        textures["tex_chassis"] = ET.Element('texture', name='tex_chassis', type='2d', builtin='gradient', rgb1='0.39 0.13 0.63', rgb2='0 0 0', width='32', height='32')
        return textures

    def enhance_robot_visuals(self, robot_body: Optional[ET.Element]) -> None:
        if robot_body is None: return
        print("Enhancing robot visuals...")
        for geom in robot_body.iter('geom'):
            if 'chassis' in geom.get('name', ''):
                geom.set('material', 'mat_chassis_violet')
        
        for body in robot_body.iter('body'):
            if body.get('name', '').startswith('wheel_'):
                for geom in body.iter('geom'):
                    if geom.get('name', '').startswith('geom_'):
                        geom.set('material', 'mat_wheel_black')
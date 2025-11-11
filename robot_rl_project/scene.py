"""
Root world scene construction and management.
"""
from typing import Optional, Any
import xml.etree.ElementTree as ET
import mujoco

from config import RENDER_WIDTH, RENDER_HEIGHT
from corridor import Corridor
from robot import Robot


class RootWorldScene:
    """Manages the complete MuJoCo scene with robot and corridor."""
    
    def __init__(
        self,
        sensor_boxes_enabled: bool = True,
        sensor_box_size: float = 0.5,
        sensor_layers: int = 2,
        boxes_per_layer: int = 8
    ) -> None:
        self.corridor: Corridor = Corridor()
        self.robot: Robot = Robot(
            sensor_boxes_enabled=sensor_boxes_enabled,
            sensor_box_size=sensor_box_size,
            sensor_layers=sensor_layers,
            boxes_per_layer=boxes_per_layer
        )
        
        self.mujoco_model: mujoco.MjModel
        self.mujoco_data: mujoco.MjData
    
    def build_combined_model(
        self,
        robot_components: dict[str, Optional[ET.Element]],
        corridor_components: dict[str, Optional[ET.Element]],
        floor_type: str = "standard",
        robot_height: float = 1.0
    ) -> mujoco.MjModel:
        """
        Build a complete MuJoCo model by combining robot components with programmatic floor.
        Environment controls physics settings (gravity, timestep, etc).
        """
        print("Building combined model...")
        print(f"Robot starting height: {robot_height}m above floor")
        
        # Create root mujoco element
        root: ET.Element = ET.Element('mujoco')
        root.set('model', 'robot_with_programmatic_floor')
        
        g: ET.Element = ET.Element('global')
        g.set("offwidth", str(RENDER_WIDTH))
        g.set("offheight", str(RENDER_HEIGHT))
        visual: ET.Element = ET.Element("visual")
        visual.append(g)
        root.append(visual)
        
        # Add compiler settings from robot XML
        if robot_components['compiler'] is not None:
            root.append(robot_components['compiler'])
        
        # CREATE ENVIRONMENT-CONTROLLED PHYSICS SETTINGS
        option: ET.Element = ET.Element('option')
        option.set('timestep', '0.001')
        option.set("gravity", "0 0 -0.20")
        option.set('solver', 'Newton')
        option.set('iterations', '500')
        root.append(option)
        print(f"  Environment physics: gravity enabled ({option.get('gravity')}), timestep={option.get('timestep')}s")
        
        # Add size settings
        size: ET.Element = ET.Element('size')
        size.set('njmax', '1000')
        size.set('nconmax', '500')
        root.append(size)
        
        if robot_components['default'] is not None:
            root.append(robot_components['default'])
        
        # Create asset section with textures and enhanced materials
        asset: ET.Element = ET.Element('asset')
        
        # Add textures first
        for texture in self.robot.create_textures().values():
            asset.append(texture)
        
        # Add enhanced materials
        enhanced_materials: dict[str, ET.Element] = self.robot.create_enhanced_materials()
        materials_used: set[str] = set()
        for material in enhanced_materials.values():
            asset.append(material)
            materials_used.add(material.get('name', ''))
        
        # Also keep any original materials from robot XML if they don't exist
        if robot_components['asset'] is not None:
            for original_material in robot_components['asset']:
                material_name = original_material.get('name', '')
                if material_name not in enhanced_materials and material_name not in materials_used:
                    asset.append(original_material)
                    materials_used.add(material_name)
        
        # Also keep any original materials from corridor XML if they don't exist
        if corridor_components['asset'] is not None:
            for original_material in corridor_components['asset']:
                material_name = original_material.get('name', '')
                if material_name not in enhanced_materials and material_name not in materials_used:
                    asset.append(original_material)
                    materials_used.add(material_name)
        
        root.append(asset)
        
        # Create worldbody with corridor and robot
        worldbody: ET.Element = ET.Element('worldbody')
        
        # Add all the corridor worldbody to this worldbody
        if corridor_components['body'] is not None:
            for body in corridor_components['body']:
                worldbody.append(body)
        
        # Add robot body with enhanced visuals and adjusted height
        if robot_components['robot_body'] is not None:
            # Enhance the robot's visual appearance
            self.robot.enhance_robot_visuals(robot_components['robot_body'])
            
            # Adjust robot starting height
            robot_z_position: float = robot_height - 0.1 + 10
            current_pos: str = robot_components['robot_body'].get('pos', '0 0 0.2')
            pos_parts: list[str] = current_pos.split()
            
            if len(pos_parts) == 3:
                new_pos: str = f"{pos_parts[0]} {pos_parts[1]} {robot_z_position}"
                robot_components['robot_body'].set('pos', new_pos)
                print(f"  Robot positioned at: {new_pos} (will fall {robot_height}m to floor)")
            
            robot_components['robot_body'].set('pos', "0 0 5")
            robot_components['robot_body'].set('euler', "0 0 0")
            
            worldbody.append(robot_components['robot_body'])
        
        # Add worldbody to root
        root.append(worldbody)
        
        # Add actuators
        if robot_components['actuators'] is not None:
            root.append(robot_components['actuators'])
        
        # Convert to XML string
        xml_string = ET.tostring(root, encoding='unicode')
        
        # Create and return MuJoCo model
        return mujoco.MjModel.from_xml_string(xml_string)
    
    def construct_scene(
        self,
        floor_type: str = "standard",
        robot_height: float = 1.0,
        corridor_params: Optional[dict[str, Any]] = None
    ) -> None:
        """Construct the complete scene."""
        # Build combined model with enhanced visuals and physics
        if corridor_params is None:
            from config import GENERATE_CORRIDOR_PARAM
            corridor_params = GENERATE_CORRIDOR_PARAM
        
        self.mujoco_model = self.build_combined_model(
            robot_components=self.robot.extract_robot_from_xml(),
            corridor_components=self.corridor.generate_corridor(**corridor_params),
            floor_type=floor_type,
            robot_height=robot_height
        )
        self.mujoco_data = mujoco.MjData(self.mujoco_model)

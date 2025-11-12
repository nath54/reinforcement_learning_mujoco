#
### Import Modules. ###
#
from typing import Any, Optional, cast
#
import os
import json
import argparse
import random
import math
#
from math import floor, sin, pi
#
from matplotlib import pyplot as plt
#
### Set environment variable to indicate mujoco to use GPU rendering. ###
#
os.environ["MUJOCO_GL"] = "egl"

#
import mujoco
from mujoco import viewer as viewer_  # type: ignore
#
import xml.etree.ElementTree as ET
#
import numpy as np
from numpy.typing import NDArray
#
from tqdm import tqdm

#
### Graphics and plotting. ###
#
import mediapy as media


#
class Vec3:

    #
    def __init__(self, x: float, y: float, z: float) -> None:

        #
        self.x: float = x
        self.y: float = y
        self.z: float = z


#
### Custom Value or Interval Type. ###
#
class ValType:

    #
    def __init__(
        self,
        from_value: float | tuple[float, float]
    ) -> None:

        #
        self.value: float | tuple[float, float] = from_value

    #
    def get_value(self) -> float:

        #
        return random.uniform(*self.value) if isinstance(self.value, tuple) else self.value

    #
    def get_max_value(self) -> float:

        #
        return max(*self.value) if isinstance(self.value, tuple) else self.value


#
viewer: Any = cast(Any, viewer_)  # fix to remove pylance type hinting errors with mujoco.viewer stubs errors


#
CTRL_SAVE_PATH: str = "saved_control.json"

#
RENDER_WIDTH: int = 1440
RENDER_HEIGHT: int = 1024

#
GENERATE_CORRIDOR_PARAM: dict[str, Any] = {
    "corridor_length": ValType(100.0),
    "corridor_width": ValType(3.0),
    "obstacles_mode": "sinusoidal",
    "obstacles_mode_param": {
        "obstacle_sep": ValType( 4.0 ),
        "obstacle_size_x": ValType( 0.4 ),
        "obstacle_size_y": ValType( 0.4 ),
        "obstacle_size_z": ValType( 0.2 ),
    }
}


#
def create_geom(name: str, geom_type: str, pos: Vec3, size: Vec3) -> ET.Element:

    #
    new_geom = ET.Element('geom')
    #
    new_geom.set('name', name)
    new_geom.set('type', geom_type)
    new_geom.set('size', f'{size.x} {size.y} {size.z}')
    new_geom.set('pos', f'{pos.x} {pos.y} {pos.z}')

    #
    return new_geom


#
class Corridor:

    #
    def __init__(self) -> None:

        #
        self.xml_file_path: str = "corridor_3x100.xml"  # "corridor_custom.xml"

    #
    def extract_corridor_from_xml(self) -> dict[str, Any]:
        """
        Extract robot components from existing XML file.
        This teaches students how to parse and reuse XML components.
        """

        #
        print(f"Extracting corridor components from {self.xml_file_path}...")

        #
        ### Parse the existing XML file. ###
        #
        tree: ET.ElementTree[ET.Element] = ET.parse(self.xml_file_path)
        root: ET.Element = tree.getroot()

        #
        ### Extract useful components. ###
        #
        components: dict[str, Optional[Any]] = {
            'compiler': None,
            'option': None,
            'default': None,
            'asset': None,
            'body': None
        }

        #
        comps_body: list[ET.Element] = []

        #
        ## Add global scene ground. ##
        #
        floor_geom = ET.Element('geom')
        #
        floor_geom.set('name', 'global_floor')
        floor_geom.set('type', 'plane')
        floor_geom.set('size', '200 200 0.1')
        floor_geom.set('pos', '0 0 0.12')
        #
        comps_body.append(floor_geom)

        #
        ### Find and extract each component. ###
        #
        for child in root:
            #
            if child.tag == 'compiler':
                #
                components['compiler'] = child
            #
            elif child.tag == 'option':
                #
                components['option'] = child
            #
            elif child.tag == 'default':
                #
                components['default'] = child
            #
            elif child.tag == 'asset':
                #
                components['asset'] = child
            #
            elif child.tag == 'worldbody':
                #
                for body in child:
                    #
                    comps_body.append(body)
                #
                components["body"] = comps_body
            #
            elif child.tag == 'actuator':
                #
                components['actuators'] = child

        #
        return components

    #
    def generate_corridor(
        self,
        corridor_length: ValType = ValType(100.0),              # Value, or random in [min_value, max_value]
        corridor_width: ValType = ValType(3.0),                 # Value, or random in [min_value, max_value]
        obstacles_mode: str = "none",                           # 'none', 'sinusoidal`, `double_sinusoidal`, `random`
        obstacles_mode_param: Optional[dict[str, Any]] = None,  # specific parameters for `obstacles_mode`, ex: `obstacle_separation` for `sinusoidal` mode
    ) -> dict[str, Any]:

        #
        ### Generate useful components. ###
        #
        components: dict[str, Optional[Any]] = {
            'compiler': None,
            'option': None,
            'default': None,
            'asset': None,
            'body': []
        }

        #
        components_body: list[ET.Element] = []


        #
        ### Generate a corridor based on parameters: ###
        #

        #
        obstacles_mode_param_: dict[str, Any] = {} if obstacles_mode_param is None else obstacles_mode_param

        #
        corridor_length_: float = corridor_length.get_value()
        corridor_width_: float  = corridor_width.get_value()

        #
        wall_width: float = 0.3
        wall_height: float = 10.0

        #
        ## Add global scene ground. ##
        #
        components_body.append(
            create_geom(
                name="global_floor",
                geom_type="plane",
                size=Vec3(x = corridor_length_ * 10.0, y = corridor_length_ * 10.0, z = 1.0),
                pos=Vec3(x = 0, y = 0, z = 0)
            )
        )

        #
        ## Add the two walls. ##
        #
        components_body.append(
            create_geom(
                name="wall_1",
                geom_type="box",
                size=Vec3(x = corridor_length_, y = wall_width, z = wall_height),
                pos=Vec3(x = corridor_length_, y = -corridor_width_, z = wall_height)
            )
        )
        components_body.append(
            create_geom(
                name="wall_2",
                geom_type="box",
                size=Vec3(x = corridor_length_, y = wall_width, z = wall_height),
                pos=Vec3(x = corridor_length_, y = +corridor_width_, z = wall_height)
            )
        )

        #
        ## Add the obstacles ##
        #

        #
        ## Sinusoidal mode. ##
        #
        if obstacles_mode == "sinusoidal":

            #
            obs_size_x: ValType = cast(ValType, obstacles_mode_param_.get("obstacle_size_x", 0.2))  # type: ignore
            obs_size_y: ValType = cast(ValType, obstacles_mode_param_.get("obstacle_size_y", 0.2))  # type: ignore
            obs_size_z: ValType = cast(ValType, obstacles_mode_param_.get("obstacle_size_z", 0.2))  # type: ignore
            obs_sep: ValType = cast(ValType, obstacles_mode_param_.get("obstacle_sep", 0.2))  # type: ignore

            #
            obs_length: float = obs_size_x.get_max_value() + obs_sep.get_max_value()

            #
            nb_obstacles: int = max( floor(corridor_length_ / obs_length) + 1, 1)

            #
            current_obs_x: float = 2 + obs_sep.get_value()

            #
            obs_idx: int
            #
            for obs_idx in range(nb_obstacles):

                #
                current_obs_size_x: float = obs_size_x.get_value()
                current_obs_size_y: float = obs_size_y.get_value()
                current_obs_size_z: float = obs_size_z.get_value()

                #
                current_obs_y: float = sin( float(obs_idx) * 0.16 * pi ) * ( corridor_width_ - 2 * current_obs_size_y )
                current_obs_z: float = 0

                #
                components_body.append(
                    create_geom(
                        name=f"obstacle_{obs_idx}",
                        geom_type="box",
                        size=Vec3(x = current_obs_size_x, y = current_obs_size_y, z = current_obs_size_z),
                        pos=Vec3(x = current_obs_x, y = current_obs_y, z = current_obs_z)
                    )
                )

                #
                current_obs_x += current_obs_size_x + obs_sep.get_max_value()


        #
        ## Double sinusoidal mode. ##
        #
        elif obstacles_mode == "double_sinusoidal":

            #
            pass


        #
        ## Random mode. ##
        #
        elif obstacles_mode == "random":

            #
            pass


        #
        components['body'] = components_body

        #
        return components


#
class Robot:

    #
    def __init__(
        self,
        sensor_num_rays: int = 8,
        sensor_ray_length: float = 10.0
    ) -> None:

        #
        self.xml_file_path: str = "four_wheels_robot.xml"

        #
        ### Sensor configuration. ###
        #
        self.sensor_num_rays: int = sensor_num_rays
        self.sensor_ray_length: float = sensor_ray_length

        #
        ### Parse the existing XML file. ###
        #
        self.tree: ET.ElementTree[ET.Element] = ET.parse(self.xml_file_path)
        self.root: ET.Element = self.tree.getroot()

    #
    def create_sensor_rays(self, robot_body: ET.Element) -> list[ET.Element]:
        """
        Create non-colliding sensor boxes around the robot for vision detection.
        Returns a list of sensor site elements.
        """

        #
        return []

    #
    def extract_robot_from_xml(self) -> dict[str, Any]:
        """
        Extract robot components from existing XML file.
        This teaches students how to parse and reuse XML components.
        """

        #
        print(f"Extracting robot components from {self.xml_file_path}...")

        #
        ### Parse the existing XML file. ###
        #
        tree: ET.ElementTree[ET.Element] = ET.parse(self.xml_file_path)
        root: ET.Element = tree.getroot()

        #
        ### Extract useful components. ###
        #
        components: dict[str, Optional[Any]] = {
            'compiler': None,
            'option': None,
            'default': None,
            'asset': None,
            'robot_body': None,
            'actuators': None,
            'sensor_boxes': []
        }

        #
        ### Find and extract each component. ###
        #
        for child in root:
            #
            if child.tag == 'compiler':
                #
                components['compiler'] = child
            #
            elif child.tag == 'option':
                #
                components['option'] = child
            #
            elif child.tag == 'default':
                #
                components['default'] = child
            #
            elif child.tag == 'asset':
                #
                components['asset'] = child
            #
            elif child.tag == 'worldbody':

                #
                ### Extract the robot body from worldbody. ###
                #
                for body in child:

                    #
                    if body.get('name') == 'robot':

                        #
                        components['robot_body'] = body

            #
            elif child.tag == 'actuator':
                #
                components['actuators'] = child

        #
        ### Add sensor rays to robot body. ###
        #
        if components['robot_body'] is not None:
            #
            sensor_rays: list[ET.Element] = self.create_sensor_rays(components['robot_body'])
            #
            # TODO: add sensor to scene

        #
        return components

    #
    def create_enhanced_materials(self) -> dict[str, ET.Element]:
        """
        Create enhanced materials with textures and visual appeal.
        Returns a list of material elements.
        """

        #
        ### Enhanced robot materials. ###
        #
        robot_materials: list[dict[str, str]] = [
            {
                'name': 'mat_chassis_violet',
                'texture': 'tex_chassis',
                'rgba': '0.39 0.13 0.63 1',  # Nice violet color
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

        #
        ### Initialize Materials Containers. ###
        #
        materials: dict[str, ET.Element] = {}

        #
        ### Create material elements. ###
        #
        for mat_config in robot_materials:
            #
            material: ET.Element = ET.Element('material')
            #
            mat_name: str = mat_config['name']
            #
            material.set('name', mat_name)
            material.set('rgba', mat_config['rgba'])
            #
            if 'texture' in mat_config:
                #
                material.set('texture', mat_config['texture'])
            #
            if 'shininess' in mat_config:
                #
                material.set('shininess', mat_config['shininess'])
            #
            if 'specular' in mat_config:
                #
                material.set('specular', mat_config['specular'])
            #
            if 'texrepeat' in mat_config:
                #
                material.set('texrepeat', mat_config['texrepeat'])

            #
            materials[mat_name] = material

        #
        return materials

    #
    def create_textures(self) -> dict[str, ET.Element]:
        """
        Create texture elements for enhanced visuals.
        Returns a list of texture elements.
        """

        #
        textures: dict[str, ET.Element] = {}

        #
        ### Wheel radius texture to show rotation. ###
        #
        wheel_texture = ET.Element('texture')
        #
        wheel_texture.set('name', 'tex_wheel_radius')
        wheel_texture.set('type', '2d')
        wheel_texture.set('builtin', 'checker')
        wheel_texture.set('rgb1', '0.8 0.2 0.2')  # Red spoke
        wheel_texture.set('rgb2', '0.1 0.1 0.1')  # Black tire
        wheel_texture.set('width', '32')
        wheel_texture.set('height', '32')
        wheel_texture.set('mark', 'random')
        wheel_texture.set('markrgb', '0.8 0.8 0.2')  # Yellow marks
        #
        textures["tex_wheel_radius"] = wheel_texture


        wheel_texture = ET.Element('texture')
        #
        wheel_texture.set('name', 'tex_chassis')
        wheel_texture.set('type', '2d')
        wheel_texture.set('builtin', 'gradient')
        wheel_texture.set('rgb1', '0.39 0.13 0.63')  # Red spoke
        wheel_texture.set('rgb2', '0 0 0')  # Black tire
        wheel_texture.set('width', '32')
        wheel_texture.set('height', '32')
        # wheel_texture.set('mark', 'random')
        # wheel_texture.set('markrgb', '1 1 1')  # Yellow marks
        #
        textures["tex_chassis"] = wheel_texture

        #
        return textures

    #
    def enhance_robot_visuals(self, robot_body: Optional[ET.Element]) -> None:
        """
        Enhance robot body with better materials and visual features.
        Also adjusts wheel positions to be outside the robot body.
        Modifies the robot body XML element in place.
        """

        #
        if robot_body is None:
            #
            return

        #
        print("Enhancing robot visuals...")

        #
        ### Find and update chassis material. ###
        #
        for geom in robot_body.iter('geom'):
            #
            if 'chassis' in geom.get('name', ''):
                #
                geom.set('material', 'mat_chassis_violet')
                #
                print("  Updated chassis color")

        #
        ### Find and update wheel materials (positions now defined in XML). ###
        #
        wheel_count: int = 0

        #
        for body in robot_body.iter('body'):
            #
            body_name: str = body.get('name', '')
            #
            if body_name.startswith('wheel_'):
                #
                ### Update wheel material. ###
                #
                for geom in body.iter('geom'):
                    #
                    if geom.get('name', '').startswith('geom_'):
                        #
                        geom.set('material', 'mat_wheel_black')
                        #
                        wheel_count += 1

                #
                ### Log the wheel position (already set in XML). ###
                #
                current_pos = body.get('pos', '0 0 0')
                #
                print(f"  {body_name} at position: {current_pos}")

        #
        print(f"  Updated {wheel_count} wheels with textured black material")
        print("  Wheel positions defined in XML (no runtime modification)")


#
class RootWorldScene:

    #
    def __init__(
        self,
        #
        ### Robot sensor parameters. ###
        #
        sensor_boxes_enabled: bool = True,
        sensor_box_size: float = 0.5,
        sensor_layers: int = 2,
        boxes_per_layer: int = 8
    ) -> None:

        #
        self.corridor: Corridor = Corridor()
        #
        self.robot: Robot = Robot()

        #
        self.mujoco_model: mujoco.MjModel
        self.mujoco_data: mujoco.MjData


    #
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

        Args:
            robot_components: Extracted robot parts from XML
            corridor_components: Extracted corridor parts from XML
            floor_type: Type of floor to create ("standard", "ice", "sand")
            robot_height: Starting height of robot above floor (meters)
        """

        #
        print("Building combined model...")
        print(f"Robot starting height: {robot_height}m above floor")

        #
        ### Create root mujoco element. ###
        #
        root: ET.Element = ET.Element('mujoco')
        #
        root.set('model', 'robot_with_programmatic_floor')

        #
        g: ET.Element = ET.Element('global')
        g.set("offwidth", str(RENDER_WIDTH))
        g.set("offheight", str(RENDER_HEIGHT))
        #
        visual: ET.Element = ET.Element("visual")
        #
        visual.append(g)
        #
        root.append(visual)

        #
        ### Add compiler settings from robot XML. ###
        #
        if robot_components['compiler'] is not None:
            #
            root.append(robot_components['compiler'])

        #
        ### CREATE ENVIRONMENT-CONTROLLED PHYSICS SETTINGS. ###
        #
        option: ET.Element = ET.Element('option')
        #
        option.set('timestep', '0.001')
        option.set("gravity", "0 0 -0.20")
        option.set('solver', 'Newton')
        option.set('iterations', '500')
        #
        root.append(option)
        #
        print(f"  Environment physics: gravity enabled ({option.get("gravity")}), timestep={option.get("timestep")}s")

        #
        ### Add size settings. ###
        #
        size: ET.Element = ET.Element('size')
        #
        size.set('njmax', '1000')
        size.set('nconmax', '500')
        #
        root.append(size)

        #
        if robot_components['default'] is not None:
            #
            root.append(robot_components['default'])

        #
        ### Create asset section with textures and enhanced materials. ###
        #
        asset: ET.Element = ET.Element('asset')

        #
        ### Add textures first. ###
        #
        _texture_id: str
        texture: ET.Element
        #
        for _texture_id, texture in self.robot.create_textures().items():
            #
            asset.append(texture)

        #
        ### Add enhanced materials. ###
        #
        _material_id: str
        material: ET.Element
        #
        enhanced_materials: dict[str, ET.Element] = self.robot.create_enhanced_materials()
        #
        materials_used: set[str] = set()
        #
        for _material_id, material in enhanced_materials.items():
            #
            asset.append(material)
            #
            materials_used.add(_material_id)

        #
        ### Also keep any original materials from robot XML if they don't exist. ###
        #
        if robot_components['asset'] is not None:
            #
            for original_material in robot_components['asset']:
                #
                ### Only add if it's not already in our enhanced materials. ###
                #
                material_name = original_material.get('name', '')
                #
                if material_name not in enhanced_materials and material_name not in materials_used:
                    #
                    asset.append(original_material)
                    #
                    materials_used.add(material_name)

        #
        ### Also keep any original materials from robot XML if they don't exist. ###
        #
        if corridor_components['asset'] is not None:
            #
            for original_material in corridor_components['asset']:
                #
                ### Only add if it's not already in our enhanced materials. ###
                #
                material_name = original_material.get('name', '')
                #
                if material_name not in enhanced_materials and material_name not in materials_used:
                    #
                    asset.append(original_material)
                    #
                    materials_used.add(material_name)

        #
        root.append(asset)

        #
        ### Create worldbody with corridor and robot. ###
        #
        worldbody: ET.Element = ET.Element('worldbody')

        #
        ### Add all the corridor worldbody to this worldbody. ###
        #
        if corridor_components['body'] is not None:
            #
            body: ET.Element
            #
            for body in corridor_components['body']:
                #
                worldbody.append(body)

        #
        ### Add robot body with enhanced visuals and adjusted height. ###
        #
        if robot_components['robot_body'] is not None:

            #
            ### Enhance the robot's visual appearance. ###
            #
            self.robot.enhance_robot_visuals(robot_components['robot_body'])

            #
            ### Adjust robot starting height (floor is at -0.1, so robot center should be at robot_height - 0.1). ###
            #
            robot_z_position: float = robot_height - 0.1 + 10  # Floor offset
            current_pos: str = robot_components['robot_body'].get('pos', '0 0 0.2')
            #
            pos_parts: list[str] = current_pos.split()
            #
            if len(pos_parts) == 3:
                #
                new_pos: str = f"{pos_parts[0]} {pos_parts[1]} {robot_z_position}"
                #
                robot_components['robot_body'].set('pos', new_pos)
                #
                print(f"  Robot positioned at: {new_pos} (will fall {robot_height}m to floor)")

            #
            robot_components['robot_body'].set('pos', "0 0 5")
            robot_components['robot_body'].set('euler', "0 0 0")

            #
            worldbody.append(robot_components['robot_body'])

        #
        ### Add worldbody to root. ###
        #
        root.append( worldbody )

        #
        ### Add actuators. ###
        #
        if robot_components['actuators'] is not None:
            #
            root.append(robot_components['actuators'])

        #
        ### Convert to XML string. ###
        #
        xml_string = ET.tostring(root, encoding='unicode')

        #
        ### Create and return MuJoCo model. ###
        #
        return mujoco.MjModel.from_xml_string(xml_string)


    #
    def construct_scene(
        self,
        floor_type: str = "standard",
        robot_height: float = 1.0
    ) -> None:

        #
        ### Build combined model with enhanced visuals and physics. ###
        #
        self.mujoco_model = self.build_combined_model(
            robot_components=self.robot.extract_robot_from_xml(),
            corridor_components=self.corridor.generate_corridor(**GENERATE_CORRIDOR_PARAM),
            floor_type=floor_type,
            robot_height=robot_height
        )
        #
        self.mujoco_data = mujoco.MjData(self.mujoco_model)


#
class Sensors:

    #
    def __init__(
        self,
        # TODO: parameters
    ) -> None:

        # TODO
        pass

        self.nb_sensors: int = 8  # TODO: link that directly with the Robot object

    # TODO: utils method if needed

    #
    def get_sensor_data(self) -> NDArray[np.float32]:

        """
        Function that very efficiently reads all the raycasting sensor data,
        returns an array of the number of sensors with:
            -1: Nothing hits the sensor
            0.0 <= v <= ray_length: distance of the nearest object that intersect the ray

        Returns:
            NDArray[np.float32]: array of sensor data
        """

        # TODO
        pass

        return np.full((self.nb_sensors,), fill_value=-1, dtype=np.float32)


#
class TrackRobot:

    #
    def __init__(
        self,
        mujoco_data_scene: mujoco.MjData,
    ) -> None:

        #
        self.mjdata: mujoco.MjData = mujoco_data_scene

        #
        self.robot_posx_track: list[float] = []
        self.robot_posy_track: list[float] = []
        self.robot_posz_track: list[float] = []

        #
        self.robot_velx_track: list[float] = []
        self.robot_vely_track: list[float] = []
        self.robot_velz_track: list[float] = []

    #
    def track(self) -> None:

        #
        self.robot_posx_track.append( self.mjdata.qpos[0] )
        self.robot_posy_track.append( self.mjdata.qpos[1] )
        self.robot_posz_track.append( self.mjdata.qpos[2] )

        #
        self.robot_velx_track.append( self.mjdata.qvel[0] )
        self.robot_vely_track.append( self.mjdata.qvel[1] )
        self.robot_velz_track.append( self.mjdata.qvel[2] )

    #
    def plot_tracking(self) -> None:

        #
        time_range: list[int] = list(range(len(self.robot_posx_track)))

        #
        plt.plot(time_range, self.robot_posx_track, label="pos x")  # type: ignore
        plt.plot(time_range, self.robot_posy_track, label="pos y")  # type: ignore
        plt.plot(time_range, self.robot_posz_track, label="pos z")  # type: ignore
        #
        plt.plot(time_range, self.robot_velx_track, label="vel x")  # type: ignore
        plt.plot(time_range, self.robot_vely_track, label="vel y")  # type: ignore
        plt.plot(time_range, self.robot_velz_track, label="vel z")  # type: ignore
        #
        plt.yscale("symlog")  # type: ignore
        plt.legend()  # type: ignore
        plt.show()  # type: ignore


#
class Physics:

    #
    def __init__(
        self,
        mujoco_model_scene: mujoco.MjModel,
        mujoco_data_scene: mujoco.MjData,
        air_drag_coefficient: float = 0.5,
        robot_deceleration_force: float = 0.01
    ) -> None:

        #
        self.model_scene: mujoco.MjModel = mujoco_model_scene
        self.data_scene: mujoco.MjData = mujoco_data_scene

        #
        self.air_drag_coefficient: float = air_drag_coefficient

        #
        self.robot_deceleration_force: float = robot_deceleration_force
        self.robot_deceleration_factor: float = 1.0 - self.robot_deceleration_force

        #
        ### Initialize simulation with zero velocities for stability. ###
        #
        self.data_scene.qvel[:] = 0  # Zero all velocities
        self.data_scene.qacc[:] = 0  # Zero all accelerations

        #
        ### Initialize wheels speed. ###
        #
        self.robot_wheels_speed: NDArray[np.float64] = np.zeros((4,), dtype=np.float64)

    #
    def apply_air_resistance(self) -> None:
        """
        Applies air drag to all bodies based on their velocity.
        F_drag = -k * v^2 * sign(v)
        """

        #
        for i in range(self.model_scene.nbody):

            #
            ### Skip world body. ###
            #
            if i == 0:
                #
                continue

            #
            ### Get linear velocity of body i. ###
            #
            ### translational velocity. ###
            #
            vel = self.data_scene.cvel[i][:3]
            #
            speed = np.linalg.norm(vel)
            #
            if speed < 1e-6:
                #
                continue

            #
            ### Simple quadratic drag model: F = -k * v * |v|. ###
            #
            F_drag = -self.air_drag_coefficient * speed * vel

            #
            ### Allocate full-sized qfrc_target vector (size nv). ###
            #
            qfrc_target = np.zeros(self.model_scene.nv)

            #
            ### Apply the force at the center of mass. ###
            #
            mujoco.mj_applyFT(
                m=self.model_scene,
                d=self.data_scene,
                force=F_drag,
                torque=np.zeros(3),
                point=self.data_scene.xpos[i],
                body=i,
                qfrc_target=qfrc_target
            )

    #
    def apply_additionnal_physics(self) -> None:

        #
        self.apply_air_resistance()

        #
        self.apply_robot_deceleration()

        #
        self.set_robot_wheel_speeds()

    #
    def apply_robot_ctrl_movement(
        self,
        acceleration_factor: float = 0.0,
        rotation_factor: float = 0.0,
        acceleration_force: float = 0.15,
        rotation_force: float = 0.05,
        decceleration_factor: float = 1.0,
        max_front_wheel_speeds: float = 200.0,
        max_back_wheel_speeds: float = 100,
    ) -> None:

        #
        if acceleration_factor > 0:
            #
            self.robot_wheels_speed[0] += acceleration_factor * acceleration_force
            self.robot_wheels_speed[1] += acceleration_factor * acceleration_force
            self.robot_wheels_speed[2] += acceleration_factor * acceleration_force * 0.2
            self.robot_wheels_speed[3] += acceleration_factor * acceleration_force * 0.2
        #
        else:
            #
            self.robot_wheels_speed[0] += acceleration_factor * acceleration_force * 0.2
            self.robot_wheels_speed[1] += acceleration_factor * acceleration_force * 0.2
            self.robot_wheels_speed[2] += acceleration_factor * acceleration_force
            self.robot_wheels_speed[3] += acceleration_factor * acceleration_force

        #
        self.robot_wheels_speed[0] -= rotation_factor * rotation_force
        self.robot_wheels_speed[1] += rotation_factor * rotation_force
        self.robot_wheels_speed[2] -= rotation_factor * rotation_force
        self.robot_wheels_speed[3] += rotation_factor * rotation_force

        #
        if decceleration_factor < 1.0:
            #
            self.robot_wheels_speed[:] *= decceleration_factor

        #
        ### Clamp values. ###
        #
        self.robot_wheels_speed[0:2] = np.clip(self.robot_wheels_speed[0:2], -max_front_wheel_speeds, max_front_wheel_speeds)
        self.robot_wheels_speed[2:5] = np.clip(self.robot_wheels_speed[2:5], -max_back_wheel_speeds, max_back_wheel_speeds)

    #
    def apply_robot_deceleration(self) -> None:

        #
        self.robot_wheels_speed[:] *= self.robot_deceleration_factor

    #
    def set_robot_wheel_speeds(self) -> None:

        #
        ### This will set the actuator values. ###
        #
        self.data_scene.ctrl = self.robot_wheels_speed


#
class Camera:

    #
    def __init__(self) -> None:

        #
        ### Modes: "free", "follow_robot", "top_down" ###
        #
        self.current_mode: str = "free"

        #
        ### Store robot ID to avoid looking it up every frame. ###
        #
        self.robot_id: int = -1

    #
    def set_mode(self, mode: str) -> None:
        #
        if mode in ["free", "follow_robot", "top_down"]:
            #
            self.current_mode = mode

    #
    def update_viewer_camera(
        self,
        cam: Any,  # viewer_instance.cam
        model: mujoco.MjModel,
        data: mujoco.MjData
    ) -> None:

        #
        ### Get robot ID once. ###
        #
        if self.robot_id == -1:
            #
            self.robot_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "robot"
            )
            #
            if self.robot_id == -1:
                #
                print("Warning: Could not find robot body named 'robot'")
                #
                return

        #
        ### Handle camera logic based on mode. ###
        #
        if self.current_mode == "free":
            #
            ### Set camera to free mode and do nothing else. ###
            #
            ### This lets the user control it manually with the mouse. ###
            #
            if cam.type != mujoco.mjtCamera.mjCAMERA_FREE:
                #
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        #
        elif self.current_mode == "follow_robot":
            #
            ### Use MuJoCo's built-in tracking camera. ###
            #
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.robot_id
            #
            ### You can set preferred distance and elevation. ###
            #
            cam.distance = 5.0
            cam.elevation = -16
            cam.azimuth = 1.01171875
            #

        #
        elif self.current_mode == "top_down":
            #
            ### Use tracking camera, but point straight down. ###
            #
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.robot_id
            #
            cam.distance = 10.0      # Zoom out
            cam.azimuth = 90.0       # Face "north"
            cam.elevation = -89.9    # Look almost straight down

    #
    def get_camera_data(self, cam: Any) -> tuple[ NDArray[np.float64], float, float, float, int, Any ]:

        #
        return (cam.lookat, cam.distance, cam.azimuth, cam.elevation, cam.trackbodyid, cam.type)


#
class Controls:

    #
    def __init__(
        self,
        physics: Physics,
        camera: Camera,
        render_mode: bool = False
    ) -> None:

        #
        self.physics: Physics = physics
        self.camera: Camera = camera

        #
        self.render_mode: bool = render_mode

        #
        self.quit_requested: bool = False
        self.display_camera_info: bool = False

        #
        self.key_pressed: set[int] = set()

        #
        self.controls_history: dict[ str, list[int] ] = {}

        #
        self.current_frame: int = 0

        #
        self.easy_control: set[int] = {32, 262, 263, 264, 265}

    #
    def new_frame(self, cam: Any) -> None:

        #
        self.current_frame += 1

    #
    def apply_controls_each_frame_render_mode(self) -> None:

        #
        if str(self.current_frame) in self.controls_history:
            #
            for k in self.controls_history[str(self.current_frame)]:
                #
                self.key_callback(keycode=k, render_mode=True)


    #
    def apply_controls_each_frame(self) -> None:

        #
        if self.render_mode:
            #
            self.apply_controls_each_frame_render_mode()

        #
        ### Robot Movements. ###
        #
        ### Forward. ###
        #
        if 265 in self.key_pressed:
            #
            self.physics.apply_robot_ctrl_movement(acceleration_factor=1.0)
        #
        ### Turn Left. ###
        #
        elif 263 in self.key_pressed:
            #
            self.physics.apply_robot_ctrl_movement(rotation_factor=1.0)
        #
        ### Turn Right. ###
        #
        elif 262 in self.key_pressed:
            #
            self.physics.apply_robot_ctrl_movement(rotation_factor=-1.0)
        #
        ### Backward. ###
        #
        elif 264 in self.key_pressed:
            #
            self.physics.apply_robot_ctrl_movement(acceleration_factor=-1.0)
        #
        ### Space Key, robot stops. ###
        #
        elif 32 in self.key_pressed:
            #
            self.physics.apply_robot_ctrl_movement(decceleration_factor=0.2)

    #
    def key_callback(self, keycode: int, render_mode: bool = False) -> None:

        #
        if not render_mode:
            #
            if not str(self.current_frame) in self.controls_history:
                #
                self.controls_history[str(self.current_frame)] = [keycode]
            #
            else:
                #
                self.controls_history[str(self.current_frame)].append(keycode)

            #
            ### Display Camera Informations. ###
            #
            if keycode == ord('c') or keycode == ord('C'):
                #
                self.display_camera_info = True

            #
            ### Save the control history. ###
            #
            elif keycode == ord('s') or keycode == ord('S'):
                #
                with open(CTRL_SAVE_PATH, "w", encoding="utf-8") as f:
                    #
                    json.dump(self.controls_history, f)
                #
                print(f"Saved control history at path : `{CTRL_SAVE_PATH}` !")

            #
            ### Quit, with 'Q' or 'Esc' Keys. ###
            #
            elif keycode == ord('q') or keycode == ord('Q'):
                #
                self.quit_requested = True
            #
            elif keycode == 256:
                #
                self.quit_requested = True

        #
        if self.render_mode and not render_mode:
            #
            return

        #
        ### Camera Mode Switching. ###
        #
        if keycode == ord('1'):
            #
            self.camera.set_mode("free")
        #
        elif keycode == ord('2'):
            #
            self.camera.set_mode("follow_robot")
        #
        elif keycode == ord('3'):
            #
            self.camera.set_mode("top_down")

        #
        else:
            #
            if keycode in self.key_pressed:
                #
                self.key_pressed.remove(keycode)
            #
            else:
                #
                self.key_pressed.add(keycode)

            #
            if keycode in self.easy_control:
                #
                for kk in list(self.key_pressed):
                    #
                    if kk != keycode and kk in self.easy_control:
                        #
                        self.key_pressed.remove(kk)


#
class State:

    #
    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        physics: Physics,
        camera: Camera,
        controls: Controls,
        viewer_instance: Any,
        robot_track: TrackRobot,
        robot: Robot,
        sensors: Sensors
    ) -> None:

        #
        self.mj_model: mujoco.MjModel = mj_model
        self.mj_data: mujoco.MjData = mj_data
        self.physics: Physics = physics
        self.camera: Camera = camera
        self.controls: Controls = controls
        self.viewer_instance: Any = viewer_instance
        #
        self.robot_track: TrackRobot = robot_track
        self.robot: Robot = robot
        self.sensors: Sensors = sensors
        #
        self.quit: bool = False

        #
        self.init_state()

    #
    def init_state(self) -> None:

        #
        ### Configure camera parameters with your good settings. ###
        #

        #
        ### Camera parameters. ###
        #
        self.viewer_instance.cam.type = mujoco.mjtCamera.mjCAMERA_FREE   # Free camera
        self.viewer_instance.cam.fixedcamid = -1                         # Not using fixed camera

        #
        ## Set good camera position and orientation parameters. ###
        #
        self.viewer_instance.cam.azimuth = 1.01171875
        self.viewer_instance.cam.elevation = -16.6640625
        self.viewer_instance.cam.lookat = np.array(
            [ 1.55633679e-04, -4.88295545e-02,  1.05485916e+00]
        )


#
class Main:

    #
    @staticmethod
    def state_step(state: State) -> State:

        #
        sensor_data: NDArray[np.float32] = state.sensors.get_sensor_data()

        #
        ### Print debug info every ~0.5 seconds. ###
        #
        if state.mj_data.time % 0.5 < state.mj_model.opt.timestep:

            #
            print(f"Sensor data:\n{sensor_data}")

        #
        state.robot_track.track()

        #
        state.controls.new_frame(state.viewer_instance.cam)

        #
        state.controls.apply_controls_each_frame()

        #
        ### Print camera info when requested. ###
        #
        if state.controls.display_camera_info:
            #
            print(state.viewer_instance.cam)
            #
            robot_id: int = mujoco.mj_name2id(
                state.mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot"
            )
            #
            pos: float | np.typing.NDArray[np.float32] = state.mj_data.xpos[robot_id]
            quat: float | np.typing.NDArray[np.float32] = state.mj_data.xquat[robot_id]
            #
            print(f"Robot position: {pos}")
            print(f"Robot orientation (quat): {quat}")
            #
            state.controls.display_camera_info = False

        #
        state.physics.apply_additionnal_physics()

        #
        mujoco.mj_step(
            m=state.mj_model,
            d=state.mj_data,
            nstep=1
        )

        #
        state.camera.update_viewer_camera(
            cam=state.viewer_instance.cam,
            model=state.mj_model,
            data=state.mj_data
        )

        #
        state.viewer_instance.sync()

        #
        return state

    #
    @staticmethod
    def main(render_mode: bool = False) -> None:

        #
        root_scene: RootWorldScene = RootWorldScene(
            sensor_boxes_enabled=True,
            sensor_box_size=0.3,  # 50cm boxes
            sensor_layers=2,       # 2 concentric circles
            boxes_per_layer=8      # 8 boxes per circle
        )

        #
        root_scene.construct_scene(
            floor_type="standard",
            robot_height=1.0
        )

        #
        sensors: Sensors = Sensors(
            # TODO: parameters
        )

        #
        physics: Physics = Physics(
            mujoco_model_scene=root_scene.mujoco_model,
            mujoco_data_scene=root_scene.mujoco_data
        )

        #
        camera: Camera = Camera()

        #
        controls: Controls = Controls(
            physics=physics,
            camera=camera,
            render_mode=render_mode
        )

        #
        robot_track: TrackRobot = TrackRobot(mujoco_data_scene=root_scene.mujoco_data)

        #
        if render_mode:
            #
            if not os.path.exists(CTRL_SAVE_PATH):
                #
                raise UserWarning(f"Error: there is no saved control files at path `{CTRL_SAVE_PATH}` !")

            #
            with open(CTRL_SAVE_PATH, "r", encoding="utf-8") as f:
                #
                controls.controls_history = json.load(f)

        #
        print("")
        print("Controls:")
        print("  'up arrow': Robot goes forward (key control)")
        print("  'down arrow': Robot goes backward (key control)")
        print("  'left arrow': Robot goes left (key control)")
        print("  'right arrow': Robot goes right (key control)")
        print("  '1': Free camera (mouse control)")
        print("  '2': Follow robot (3rd person)")
        print("  '3': Top-down camera")
        print("  'c' or 'C': Print camera parameters")
        print("  'q' or 'Q': Quit")
        print("  ESC: Quit")
        print("")

        #
        viewer_instance: Any

        #
        with viewer.launch_passive(
                root_scene.mujoco_model,
                root_scene.mujoco_data,
                key_callback=controls.key_callback
            ) as viewer_instance:

            #
            state: State = State(
                mj_model=root_scene.mujoco_model,
                mj_data=root_scene.mujoco_data,
                physics=physics,
                camera=camera,
                controls=controls,
                viewer_instance=viewer_instance,
                robot_track=robot_track,
                robot=root_scene.robot,
                sensors=sensors
            )

            #
            ### Mainloop. ###
            #
            while viewer_instance.is_running() and not state.controls.quit_requested:

                #
                state = Main.state_step(state)

        #
        robot_track.plot_tracking()


    #
    @staticmethod
    def main_video_render() -> None:

        #
        root_scene: RootWorldScene = RootWorldScene()
        #
        root_scene.construct_scene(
            floor_type="standard",
            robot_height=1.0
        )

        #
        physics: Physics = Physics(
            mujoco_model_scene=root_scene.mujoco_model,
            mujoco_data_scene=root_scene.mujoco_data
        )

        #
        camera: Camera = Camera()

        #
        controls: Controls = Controls(
            physics=physics,
            camera=camera,
            render_mode=True
        )

        #
        if not os.path.exists(CTRL_SAVE_PATH):
            #
            raise UserWarning(f"Error: there is no saved control files at path `{CTRL_SAVE_PATH}` !")

        #
        with open(CTRL_SAVE_PATH, "r", encoding="utf-8") as f:
            #
            controls.controls_history = json.load(f)

        #
        ### Create a camera. ###
        #
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)

        #
        ### Camera parameters. ###
        #
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE   # Free camera
        cam.fixedcamid = -1                         # Not using fixed camera

        #
        ## Set good camera position and orientation parameters. ###
        #
        cam.azimuth = 1.01171875
        cam.elevation = -16.6640625
        cam.lookat = np.array(
            [ 1.55633679e-04, -4.88295545e-02,  1.05485916e+00]
        )


        #
        ### Set up parameters. ###
        #
        framerate: int = 60  # (Hz)
        #
        skip_factor: int = framerate
        #
        n_frames: int = max([int(f_id) for f_id in controls.controls_history.keys()])

        #
        ### Simulate and display video. ###
        #
        mujoco.mj_resetData(root_scene.mujoco_model, root_scene.mujoco_data)

        #
        with media.VideoWriter('rendered_video.mp4', shape=(RENDER_HEIGHT, RENDER_WIDTH), fps=framerate) as writer:
            #
            with mujoco.Renderer(root_scene.mujoco_model, RENDER_HEIGHT, RENDER_WIDTH) as renderer:  # type: ignore

                #
                for frame_idx in tqdm(range(min(2000000, n_frames))):

                    #
                    controls.new_frame(cam)

                    #
                    controls.apply_controls_each_frame()

                    #
                    ### Quit if requested. ###
                    #
                    if controls.quit_requested:
                        #
                        break

                    #
                    physics.apply_additionnal_physics()

                    #
                    mujoco.mj_step(
                        m=root_scene.mujoco_model,
                        d=root_scene.mujoco_data,
                        nstep=1
                    )

                    #
                    camera.update_viewer_camera(
                        cam=cam,
                        model=root_scene.mujoco_model,
                        data=root_scene.mujoco_data
                    )

                    #
                    if frame_idx % skip_factor == 0:

                        #
                        renderer.update_scene(root_scene.mujoco_data, cam)  # type: ignore
                        #
                        pixels: NDArray[np.int8] = renderer.render()  # type: ignore
                        #
                        writer.add_image(pixels)  # type: ignore


#
if __name__ == "__main__":
    #
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    #
    parser.add_argument('--render_mode', action="store_true", default=False)
    parser.add_argument('--render_video', action="store_true", default=False)
    #
    args: argparse.Namespace = parser.parse_args()

    #
    if args.render_video:
        #
        Main.main_video_render()
    #
    else:
        #
        Main.main(render_mode=args.render_mode)

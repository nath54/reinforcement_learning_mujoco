import xml.etree.ElementTree as ET
from typing import Any, Optional, Dict, List, cast
from math import floor, sin, pi
import mujoco

from src.core.types import Vec3, Point2d, Rect2d, ValType, GlobalConfig
from src.simulation.robot import Robot

def create_geom(name: str, geom_type: str, pos: Vec3, size: Vec3) -> ET.Element:
    new_geom = ET.Element('geom')
    new_geom.set('name', name)
    new_geom.set('type', geom_type)
    new_geom.set('size', f'{size.x} {size.y} {size.z}')
    new_geom.set('pos', f'{pos.x} {pos.y} {pos.z}')
    return new_geom

class Corridor:
    def generate_corridor(
        self,
        corridor_length: float,
        corridor_width: float,
        obstacles_mode: str = "none",
        obstacles_mode_param: Optional[Dict[str, Any]] = None,
        corridor_shift_x: float = -3.0,
    ) -> Dict[str, Any]:

        components_body: List[ET.Element] = []
        environment_rects: List[Rect2d] = []
        obstacles_mode_param_ = obstacles_mode_param if obstacles_mode_param else {}

        # Ground
        components_body.append(create_geom("global_floor", "plane", Vec3(0,0,0), Vec3(corridor_length*10, corridor_length*10, 1.0)))

        # Walls logic (Simplified from original for brevity, but logically identical)
        wall_height = 3.0
        wall_width = 0.3

        # Wall 1 (Negative Y)
        w1_pos = Vec3(corridor_length, -corridor_width, wall_height)
        w1_size = Vec3(corridor_length - corridor_shift_x, wall_width, wall_height)
        components_body.append(create_geom("wall_1", "box", w1_pos, w1_size))
        environment_rects.append(Rect2d(Point2d(w1_pos.x - w1_size.x, w1_pos.y - w1_size.y), Point2d(w1_pos.x + w1_size.x, w1_pos.y + w1_size.y), wall_height * 2))

        # Wall 2 (Positive Y)
        w2_pos = Vec3(corridor_length, corridor_width, wall_height)
        components_body.append(create_geom("wall_2", "box", w2_pos, w1_size))
        environment_rects.append(Rect2d(Point2d(w2_pos.x - w1_size.x, w2_pos.y - w1_size.y), Point2d(w2_pos.x + w1_size.x, w2_pos.y + w1_size.y), wall_height * 2))

        # Obstacles
        if obstacles_mode == "sinusoidal":
            obs_sep = ValType(obstacles_mode_param_.get("obstacle_sep", 5.0))
            obs_sz_x = ValType(obstacles_mode_param_.get("obstacle_size_x", 0.4))
            obs_sz_y = ValType(obstacles_mode_param_.get("obstacle_size_y", 0.4))
            obs_sz_z = ValType(obstacles_mode_param_.get("obstacle_size_z", 0.5))

            current_x = 2.0 + obs_sep.get_value()
            obs_len_approx = obs_sz_x.get_max_value() + obs_sep.get_max_value()
            nb_obstacles = max(floor(corridor_length / obs_len_approx) + 1, 1)

            for i in range(nb_obstacles):
                sx, sy, sz = obs_sz_x.get_value(), obs_sz_y.get_value(), obs_sz_z.get_value()
                y_pos = sin(i * 0.16 * pi) * (corridor_width - 2 * sy)
                pos = Vec3(current_x, y_pos, 0)
                components_body.append(create_geom(f"obstacle_{i}", "box", pos, Vec3(sx, sy, sz)))
                environment_rects.append(Rect2d(Point2d(current_x - sx, y_pos - sy), Point2d(current_x + sx, y_pos + sy), sz * 2))
                current_x += sx + obs_sep.get_max_value()

        return {'body': components_body, 'environment_rects': environment_rects, 'asset': None}

class SceneBuilder:
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.corridor = Corridor()
        self.robot = Robot(config.robot)
        self.environment_rects: List[Rect2d] = []
        # Approximate bounds
        margin = 250.0
        self.env_bounds = Rect2d(
            Point2d(-margin, -config.simulation.corridor_width - margin),
            Point2d(2 * config.simulation.corridor_length + margin, config.simulation.corridor_width + margin)
        )
        self.mujoco_model: Optional[mujoco.MjModel] = None
        self.mujoco_data: Optional[mujoco.MjData] = None

    def build(self) -> None:
        # Generate components
        corridor_comps = self.corridor.generate_corridor(
            self.config.simulation.corridor_length,
            self.config.simulation.corridor_width,
            self.config.simulation.obstacles_mode,
            self.config.simulation.obstacles_mode_param
        )
        self.environment_rects = cast(List[Rect2d], corridor_comps.get('environment_rects', []))
        robot_comps = self.robot.extract_robot_from_xml()

        # Assemble XML
        root = ET.Element('mujoco', model='robot_scene')
        root.append(ET.Element('size', njmax='1000', nconmax='500'))

        # Physics Option
        root.append(ET.Element('option', timestep='0.01', gravity='0 0 -9.81', solver='Newton', iterations='500'))

        # Assets
        asset = ET.Element('asset')
        for texture in self.robot.create_textures().values(): asset.append(texture)
        for mat in self.robot.create_enhanced_materials().values(): asset.append(mat)
        if robot_comps['asset']:
            for a in robot_comps['asset']: asset.append(a)
        root.append(asset)

        # Worldbody
        worldbody = ET.Element('worldbody')
        if corridor_comps['body']:
            for b in corridor_comps['body']: worldbody.append(b)

        if robot_comps['robot_body']:
            self.robot.enhance_robot_visuals(robot_comps['robot_body'])
            robot_comps['robot_body'].set('pos', '0 0 0.5') # Reset pos
            worldbody.append(robot_comps['robot_body'])
        root.append(worldbody)

        if robot_comps['actuators']: root.append(robot_comps['actuators'])

        # Compile
        xml_str = ET.tostring(root, encoding='unicode')
        self.mujoco_model = mujoco.MjModel.from_xml_string(xml_str)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)
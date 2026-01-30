"""
Scene Generator Module

This module is responsible for procedurally generating the MuJoCo XML scene,
including the corridor, walls, and obstacles.
"""


from typing import Any, Optional, cast

import random
import mujoco
import xml.etree.ElementTree as ET

from math import floor, sin, pi

from src.core.types import Vec3, Point2d, Rect2d, ValType, GlobalConfig
from src.simulation.robot import Robot

from .utils_geom import create_geom
from .corridor import Corridor
from .flatworld import FlatWorld


# Scene Builder class to assemble the full model
class SceneBuilder:
    """
    Builds the complete MuJoCo scene from configuration.
    """

    # Initialize SceneBuilder
    def __init__(self, config: GlobalConfig) -> None:

        self.config = config
        self.corridor = Corridor()
        self.flat_world = FlatWorld()
        self.robot = Robot(config.robot)
        self.environment_rects: list[Rect2d] = []

        # Goal position (will be set during build)
        self.goal_position: Optional[Vec3] = None

        # Approximate bounds
        margin: float = 250.0
        if config.simulation.scene_type == "flat_world":
            # Symmetric arena bounds
            self.env_bounds: Rect2d = Rect2d(
                Point2d(-config.simulation.corridor_length - margin, -config.simulation.corridor_width - margin),
                Point2d(config.simulation.corridor_length + margin, config.simulation.corridor_width + margin)
            )
        else:
            # Corridor bounds (original)
            self.env_bounds: Rect2d = Rect2d(
                Point2d(-margin, -config.simulation.corridor_width - margin),
                Point2d(2 * config.simulation.corridor_length + margin, config.simulation.corridor_width + margin)
            )

        self.mujoco_model: Optional[mujoco.MjModel] = None
        self.mujoco_data: Optional[mujoco.MjData] = None

    #
    def _setup_goal_position(self) -> None:
        """
        Set up goal position based on config or randomize for flat_world.
        Called once during initial build.
        """

        if self.config.simulation.goal_position is not None:
            # Use configured goal position
            gp = self.config.simulation.goal_position
            self.goal_position = Vec3(gp[0], gp[1], gp[2])
        else:
            # Random goal position within arena (for flat_world)
            self._randomize_goal_position()

    #
    def _randomize_goal_position(self) -> None:
        """
        Randomize goal position within arena bounds.
        """

        arena_x: float = self.config.simulation.corridor_length
        arena_y: float = self.config.simulation.corridor_width
        goal_radius: float = self.config.simulation.goal_radius

        # Random position, avoiding spawn area (center) and edges
        margin: float = goal_radius + 2.0
        min_dist_from_spawn: float = 10.0

        while True:
            x: float = random.uniform(-arena_x + margin, arena_x - margin)
            y: float = random.uniform(-arena_y + margin, arena_y - margin)
            dist_from_spawn: float = (x**2 + y**2)**0.5

            if dist_from_spawn > min_dist_from_spawn:
                break

        self.goal_position = Vec3(x, y, 0.5)

    #
    def reset_goal_position(self) -> Vec3:
        """
        Called on episode reset to randomize goal for flat_world (if randomize_goal is True).
        Returns the new goal position.
        """

        if self.config.simulation.scene_type == "flat_world":
            if self.config.simulation.randomize_goal or self.config.simulation.goal_position is None:
                self._randomize_goal_position()

        return self.goal_position



    #
    def build_flatworld(self) -> dict[str, Any]:

        #
        self._setup_goal_position()
        #
        return self.flat_world.generate_flat_world(
            arena_size_x=self.config.simulation.corridor_length,
            arena_size_y=self.config.simulation.corridor_width,
            goal_position=self.goal_position,
            goal_radius=self.config.simulation.goal_radius,
            obstacles_mode=self.config.simulation.obstacles_mode,
            obstacles_mode_param=self.config.simulation.obstacles_mode_param,
            ground_friction=self.config.simulation.ground_friction
        )


    #
    def build_corridor(self) -> dict[str, Any]:

        #
        scene_comps: dict[str, Any]

        # Set goal beyond corridor end (obstacles stop at corridor_length)
        goal_radius: float = self.config.simulation.goal_radius
        goal_x: float = self.config.simulation.corridor_length + goal_radius + 5.0
        self.goal_position = Vec3(goal_x, 0.0, 0.5)
        scene_comps = self.corridor.generate_corridor(
            self.config.simulation.corridor_length,
            self.config.simulation.corridor_width,
            self.config.simulation.obstacles_mode,
            self.config.simulation.obstacles_mode_param,
            ground_friction=self.config.simulation.ground_friction
        )

        # Add goal marker sphere to corridor scene
        goal_radius: float = self.config.simulation.goal_radius
        goal_geom: ET.Element = ET.Element('geom')
        goal_geom.set('name', 'goal_marker')
        goal_geom.set('type', 'sphere')
        goal_geom.set('pos', f'{self.goal_position.x} {self.goal_position.y} {self.goal_position.z}')
        goal_geom.set('size', str(goal_radius))
        goal_geom.set('rgba', '1 0.2 0.2 0.4')  # Semi-transparent red
        goal_geom.set('contype', '0')  # No collision type
        goal_geom.set('conaffinity', '0')  # No collision affinity
        scene_comps['body'].append(goal_geom)

        #
        return scene_comps


    # Build the scene
    def build(self) -> None:
        """
        Build the complete scene based on scene_type
        """

        #
        scene_comps: dict[str, Any]
        #
        if self.config.simulation.scene_type == "flat_world":
            #
            scene_comps = self.build_flatworld()
        #
        elif self.config.simulation.scene_type == "corridor":
            #
            scene_comps = self.build_corridor()
        #
        else:
            # Default Scene
            scene_comps = self.build_corridor()


        self.environment_rects = cast(list[Rect2d], scene_comps.get('environment_rects', []))
        robot_comps: dict[str, Any] = self.robot.extract_robot_from_xml()

        # Assemble XML
        root: ET.Element = ET.Element('mujoco', model='robot_scene')

        # Visual settings for rendering
        visual: ET.Element = ET.Element('visual')
        global_elem: ET.Element = ET.Element('global')
        global_elem.set('offwidth', '1440')
        global_elem.set('offheight', '1024')
        visual.append(global_elem)
        root.append(visual)

        # Size limits
        root.append(ET.Element('size', njmax='1000', nconmax='500'))

        # Physics Option
        root.append(ET.Element('option',
            timestep=str(self.config.simulation.dt),
            gravity=self.config.simulation.gravity,
            solver=self.config.simulation.solver,
            iterations=str(self.config.simulation.iterations)
        ))

        # Assets
        asset: ET.Element = ET.Element('asset')
        #
        texture: ET.Element
        mat: ET.Element
        a: ET.Element
        #
        for texture in self.robot.create_textures().values():
            asset.append(texture)
        for mat in self.robot.create_enhanced_materials().values():
            asset.append(mat)
        if robot_comps['asset']:
            for a in robot_comps['asset']:
                asset.append(a)
        root.append(asset)

        # Worldbody
        worldbody: ET.Element = ET.Element('worldbody')
        #
        b: ET.Element
        #
        if scene_comps['body']:
            for b in scene_comps['body']:
                worldbody.append(b)

        if robot_comps['robot_body']:
            self.robot.enhance_robot_visuals(robot_comps['robot_body'])
            robot_comps['robot_body'].set('pos', '0 0 0.5')  # Reset pos
            worldbody.append(robot_comps['robot_body'])
        root.append(worldbody)

        if robot_comps['actuators']:
            root.append(robot_comps['actuators'])

        # Compile
        xml_str: str = ET.tostring(root, encoding='unicode')
        self.mujoco_model = mujoco.MjModel.from_xml_string(xml_str)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)

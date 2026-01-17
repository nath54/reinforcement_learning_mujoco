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


# Helper to create a MuJoCo geom element
def create_geom(
    name: str,
    geom_type: str,
    pos: Vec3,
    size: Vec3,
    extra_attribs: Optional[dict[str, str]] = None
) -> ET.Element:
    """
    Helper function to create a MuJoCo geom element.
    """

    new_geom: ET.Element = ET.Element('geom')
    new_geom.set('name', name)
    new_geom.set('type', geom_type)
    new_geom.set('size', f'{size.x} {size.y} {size.z}')
    new_geom.set('pos', f'{pos.x} {pos.y} {pos.z}')

    if extra_attribs:
        #
        k: str
        v: str
        #
        for k, v in extra_attribs.items():
            new_geom.set(k, v)

    return new_geom


# Corridor generator class
class Corridor:
    """
    Generates the corridor environment geometry.
    """

    # Generate corridor with walls and obstacles
    def generate_corridor(
        self,
        corridor_length: float,
        corridor_width: float,
        obstacles_mode: str = "none",
        obstacles_mode_param: Optional[dict[str, Any]] = None,
        corridor_shift_x: float = -3.0,
        ground_friction: str = "1 0.005 0.0001"
    ) -> dict[str, Any]:

        """
        Generate corridor with walls and obstacles.
        """

        # Initialize lists
        components_body: list[ET.Element] = []
        environment_rects: list[Rect2d] = []
        obstacles_mode_param_: dict[str, Any] = obstacles_mode_param if obstacles_mode_param else {}

        # Ground
        components_body.append(
            create_geom(
                "global_floor", "plane", Vec3(0, 0, 0),
                Vec3(corridor_length * 10, corridor_length * 10, 1.0),
                extra_attribs={"friction": ground_friction}
            )
        )

        # Walls dimensions
        wall_height: float = 3.0
        wall_width: float = 0.3

        # Wall 1 (Negative Y)
        w1_pos: Vec3 = Vec3(corridor_length, -corridor_width, wall_height)
        w1_size: Vec3 = Vec3(corridor_length - corridor_shift_x, wall_width, wall_height)
        components_body.append(create_geom("wall_1", "box", w1_pos, w1_size))
        environment_rects.append(
            Rect2d(
                Point2d(w1_pos.x - w1_size.x, w1_pos.y - w1_size.y),
                Point2d(w1_pos.x + w1_size.x, w1_pos.y + w1_size.y),
                wall_height * 2
            )
        )

        # Wall 2 (Positive Y)
        w2_pos: Vec3 = Vec3(corridor_length, corridor_width, wall_height)
        components_body.append(create_geom("wall_2", "box", w2_pos, w1_size))
        environment_rects.append(
            Rect2d(
                Point2d(w2_pos.x - w1_size.x, w2_pos.y - w1_size.y),
                Point2d(w2_pos.x + w1_size.x, w2_pos.y + w1_size.y),
                wall_height * 2
            )
        )

        # Wall 3 (Negative X side)
        w3_pos: Vec3 = Vec3(corridor_shift_x, 0, wall_height)
        w3_size: Vec3 = Vec3(wall_width, corridor_width, wall_height)
        components_body.append(create_geom("wall_3", "box", w3_pos, w3_size))
        environment_rects.append(
            Rect2d(
                Point2d(w3_pos.x - w3_size.x, w3_pos.y - w3_size.y),
                Point2d(w3_pos.x + w3_size.x, w3_pos.y + w3_size.y),
                wall_height * 2
            )
        )

        # Wall 4 (Positive X side)
        w4_pos: Vec3 = Vec3(2 * corridor_length - corridor_shift_x, 0, wall_height)
        components_body.append(create_geom("wall_4", "box", w4_pos, w3_size))
        environment_rects.append(
            Rect2d(
                Point2d(w4_pos.x - w3_size.x, w4_pos.y - w3_size.y),
                Point2d(w4_pos.x + w3_size.x, w4_pos.y + w3_size.y),
                wall_height * 2
            )
        )

        # Obstacles Generation
        if obstacles_mode == "sinusoidal":
            #
            obs_sep: ValType = ValType(obstacles_mode_param_.get("obstacle_sep", 5.0))
            obs_sz_x: ValType = ValType(obstacles_mode_param_.get("obstacle_size_x", 0.4))
            obs_sz_y: ValType = ValType(obstacles_mode_param_.get("obstacle_size_y", 0.4))
            obs_sz_z: ValType = ValType(obstacles_mode_param_.get("obstacle_size_z", 0.5))

            current_x: float = 2.0 + obs_sep.get_value()
            obs_len_approx: float = obs_sz_x.get_max_value() + obs_sep.get_max_value()
            nb_obstacles: int = max(floor(corridor_length / obs_len_approx) + 1, 1)

            # Loop through obstacles
            for i in range(nb_obstacles):
                #
                sx: float = obs_sz_x.get_value()
                sy: float = obs_sz_y.get_value()
                sz: float = obs_sz_z.get_value()
                #
                y_pos: float = sin(i * 0.16 * pi) * (corridor_width - 2 * sy)
                pos: Vec3 = Vec3(current_x, y_pos, 0)
                #
                components_body.append(create_geom(f"obstacle_{i}", "box", pos, Vec3(sx, sy, sz)))
                environment_rects.append(
                    Rect2d(
                        Point2d(current_x - sx, y_pos - sy),
                        Point2d(current_x + sx, y_pos + sy),
                        sz * 2
                    )
                )
                current_x += sx + obs_sep.get_max_value()

        elif obstacles_mode == "double_sinusoidal":
            # Two sinusoidal patterns offset
            obs_sep = ValType(obstacles_mode_param_.get("obstacle_sep", 5.0))
            obs_sz_x = ValType(obstacles_mode_param_.get("obstacle_size_x", 0.4))
            obs_sz_y = ValType(obstacles_mode_param_.get("obstacle_size_y", 0.4))
            obs_sz_z = ValType(obstacles_mode_param_.get("obstacle_size_z", 0.5))

            current_x = 2.0 + obs_sep.get_value()
            obs_len_approx = obs_sz_x.get_max_value() + obs_sep.get_max_value()
            nb_obstacles = max(floor(corridor_length / obs_len_approx) + 1, 1)

            # Loop through obstacles
            for i in range(nb_obstacles):
                #
                sx = obs_sz_x.get_value()
                sy = obs_sz_y.get_value()
                sz = obs_sz_z.get_value()

                # First wave
                y_pos1: float = sin(i * 0.16 * pi) * (corridor_width - 2 * sy) * 0.5
                pos1: Vec3 = Vec3(current_x, y_pos1, 0)
                components_body.append(create_geom(f"obstacle_{i}_a", "box", pos1, Vec3(sx, sy, sz)))
                environment_rects.append(
                    Rect2d(Point2d(current_x - sx, y_pos1 - sy), Point2d(current_x + sx, y_pos1 + sy), sz * 2)
                )

                # Second wave (offset)
                y_pos2: float = -sin(i * 0.16 * pi + pi/2) * (corridor_width - 2 * sy) * 0.5
                pos2: Vec3 = Vec3(current_x + sx, y_pos2, 0)
                components_body.append(create_geom(f"obstacle_{i}_b", "box", pos2, Vec3(sx, sy, sz)))
                environment_rects.append(
                    Rect2d(Point2d(current_x + sx - sx, y_pos2 - sy), Point2d(current_x + sx + sx, y_pos2 + sy), sz * 2)
                )

                current_x += 2 * sx + obs_sep.get_max_value()

        elif obstacles_mode == "random":
            #
            obs_sep = ValType(obstacles_mode_param_.get("obstacle_sep", 3.0))
            obs_sz_x = ValType(obstacles_mode_param_.get("obstacle_size_x", 0.5))
            obs_sz_y = ValType(obstacles_mode_param_.get("obstacle_size_y", 0.5))
            obs_sz_z = ValType(obstacles_mode_param_.get("obstacle_size_z", 0.5))

            current_x = 2.0 + obs_sep.get_value()
            obs_len_approx = obs_sz_x.get_max_value() + obs_sep.get_max_value()
            nb_obstacles = max(floor(corridor_length / obs_len_approx) + 1, 1)

            # Loop through obstacles
            for i in range(nb_obstacles):
                #
                sx = obs_sz_x.get_value()
                sy = obs_sz_y.get_value()
                sz = obs_sz_z.get_value()
                #
                y_pos: float = random.uniform(-(corridor_width - 2 * sy), corridor_width - 2 * sy)
                pos = Vec3(current_x, y_pos, 0)
                #
                components_body.append(create_geom(f"obstacle_{i}", "box", pos, Vec3(sx, sy, sz)))
                environment_rects.append(
                    Rect2d(Point2d(current_x - sx, y_pos - sy), Point2d(current_x + sx, y_pos + sy), sz * 2)
                )
                current_x += sx + obs_sep.get_max_value()

        return {'body': components_body, 'environment_rects': environment_rects, 'asset': None}


# FlatWorld generator class
class FlatWorld:
    """
    Generates an open arena environment with optional boundary walls and goal marker.
    """

    #
    def generate_flat_world(
        self,
        arena_size_x: float,
        arena_size_y: float,
        goal_position: Vec3,
        goal_radius: float = 3.0,
        obstacles_mode: str = "none",
        obstacles_mode_param: Optional[dict[str, Any]] = None,
        ground_friction: str = "1 0.005 0.0001"
    ) -> dict[str, Any]:

        """
        Generate flat world arena with goal marker.

        Args:
            arena_size_x: Arena size in X direction
            arena_size_y: Arena size in Y direction
            goal_position: Position of the goal marker
            goal_radius: Radius of the goal zone
            obstacles_mode: "none" or "random"
            obstacles_mode_param: Parameters for obstacle generation
            ground_friction: Friction string for ground
        """

        # Initialize lists
        components_body: list[ET.Element] = []
        environment_rects: list[Rect2d] = []
        obstacles_mode_param_: dict[str, Any] = obstacles_mode_param if obstacles_mode_param else {}

        # Ground plane (large)
        components_body.append(
            create_geom(
                "ground_floor", "plane", Vec3(0, 0, 0),
                Vec3(arena_size_x * 10, arena_size_y * 10, 1.0),
                {'rgba': '0.35 0.45 0.35 1', 'friction': ground_friction}
            )
        )

        # Goal marker (semi-transparent red sphere, no collision)
        goal_geom: ET.Element = ET.Element('geom')
        goal_geom.set('name', 'goal_marker')
        goal_geom.set('type', 'sphere')
        goal_geom.set('pos', f'{goal_position.x} {goal_position.y} {goal_position.z}')
        goal_geom.set('size', str(goal_radius))
        goal_geom.set('rgba', '1 0.2 0.2 0.4')  # Semi-transparent red
        goal_geom.set('contype', '0')  # No collision type
        goal_geom.set('conaffinity', '0')  # No collision affinity
        components_body.append(goal_geom)

        # Optional boundary walls (soft boundaries, not blocking)
        wall_height: float = 0.5
        wall_thickness: float = 0.2

        # Boundary walls (optional, can be disabled)
        # Left wall
        components_body.append(
            create_geom(
                "boundary_left", "box",
                Vec3(-arena_size_x - wall_thickness, 0, wall_height / 2),
                Vec3(wall_thickness, arena_size_y + wall_thickness, wall_height),
                {'rgba': '0.5 0.5 0.5 0.3'}
            )
        )
        # Right wall
        components_body.append(
            create_geom(
                "boundary_right", "box",
                Vec3(arena_size_x + wall_thickness, 0, wall_height / 2),
                Vec3(wall_thickness, arena_size_y + wall_thickness, wall_height),
                {'rgba': '0.5 0.5 0.5 0.3'}
            )
        )
        # Front wall
        components_body.append(
            create_geom(
                "boundary_front", "box",
                Vec3(0, arena_size_y + wall_thickness, wall_height / 2),
                Vec3(arena_size_x + wall_thickness, wall_thickness, wall_height),
                {'rgba': '0.5 0.5 0.5 0.3'}
            )
        )
        # Back wall
        components_body.append(
            create_geom(
                "boundary_back", "box",
                Vec3(0, -arena_size_y - wall_thickness, wall_height / 2),
                Vec3(arena_size_x + wall_thickness, wall_thickness, wall_height),
                {'rgba': '0.5 0.5 0.5 0.3'}
            )
        )

        # Random obstacles if specified
        if obstacles_mode == "random":
            num_obstacles: int = obstacles_mode_param_.get('num_obstacles', 10)
            sx: float = obstacles_mode_param_.get('obstacle_size_x', 0.5)
            sy: float = obstacles_mode_param_.get('obstacle_size_y', 0.5)
            sz: float = obstacles_mode_param_.get('obstacle_size_z', 0.5)

            for i in range(num_obstacles):
                # Random position within arena, avoiding goal and spawn
                x_pos: float = random.uniform(-arena_size_x + sx * 2, arena_size_x - sx * 2)
                y_pos: float = random.uniform(-arena_size_y + sy * 2, arena_size_y - sy * 2)

                # Skip if too close to spawn (0, 0) or goal
                dist_to_spawn: float = (x_pos**2 + y_pos**2)**0.5
                dist_to_goal: float = ((x_pos - goal_position.x)**2 + (y_pos - goal_position.y)**2)**0.5

                if dist_to_spawn < 5.0 or dist_to_goal < goal_radius + 2.0:
                    continue

                pos = Vec3(x_pos, y_pos, sz)
                components_body.append(
                    create_geom(f"obstacle_{i}", "box", pos, Vec3(sx, sy, sz))
                )
                environment_rects.append(
                    Rect2d(Point2d(x_pos - sx, y_pos - sy), Point2d(x_pos + sx, y_pos + sy), sz * 2)
                )

        return {'body': components_body, 'environment_rects': environment_rects, 'asset': None}


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

    # Build the scene
    def build(self) -> None:
        """
        Build the complete scene based on scene_type
        """

        scene_comps: dict[str, Any]

        if self.config.simulation.scene_type == "flat_world":
            # Flat world arena
            self._setup_goal_position()
            scene_comps = self.flat_world.generate_flat_world(
                arena_size_x=self.config.simulation.corridor_length,
                arena_size_y=self.config.simulation.corridor_width,
                goal_position=self.goal_position,
                goal_radius=self.config.simulation.goal_radius,
                obstacles_mode=self.config.simulation.obstacles_mode,
                obstacles_mode_param=self.config.simulation.obstacles_mode_param,
                ground_friction=self.config.simulation.ground_friction
            )
        else:
            # Corridor (default)
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

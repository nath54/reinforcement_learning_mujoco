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





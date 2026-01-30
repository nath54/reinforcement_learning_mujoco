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





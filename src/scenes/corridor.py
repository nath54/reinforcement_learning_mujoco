"""
Scene Generator Module

This module is responsible for procedurally generating the MuJoCo XML scene,
including the corridor, walls, and obstacles.
"""

from typing import Any, Optional

import random
import xml.etree.ElementTree as ET

from math import floor, sin, pi

from src.core.types import Vec3, Point2d, Rect2d, ValType

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
        ground_friction: str = "1 0.005 0.0001",
    ) -> dict[str, Any]:
        """
        Generate corridor with walls and obstacles.
        """

        # Initialize lists
        components_body: list[ET.Element] = []
        environment_rects: list[Rect2d] = []
        obstacles_mode_param_: dict[str, Any] = (
            obstacles_mode_param if obstacles_mode_param else {}
        )

        # Ground
        components_body.append(
            create_geom(
                "global_floor",
                "plane",
                Vec3(0, 0, 0),
                Vec3(corridor_length * 10, corridor_length * 10, 1.0),
                extra_attribs={"friction": ground_friction},
            )
        )

        # Walls dimensions
        wall_height: float = 3.0
        wall_width: float = 0.3

        # Wall 1 (Negative Y)
        w1_pos: Vec3 = Vec3(corridor_length, -corridor_width, wall_height)
        w1_size: Vec3 = Vec3(
            corridor_length - corridor_shift_x, wall_width, wall_height
        )
        components_body.append(create_geom("wall_1", "box", w1_pos, w1_size))
        environment_rects.append(
            Rect2d(
                Point2d(w1_pos.x - w1_size.x, w1_pos.y - w1_size.y),
                Point2d(w1_pos.x + w1_size.x, w1_pos.y + w1_size.y),
                wall_height * 2,
            )
        )

        # Wall 2 (Positive Y)
        w2_pos: Vec3 = Vec3(corridor_length, corridor_width, wall_height)
        components_body.append(create_geom("wall_2", "box", w2_pos, w1_size))
        environment_rects.append(
            Rect2d(
                Point2d(w2_pos.x - w1_size.x, w2_pos.y - w1_size.y),
                Point2d(w2_pos.x + w1_size.x, w2_pos.y + w1_size.y),
                wall_height * 2,
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
                wall_height * 2,
            )
        )

        # Wall 4 (Positive X side)
        w4_pos: Vec3 = Vec3(2 * corridor_length - corridor_shift_x, 0, wall_height)
        components_body.append(create_geom("wall_4", "box", w4_pos, w3_size))
        environment_rects.append(
            Rect2d(
                Point2d(w4_pos.x - w3_size.x, w4_pos.y - w3_size.y),
                Point2d(w4_pos.x + w3_size.x, w4_pos.y + w3_size.y),
                wall_height * 2,
            )
        )

        # Obstacles Generation
        if obstacles_mode == "sinusoidal":
            #
            obs_sep: ValType = ValType(obstacles_mode_param_.get("obstacle_sep", 5.0))
            obs_sz_x: ValType = ValType(
                obstacles_mode_param_.get("obstacle_size_x", 0.4)
            )
            obs_sz_y: ValType = ValType(
                obstacles_mode_param_.get("obstacle_size_y", 0.4)
            )
            obs_sz_z: ValType = ValType(
                obstacles_mode_param_.get("obstacle_size_z", 0.5)
            )

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
                components_body.append(
                    create_geom(f"obstacle_{i}", "box", pos, Vec3(sx, sy, sz))
                )
                environment_rects.append(
                    Rect2d(
                        Point2d(current_x - sx, y_pos - sy),
                        Point2d(current_x + sx, y_pos + sy),
                        sz * 2,
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
                components_body.append(
                    create_geom(f"obstacle_{i}_a", "box", pos1, Vec3(sx, sy, sz))
                )
                environment_rects.append(
                    Rect2d(
                        Point2d(current_x - sx, y_pos1 - sy),
                        Point2d(current_x + sx, y_pos1 + sy),
                        sz * 2,
                    )
                )

                # Second wave (offset)
                y_pos2: float = (
                    -sin(i * 0.16 * pi + pi / 2) * (corridor_width - 2 * sy) * 0.5
                )
                pos2: Vec3 = Vec3(current_x + sx, y_pos2, 0)
                components_body.append(
                    create_geom(f"obstacle_{i}_b", "box", pos2, Vec3(sx, sy, sz))
                )
                environment_rects.append(
                    Rect2d(
                        Point2d(current_x + sx - sx, y_pos2 - sy),
                        Point2d(current_x + sx + sx, y_pos2 + sy),
                        sz * 2,
                    )
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
                y_pos: float = random.uniform(
                    -(corridor_width - 2 * sy), corridor_width - 2 * sy
                )
                pos = Vec3(current_x, y_pos, 0)
                #
                components_body.append(
                    create_geom(f"obstacle_{i}", "box", pos, Vec3(sx, sy, sz))
                )
                environment_rects.append(
                    Rect2d(
                        Point2d(current_x - sx, y_pos - sy),
                        Point2d(current_x + sx, y_pos + sy),
                        sz * 2,
                    )
                )
                current_x += sx + obs_sep.get_max_value()

        elif obstacles_mode == "holes":
            # Holes mode: partial-width gaps in the floor
            # Instead of the global ground plane, we generate segmented floor tiles
            # with gaps where holes appear. Holes are positioned sinusoidally
            # so the robot can navigate around them.

            # Remove the global floor (it was already added above)
            # We replace it with segmented floor tiles
            components_body.clear()
            environment_rects.clear()

            # Re-add walls (same as above)
            # Wall 1 (Negative Y)
            w1_pos_h: Vec3 = Vec3(corridor_length, -corridor_width, wall_height)
            w1_size_h: Vec3 = Vec3(
                corridor_length - corridor_shift_x, wall_width, wall_height
            )
            components_body.append(create_geom("wall_1", "box", w1_pos_h, w1_size_h))
            environment_rects.append(
                Rect2d(
                    Point2d(w1_pos_h.x - w1_size_h.x, w1_pos_h.y - w1_size_h.y),
                    Point2d(w1_pos_h.x + w1_size_h.x, w1_pos_h.y + w1_size_h.y),
                    wall_height * 2,
                )
            )

            # Wall 2 (Positive Y)
            w2_pos_h: Vec3 = Vec3(corridor_length, corridor_width, wall_height)
            components_body.append(create_geom("wall_2", "box", w2_pos_h, w1_size_h))
            environment_rects.append(
                Rect2d(
                    Point2d(w2_pos_h.x - w1_size_h.x, w2_pos_h.y - w1_size_h.y),
                    Point2d(w2_pos_h.x + w1_size_h.x, w2_pos_h.y + w1_size_h.y),
                    wall_height * 2,
                )
            )

            # Wall 3 (Negative X side)
            w3_pos_h: Vec3 = Vec3(corridor_shift_x, 0, wall_height)
            w3_size_h: Vec3 = Vec3(wall_width, corridor_width, wall_height)
            components_body.append(create_geom("wall_3", "box", w3_pos_h, w3_size_h))
            environment_rects.append(
                Rect2d(
                    Point2d(w3_pos_h.x - w3_size_h.x, w3_pos_h.y - w3_size_h.y),
                    Point2d(w3_pos_h.x + w3_size_h.x, w3_pos_h.y + w3_size_h.y),
                    wall_height * 2,
                )
            )

            # Wall 4 (Positive X side)
            w4_pos_h: Vec3 = Vec3(
                2 * corridor_length - corridor_shift_x, 0, wall_height
            )
            components_body.append(create_geom("wall_4", "box", w4_pos_h, w3_size_h))
            environment_rects.append(
                Rect2d(
                    Point2d(w4_pos_h.x - w3_size_h.x, w4_pos_h.y - w3_size_h.y),
                    Point2d(w4_pos_h.x + w3_size_h.x, w4_pos_h.y + w3_size_h.y),
                    wall_height * 2,
                )
            )

            # Hole parameters
            hole_sep: ValType = ValType(obstacles_mode_param_.get("obstacle_sep", 5.0))
            hole_width_x: ValType = ValType(
                obstacles_mode_param_.get("obstacle_size_x", 1.0)
            )
            hole_width_y: ValType = ValType(
                obstacles_mode_param_.get("obstacle_size_y", 1.0)
            )

            # Floor tile height
            floor_thickness: float = 0.05
            floor_z: float = floor_thickness

            # Generate floor segments with holes
            current_x = corridor_shift_x + 0.5
            segment_idx: int = 0
            hole_idx: int = 0

            while current_x < 2 * corridor_length - corridor_shift_x:
                #
                sep: float = hole_sep.get_value()
                segment_end_x: float = min(
                    current_x + sep,
                    2 * corridor_length - corridor_shift_x,
                )

                # Add solid floor segment (full width)
                seg_center_x: float = (current_x + segment_end_x) / 2.0
                seg_half_x: float = (segment_end_x - current_x) / 2.0
                #
                if seg_half_x > 0.01:
                    components_body.append(
                        create_geom(
                            f"floor_seg_{segment_idx}",
                            "box",
                            Vec3(seg_center_x, 0, floor_z),
                            Vec3(seg_half_x, corridor_width, floor_thickness),
                            extra_attribs={"friction": ground_friction},
                        )
                    )
                    segment_idx += 1

                # Move to hole position
                current_x = segment_end_x

                # Check if there is room for a hole
                hw_x: float = hole_width_x.get_value()
                #
                if current_x + hw_x >= 2 * corridor_length - corridor_shift_x:
                    break

                hw_y: float = hole_width_y.get_value()

                # Hole Y position follows sinusoidal pattern (partial-width)
                hole_y: float = sin(hole_idx * 0.16 * pi) * (corridor_width - 2 * hw_y)

                # Create floor segments on both sides of the hole
                # The hole is a gap — we place floor on either side of it

                # Floor on the left side of the hole (negative Y from hole)
                left_floor_half_y: float = (corridor_width + (hole_y - hw_y)) / 2.0
                left_floor_center_y: float = (-corridor_width + (hole_y - hw_y)) / 2.0
                #
                if left_floor_half_y > 0.01:
                    components_body.append(
                        create_geom(
                            f"floor_hole_{hole_idx}_left",
                            "box",
                            Vec3(current_x + hw_x / 2.0, left_floor_center_y, floor_z),
                            Vec3(hw_x / 2.0, left_floor_half_y, floor_thickness),
                            extra_attribs={"friction": ground_friction},
                        )
                    )

                # Floor on the right side of the hole (positive Y from hole)
                right_floor_half_y: float = (corridor_width - (hole_y + hw_y)) / 2.0
                right_floor_center_y: float = (corridor_width + (hole_y + hw_y)) / 2.0
                #
                if right_floor_half_y > 0.01:
                    components_body.append(
                        create_geom(
                            f"floor_hole_{hole_idx}_right",
                            "box",
                            Vec3(
                                current_x + hw_x / 2.0,
                                right_floor_center_y,
                                floor_z,
                            ),
                            Vec3(hw_x / 2.0, right_floor_half_y, floor_thickness),
                            extra_attribs={"friction": ground_friction},
                        )
                    )

                # Add a visual marker for the hole edges (thin red borders)
                components_body.append(
                    create_geom(
                        f"hole_marker_{hole_idx}",
                        "box",
                        Vec3(current_x + hw_x / 2.0, hole_y, floor_z + 0.001),
                        Vec3(hw_x / 2.0, hw_y, 0.002),
                        extra_attribs={
                            "rgba": "0.8 0.2 0.2 0.5",
                            "contype": "0",
                            "conaffinity": "0",
                        },
                    )
                )

                #
                current_x += hw_x
                hole_idx += 1

        return {
            "body": components_body,
            "environment_rects": environment_rects,
            "asset": None,
        }

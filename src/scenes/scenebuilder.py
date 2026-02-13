"""
Scene Generator Module

This module is responsible for procedurally generating the MuJoCo XML scene,
including the corridor, walls, and obstacles.
"""

from typing import Any, Optional, cast

import random
import mujoco
import xml.etree.ElementTree as ET

from math import sin, cos

from src.core.types import Vec3, Point2d, Rect2d, GlobalConfig
from src.simulation.robot import Robot

from .corridor import Corridor
from .flatworld import FlatWorld
from .load_from_custom_xml import CustomXmlScene


# Helper function to convert euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)
def euler_to_quaternion(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    """
    Convert euler angles (roll, pitch, yaw) in radians to quaternion (w, x, y, z).

    Args:
        roll: Rotation around X axis (radians)
        pitch: Rotation around Y axis (radians)
        yaw: Rotation around Z axis (radians)

    Returns:
        Tuple (w, x, y, z) quaternion components
    """

    #
    cr: float = cos(roll * 0.5)
    sr: float = sin(roll * 0.5)
    cp: float = cos(pitch * 0.5)
    sp: float = sin(pitch * 0.5)
    cy: float = cos(yaw * 0.5)
    sy: float = sin(yaw * 0.5)

    #
    w: float = cr * cp * cy + sr * sp * sy
    x: float = sr * cp * cy - cr * sp * sy
    y: float = cr * sp * cy + sr * cp * sy
    z: float = cr * cp * sy - sr * sp * cy

    return (w, x, y, z)


# Helper function to scale a geom element's position and size
def _scale_geom_element(geom: ET.Element, scale: float) -> None:
    """
    Scale a geom element's position and size attributes in-place.

    Args:
        geom: The XML geom element to scale
        scale: Scale factor to apply
    """

    # Scale position
    pos_str: str = geom.get("pos", "")
    if pos_str:
        parts: list[float] = [float(v) * scale for v in pos_str.split()]
        geom.set("pos", " ".join(str(v) for v in parts))

    # Scale size
    size_str: str = geom.get("size", "")
    if size_str:
        parts = [float(v) * scale for v in size_str.split()]
        geom.set("size", " ".join(str(v) for v in parts))


# Helper function to scale all geoms within a body element (recursively)
def _scale_body_element(body: ET.Element, scale: float) -> None:
    """
    Scale all geom children and nested bodies within a body element.
    Also scales the body's own position.

    Args:
        body: The XML body element to scale
        scale: Scale factor to apply
    """

    # Scale body position
    pos_str: str = body.get("pos", "")
    if pos_str:
        parts: list[float] = [float(v) * scale for v in pos_str.split()]
        body.set("pos", " ".join(str(v) for v in parts))

    # Scale all child geoms
    child: ET.Element
    for child in body.findall("geom"):
        _scale_geom_element(child, scale)

    # Scale nested bodies recursively
    for child in body.findall("body"):
        _scale_body_element(child, scale)


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
        self.custom_xml_scene = CustomXmlScene()
        self.robot = Robot(config.robot)
        self.environment_rects: list[Rect2d] = []

        # Goal position (will be set during build)
        self.goal_position: Optional[Vec3] = None

        # Robot spawn position from config (default: (0, 0, 0.5))
        rsp: Optional[tuple[float, float, float]] = (
            config.simulation.robot_start_position
        )
        self.robot_spawn_pos: Vec3 = (
            Vec3(rsp[0], rsp[1], rsp[2]) if rsp is not None else Vec3(0.0, 0.0, 0.5)
        )

        # Approximate bounds
        margin: float = 250.0
        if config.simulation.scene_type == "flat_world":
            # Symmetric arena bounds
            self.env_bounds: Rect2d = Rect2d(
                Point2d(
                    -config.simulation.corridor_length - margin,
                    -config.simulation.corridor_width - margin,
                ),
                Point2d(
                    config.simulation.corridor_length + margin,
                    config.simulation.corridor_width + margin,
                ),
            )
        elif config.simulation.scene_type == "custom_xml":
            # Will be computed after loading the custom XML
            self.env_bounds: Rect2d = Rect2d(
                Point2d(-margin, -margin),
                Point2d(margin, margin),
            )
        else:
            # Corridor bounds (original)
            self.env_bounds: Rect2d = Rect2d(
                Point2d(-margin, -config.simulation.corridor_width - margin),
                Point2d(
                    2 * config.simulation.corridor_length + margin,
                    config.simulation.corridor_width + margin,
                ),
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
            dist_from_spawn: float = (x**2 + y**2) ** 0.5

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
            if (
                self.config.simulation.randomize_goal
                or self.config.simulation.goal_position is None
            ):
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
            ground_friction=self.config.simulation.ground_friction,
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
            ground_friction=self.config.simulation.ground_friction,
        )

        # Add goal marker sphere to corridor scene
        goal_radius: float = self.config.simulation.goal_radius
        goal_geom: ET.Element = ET.Element("geom")
        goal_geom.set("name", "goal_marker")
        goal_geom.set("type", "sphere")
        goal_geom.set(
            "pos",
            f"{self.goal_position.x} {self.goal_position.y} {self.goal_position.z}",
        )
        goal_geom.set("size", str(goal_radius))
        goal_geom.set("rgba", "1 0.2 0.2 0.4")  # Semi-transparent red
        goal_geom.set("contype", "0")  # No collision type
        goal_geom.set("conaffinity", "0")  # No collision affinity
        scene_comps["body"].append(goal_geom)

        #
        return scene_comps

    #
    def build_custom_xml(self) -> dict[str, Any]:
        """
        Build scene from a custom pre-authored MuJoCo XML file.
        """

        # Get custom XML path from config
        custom_xml_path: Optional[str] = self.config.simulation.custom_xml_path
        #
        if custom_xml_path is None:
            raise ValueError(
                "custom_xml_path must be set in simulation config "
                "when using scene_type 'custom_xml'"
            )

        # Load scene from custom XML
        scene_comps: dict[str, Any] = self.custom_xml_scene.load_scene(custom_xml_path)

        # Update env_bounds from the loaded scene geometry
        self.env_bounds = self.custom_xml_scene.compute_scene_bounds(
            scene_comps["body"]
        )

        # Set goal position from custom config or compute from scene bounds
        custom_goal: Optional[tuple[float, float, float]] = (
            self.config.simulation.custom_xml_goal_position
        )
        #
        if custom_goal is not None:
            #
            self.goal_position = Vec3(custom_goal[0], custom_goal[1], custom_goal[2])
        else:
            # Place goal at the far end of the scene bounds
            goal_x: float = self.env_bounds.corner_bottom_right.x - 15.0
            self.goal_position = Vec3(goal_x, 0.0, 0.5)

        # Add goal marker sphere
        goal_radius: float = self.config.simulation.goal_radius
        goal_geom: ET.Element = ET.Element("geom")
        goal_geom.set("name", "goal_marker")
        goal_geom.set("type", "sphere")
        goal_geom.set(
            "pos",
            f"{self.goal_position.x} {self.goal_position.y} {self.goal_position.z}",
        )
        goal_geom.set("size", str(goal_radius))
        goal_geom.set("rgba", "1 0.2 0.2 0.4")  # Semi-transparent red
        goal_geom.set("contype", "0")  # No collision type
        goal_geom.set("conaffinity", "0")  # No collision affinity
        scene_comps["body"].append(goal_geom)

        #
        return scene_comps

    # Build the scene
    def build(self) -> None:
        """
        Build the complete scene based on scene_type.
        First builds the scene geometry, then injects the robot separately.
        """

        #
        scene_comps: dict[str, Any]
        #
        if self.config.simulation.scene_type == "flat_world":
            #
            scene_comps = self.build_flatworld()
        #
        elif self.config.simulation.scene_type == "custom_xml":
            #
            scene_comps = self.build_custom_xml()
        #
        elif self.config.simulation.scene_type == "corridor":
            #
            scene_comps = self.build_corridor()
        #
        else:
            # Default Scene
            scene_comps = self.build_corridor()

        self.environment_rects = cast(
            list[Rect2d], scene_comps.get("environment_rects", [])
        )

        # Extract robot components (separate from scene)
        robot_comps: dict[str, Any] = self.robot.extract_robot_from_xml()

        # Assemble XML
        root: ET.Element = ET.Element("mujoco", model="robot_scene")

        # Visual settings for rendering
        visual: ET.Element = ET.Element("visual")
        global_elem: ET.Element = ET.Element("global")
        global_elem.set("offwidth", "1440")
        global_elem.set("offheight", "1024")
        visual.append(global_elem)
        root.append(visual)

        # Size limits
        root.append(ET.Element("size", njmax="1000", nconmax="500"))

        # Physics Option
        root.append(
            ET.Element(
                "option",
                timestep=str(self.config.simulation.dt),
                gravity=self.config.simulation.gravity,
                solver=self.config.simulation.solver,
                iterations=str(self.config.simulation.iterations),
            )
        )

        # Assets
        asset: ET.Element = ET.Element("asset")
        #
        texture: ET.Element
        mat: ET.Element
        a: ET.Element
        #
        # Track added asset names to avoid duplicates (custom XML may share names)
        added_asset_names: set[str] = set()
        #
        for texture in self.robot.create_textures().values():
            asset.append(texture)
            added_asset_names.add(texture.get("name", ""))
        for mat in self.robot.create_enhanced_materials().values():
            asset.append(mat)
            added_asset_names.add(mat.get("name", ""))
        if robot_comps["asset"]:
            for a in robot_comps["asset"]:
                asset_name: str = a.get("name", "")
                if asset_name not in added_asset_names:
                    asset.append(a)
                    added_asset_names.add(asset_name)
        # Add scene assets (from custom XML), skipping duplicates
        if scene_comps.get("asset"):
            for a in scene_comps["asset"]:
                asset_name = a.get("name", "")
                if asset_name not in added_asset_names:
                    asset.append(a)
                    added_asset_names.add(asset_name)
        root.append(asset)

        # Scale factors
        scene_scale: float = self.config.simulation.scene_scale

        # Worldbody
        worldbody: ET.Element = ET.Element("worldbody")
        #
        b: ET.Element
        #
        if scene_comps["body"]:
            for b in scene_comps["body"]:
                # Apply scene scale to all scene geoms
                if scene_scale != 1.0:
                    _scale_geom_element(b, scene_scale)
                worldbody.append(b)

        # Inject robot into scene (separate from scene generation)
        if robot_comps["robot_body"]:
            self.robot.enhance_robot_visuals(robot_comps["robot_body"])
            # Apply robot scale
            robot_scale: float = self.config.robot.robot_scale
            if robot_scale != 1.0:
                _scale_body_element(robot_comps["robot_body"], robot_scale)
            # Use configurable robot spawn position
            robot_comps["robot_body"].set(
                "pos",
                f"{self.robot_spawn_pos.x} {self.robot_spawn_pos.y} "
                f"{self.robot_spawn_pos.z}",
            )
            # Apply orientation if configured (euler -> quaternion)
            rso: Optional[tuple[float, float, float]] = (
                self.config.simulation.robot_start_orientation
            )
            if rso is not None:
                qw, qx, qy, qz = euler_to_quaternion(rso[0], rso[1], rso[2])
                robot_comps["robot_body"].set("quat", f"{qw} {qx} {qy} {qz}")
            worldbody.append(robot_comps["robot_body"])
        root.append(worldbody)

        if robot_comps["actuators"]:
            root.append(robot_comps["actuators"])

        # Compile
        xml_str: str = ET.tostring(root, encoding="unicode")
        self.mujoco_model = mujoco.MjModel.from_xml_string(xml_str)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)

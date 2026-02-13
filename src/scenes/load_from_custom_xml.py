"""
Custom XML Scene Loader Module

This module handles loading pre-authored MuJoCo XML scenes (e.g. corridor_3x100.xml)
and extracting their geometry components for integration with the SceneBuilder.

The custom XML's physics settings (<option>, <compiler>, <size>) are ignored —
the SceneBuilder supplies its own physics from the global config.
Only <worldbody> geoms and <asset> elements are extracted.
"""

from typing import Any, Optional

import os
import xml.etree.ElementTree as ET

from src.core.types import Point2d, Rect2d


# Custom XML scene loader class
class CustomXmlScene:
    """
    Loads a pre-authored MuJoCo XML scene file and extracts its geometry
    components for use with the SceneBuilder.
    """

    # Load scene from a custom XML file
    def load_scene(
        self,
        xml_path: str,
    ) -> dict[str, Any]:
        """
        Load and parse a custom MuJoCo XML scene file.

        Extracts worldbody geoms and asset elements. Classifies geoms into:
        - Walls: large geoms with "wall" in name -> environment_rects
        - Bumps/obstacles: geoms with material "mat_bump" -> environment_rects
        - Floor tiles: remaining geoms -> geometry only (no collision rect)
        - Other geoms with significant height -> environment_rects

        Args:
            xml_path: Path to the custom MuJoCo XML file

        Returns:
            Dict with keys "body", "environment_rects", "asset"
        """

        # Validate file exists
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Custom XML scene file not found: {xml_path}")

        # Parse the XML file
        tree: ET.ElementTree = ET.parse(xml_path)
        root: ET.Element = tree.getroot()

        # Initialize output lists
        components_body: list[ET.Element] = []
        environment_rects: list[Rect2d] = []
        asset_elements: list[ET.Element] = []

        # Extract assets (materials, textures)
        asset_elem: Optional[ET.Element] = root.find("asset")
        #
        if asset_elem is not None:
            #
            child: ET.Element
            #
            for child in asset_elem:
                asset_elements.append(child)

        # Extract worldbody geoms
        worldbody: Optional[ET.Element] = root.find("worldbody")
        #
        if worldbody is not None:
            #
            geom: ET.Element
            #
            for geom in worldbody.findall("geom"):
                #
                components_body.append(geom)

                # Classify geom for collision detection
                self._classify_geom(geom, environment_rects)

        #
        return {
            "body": components_body,
            "environment_rects": environment_rects,
            "asset": asset_elements if asset_elements else None,
        }

    # Classify a geom element and add to environment_rects if needed
    def _classify_geom(
        self,
        geom: ET.Element,
        environment_rects: list[Rect2d],
    ) -> None:
        """
        Classify a geom element and add it to environment_rects if it
        represents a wall, bump, or other obstacle.

        Args:
            geom: The XML geom element to classify
            environment_rects: List to append collision rects to
        """

        # Get geom attributes
        geom_name: str = geom.get("name", "")
        geom_material: str = geom.get("material", "")
        geom_type: str = geom.get("type", "")
        pos_str: str = geom.get("pos", "0 0 0")
        size_str: str = geom.get("size", "0 0 0")

        # Parse position and size
        pos_parts: list[float] = [float(v) for v in pos_str.split()]
        size_parts: list[float] = [float(v) for v in size_str.split()]

        # Skip planes (infinite ground)
        if geom_type == "plane":
            return

        # Ensure we have enough components
        if len(pos_parts) < 3 or len(size_parts) < 3:
            return

        #
        px: float = pos_parts[0]
        py: float = pos_parts[1]
        sx: float = size_parts[0]
        sy: float = size_parts[1]
        sz: float = size_parts[2]

        # Walls: large geoms with "wall" in name
        is_wall: bool = "wall" in geom_name.lower()

        # Bumps/obstacles: geoms with bump material
        is_bump: bool = geom_material == "mat_bump"

        # Other significant obstacles: tall geoms that aren't floor tiles
        is_tall: bool = sz > 0.03  # Floor tiles have sz ≈ 0.025

        #
        if is_wall or is_bump or is_tall:
            #
            environment_rects.append(
                Rect2d(
                    Point2d(px - sx, py - sy),
                    Point2d(px + sx, py + sy),
                    sz * 2,
                )
            )

    # Compute scene bounds from geom elements
    def compute_scene_bounds(
        self,
        components_body: list[ET.Element],
    ) -> Rect2d:
        """
        Compute the bounding rectangle of all geoms in the scene.

        Args:
            components_body: List of geom XML elements

        Returns:
            Rect2d representing the scene bounds
        """

        #
        min_x: float = float("inf")
        max_x: float = float("-inf")
        min_y: float = float("inf")
        max_y: float = float("-inf")

        #
        geom: ET.Element
        #
        for geom in components_body:
            #
            pos_str: str = geom.get("pos", "0 0 0")
            size_str: str = geom.get("size", "0 0 0")
            #
            pos_parts: list[float] = [float(v) for v in pos_str.split()]
            size_parts: list[float] = [float(v) for v in size_str.split()]

            #
            if len(pos_parts) >= 2 and len(size_parts) >= 2:
                #
                px: float = pos_parts[0]
                py: float = pos_parts[1]
                sx: float = size_parts[0]
                sy: float = size_parts[1]
                #
                min_x = min(min_x, px - sx)
                max_x = max(max_x, px + sx)
                min_y = min(min_y, py - sy)
                max_y = max(max_y, py + sy)

        # Fallback if no geoms found
        if min_x == float("inf"):
            return Rect2d(Point2d(-100, -100), Point2d(100, 100))

        # Add margin
        margin: float = 10.0
        #
        return Rect2d(
            Point2d(min_x - margin, min_y - margin),
            Point2d(max_x + margin, max_y + margin),
        )

"""
Scene Generator Module

This module is responsible for procedurally generating the MuJoCo XML scene,
including the corridor, walls, and obstacles.
"""

from typing import Optional

import xml.etree.ElementTree as ET

from src.core.types import Vec3


# Helper to create a MuJoCo geom element
def create_geom(
    name: str,
    geom_type: str,
    pos: Vec3,
    size: Vec3,
    extra_attribs: Optional[dict[str, str]] = None,
) -> ET.Element:
    """
    Helper function to create a MuJoCo geom element.
    """

    new_geom: ET.Element = ET.Element("geom")
    new_geom.set("name", name)
    new_geom.set("type", geom_type)
    new_geom.set("size", f"{size.x} {size.y} {size.z}")
    new_geom.set("pos", f"{pos.x} {pos.y} {pos.z}")

    if extra_attribs:
        #
        k: str
        v: str
        #
        for k, v in extra_attribs.items():
            new_geom.set(k, v)

    return new_geom

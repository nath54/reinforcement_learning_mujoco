# type: ignore

#
### Mujoco Preparation. ###
#
import os
import io
from typing import Any

#
### Set environment variable to indicate mujoco to use GPU rendering. ###
#
os.environ["MUJOCO_GL"]: str = "egl"

#
### Import Mujoco library. ###
#
import mujoco
import mujoco.mjcf as mjcf
import numpy as np
from numpy.typing import NDArray

#
### Graphics and plotting. ###
#
import mediapy as media

#
### More legible printing from numpy. ###
#
np.set_printoptions(precision=3, suppress=True, linewidth=100)

#
### Type Aliases for Clarity ###
#
Model: type[mujoco.MjModel] = mujoco.MjModel
Data: type[mujoco.MjData] = mujoco.MjData
Renderer: type[mujoco.Renderer] = mujoco.Renderer

# Define a Type for the XML Root Element
MjcfRoot: type[mjcf.RootElement] = mjcf.RootElement

#
### Python Scene Definition (Replacing XML String) ###
#

def create_dominos_model() -> MjcfRoot:
    """Constructs the dominos scene using the MuJoCo MjcfElement API."""

    root: MjcfRoot = MjcfRoot()

    # --- Asset/Visual/Default ---
    asset_element: mjcf.Asset = root.asset
    asset_element.add('texture', type="skybox", builtin="gradient", rgb1=".3 .5 .7", rgb2="0 0 0", width="32", height="512")
    asset_element.add('texture', name="grid", type="2d", builtin="checker", width="512", height="512", rgb1=".1 .2 .3", rgb2=".2 .3 .4")
    asset_element.add('material', name="grid", texture="grid", texrepeat="2 2", texuniform="true", reflectance=".2")

    # MuJoCo 3.0+ separates statistics and visual options
    root.statistic.meansize = 0.01

    visual_element: mjcf.Visual = root.visual
    visual_element.global_.offheight = 2160
    visual_element.global_.offwidth = 3840
    visual_element.quality.offsamples = 8

    # Default elements (must be added via the 'default' tag)
    default_geom: mjcf.Geom = root.default.add('geom', type="box", solref=[0.005, 1.0])
    static_class: mjcf.Default = root.default.add('default', class_="static")
    static_class.geom.rgba = [0.3, 0.5, 0.7, 1.0]

    # --- Option ---
    root.option.timestep = 5e-4

    # --- Worldbody ---
    worldbody: mjcf.WorldBody = root.worldbody
    worldbody.add('light', pos=[0.3, -0.3, 0.8], mode="trackcom", diffuse=[1, 1, 1], specular=[0.3, 0.3, 0.3])
    worldbody.add('light', pos=[0.0, -0.3, 0.4], mode="targetbodycom", target="box", diffuse=[0.8, 0.8, 0.8], specular=[0.3, 0.3, 0.3])
    worldbody.add('geom', name="floor", type="plane", size=[3, 3, 0.01], pos=[-0.025, -0.295, 0.0], material="grid")
    worldbody.add('geom', name="ramp", pos=[0.25, -0.45, -0.03], size=[0.04, 0.1, 0.07], euler=[-30, 0, 0], class_="static")
    worldbody.add('camera', name="top", pos=[-0.37, -0.78, 0.49], xyaxes=[0.78, -0.63, 0.0, 0.27, 0.33, 0.9])

    # Ball
    ball_body: mjcf.Body = worldbody.add('body', name="ball", pos=[0.25, -0.45, 0.1])
    ball_body.add('freejoint', name="ball")
    ball_body.add('geom', name="ball", type="sphere", size=[0.02], rgba=[0.65, 0.81, 0.55, 1.0])

    # Dominos (using tuples/lists for positional arguments)
    dominos_data: list[tuple[list[float], float, list[float]]] = [
        ([0.26, -0.3, 0.03], -90.0, [0.0015, 0.015, 0.03], [1.0, 0.5, 0.5, 1.0]),
        ([0.26, -0.27, 0.04], -81.0, [0.002, 0.02, 0.04], [1.0, 1.0, 0.5, 1.0]),
        ([0.24, -0.21, 0.06], -63.0, [0.003, 0.03, 0.06], [0.5, 1.0, 0.5, 1.0]),
        ([0.2, -0.16, 0.08], -45.0, [0.004, 0.04, 0.08], [0.5, 1.0, 1.0, 1.0]),
        ([0.15, -0.12, 0.1], -27.0, [0.005, 0.05, 0.1], [0.5, 0.5, 1.0, 1.0]),
        ([0.09, -0.1, 0.12], -9.0, [0.006, 0.06, 0.12], [1.0, 0.5, 1.0, 1.0]),
    ]

    for pos, angle_deg, size, rgba in dominos_data:
        domino_body: mjcf.Body = worldbody.add('body', pos=pos, euler=[0, 0, angle_deg])
        domino_body.add('freejoint')
        domino_body.add('geom', size=size, rgba=rgba)

    # Seesaw
    seasaw_wrapper: mjcf.Body = worldbody.add('body', name="seasaw_wrapper", pos=[-0.23, -0.1, 0.0], euler=[0, 0, 30.0])
    seasaw_wrapper.add('geom', size=[0.01, 0.01, 0.015], pos=[0.0, 0.05, 0.015], class_="static")
    seasaw_wrapper.add('geom', size=[0.01, 0.01, 0.015], pos=[0.0, -0.05, 0.015], class_="static")
    seasaw_wrapper.add('geom', type="cylinder", size=[0.01, 0.0175], pos=[-0.09, 0.0, 0.0175], class_="static")

    seasaw_body: mjcf.Body = seasaw_wrapper.add('body', name="seasaw", pos=[0.0, 0.0, 0.03])
    seasaw_body.add('joint', axis=[0, 1, 0])
    seasaw_body.add('geom', type="cylinder", size=[0.005, 0.039], zaxis=[0, 1, 0], rgba=[0.84, 0.15, 0.33, 1.0])
    seasaw_body.add('geom', size=[0.1, 0.02, 0.005], pos=[0.0, 0.0, 0.01], rgba=[0.84, 0.15, 0.33, 1.0])

    # Box
    box_body: mjcf.Body = worldbody.add('body', name="box", pos=[-0.3, -0.14, 0.05501], euler=[0, 0, -30.0])
    box_body.add('freejoint', name="box")
    box_body.add('geom', name="box", size=[0.01, 0.01, 0.01], rgba=[0.0, 0.7, 0.79, 1.0])

    return root

#
### Build Model and Initialize Data ###
#
# Create the root element
root_element: MjcfRoot = create_dominos_model()

# Compile the MjcfRoot into an MjModel
model: Model = mjcf.export_with_assets(root_element)
data: Data = mujoco.MjData(model)

#
### Render from fixed camera ###
#
duration: float = 2.5  # (seconds)
framerate: int = 60  # (Hz)
height: int = 1024
width: int = 1440

#
### Simulate and display video. ###
#

#
# frames will store the rendered pixels, which are numpy arrays.
frames: list[NDArray[np.int8]] = []

#
### Reset state and time. ###
#
mujoco.mj_resetData(model, data)

#
renderer: Renderer
#
with mujoco.Renderer(model, height, width) as renderer:

    #
    while data.time < duration:

        #
        mujoco.mj_step(model, data)

        #
        ### data.time is a float ###
        #
        required_frames: float = data.time * framerate

        #
        if len(frames) < required_frames:

            #
            renderer.update_scene(data, camera='top')

            #
            ### renderer.render() returns an NDArray representing the image pixels ###
            #
            pixels: NDArray[np.int8] = renderer.render()
            #
            frames.append(pixels)

#
### Render video at half real-time. ###
#
output_filename: str = "tmp3.mp4"
output_fps: float = framerate / 2.0
#
media.write_video(output_filename, frames, fps=output_fps)
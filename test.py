# type: ignore

#
### Mujoco Preparation. ###
#
import os
import io
#
### Set environment variable to indicate mujoco to use GPU rendering. ###
#
os.environ["MUJOCO_GL"] = "egl"

#
### Import Mujoco library. ###
#
import mujoco  # type: ignore

#
### Other imports and helper functions. ###
#
# import time
# import itertools
import numpy as np
from numpy.typing import NDArray
#
### Graphics and plotting. ###
#
import mediapy as media
# import matplotlib.pyplot as plt

#
### More legible printing from numpy. ###
#
# np.set_printoptions does not return a value.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

#
### Type Aliases for Clarity ###
#
#
### Define common types for the simulation data ###
#
Model: type[mujoco.MjModel] = mujoco.MjModel
Data: type[mujoco.MjData] = mujoco.MjData
#
Renderer: type[mujoco.Renderer] = mujoco.Renderer
#
VisionOption: type[mujoco.MjvOption] = mujoco.MjvOption
Perturbation: type[mujoco.MjvPerturb] = mujoco.MjvPerturb

#
### Get MuJoCo's standard humanoid model. ###
### print('Getting MuJoCo humanoid XML description from GitHub:') ###
#
file_path: str | os.PathLike[str] = 'mujoco/model/humanoid/humanoid.xml'

#
### The file object is opened in read mode ('r'). ###
### The result of open is a TextIOWrapper which is a subclass of io.TextIOBase ###
#
f: io.TextIOBase
#
with open(file_path, 'r') as f:
    #
    xml: str = f.read()

#
### Load the model, make two MjData's. ###
#
model: Model = mujoco.MjModel.from_xml_string(xml)
data: Data = mujoco.MjData(model)
data2: Data = mujoco.MjData(model)

#
### Episode parameters. ###
#
duration: int = 30       # (seconds)
framerate: int = 60     # (Hz)

#
### data.qpos is an NDArray of floats ###
### NDArray[np.float64] is used because the specific dtype (e.g., np.float64) isn't strictly enforced by the type checker without more advanced tools ###
#
qpos_initial: NDArray[np.float64] = np.array([-.5, -.5])
#
### Initial x-y position (m) ###
#
data.qpos[0:2] = qpos_initial

#
### data.qvel is an NDArray of floats ###
### Initial vertical velocity (m/s) ###
#
data.qvel[2] = 4.0

#
### ctrl_phase is an NDArray of floats of shape (model.nu,) ###
#
ctrl_phase: NDArray[np.float64] = 2 * np.pi * np.random.rand(model.nu)  # Control phase
ctrl_freq: float = 1.0     # Control frequency

#
### Visual options for the "ghost" model. ###
#
vopt2: VisionOption = mujoco.MjvOption()

# mujoco.mjtVisFlag is an enum, its values are of type int
flag_value: int = mujoco.mjtVisFlag.mjVIS_TRANSPARENT
vopt2.flags[flag_value] = True  # Transparent.

pert: Perturbation = mujoco.MjvPerturb()  # Empty MjvPerturb object

#
### We only want dynamic objects (the humanoid). Static objects (the floor) ###
### should not be re-drawn. The mjtCatBit flag lets us do that, though we could ###
### equivalently use mjtVisFlag.mjVIS_STATIC ###
#
# mujoco.mjtCatBit is an enum, its value is of type int
catmask: int = mujoco.mjtCatBit.mjCAT_DYNAMIC

#
### Simulate and render. ###
#
# frames will store the rendered pixels, which are numpy arrays.
# media.write_video expects a sequence of frames, often a list[NDArray[np.float64]]
frames: list[NDArray[np.float64]] = []

#
# The Renderer object is used as a context manager
width: int = 480
height: int = 640
with mujoco.Renderer(model, width, height) as renderer:
    renderer: Renderer
    #
    while data.time < duration:
        # data.time is a float
        current_time: float = data.time
        #
        ### Sinusoidal control signal. ###
        #
        control_signal: NDArray[np.float64] = np.sin(ctrl_phase + 2 * np.pi * current_time * ctrl_freq)
        data.ctrl: NDArray[np.float64] = control_signal

        #
        # mj_step mutates the MjData object and returns None
        mujoco.mj_step(model, data)
        #
        # len(frames) is an int
        required_frames: float = current_time * framerate
        if len(frames) < required_frames:
            #
            ### This draws the regular humanoid from `data`. ###
            #
            # renderer.update_scene mutates renderer.scene and returns None
            renderer.update_scene(data)

            #
            ### Copy qpos to data2, move the humanoid sideways, call mj_forward. ###
            #
            # data.qpos and data2.qpos are views into the respective data buffers
            data2.qpos: NDArray[np.float64] = data.qpos
            data2.qpos[0] += 1.5
            data2.qpos[1] += 1
            # mj_forward mutates the MjData object and returns None
            mujoco.mj_forward(model, data2)

            #
            ### Call mjv_addGeoms to add the ghost humanoid to the scene. ###
            #
            # renderer.scene is an MjvScene object, this function mutates it and returns None
            mujoco.mjv_addGeoms(model, data2, vopt2, pert, catmask, renderer.scene)

            #
            ### Render and add the frame. ###
            #
            # renderer.render() returns an NDArray representing the image pixels
            pixels: NDArray[np.float64] = renderer.render()
            frames.append(pixels)

#
### Render video at half real-time. ###
#
output_filename: str = "tmp.mp4"
output_fps: float = framerate / 2.0
#
media.write_video(output_filename, frames, fps=output_fps)
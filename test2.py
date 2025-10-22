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
#@title Load the "dominos" model

dominos_xml = """
<mujoco>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="32" height="512"/>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="2 2" texuniform="true" reflectance=".2"/>
    </asset>

    <statistic meansize=".01"/>

    <visual>
        <global offheight="2160" offwidth="3840"/>
        <quality offsamples="8"/>
    </visual>

    <default>
        <geom type="box" solref=".005 1"/>
        <default class="static">
            <geom rgba=".3 .5 .7 1"/>
        </default>
    </default>

    <option timestep="5e-4"/>

    <worldbody>
        <light pos=".3 -.3 .8" mode="trackcom" diffuse="1 1 1" specular=".3 .3 .3"/>
        <light pos="0 -.3 .4" mode="targetbodycom" target="box" diffuse=".8 .8 .8" specular=".3 .3 .3"/>
        <geom name="floor" type="plane" size="3 3 .01" pos="-0.025 -0.295  0" material="grid"/>
        <geom name="ramp" pos=".25 -.45 -.03" size=".04 .1 .07" euler="-30 0 0" class="static"/>
        <camera name="top" pos="-0.37 -0.78 0.49" xyaxes="0.78 -0.63 0 0.27 0.33 0.9"/>

        <body name="ball" pos=".25 -.45 .1">
            <freejoint name="ball"/>
            <geom name="ball" type="sphere" size=".02" rgba=".65 .81 .55 1"/>
        </body>

        <body pos=".26 -.3 .03" euler="0 0 -90.0">
            <freejoint/>
            <geom size=".0015 .015 .03" rgba="1 .5 .5 1"/>
        </body>

        <body pos=".26 -.27 .04" euler="0 0 -81.0">
            <freejoint/>
            <geom size=".002 .02 .04" rgba="1 1 .5 1"/>
        </body>

        <body pos=".24 -.21 .06" euler="0 0 -63.0">
            <freejoint/>
            <geom size=".003 .03 .06" rgba=".5 1 .5 1"/>
        </body>

        <body pos=".2 -.16 .08" euler="0 0 -45.0">
            <freejoint/>
            <geom size=".004 .04 .08" rgba=".5 1 1 1"/>
        </body>

        <body pos=".15 -.12 .1" euler="0 0 -27.0">
            <freejoint/>
            <geom size=".005 .05 .1" rgba=".5 .5 1 1"/>
        </body>

        <body pos=".09 -.1 .12" euler="0 0 -9.0">
            <freejoint/>
            <geom size=".006 .06 .12" rgba="1 .5 1 1"/>
        </body>

        <body name="seasaw_wrapper" pos="-.23 -.1 0" euler="0 0 30">
            <geom size=".01 .01 .015" pos="0 .05 .015" class="static"/>
            <geom size=".01 .01 .015" pos="0 -.05 .015" class="static"/>
            <geom type="cylinder" size=".01 .0175" pos="-.09 0 .0175" class="static"/>
            <body name="seasaw" pos="0 0 .03">
                <joint axis="0 1 0"/>
                <geom type="cylinder" size=".005 .039" zaxis="0 1 0" rgba=".84 .15 .33 1"/>
                <geom size=".1 .02 .005" pos="0 0 .01" rgba=".84 .15 .33 1"/>
            </body>
        </body>

        <body name="box" pos="-.3 -.14 .05501" euler="0 0 -30">
            <freejoint name="box"/>
            <geom name="box" size=".01 .01 .01" rgba=".0 .7 .79 1"/>
        </body>

    </worldbody>

</mujoco>
"""

#
model: Model = mujoco.MjModel.from_xml_string(dominos_xml)
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
frames: list[NDArray[np.float64]] = []

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
        if len(frames) < data.time * framerate:

            #
            renderer.update_scene(data, camera='top')
            pixels = renderer.render()
            frames.append(pixels)

#
### Render video at half real-time. ###
#
output_filename: str = "tmp2.mp4"
output_fps: float = framerate / 2.0
#
media.write_video(output_filename, frames, fps=output_fps)
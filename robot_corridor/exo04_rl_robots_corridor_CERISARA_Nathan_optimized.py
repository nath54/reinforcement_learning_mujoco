#
### Import Modules. ###
#
from typing import Any, Optional, cast
#
import os
import json
import argparse
import random
import time
import yaml
from dataclasses import dataclass, field
#
from math import floor, sin, pi
#
from matplotlib import pyplot as plt
#
### Set environment variable to indicate mujoco to use GPU rendering. ###
#
os.environ["MUJOCO_GL"] = "egl"

#
import mujoco
from mujoco import viewer as viewer_  # type: ignore
#
import xml.etree.ElementTree as ET
#
import numpy as np
from numpy.typing import NDArray
#
from tqdm import tqdm

#
### Graphics and plotting. ###
#
import mediapy as media
#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import multiprocessing as mp
from multiprocessing import Process, Pipe
import gymnasium as gym
from gymnasium import spaces
import cv2



def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return Vec3(roll_x, pitch_y, yaw_z)

@dataclass
class SimulationConfig:
    corridor_length: float = 100.0
    corridor_width: float = 3.0
    robot_view_range: float = 3.0
    max_steps: int = 2000
    env_precision: float = 0.1

@dataclass
class RobotConfig:
    xml_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "four_wheels_robot.xml")

@dataclass
class RewardConfig:
    goal: float = 100.0
    collision: float = -0.05
    straight_line_penalty: float = 0.5
    straight_line_dist: float = 15.0

@dataclass
class TrainingConfig:
    agent_type: str = "ppo"
    max_episodes: int = 1000
    model_path: str = "ppo_robot_corridor.pth"
    load_weights_from: Optional[str] = None
    learning_rate: float = 0.02
    gamma: float = 0.99
    k_epochs: int = 4
    eps_clip: float = 0.2
    update_timestep: int = 4000
    gae_lambda: float = 0.95

@dataclass
class SimConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @staticmethod
    def load(path: str) -> "SimConfig":
        if not os.path.exists(path):
            print(f"Config file {path} not found, using defaults.")
            return SimConfig()
        with open(path, 'r') as f:
            try:
                data = yaml.safe_load(f)
                # Handle nested loading
                sim_config = SimulationConfig(**data.get('simulation', {}))
                robot_config = RobotConfig(**data.get('robot', {}))
                reward_config = RewardConfig(**data.get('rewards', {}))
                train_config = TrainingConfig(**data.get('training', {}))
                return SimConfig(
                    simulation=sim_config,
                    robot=robot_config,
                    rewards=reward_config,
                    training=train_config
                )
            except Exception as e:
                print(f"Error loading config: {e}, using defaults.")
                return SimConfig()


def worker(remote: Any, parent_remote: Any, env_fn_wrapper: Any) -> None:
    parent_remote.close()
    try:
        env: CorridorEnv = env_fn_wrapper()
    except Exception as e:
        print(f"Worker initialization failed: {e}")
        import traceback
        traceback.print_exc()
        parent_remote.close()
        return
    while True:
        try:
            cmd: str
            data: Any
            cmd, data = remote.recv()
            if cmd == 'step':
                ob: Any
                reward: float
                terminated: bool
                truncated: bool
                info: dict[str, Any]
                ob, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    ob, _ = env.reset()
                remote.send((ob, reward, terminated, truncated, info))
            elif cmd == 'reset':
                ob, info = env.reset()
                remote.send(ob)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError
        except Exception as e:
            print(f"Worker error: {e}")
            import traceback
            traceback.print_exc()
            import sys
            sys.stdout.flush()
            break



class SubprocVecEnv:
    def __init__(self, env_fns: list[Any]) -> None:
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting: bool = False
        self.closed: bool = False
        nenvs: int = len(env_fns)
        self.remotes: list[Any]
        self.work_remotes: list[Any]
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps: list[Process] = [Process(target=worker, args=(work_remote, remote, env_fn))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions: Any) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self) -> tuple[Any, Any, Any, Any, Any]:
        results: list[Any] = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, terms, truncs, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(terms), np.stack(truncs), infos

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, Any]:
        self.step_async(actions)
        return self.step_wait()

    def reset(self) -> Any:
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


#
class Vec3:

    #
    def __init__(self, x: float, y: float, z: float) -> None:

        #
        self.x: float = x
        self.y: float = y
        self.z: float = z


#
### Custom Value or Interval Type. ###
#
class ValType:

    #
    def __init__(
        self,
        from_value: float | tuple[float, float]
    ) -> None:

        #
        self.value: float | tuple[float, float] = from_value

    #
    def get_value(self) -> float:

        #
        return random.uniform(*self.value) if isinstance(self.value, tuple) else self.value

    #
    def get_max_value(self) -> float:

        #
        return max(*self.value) if isinstance(self.value, tuple) else self.value


#
### Point 2d ###
#
class Point2d:

    #
    def __init__(self, x: float, y: float) -> float:

        #
        self.x: float = x
        self.y: float = y

    #
    def dist(self, p: "Point2d") -> float:

        #
        return (p.x - self.x) ** 2 + (p.y - self.y) ** 2

    #
    def __add__(self, p: "Point2d") -> "Point2d":

        #
        return Point2d(x=self.x+p.x, y=self.y+p.y)

    #
    def __substract__(self, p: "Point2d") -> "Point2d":

        #
        return Point2d(x=self.x-p.x, y=self.y-p.y)


#   
### Rect 2d ###
#
class Rect2d:

    #
    def __init__(self, corner_top_left: Point2d, corner_bottom_right: Point2d, height: float = 0) -> float:

        #
        self.corner_top_left: Point2d = corner_top_left
        self.corner_bottom_right: Point2d = corner_bottom_right
        self.corner_top_right: Point2d = Point2d(x=corner_bottom_right.x, y=corner_top_left.y)
        self.corner_bottom_left: Point2d = Point2d(x=corner_top_left.x, y=corner_bottom_right.y)
        self.height: float = height

    #
    def point_inside(self, p: Point2d) -> bool:

        #
        return (p.x >= self.corner_top_left.x and p.x <= self.corner_top_right) and (p.y >= self.corner_top_left.y and p.y <= self.corner_bottom_left.y)
        return (p.x >= self.corner_top_left.x and p.x <= self.corner_top_right.x) and (p.y >= self.corner_top_left.y and p.y <= self.corner_bottom_left.y)

    #
    def rect_collision(self, r: "Rect2d") -> bool:

        #
        return (
            self.point_inside(r.corner_top_left) or \
            self.point_inside(r.corner_top_right) or \
            self.point_inside(r.corner_bottom_left) or \
            self.point_inside(r.corner_bottom_right)
        ) or \
        (
            r.point_inside(self.corner_top_left) or \
            r.point_inside(self.corner_top_right) or \
            r.point_inside(self.corner_bottom_left) or \
            r.point_inside(self.corner_bottom_right)
        )

#
### Efficient Collision System Between Rectangles. ###
#
class EfficientCollisionSystemBetweenEnvAndAgent:

    #
    def __init__(
        self,
        environment_obstacles: list[Rect2d],
        env_bounds: Rect2d,
        env_precision: float = 0.1
    ) -> None:

        #
        self.env_precision: float = env_precision
        self.env_bounds: Rect2d = env_bounds

        #
        self.env_matrix: NDArray[np.float64] = np.zeros(
            (
                int((env_bounds.corner_bottom_right.x - env_bounds.corner_top_left.x) / env_precision),
                int((env_bounds.corner_bottom_right.y - env_bounds.corner_top_left.y) / env_precision)
            ),
            dtype=np.float64
        )

        #
        for rect in environment_obstacles:
            #
            start_x: int = int((rect.corner_top_left.x - env_bounds.corner_top_left.x) / env_precision)
            end_x: int = int((rect.corner_bottom_right.x - env_bounds.corner_top_left.x) / env_precision)
            start_y: int = int((rect.corner_top_left.y - env_bounds.corner_top_left.y) / env_precision)
            end_y: int = int((rect.corner_bottom_right.y - env_bounds.corner_top_left.y) / env_precision)
            
            # Clip to bounds
            start_x = max(0, start_x)
            end_x = min(self.env_matrix.shape[0], end_x)
            start_y = max(0, start_y)
            end_y = min(self.env_matrix.shape[1], end_y)
            
            self.env_matrix[start_x:end_x, start_y:end_y] = rect.height

    #
    def get_robot_vision_and_state(
        self,
        robot_pos: Vec3,
        robot_rot: Vec3,
        robot_speed: Vec3,
        previous_action: NDArray[np.float64],
        robot_view_range: float,
    ) -> NDArray[np.float64]:

        #
        ### Return a matrix or a vector of the environment seen by the robot and agent state. ###
        ### This data will be used as the observation space for the agent. ###
        #

        #
        ### 1. Get robot position in the grid. ###
        #
        robot_grid_x: int = int((robot_pos.x - self.env_bounds.corner_top_left.x) / self.env_precision)
        robot_grid_y: int = int((robot_pos.y - self.env_bounds.corner_top_left.y) / self.env_precision)

        #
        ### 2. Determine the range of indices to extract. ###
        #
        view_range_grid: int = int(robot_view_range / self.env_precision)
        
        #
        start_x: int = robot_grid_x - view_range_grid
        end_x: int = robot_grid_x + view_range_grid
        start_y: int = robot_grid_y - view_range_grid
        end_y: int = robot_grid_y + view_range_grid
        
        #
        ### Calculate vision size dynamically. ###
        #
        vision_width = end_x - start_x
        vision_height = end_y - start_y
        #
        # print(f"Vision size: {vision_width}x{vision_height} = {vision_width * vision_height}")

        #
        ### 3. Extract the sub-matrix. ###
        #
        # Create a zero-filled matrix of the desired size
        vision_matrix_size_x: int = end_x - start_x
        vision_matrix_size_y: int = end_y - start_y
        vision_matrix: NDArray[np.float64] = np.zeros((vision_matrix_size_x, vision_matrix_size_y), dtype=np.float64)

        # Calculate overlap with the environment matrix
        env_max_x, env_max_y = self.env_matrix.shape
        
        # Intersection in environment coordinates
        inter_start_x: int = max(0, start_x)
        inter_end_x: int = min(env_max_x, end_x)
        inter_start_y: int = max(0, start_y)
        inter_end_y: int = min(env_max_y, end_y)

        # If there is an overlap, copy the data
        if inter_start_x < inter_end_x and inter_start_y < inter_end_y:
            # Calculate where to paste in the vision matrix
            paste_start_x: int = inter_start_x - start_x
            paste_end_x: int = paste_start_x + (inter_end_x - inter_start_x)
            paste_start_y: int = inter_start_y - start_y
            paste_end_y: int = paste_start_y + (inter_end_y - inter_start_y)
            
            vision_matrix[paste_start_x:paste_end_x, paste_start_y:paste_end_y] = \
                self.env_matrix[inter_start_x:inter_end_x, inter_start_y:inter_end_y]

        #
        ### 4. Flatten vision and concatenate with state. ###
        #
        state_vector: NDArray[np.float64] = np.array([
            robot_pos.x, robot_pos.y, robot_pos.z,
            robot_rot.x, robot_rot.y, robot_rot.z,
            robot_speed.x, robot_speed.y, robot_speed.z,
            previous_action[0], previous_action[1], previous_action[2], previous_action[3]
        ], dtype=np.float64)

        return np.concatenate((vision_matrix.flatten(), state_vector))


#
viewer: Any = cast(Any, viewer_)  # fix to remove pylance type hinting errors with mujoco.viewer stubs errors

#
CTRL_SAVE_PATH: str = "saved_control.json"

#
RENDER_WIDTH: int = 1440
RENDER_HEIGHT: int = 1024

#
GENERATE_CORRIDOR_PARAM: dict[str, Any] = {
    "corridor_length": ValType(200.0),
    "corridor_width": ValType(4.0),
    "obstacles_mode": "sinusoidal",
    "obstacles_mode_param": {
        "obstacle_sep": ValType( 10.0 ),
        "obstacle_size_x": ValType( 0.4 ),
        "obstacle_size_y": ValType( 0.4 ),
        "obstacle_size_z": ValType( 0.2 ),
    }
}


#
def create_geom(name: str, geom_type: str, pos: Vec3, size: Vec3) -> ET.Element:

    #
    new_geom = ET.Element('geom')
    #
    new_geom.set('name', name)
    new_geom.set('type', geom_type)
    new_geom.set('size', f'{size.x} {size.y} {size.z}')
    new_geom.set('pos', f'{pos.x} {pos.y} {pos.z}')

    #
    return new_geom


#
class Corridor:

    #
    def __init__(self) -> None:

        #
        self.xml_file_path: str = "corridor_3x100.xml"  # "corridor_custom.xml"

        # OBJECTIVE WILL BE TO HAVE A CUSTOM AND VERY EFFICIENT TOP DOWN 2D RECTANGULAR VISION SYSTEM FOR THE ROBOT AGENT


    #
    def extract_corridor_from_xml(self) -> dict[str, Any]:
        """
        Extract robot components from existing XML file.
        This teaches students how to parse and reuse XML components.
        """

        #
        print(f"Extracting corridor components from {self.xml_file_path}...")

        #
        ### Parse the existing XML file. ###
        #
        tree: ET.ElementTree[ET.Element] = ET.parse(self.xml_file_path)
        root: ET.Element = tree.getroot()

        #
        ### Extract useful components. ###
        #
        components: dict[str, Optional[Any]] = {
            'compiler': None,
            'option': None,
            'default': None,
            'asset': None,
            'body': None
        }

        #
        comps_body: list[ET.Element] = []

        #
        ## Add global scene ground. ##
        #
        floor_geom = ET.Element('geom')
        #
        floor_geom.set('name', 'global_floor')
        floor_geom.set('type', 'plane')
        floor_geom.set('size', '200 200 0.1')
        floor_geom.set('pos', '0 0 0.12')
        #
        comps_body.append(floor_geom)

        #
        ### Find and extract each component. ###
        #
        for child in root:
            #
            if child.tag == 'compiler':
                #
                components['compiler'] = child
            #
            elif child.tag == 'option':
                #
                components['option'] = child
            #
            elif child.tag == 'default':
                #
                components['default'] = child
            #
            elif child.tag == 'asset':
                #
                components['asset'] = child
            #
            elif child.tag == 'worldbody':
                #
                for body in child:
                    #
                    comps_body.append(body)
                #
                components["body"] = comps_body
            #
            elif child.tag == 'actuator':
                #
                components['actuators'] = child

        #
        return components

    #
    def generate_corridor(
        self,
        corridor_length: ValType = ValType(100.0),              # Value, or random in [min_value, max_value]
        corridor_width: ValType = ValType(3.0),                 # Value, or random in [min_value, max_value]
        obstacles_mode: str = "none",                           # 'none', 'sinusoidal`, `double_sinusoidal`, `random`
        obstacles_mode_param: Optional[dict[str, Any]] = None,  # specific parameters for `obstacles_mode`, ex: `obstacle_separation` for `sinusoidal` mode
        corridor_shift_x: float = -25.0,
    ) -> dict[str, Any]:

        #
        ### Generate useful components. ###
        #
        components: dict[str, Optional[Any]] = {
            'compiler': None,
            'option': None,
            'default': None,
            'asset': None,
            'body': []
        }

        #
        components_body: list[ET.Element] = []
        environment_rects: list[Rect2d] = []

        #
        ### Generate a corridor based on parameters: ###
        #

        #
        obstacles_mode_param_: dict[str, Any] = {} if obstacles_mode_param is None else obstacles_mode_param

        #
        corridor_length_: float = corridor_length.get_value()
        corridor_width_: float  = corridor_width.get_value()

        #
        wall_width: float = 0.3
        wall_height: float = 10.0

        #
        ## Add global scene ground. ##
        #
        components_body.append(
            create_geom(
                name="global_floor",
                geom_type="plane",
                size=Vec3(x = corridor_length_ * 10.0, y = corridor_length_ * 10.0, z = 1.0),
                pos=Vec3(x = 0, y = 0, z = 0)
            )
        )

        #
        ## Add the four walls. ##
        #
        ## Wall 1 (Negative Y side) ##
        #
        wall_1_x: float = corridor_length_
        wall_1_y_center: float = -corridor_width_
        wall_1_half_x: float = corridor_length_ - corridor_shift_x
        wall_1_half_y: float = wall_width
        wall_1_height: float = wall_height

        #
        components_body.append(
            create_geom(
                name="wall_1",
                geom_type="box",
                size=Vec3(x = wall_1_half_x, y = wall_1_half_y, z = wall_1_height),
                pos=Vec3(x = wall_1_x, y = wall_1_y_center, z = wall_1_height)
            )
        )
        #
        environment_rects.append(
            Rect2d(
                corner_top_left=Point2d(x = wall_1_x - wall_1_half_x, y = wall_1_y_center - wall_1_half_y),
                corner_bottom_right=Point2d(x = wall_1_x + wall_1_half_x, y = wall_1_y_center + wall_1_half_y),
                height=wall_1_height * 2 # The size is half-extent, height is full extent
            )
        )
        
        #
        ## Wall 2 (Positive Y side) ##
        #
        wall_2_x: float = corridor_length_
        wall_2_y_center: float = +corridor_width_
        #
        components_body.append(
            create_geom(
                name="wall_2",
                geom_type="box",
                size=Vec3(x = wall_1_half_x, y = wall_1_half_y, z = wall_1_height),
                pos=Vec3(x = wall_2_x, y = wall_2_y_center, z = wall_1_height)
            )
        )
        #
        environment_rects.append(
            Rect2d(
                corner_top_left=Point2d(x = wall_2_x - wall_1_half_x, y = wall_2_y_center - wall_1_half_y),
                corner_bottom_right=Point2d(x = wall_2_x + wall_1_half_x, y = wall_2_y_center + wall_1_half_y),
                height=wall_1_height * 2
            )
        )

        #
        ## Wall 3 (Negative X side) ##
        #
        wall_3_x_center: float = corridor_shift_x
        wall_3_y: float = 0
        wall_3_half_x: float = wall_width
        wall_3_half_y: float = corridor_width_
        wall_3_height: float = wall_height

        #
        components_body.append(
            create_geom(
                name="wall_3",
                geom_type="box",
                size=Vec3(x = wall_3_half_x, y = wall_3_half_y, z = wall_3_height),
                pos=Vec3(x = wall_3_x_center, y = wall_3_y, z = wall_3_height)
            )
        )
        #
        environment_rects.append(
            Rect2d(
                corner_top_left=Point2d(x = wall_3_x_center - wall_3_half_x, y = wall_3_y - wall_3_half_y),
                corner_bottom_right=Point2d(x = wall_3_x_center + wall_3_half_x, y = wall_3_y + wall_3_half_y),
                height=wall_3_height * 2
            )
        )

        #
        ## Wall 4 (Positive X side) ##
        #
        wall_4_x_center: float = 2 * corridor_length_ - corridor_shift_x
        wall_4_y: float = 0
        wall_4_half_x: float = wall_width
        wall_4_half_y: float = corridor_width_
        wall_4_height: float = wall_height

        #
        components_body.append(
            create_geom(
                name="wall_4",
                geom_type="box",
                size=Vec3(x = wall_4_half_x, y = wall_4_half_y, z = wall_4_height),
                pos=Vec3(x = wall_4_x_center, y = wall_4_y, z = wall_4_height)
            )
        )
        #
        environment_rects.append(
            Rect2d(
                corner_top_left=Point2d(x = wall_4_x_center - wall_4_half_x, y = wall_4_y - wall_4_half_y),
                corner_bottom_right=Point2d(x = wall_4_x_center + wall_4_half_x, y = wall_4_y + wall_4_half_y),
                height=wall_4_height * 2
            )
        )

        #
        ## Add the obstacles ##
        #

        #
        ## Sinusoidal mode. ##
        #
        if obstacles_mode == "sinusoidal":

            #
            obs_size_x: ValType = cast(ValType, obstacles_mode_param_.get("obstacle_size_x", 0.2))  # type: ignore
            obs_size_y: ValType = cast(ValType, obstacles_mode_param_.get("obstacle_size_y", 0.2))  # type: ignore
            obs_size_z: ValType = cast(ValType, obstacles_mode_param_.get("obstacle_size_z", 0.2))  # type: ignore
            obs_sep: ValType = cast(ValType, obstacles_mode_param_.get("obstacle_sep", 0.2))  # type: ignore

            #
            obs_length: float = obs_size_x.get_max_value() + obs_sep.get_max_value()

            #
            nb_obstacles: int = max( floor(corridor_length_ / obs_length) + 1, 1)

            #
            current_obs_x: float = 2 + obs_sep.get_value()

            #
            obs_idx: int
            #
            for obs_idx in range(nb_obstacles):

                #
                current_obs_size_x: float = obs_size_x.get_value()
                current_obs_size_y: float = obs_size_y.get_value()
                current_obs_size_z: float = obs_size_z.get_value()

                #
                current_obs_y: float = sin( float(obs_idx) * 0.16 * pi ) * ( corridor_width_ - 2 * current_obs_size_y )
                current_obs_z: float = 0

                #
                components_body.append(
                    create_geom(
                        name=f"obstacle_{obs_idx}",
                        geom_type="box",
                        size=Vec3(x = current_obs_size_x, y = current_obs_size_y, z = current_obs_size_z),
                        pos=Vec3(x = current_obs_x, y = current_obs_y, z = current_obs_z)
                    )
                )

                environment_rects.append(
                    Rect2d(
                        corner_top_left=Point2d(x = current_obs_x - current_obs_size_x, y = current_obs_y - current_obs_size_y),
                        corner_bottom_right=Point2d(x = current_obs_x + current_obs_size_x, y = current_obs_y + current_obs_size_y),
                        height=current_obs_size_z * 2 # Size is half-extent, height is full extent
                    )
                )

                #
                current_obs_x += current_obs_size_x + obs_sep.get_max_value()


        #
        ## Double sinusoidal mode. ##
        #
        elif obstacles_mode == "double_sinusoidal":

            #
            pass


        #
        ## Random mode. ##
        #
        elif obstacles_mode == "random":

            #
            pass


        #
        components['body'] = components_body
        components['environment_rects'] = environment_rects

        #
        return components


#
class Robot:

    #
    def __init__(self, config: SimConfig = SimConfig()) -> None:

        #
        self.xml_file_path: str = config.robot.xml_path

        #
        ### Parse the existing XML file. ###
        #
        self.tree: ET.ElementTree[ET.Element] = ET.parse(self.xml_file_path)
        #
        self.root: ET.Element = self.tree.getroot()

    #
    def extract_robot_from_xml(self) -> dict[str, Any]:
        """
        Extract robot components from existing XML file.
        This teaches students how to parse and reuse XML components.
        """

        #
        print(f"Extracting robot components from {self.xml_file_path}...")

        #
        ### Parse the existing XML file. ###
        #
        tree: ET.ElementTree[ET.Element] = ET.parse(self.xml_file_path)
        root: ET.Element = tree.getroot()

        #
        ### Extract useful components. ###
        #
        components: dict[str, Optional[Any]] = {
            'compiler': None,
            'option': None,
            'default': None,
            'asset': None,
            'robot_body': None,
            'actuators': None
        }

        #
        ### Find and extract each component. ###
        #
        for child in root:
            #
            if child.tag == 'compiler':
                #
                components['compiler'] = child
            #
            elif child.tag == 'option':
                #
                components['option'] = child
            #
            elif child.tag == 'default':
                #
                components['default'] = child
            #
            elif child.tag == 'asset':
                #
                components['asset'] = child
            #
            elif child.tag == 'worldbody':
                #
                ### Extract the robot body from worldbody. ###
                #
                for body in child:
                    #
                    if body.get('name') == 'robot':
                        #
                        components['robot_body'] = body
            #
            elif child.tag == 'actuator':
                #
                components['actuators'] = child

        #
        return components

    #
    def create_enhanced_materials(self) -> dict[str, ET.Element]:
        """
        Create enhanced materials with textures and visual appeal.
        Returns a list of material elements.
        """

        #
        ### Enhanced robot materials. ###
        #
        robot_materials: list[dict[str, str]] = [
            {
                'name': 'mat_chassis_violet',
                'texture': 'tex_chassis',
                'rgba': '0.39 0.13 0.63 1',  # Nice violet color
                'shininess': '0.3',
                'specular': '0.5'
            },
            {
                'name': 'mat_wheel_black',
                'texture': 'tex_wheel_radius',
                'rgba': '0.1 0.1 0.1 1',
                'shininess': '0.6',
                'specular': '0.3'
            }
        ]

        #
        ### Initialize Materials Containers. ###
        #
        materials: dict[str, ET.Element] = {}

        #
        ### Create material elements. ###
        #
        for mat_config in robot_materials:
            #
            material: ET.Element = ET.Element('material')
            #
            mat_name: str = mat_config['name']
            #
            material.set('name', mat_name)
            material.set('rgba', mat_config['rgba'])
            #
            if 'texture' in mat_config:
                #
                material.set('texture', mat_config['texture'])
            #
            if 'shininess' in mat_config:
                #
                material.set('shininess', mat_config['shininess'])
            #
            if 'specular' in mat_config:
                #
                material.set('specular', mat_config['specular'])
            #
            if 'texrepeat' in mat_config:
                #
                material.set('texrepeat', mat_config['texrepeat'])

            #
            materials[mat_name] = material

        #
        return materials

    #
    def create_textures(self) -> dict[str, ET.Element]:
        """
        Create texture elements for enhanced visuals.
        Returns a list of texture elements.
        """

        #
        textures: dict[str, ET.Element] = {}

        #
        ### Wheel radius texture to show rotation. ###
        #
        wheel_texture = ET.Element('texture')
        #
        wheel_texture.set('name', 'tex_wheel_radius')
        wheel_texture.set('type', '2d')
        wheel_texture.set('builtin', 'checker')
        wheel_texture.set('rgb1', '0.8 0.2 0.2')  # Red spoke
        wheel_texture.set('rgb2', '0.1 0.1 0.1')  # Black tire
        wheel_texture.set('width', '32')
        wheel_texture.set('height', '32')
        wheel_texture.set('mark', 'random')
        wheel_texture.set('markrgb', '0.8 0.8 0.2')  # Yellow marks
        #
        textures["tex_wheel_radius"] = wheel_texture


        wheel_texture = ET.Element('texture')
        #
        wheel_texture.set('name', 'tex_chassis')
        wheel_texture.set('type', '2d')
        wheel_texture.set('builtin', 'gradient')
        wheel_texture.set('rgb1', '0.39 0.13 0.63')  # Red spoke
        wheel_texture.set('rgb2', '0 0 0')  # Black tire
        wheel_texture.set('width', '32')
        wheel_texture.set('height', '32')
        # wheel_texture.set('mark', 'random')
        # wheel_texture.set('markrgb', '1 1 1')  # Yellow marks
        #
        textures["tex_chassis"] = wheel_texture

        #
        return textures

    #
    def enhance_robot_visuals(self, robot_body: Optional[ET.Element]) -> None:
        """
        Enhance robot body with better materials and visual features.
        Also adjusts wheel positions to be outside the robot body.
        Modifies the robot body XML element in place.
        """

        #
        if robot_body is None:
            #
            return

        #
        print("Enhancing robot visuals...")

        #
        ### Find and update chassis material. ###
        #
        for geom in robot_body.iter('geom'):
            #
            if 'chassis' in geom.get('name', ''):
                #
                geom.set('material', 'mat_chassis_violet')
                #
                print("  Updated chassis color")

        #
        ### Find and update wheel materials (positions now defined in XML). ###
        #
        wheel_count: int = 0

        #
        for body in robot_body.iter('body'):
            #
            body_name: str = body.get('name', '')
            #
            if body_name.startswith('wheel_'):
                #
                ### Update wheel material. ###
                #
                for geom in body.iter('geom'):
                    #
                    if geom.get('name', '').startswith('geom_'):
                        #
                        geom.set('material', 'mat_wheel_black')
                        #
                        wheel_count += 1

                #
                ### Log the wheel position (already set in XML). ###
                #
                current_pos = body.get('pos', '0 0 0')
                #
                print(f"  {body_name} at position: {current_pos}")

        #
        print(f"  Updated {wheel_count} wheels with textured black material")
        print("  Wheel positions defined in XML (no runtime modification)")


#
class RootWorldScene:

    #
    def __init__(self, config: Optional[SimConfig] = None) -> None:

        #
        self.corridor: Corridor = Corridor()
        #
        self.robot: Robot = Robot(config=config) if config else Robot()

        #
        self.mujoco_model: mujoco.MjModel
        self.mujoco_data: mujoco.MjData
        
        self.environment_rects: list[Rect2d] = []
        self.env_bounds: Rect2d = Rect2d(Point2d(0, -10), Point2d(100, 10)) # Approximate bounds, will be updated


    #
    def build_combined_model(
        self,
        robot_components: dict[str, Optional[ET.Element]],
        corridor_components: dict[str, Optional[ET.Element]],
        floor_type: str = "standard",
        robot_height: float = 1.0
    ) -> mujoco.MjModel:

        """
        Build a complete MuJoCo model by combining robot components with programmatic floor.
        Environment controls physics settings (gravity, timestep, etc).

        Args:
            robot_components: Extracted robot parts from XML
            corridor_components: Extracted corridor parts from XML
            floor_type: Type of floor to create ("standard", "ice", "sand")
            robot_height: Starting height of robot above floor (meters)
        """

        #
        print("Building combined model...")
        print(f"Robot starting height: {robot_height}m above floor")

        #
        ### Create root mujoco element. ###
        #
        root: ET.Element = ET.Element('mujoco')
        #
        root.set('model', 'robot_with_programmatic_floor')

        #
        g: ET.Element = ET.Element('global')
        g.set("offwidth", str(RENDER_WIDTH))
        g.set("offheight", str(RENDER_HEIGHT))
        #
        visual: ET.Element = ET.Element("visual")
        #
        visual.append(g)
        #
        root.append(visual)

        #
        ### Add compiler settings from robot XML. ###
        #
        if robot_components['compiler'] is not None:
            #
            root.append(robot_components['compiler'])

        #
        ### CREATE ENVIRONMENT-CONTROLLED PHYSICS SETTINGS. ###
        #
        option: ET.Element = ET.Element('option')
        #
        option.set('timestep', '0.01')
        option.set("gravity", "0 0 -0.20")
        option.set('solver', 'Newton')
        option.set('iterations', '500')
        #
        root.append(option)
        #
        print(f"  Environment physics: gravity enabled ({option.get("gravity")}), timestep={option.get("timestep")}s")

        #
        ### Add size settings. ###
        #
        size: ET.Element = ET.Element('size')
        #
        size.set('njmax', '1000')
        size.set('nconmax', '500')
        #
        root.append(size)

        #
        if robot_components['default'] is not None:
            #
            root.append(robot_components['default'])

        #
        ### Create asset section with textures and enhanced materials. ###
        #
        asset: ET.Element = ET.Element('asset')

        #
        ### Add textures first. ###
        #
        _texture_id: str
        texture: ET.Element
        #
        for _texture_id, texture in self.robot.create_textures().items():
            #
            asset.append(texture)

        #
        ### Add enhanced materials. ###
        #
        _material_id: str
        material: ET.Element
        #
        enhanced_materials: dict[str, ET.Element] = self.robot.create_enhanced_materials()
        #
        materials_used: set[str] = set()
        #
        for _material_id, material in enhanced_materials.items():
            #
            asset.append(material)
            #
            materials_used.add(_material_id)

        #
        ### Also keep any original materials from robot XML if they don't exist. ###
        #
        if robot_components['asset'] is not None:
            #
            for original_material in robot_components['asset']:
                #
                ### Only add if it's not already in our enhanced materials. ###
                #
                material_name = original_material.get('name', '')
                #
                if material_name not in enhanced_materials and material_name not in materials_used:
                    #
                    asset.append(original_material)
                    #
                    materials_used.add(material_name)

        #
        ### Also keep any original materials from robot XML if they don't exist. ###
        #
        if corridor_components['asset'] is not None:
            #
            for original_material in corridor_components['asset']:
                #
                ### Only add if it's not already in our enhanced materials. ###
                #
                material_name = original_material.get('name', '')
                #
                if material_name not in enhanced_materials and material_name not in materials_used:
                    #
                    asset.append(original_material)
                    #
                    materials_used.add(material_name)

        #
        root.append(asset)

        #
        ### Create worldbody with corridor and robot. ###
        #
        worldbody: ET.Element = ET.Element('worldbody')

        #
        ### Add all the corridor worldbody to this worldbody. ###
        #
        if corridor_components['body'] is not None:
            #
            body: ET.Element
            #
            for body in corridor_components['body']:
                #
                worldbody.append(body)

        #
        ### Add robot body with enhanced visuals and adjusted height. ###
        #
        if robot_components['robot_body'] is not None:

            #
            ### Enhance the robot's visual appearance. ###
            #
            self.robot.enhance_robot_visuals(robot_components['robot_body'])

            #
            ### Adjust robot starting height (floor is at -0.1, so robot center should be at robot_height - 0.1). ###
            #
            robot_z_position: float = robot_height - 0.1 + 10  # Floor offset
            current_pos: str = robot_components['robot_body'].get('pos', '0 0 0.2')
            #
            pos_parts: list[str] = current_pos.split()
            #
            if len(pos_parts) == 3:
                #
                new_pos: str = f"{pos_parts[0]} {pos_parts[1]} {robot_z_position}"
                #
                robot_components['robot_body'].set('pos', new_pos)
                #
                print(f"  Robot positioned at: {new_pos} (will fall {robot_height}m to floor)")

            #
            robot_components['robot_body'].set('pos', "0 0 5")
            robot_components['robot_body'].set('euler', "0 0 0")

            #
            worldbody.append(robot_components['robot_body'])

        #
        ### Add worldbody to root. ###
        #
        root.append( worldbody )

        #
        ### Add actuators. ###
        #
        if robot_components['actuators'] is not None:
            #
            root.append(robot_components['actuators'])

        #
        ### Convert to XML string. ###
        #
        xml_string = ET.tostring(root, encoding='unicode')

        #
        ### Create and return MuJoCo model. ###
        #
        return mujoco.MjModel.from_xml_string(xml_string)


    #
    def construct_scene(
        self,
        config: SimConfig,
        floor_type: str = "standard",
        robot_height: float = 1.0
    ) -> None:

        #
        ### Build combined model with enhanced visuals and physics. ###
        #
        corridor_components = self.corridor.generate_corridor(
            corridor_length=ValType(config.simulation.corridor_length),
            corridor_width=ValType(config.simulation.corridor_width),
            obstacles_mode="sinusoidal",
            obstacles_mode_param=GENERATE_CORRIDOR_PARAM["obstacles_mode_param"]
        )
        
        #
        ### Store rects. ##
        ##
        if 'environment_rects' in corridor_components:
            #
            self.environment_rects = cast(list[Rect2d], corridor_components['environment_rects'])
  
        #
        ### Large bounds for the scene. ###
        #
        margin = 50.0
        self.env_bounds = Rect2d(
            Point2d(-margin, -config.simulation.corridor_width - margin), 
            Point2d(config.simulation.corridor_length + margin, config.simulation.corridor_width + margin)
        )
        
        #
        self.mujoco_model = self.build_combined_model(
            robot_components=self.robot.extract_robot_from_xml(),
            corridor_components=corridor_components,
            floor_type=floor_type,
            robot_height=robot_height
        )
        #
        self.mujoco_data = mujoco.MjData(self.mujoco_model)


#
class TrackRobot:

    #
    def __init__(
        self,
        mujoco_data_scene: mujoco.MjData,
    ) -> None:

        #
        self.mjdata: mujoco.MjData = mujoco_data_scene

        #
        self.robot_posx_track: list[float] = []
        self.robot_posy_track: list[float] = []
        self.robot_posz_track: list[float] = []

        #
        self.robot_velx_track: list[float] = []
        self.robot_vely_track: list[float] = []
        self.robot_velz_track: list[float] = []

    #
    def track(self) -> None:

        #
        self.robot_posx_track.append( self.mjdata.qpos[0] )
        self.robot_posy_track.append( self.mjdata.qpos[1] )
        self.robot_posz_track.append( self.mjdata.qpos[2] )

        #
        self.robot_velx_track.append( self.mjdata.qvel[0] )
        self.robot_vely_track.append( self.mjdata.qvel[1] )
        self.robot_velz_track.append( self.mjdata.qvel[2] )

    #
    def plot_tracking(self) -> None:

        #
        time_range: list[int] = list(range(len(self.robot_posx_track)))

        #
        plt.plot(time_range, self.robot_posx_track, label="pos x")  # type: ignore
        plt.plot(time_range, self.robot_posy_track, label="pos y")  # type: ignore
        plt.plot(time_range, self.robot_posz_track, label="pos z")  # type: ignore
        #
        plt.plot(time_range, self.robot_velx_track, label="vel x")  # type: ignore
        plt.plot(time_range, self.robot_vely_track, label="vel y")  # type: ignore
        plt.plot(time_range, self.robot_velz_track, label="vel z")  # type: ignore
        #
        plt.legend()  # type: ignore
        plt.show()  # type: ignore


#
class Physics:

    #
    def __init__(
        self,
        mujoco_model_scene: mujoco.MjModel,
        mujoco_data_scene: mujoco.MjData,
        air_drag_coefficient: float = 0.5,
        robot_deceleration_force: float = 0.01
    ) -> None:

        #
        self.model_scene: mujoco.MjModel = mujoco_model_scene
        self.data_scene: mujoco.MjData = mujoco_data_scene

        #
        self.air_drag_coefficient: float = air_drag_coefficient

        #
        self.robot_deceleration_force: float = robot_deceleration_force
        self.robot_deceleration_factor: float = 1.0 - self.robot_deceleration_force

        #
        ### Initialize simulation with zero velocities for stability. ###
        #
        self.data_scene.qvel[:] = 0  # Zero all velocities
        self.data_scene.qacc[:] = 0  # Zero all accelerations

        #
        ### Initialize wheels speed. ###
        #
        self.robot_wheels_speed: NDArray[np.float64] = np.zeros((4,), dtype=np.float64)

    #
    def apply_air_resistance(self) -> None:
        """
        Applies air drag to all bodies based on their velocity.
        F_drag = -k * v^2 * sign(v)
        """

        #
        for i in range(self.model_scene.nbody):

            #
            ### Skip world body. ###
            #
            if i == 0:
                #
                continue

            #
            ### Get linear velocity of body i. ###
            #
            ### translational velocity. ###
            #
            vel = self.data_scene.cvel[i][:3]
            #
            speed = np.linalg.norm(vel)
            #
            if speed < 1e-6:
                #
                continue

            #
            ### Simple quadratic drag model: F = -k * v * |v|. ###
            #
            F_drag = -self.air_drag_coefficient * speed * vel

            #
            ### Allocate full-sized qfrc_target vector (size nv). ###
            #
            qfrc_target = np.zeros(self.model_scene.nv)

            #
            ### Apply the force at the center of mass. ###
            #
            mujoco.mj_applyFT(
                m=self.model_scene,
                d=self.data_scene,
                force=F_drag,
                torque=np.zeros(3),
                point=self.data_scene.xpos[i],
                body=i,
                qfrc_target=qfrc_target
            )

    #
    def apply_additionnal_physics(self) -> None:

        #
        self.apply_air_resistance()

        #
        self.apply_robot_deceleration()

        #
        self.set_robot_wheel_speeds()

    #
    def apply_robot_ctrl_movement(
        self,
        acceleration_factor: float = 0.0,
        rotation_factor: float = 0.0,
        acceleration_force: float = 0.15,
        rotation_force: float = 0.05,
        decceleration_factor: float = 1.0,
        max_front_wheel_speeds: float = 200.0,
        max_back_wheel_speeds: float = 100,
    ) -> None:

        #
        if acceleration_factor > 0:
            #
            self.robot_wheels_speed[0] += acceleration_factor * acceleration_force
            self.robot_wheels_speed[1] += acceleration_factor * acceleration_force
            self.robot_wheels_speed[2] += acceleration_factor * acceleration_force * 0.2
            self.robot_wheels_speed[3] += acceleration_factor * acceleration_force * 0.2
        #
        else:
            #
            self.robot_wheels_speed[0] += acceleration_factor * acceleration_force * 0.2
            self.robot_wheels_speed[1] += acceleration_factor * acceleration_force * 0.2
            self.robot_wheels_speed[2] += acceleration_factor * acceleration_force
            self.robot_wheels_speed[3] += acceleration_factor * acceleration_force

        #
        self.robot_wheels_speed[0] -= rotation_factor * rotation_force
        self.robot_wheels_speed[1] += rotation_factor * rotation_force
        self.robot_wheels_speed[2] -= rotation_factor * rotation_force
        self.robot_wheels_speed[3] += rotation_factor * rotation_force

        #
        if decceleration_factor < 1.0:
            #
            self.robot_wheels_speed[:] *= decceleration_factor

        #
        ### Clamp values. ###
        #
        self.robot_wheels_speed[0:2] = np.clip(self.robot_wheels_speed[0:2], -max_front_wheel_speeds, max_front_wheel_speeds)
        self.robot_wheels_speed[2:5] = np.clip(self.robot_wheels_speed[2:5], -max_back_wheel_speeds, max_back_wheel_speeds)

    #
    def apply_robot_deceleration(self) -> None:

        #
        self.robot_wheels_speed[:] *= self.robot_deceleration_factor

    #
    def set_robot_wheel_speeds(self) -> None:

        #
        ### This will set the actuator values. ###
        #
        self.data_scene.ctrl = self.robot_wheels_speed


#
class Camera:

    #
    def __init__(self) -> None:

        #
        ### Modes: "free", "follow_robot", "top_down" ###
        #
        self.current_mode: str = "free"

        #
        ### Store robot ID to avoid looking it up every frame. ###
        #
        self.robot_id: int = -1

    #
    def set_mode(self, mode: str) -> None:
        #
        if mode in ["free", "follow_robot", "top_down"]:
            #
            self.current_mode = mode

    #
    def update_viewer_camera(
        self,
        cam: Any,  # viewer_instance.cam
        model: mujoco.MjModel,
        data: mujoco.MjData
    ) -> None:

        #
        ### Get robot ID once. ###
        #
        if self.robot_id == -1:
            #
            self.robot_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "robot"
            )
            #
            if self.robot_id == -1:
                #
                print("Warning: Could not find robot body named 'robot'")
                #
                return

        #
        ### Handle camera logic based on mode. ###
        #
        if self.current_mode == "free":
            #
            ### Set camera to free mode and do nothing else. ###
            #
            ### This lets the user control it manually with the mouse. ###
            #
            if cam.type != mujoco.mjtCamera.mjCAMERA_FREE:
                #
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        #
        elif self.current_mode == "follow_robot":
            #
            ### Use MuJoCo's built-in tracking camera. ###
            #
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.robot_id
            #
            ### You can set preferred distance and elevation. ###
            #
            cam.distance = 5.0
            cam.elevation = -16
            cam.azimuth = 1.01171875
            #

        #
        elif self.current_mode == "top_down":
            #
            ### Use tracking camera, but point straight down. ###
            #
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.robot_id
            #
            cam.distance = 10.0      # Zoom out
            cam.azimuth = 90.0       # Face "north"
            cam.elevation = -89.9    # Look almost straight down

    #
    def get_camera_data(self, cam: Any) -> tuple[ NDArray[np.float64], float, float, float, int, Any ]:

        #
        return (cam.lookat, cam.distance, cam.azimuth, cam.elevation, cam.trackbodyid, cam.type)


#
class Controls:

    #
    def __init__(
        self,
        physics: Physics,
        camera: Camera,
        render_mode: bool = False
    ) -> None:

        #
        self.physics: Physics = physics
        self.camera: Camera = camera

        #
        self.render_mode: bool = render_mode

        #
        self.quit_requested: bool = False
        self.display_camera_info: bool = False

        #
        self.key_pressed: set[int] = set()

        #
        self.controls_history: dict[ str, list[int] ] = {}

        #
        self.current_frame: int = 0

        #
        self.easy_control: set[int] = {32, 262, 263, 264, 265}

    #
    def new_frame(self, cam: Any) -> None:

        #
        self.current_frame += 1

    #
    def apply_controls_each_frame_render_mode(self) -> None:

        #
        if str(self.current_frame) in self.controls_history:
            #
            for k in self.controls_history[str(self.current_frame)]:
                #
                self.key_callback(keycode=k, render_mode=True)


    #
    def apply_controls_each_frame(self) -> None:

        #
        if self.render_mode:
            #
            self.apply_controls_each_frame_render_mode()

        #
        ### Robot Movements. ###
        #
        ### Forward. ###
        #
        if 265 in self.key_pressed:
            #
            self.physics.apply_robot_ctrl_movement(acceleration_factor=1.0)
        #
        ### Turn Left. ###
        #
        elif 263 in self.key_pressed:
            #
            self.physics.apply_robot_ctrl_movement(rotation_factor=1.0)
        #
        ### Turn Right. ###
        #
        elif 262 in self.key_pressed:
            #
            self.physics.apply_robot_ctrl_movement(rotation_factor=-1.0)
        #
        ### Backward. ###
        #
        elif 264 in self.key_pressed:
            #
            self.physics.apply_robot_ctrl_movement(acceleration_factor=-1.0)
        #
        ### Space Key, robot stops. ###
        #
        elif 32 in self.key_pressed:
            #
            self.physics.apply_robot_ctrl_movement(decceleration_factor=0.2)

    #
    def key_callback(self, keycode: int, render_mode: bool = False) -> None:

        #
        if not render_mode:
            #
            if not str(self.current_frame) in self.controls_history:
                #
                self.controls_history[str(self.current_frame)] = [keycode]
            #
            else:
                #
                self.controls_history[str(self.current_frame)].append(keycode)

            #
            ### Display Camera Informations. ###
            #
            if keycode == ord('c') or keycode == ord('C'):
                #
                self.display_camera_info = True

            #
            ### Save the control history. ###
            #
            elif keycode == ord('s') or keycode == ord('S'):
                #
                with open(CTRL_SAVE_PATH, "w", encoding="utf-8") as f:
                    #
                    json.dump(self.controls_history, f)
                #
                print(f"Saved control history at path : `{CTRL_SAVE_PATH}` !")

            #
            ### Quit, with 'Q' or 'Esc' Keys. ###
            #
            elif keycode == ord('q') or keycode == ord('Q'):
                #
                self.quit_requested = True
            #
            elif keycode == 256:
                #
                self.quit_requested = True

        #
        if self.render_mode and not render_mode:
            #
            return

        #
        ### Camera Mode Switching. ###
        #
        if keycode == ord('1'):
            #
            self.camera.set_mode("free")
        #
        elif keycode == ord('2'):
            #
            self.camera.set_mode("follow_robot")
        #
        elif keycode == ord('3'):
            #
            self.camera.set_mode("top_down")

        #
        else:
            #
            if keycode in self.key_pressed:
                #
                self.key_pressed.remove(keycode)
            #
            else:
                #
                self.key_pressed.add(keycode)

            #
            if keycode in self.easy_control:
                #
                for kk in list(self.key_pressed):
                    #
                    if kk != keycode and kk in self.easy_control:
                        #
                        self.key_pressed.remove(kk)


#
class State:

    #
    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        physics: Physics,
        camera: Camera,
        controls: Controls,
        viewer_instance: Any,
        robot_track: TrackRobot,
    ) -> None:

        #
        self.mj_model: mujoco.MjModel = mj_model
        self.mj_data: mujoco.MjData = mj_data
        self.physics: Physics = physics
        self.camera: Camera = camera
        self.controls: Controls = controls
        self.viewer_instance: Any = viewer_instance
        #
        self.robot_track: TrackRobot = robot_track
        #
        self.quit: bool = False

        #
        self.init_state()

    #
    def init_state(self) -> None:

        #
        ### Configure camera parameters with your good settings. ###
        #

        #
        ### Camera parameters. ###
        #
        self.viewer_instance.cam.type = mujoco.mjtCamera.mjCAMERA_FREE   # Free camera
        self.viewer_instance.cam.fixedcamid = -1                         # Not using fixed camera

        #
        ## Set good camera position and orientation parameters. ###
        #
        self.viewer_instance.cam.azimuth = 1.01171875
        self.viewer_instance.cam.elevation = -16.6640625
        self.viewer_instance.cam.lookat = np.array(
            [ 1.55633679e-04, -4.88295545e-02,  1.05485916e+00]
        )


#
class CorridorEnv(gym.Env):

    #
    def __init__(self, config: SimConfig = SimConfig(), render_mode: bool = False) -> None:

        #
        self.config = config
        self.render_mode = render_mode
        self.root_scene = RootWorldScene(config=config)
        self.root_scene.construct_scene(config=config, floor_type="standard", robot_height=1.0)
        
        #
        self.model = self.root_scene.mujoco_model
        self.data = self.root_scene.mujoco_data
        
        #
        self.physics = Physics(self.model, self.data)
        
        #
        ### Initialize collision/vision system. ###
        #
        self.collision_system = EfficientCollisionSystemBetweenEnvAndAgent(
            environment_obstacles=self.root_scene.environment_rects,
            env_bounds=self.root_scene.env_bounds,
            env_precision=config.simulation.env_precision
        )
        
        #
        ### Calculate observation space size. ###
        #
        view_range_grid = int(config.simulation.robot_view_range / config.simulation.env_precision)
        vision_width = 2 * view_range_grid
        vision_height = 2 * view_range_grid
        self.vision_size = vision_width * vision_height
        self.state_dim = self.vision_size + 13 # 13 for state vector
        
        #
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        #
        ### Variables for straight line penalty. ###
        #
        self.straight_start_x: float = 0.0
        self.straight_y: float = 0.0
        
        #
        self.previous_action = np.zeros(4, dtype=np.float64)

    #
    def set_collision_system(self, collision_system: EfficientCollisionSystemBetweenEnvAndAgent):
        #
        self.collision_system = collision_system

    #
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)

        #
        mujoco.mj_resetData(self.model, self.data)
        #
        self.physics.data_scene.qvel[:] = 0
        self.physics.data_scene.qacc[:] = 0
        self.physics.robot_wheels_speed[:] = 0
        
        #
        ### Set robot to start position. ###
        #
        robot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        #
        if robot_id != -1:
            #
            ### Lift it up a bit to avoid stuck in floor. ###
            #
            self.data.xpos[robot_id][2] = 0.2
            
        #
        ### Initial observation. ###
        #
        
        #
        ### Reset straight line tracking. ###
        #
        robot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        self.straight_start_x = self.data.xpos[robot_id][0]
        self.straight_y = self.data.xpos[robot_id][1]
        
        #
        self.previous_action = np.zeros(4, dtype=np.float64)

        return self.get_observation(), {}

    #
    def step(self, action):

        #
        ### Action is 4 continuous values for wheel speeds ###
        ### Clip action to reasonable range ###
        #
        action = np.clip(action, -1.0, 1.0)
        
        #
        ### Map action to wheel speeds (e.g. max speed 200). ###
        #
        max_speed = 200.0
        target_speeds = action * max_speed
        
        #
        ### Apply to physics. ###
        #
        self.physics.robot_wheels_speed[:] = target_speeds
        
        #
        ### Step physics. ###
        #
        self.physics.apply_additionnal_physics()
        mujoco.mj_step(self.model, self.data)
        
        #
        ### Update previous action. ###
        #
        self.previous_action = action
        
        #
        ### Observation. ##
        ##
        obs = self.get_observation()
        
        #
        ### Reward ###
        ### 1. Progress reward (Continuous) ###
        ### Use velocity in X direction for smoother continuous reward ###
        #
        robot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        x_vel = self.data.cvel[robot_id][0]
        x_pos = self.data.xpos[robot_id][0]
        y_pos = self.data.xpos[robot_id][1]
        
        #
        ### Reward is proportional to forward velocity ###
        ### Increased weight for x_vel from 1.0 to 2.0 to incentivize speed ###
        #
        # reward = x_vel * 2.0
        # reward = (x_pos - 20) * 2.0 # OLD REWARD causing negative scores
        
        #
        ### NEW REWARD: Velocity based to avoid negative accumulation ###
        #
        reward = x_vel * 2.0
        
        #
        reward += (x_pos) / 1000.0

        #
        ### 2. Survival reward (optional) ###
        ### Increased to encourage staying alive longer ###
        #
        # reward += 0.1

        #
        ### 3. Center corridor reward ###
        ### Penalize for deviating from y=0 ###
        ### Reduced weight from 0.1 to 0.05 per user plan to avoid suicidal behavior ###
        #
        # reward -= 0.05 * abs(y_pos)

        #
        ### 4. Straight line penalty. ###
        ### Penalize if staying in same Y radius for too long (> 15m). ###
        #
        y_tolerance: float = 0.5
        if abs(y_pos - self.straight_y) < y_tolerance:
            #
            dist_straight: float = x_pos - self.straight_start_x
            #
            if dist_straight > self.config.rewards.straight_line_dist:
                #
                reward -= self.config.rewards.straight_line_penalty
        else:
            #
            ### Reset tracking if moved out of Y radius. ###
            #
            self.straight_start_x = x_pos
            self.straight_y = y_pos
        
        #
        ### Done condition. ###
        #
        terminated = False
        truncated = False
        
        #
        ### Bounded environment checks (User requested margins: x < -100, |y| > 40). ###
        ### If driving backwards too far. ###
        #
        if self.data.xpos[robot_id][0] < -self.config.simulation.corridor_length:
            #
            truncated = True
        
        #
        ### If wandering off sideways too far. ###
        #
        if abs(self.data.xpos[robot_id][1]) > self.config.simulation.corridor_width + 10.0: # Margin
            #
            truncated = True
            
        #
        ### If fell off (z check). ###
        #
        if self.data.xpos[robot_id][2] < -5.0:
            #
            terminated = True
        
        #
        ### If reached end (e.g. 100m). ###
        #
        if self.data.xpos[robot_id][0] > self.config.simulation.corridor_length:
            #
            terminated = True
            #
            reward += self.config.rewards.goal
            
        #
        return obs, reward, terminated, truncated, {}

    #
    def get_observation(self):

        #
        if self.collision_system is None:
            #
            ### Fallback. ###
            #
            return np.zeros(48)
        
        #
        robot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        #
        pos = Vec3(self.data.xpos[robot_id][0], self.data.xpos[robot_id][1], self.data.xpos[robot_id][2])
        
        # Get rotation from quaternion
        quat = self.data.xquat[robot_id]
        rot = quaternion_to_euler(quat[1], quat[2], quat[3], quat[0]) # Mujoco quat is w, x, y, z
        
        vel = Vec3(self.data.cvel[robot_id][0], self.data.cvel[robot_id][1], self.data.cvel[robot_id][2])
        # acc = Vec3(self.data.cacc[robot_id][0], self.data.cacc[robot_id][1], self.data.cacc[robot_id][2])

        #
        return self.collision_system.get_robot_vision_and_state(pos, rot, vel, self.previous_action, robot_view_range=self.config.simulation.robot_view_range)


#
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, vision_shape: tuple[int, int], state_vector_dim: int = 13, action_std_init: float = 0.6) -> None:
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)
        self.vision_shape = vision_shape
        self.state_vector_dim = state_vector_dim
        self.vision_size = vision_shape[0] * vision_shape[1]

        # CNN for Vision
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, vision_shape[0], vision_shape[1])
            cnn_out_size = self.cnn(dummy_input).shape[1]

        # Encoder for State Vector
        self.state_encoder = nn.Sequential(
            nn.Linear(state_vector_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Fusion Dimension
        fusion_dim = cnn_out_size + 64

        # Actor Head
        self.actor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

        # Critic Head
        self.critic = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def _process_input(self, state: torch.Tensor):
        # Split input into vision and state vector
        # state shape: (batch_size, vision_size + state_vector_dim)
        
        vision_flat = state[:, :self.vision_size]
        state_vec = state[:, self.vision_size:]

        # Reshape vision to (batch_size, 1, height, width)
        vision_img = vision_flat.view(-1, 1, self.vision_shape[0], self.vision_shape[1])

        # Encode
        vision_features = self.cnn(vision_img)
        state_features = self.state_encoder(state_vec)

        # Fuse
        return torch.cat((vision_features, state_features), dim=1)

    def act(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._process_input(state)
        action_mean = self.actor(features)
        
        action_std = torch.sqrt(self.action_var).to(state.device)
        dist = Normal(action_mean, action_std)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(axis=-1)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._process_input(state)
        action_mean = self.actor(features)
        
        action_std = torch.sqrt(self.action_var).to(state.device)
        dist = Normal(action_mean, action_std)
        
        action_logprobs = dist.log_prob(action).sum(axis=-1)
        dist_entropy = dist.entropy().sum(axis=-1)
        state_values = self.critic(features)
        
        return action_logprobs, state_values, dist_entropy


#
class PPOAgent:

    #
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        vision_shape: tuple[int, int],
        lr: float = 0.02,
        gamma: float = 0.99,
        K_epochs: int = 4,
        eps_clip: float = 0.2,
        gae_lambda: float = 0.95
    ) -> None:

        #
        self.gamma: float = gamma
        self.eps_clip: float = eps_clip
        self.K_epochs: int = K_epochs
        self.gae_lambda: float = gae_lambda
        
        #
        ### Detect device. ###
        #
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #
        print(f"PPOAgent using device: {self.device}")
        
        #
        self.action_dim: int = action_dim
        #
        self.policy: 'ActorCritic' = cast('ActorCritic', ActorCritic(state_dim, action_dim, vision_shape).float().to(self.device))
        self.optimizer: optim.Adam = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old: 'ActorCritic' = cast('ActorCritic', ActorCritic(state_dim, action_dim, vision_shape).float().to(self.device))
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        #
        self.MseLoss: nn.MSELoss = nn.MSELoss()
        
    #
    def select_action(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        #
        with torch.no_grad():
            #
            state_tensor: torch.Tensor = torch.FloatTensor(state).to(self.device)
            
            # Handle single observation (1D) by adding batch dimension
            is_single_obs = state.ndim == 1
            if is_single_obs:
                state_tensor = state_tensor.unsqueeze(0)
                
            action_tensor, action_logprob_tensor = self.policy_old.act(state_tensor)

        #
        action = action_tensor.cpu().numpy()
        action_logprob = action_logprob_tensor.cpu().numpy()
        
        # If input was single observation, flatten output to match
        if is_single_obs:
            return action.flatten(), action_logprob.flatten()
            
        return action, action_logprob
        
    #
    def update(self, memory: 'Memory', next_state: np.ndarray, next_done: np.ndarray) -> None:

        #
        ### GAE (Generalized Advantage Estimation) ###
        #
        
        # 1. Convert memory to tensors
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.tensor(np.array(memory.logprobs), dtype=torch.float32).to(self.device).detach()
        
        # Flattened versions for training later
        # (T * num_envs, ...)
        old_states_flat = old_states.view(-1, self.policy.vision_size + self.policy.state_vector_dim)
        old_actions_flat = old_actions.view(-1, self.action_dim)
        old_logprobs_flat = old_logprobs.view(-1)
        
        # 2. Get values for all states
        # We process in batches to avoid OOM if needed, but for simplicity we do full batch now
        # shape: (T, num_envs, 1)
        with torch.no_grad():
            # Get values of all current states
            values = self.policy.critic(self.policy._process_input(old_states_flat)).view(len(memory.states), -1)
            
            # Get value of next state (bootstrapping)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            if next_state.ndim == 1:
                next_state_tensor = next_state_tensor.unsqueeze(0)
            next_value = self.policy.critic(self.policy._process_input(next_state_tensor))
            
        # 3. Calculate GAE Advantages
        # rewards shape: list of (num_envs,)
        # is_terminals shape: list of (num_envs,)
        
        rewards = np.array(memory.rewards)      # (T, num_envs)
        is_terminals = np.array(memory.is_terminals) # (T, num_envs)
        
        # Convert to tensor for GAE calculation
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        is_terminals_t = torch.tensor(is_terminals, dtype=torch.float32).to(self.device)
        next_done_t = torch.tensor(next_done, dtype=torch.float32).to(self.device) # (num_envs,)
        
        returns = torch.zeros_like(rewards_t).to(self.device)
        advantages = torch.zeros_like(rewards_t).to(self.device)
        
        gae = 0
        
        num_steps = len(memory.rewards)
        
        for t in range(num_steps - 1, -1, -1):
            
            # If it is the last step
            if t == num_steps - 1:
                next_non_terminal = 1.0 - next_done_t
                next_val = next_value.squeeze()
            else:
                next_non_terminal = 1.0 - is_terminals_t[t + 1]
                next_val = values[t + 1].squeeze()
                
            delta = rewards_t[t] + self.gamma * next_val * next_non_terminal - values[t].squeeze()
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages[t] = gae
            returns[t] = advantages[t] + values[t].squeeze() # Return = Advantage + Value
            
        # Flatten for training
        returns = returns.view(-1)
        advantages = advantages.view(-1)
        
        #
        ### Normalize Advantages (Crucial for PPO stability) ###
        #
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        #
        ### Optimize policy for K epochs. ###
        #
        for _ in range(self.K_epochs):

            #
            logprobs: torch.Tensor
            state_values: torch.Tensor
            dist_entropy: torch.Tensor
            #
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_flat, old_actions_flat)
            
            #
            state_values = torch.squeeze(state_values)
            
            #
            ratios: torch.Tensor = torch.exp(logprobs - old_logprobs_flat)
            
            #
            # Use GAE Advantages here
            surr1: torch.Tensor = ratios * advantages
            surr2: torch.Tensor = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            #
            # Critic loss is MSE(Value, Returns)
            loss: torch.Tensor = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, returns) - 0.01*dist_entropy
            
            #
            self.optimizer.zero_grad()
            loss.mean().backward()
            #
            self.optimizer.step()

        #
        self.policy_old.load_state_dict(self.policy.state_dict())


#
class Memory:

    #
    def __init__(self) -> None:

        #
        self.actions: List[torch.Tensor] = []
        self.states: List[torch.Tensor] = []
        self.logprobs: List[np.ndarray] = []        # Stored as numpy arrays in PPOAgent.update
        self.rewards: List[np.ndarray] = []         # Stored as numpy arrays in PPOAgent.update
        self.is_terminals: List[np.ndarray] = []    # Stored as numpy arrays in PPOAgent.update
    
    #
    def clear_memory(self) -> None:

        #
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


#
def make_env(config: SimConfig = SimConfig()) -> CorridorEnv:
    #
    return cast(CorridorEnv, CorridorEnv(config=config))


#
class Main:

    #
    @staticmethod
    def state_step(state: State) -> State:

        #
        state.robot_track.track()

        #
        state.controls.new_frame(state.viewer_instance.cam)

        #
        state.controls.apply_controls_each_frame()

        #
        ### Print camera info when requested. ###
        #
        if state.controls.display_camera_info:
            #
            print(state.viewer_instance.cam)
            #
            robot_id: int = mujoco.mj_name2id(
                state.mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot"
            )
            #
            pos: float | np.typing.NDArray[np.float32] = state.mj_data.xpos[robot_id]
            quat: float | np.typing.NDArray[np.float32] = state.mj_data.xquat[robot_id]
            #
            print(f"Robot position: {pos}")
            print(f"Robot orientation (quat): {quat}")
            #
            state.controls.display_camera_info = False

        #
        state.physics.apply_additionnal_physics()

        #
        mujoco.mj_step(
            m=state.mj_model,
            d=state.mj_data,
            nstep=1
        )

        #
        state.camera.update_viewer_camera(
            cam=state.viewer_instance.cam,
            model=state.mj_model,
            data=state.mj_data
        )

        #
        state.viewer_instance.sync()

        #
        return state

    #
    @staticmethod
    def main(config: SimConfig = SimConfig(), render_mode: bool = False) -> None:

        #
        root_scene: RootWorldScene = RootWorldScene()
        #
        root_scene.construct_scene(
            config=config,
            floor_type="standard",
            robot_height=1.0
        )

        #
        physics: Physics = Physics(
            mujoco_model_scene=root_scene.mujoco_model,
            mujoco_data_scene=root_scene.mujoco_data
        )

        #
        camera: Camera = Camera()

        #
        controls: Controls = Controls(
            physics=physics,
            camera=camera,
            render_mode=render_mode
        )

        #
        robot_track: TrackRobot = TrackRobot(mujoco_data_scene=root_scene.mujoco_data)

        #
        if render_mode:
            #
            if not os.path.exists(CTRL_SAVE_PATH):
                #
                raise UserWarning(f"Error: there is no saved control files at path `{CTRL_SAVE_PATH}` !")

            #
            with open(CTRL_SAVE_PATH, "r", encoding="utf-8") as f:
                #
                controls.controls_history = json.load(f)

        #
        print("")
        print("Controls:")
        print("  'up arrow': Robot goes forward (key control)")
        print("  'down arrow': Robot goes backward (key control)")
        print("  'left arrow': Robot goes left (key control)")
        print("  'right arrow': Robot goes right (key control)")
        print("  '1': Free camera (mouse control)")
        print("  '2': Follow robot (3rd person)")
        print("  '3': Top-down camera")
        print("  'c' or 'C': Print camera parameters")
        print("  'q' or 'Q': Quit")
        print("  ESC: Quit")
        print("")

        #
        viewer_instance: Any

        #
        with viewer.launch_passive(
                root_scene.mujoco_model,
                root_scene.mujoco_data,
                key_callback=controls.key_callback
            ) as viewer_instance:

            #
            state: State = State(
                mj_model=root_scene.mujoco_model,
                mj_data=root_scene.mujoco_data,
                physics=physics,
                camera=camera,
                controls=controls,
                viewer_instance=viewer_instance,
                robot_track=robot_track
            )

            #
            ### Mainloop. ###
            #
            target_dt: float = 1.0 / 240.0
            #
            while viewer_instance.is_running() and not state.controls.quit_requested:

                #
                start_time = time.time()

                #
                state = Main.state_step(state)

                #
                ### Sync with real time. ###
                #
                elapsed = time.time() - start_time
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

        #
        robot_track.plot_tracking()


    #
    @staticmethod
    def main_video_render() -> None:

        #
        root_scene: RootWorldScene = RootWorldScene()
        #
        root_scene.construct_scene(
            floor_type="standard",
            robot_height=1.0
        )

        #
        physics: Physics = Physics(
            mujoco_model_scene=root_scene.mujoco_model,
            mujoco_data_scene=root_scene.mujoco_data
        )

        #
        camera: Camera = Camera()

        #
        controls: Controls = Controls(
            physics=physics,
            camera=camera,
            render_mode=True
        )

        #
        if not os.path.exists(CTRL_SAVE_PATH):
            #
            raise UserWarning(f"Error: there is no saved control files at path `{CTRL_SAVE_PATH}` !")

        #
        with open(CTRL_SAVE_PATH, "r", encoding="utf-8") as f:
            #
            controls.controls_history = json.load(f)

        #
        ### Create a camera. ###
        #
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)

        #
        ### Camera parameters. ###
        #
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE   # Free camera
        cam.fixedcamid = -1                         # Not using fixed camera

        #
        ## Set good camera position and orientation parameters. ###
        #
        cam.azimuth = 1.01171875
        cam.elevation = -16.6640625
        cam.lookat = np.array(
            [ 1.55633679e-04, -4.88295545e-02,  1.05485916e+00]
        )


        #
        ### Set up parameters. ###
        #
        framerate: int = 60  # (Hz)
        #
        skip_factor: int = framerate
        #
        n_frames: int = max([int(f_id) for f_id in controls.controls_history.keys()])

        #
        ### Simulate and display video. ###
        #
        mujoco.mj_resetData(root_scene.mujoco_model, root_scene.mujoco_data)

        #
        with media.VideoWriter('rendered_video.mp4', shape=(RENDER_HEIGHT, RENDER_WIDTH), fps=framerate) as writer:
            #
            with mujoco.Renderer(root_scene.mujoco_model, RENDER_HEIGHT, RENDER_WIDTH) as renderer:  # type: ignore

                #
                for frame_idx in tqdm(range(min(2000000, n_frames))):

                    #
                    controls.new_frame(cam)

                    #
                    controls.apply_controls_each_frame()

                    #
                    ### Quit if requested. ###
                    #
                    if controls.quit_requested:
                        #
                        break

                    #
                    physics.apply_additionnal_physics()

                    #
                    mujoco.mj_step(
                        m=root_scene.mujoco_model,
                        d=root_scene.mujoco_data,
                        nstep=1
                    )

                    #
                    camera.update_viewer_camera(
                        cam=cam,
                        model=root_scene.mujoco_model,
                        data=root_scene.mujoco_data
                    )

                    #
                    if frame_idx % skip_factor == 0:

                        #
                        renderer.update_scene(root_scene.mujoco_data, cam)  # type: ignore
                        #
                        pixels: NDArray[np.int8] = renderer.render()  # type: ignore
                        #
                        writer.add_image(pixels)  # type: ignore




    #
    @staticmethod
    def train(
        config: SimConfig = SimConfig(),
        max_episodes: Optional[int] = None,
        max_timesteps: Optional[int] = None,
        update_timestep: Optional[int] = None,
        lr: Optional[float] = None,
        gamma: Optional[float] = None,
        K_epochs: Optional[int] = None,
        eps_clip: Optional[float] = None,
        model_path: Optional[str] = None,
        load_weights_from: Optional[str] = None
    ) -> None:

        #
        ### Set defaults from config if not provided. ###
        #
        max_episodes = max_episodes if max_episodes is not None else config.training.max_episodes
        max_timesteps = max_timesteps if max_timesteps is not None else config.simulation.max_steps
        update_timestep = update_timestep if update_timestep is not None else config.training.update_timestep
        lr = lr if lr is not None else config.training.learning_rate
        gamma = gamma if gamma is not None else config.training.gamma
        K_epochs = K_epochs if K_epochs is not None else config.training.k_epochs
        eps_clip = eps_clip if eps_clip is not None else config.training.eps_clip
        model_path = model_path if model_path is not None else config.training.model_path
        load_weights_from = load_weights_from if load_weights_from is not None else config.training.load_weights_from

        #
        print("Starting training...")
        
        #
        ### Determine number of environments. ###
        #
        num_envs: int = min(8, mp.cpu_count())
        #
        print(f"Using {num_envs} parallel environments.")

        #   
        #   
        from functools import partial
        envs = SubprocVecEnv([partial(make_env, config=config) for _ in range(num_envs)])
        
        #
        # state_dim = 3612 # Vision (60x60=3600) + State (12)
        #
        ### Calculate state dim dynamically. ###
        #
        view_range_grid = int(config.simulation.robot_view_range / config.simulation.env_precision)
        vision_width = 2 * view_range_grid
        vision_height = 2 * view_range_grid
        vision_size = vision_width * vision_height
        state_dim = vision_size + 13
        #
        action_dim = 4
        
        #
        agent = PPOAgent(state_dim, action_dim, (vision_width, vision_height), lr=config.training.learning_rate, gamma=config.training.gamma, K_epochs=config.training.k_epochs, eps_clip=config.training.eps_clip, gae_lambda=config.training.gae_lambda)
        memory = Memory()
        
        #
        print_running_reward = 0
        print_running_episodes = 0
        
        #
        time_step = 0
        i_episode = 0
        
        #
        ### Initial reset. ###
        #
        state = envs.reset()

        #
        ### Load weights if requested. ###
        #
        if load_weights_from and os.path.exists(load_weights_from):

            #
            print(f"Loading initial weights from {load_weights_from}...")

            #
            try:

                #
                agent.policy.load_state_dict(torch.load(load_weights_from, map_location=agent.device))
                #
                agent.policy_old.load_state_dict(agent.policy.state_dict())
                #
                print("Weights loaded successfully.")

            #
            except Exception as e:
                #
                print(f"Error loading weights: {e}")

        #
        ### Initialize best reward tracking. ###
        #
        best_reward = -float('inf')
        best_model_path = model_path.replace(".pth", "_best.pth")
        best_model_info_path = model_path.replace(".pth", "_best.json")
        
        #
        ### Try to load previous best score if exists. ###
        #
        if os.path.exists(best_model_info_path):

            #
            try:

                #
                with open(best_model_info_path, 'r') as f:
                    #
                    info = json.load(f)
                    best_reward = info.get('best_reward', -float('inf'))

                #
                print(f"Resuming with previous best reward record: {best_reward}")

            #
            except Exception as e:
                #
                print(f"Could not load previous best info: {e}")
        
        #
        while i_episode < max_episodes:

            #   
            ### Run for update_timestep steps (total across all envs) ###
            ### We want to collect update_timestep transitions. ###
            ### Each step collects num_envs transitions. ###
            ### So we run for update_timestep // num_envs iterations. ###
            #
            steps_per_update: int = update_timestep // num_envs
            
            #
            for t in range(steps_per_update):

                #   
                ### Select action. ###
                #
                action, action_logprob = agent.select_action(state)
                
                #
                ### Step. ###
                #
                next_state, reward, terminated, truncated, _ = envs.step(action)
                done = np.logical_or(terminated, truncated)
                
                #
                ### Save to memory. ###
                #
                memory.states.append(torch.FloatTensor(state))
                memory.actions.append(torch.FloatTensor(action))
                memory.logprobs.append(action_logprob)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                
                #
                state = next_state
                
                #
                time_step += num_envs
                
                #
                ### Track rewards (approximate for printing). ###
                #
                print_running_reward += np.sum(reward)
                #
                ### We count "episodes" as number of done flags. ###
                #
                print_running_episodes += np.sum(done)
            
            #
            ### Update PPO. ###
            #
            # PASS CURRENT STATE (which is next_state of the last step) AND DONE FLAGS FOR BOOTSTRAPPING
            agent.update(memory, state, done)
            memory.clear_memory()
            time_step = 0
            
            #
            i_episode += 1
            
            #
            if i_episode % 10 == 0:

                #
                ### Avoid division by zero. ###
                #
                if print_running_episodes == 0:
                    #
                    print_running_episodes = 1
                
                #
                ### Also calculate avg reward per step to detect if episodes are just getting very long ###
                ### We know exactly how many steps passed: `steps_per_update * 10` ###
                #
                total_steps_in_log_interval = steps_per_update * 10 
                avg_reward = print_running_reward / print_running_episodes
                avg_reward_per_step = print_running_reward / total_steps_in_log_interval
                
                #
                print(f"Update {i_episode} \t Avg Reward/Episode: {avg_reward:.2f} \t Avg Reward/Step: {avg_reward_per_step:.4f} \t Total Episodes: {print_running_episodes}")
                #
                print_running_reward = 0
                print_running_episodes = 0
                
                #
                ### Save latest model. ###
                #
                torch.save(agent.policy.state_dict(), model_path)
                
                #
                ### Check for best model. ###
                #
                if avg_reward > best_reward:
                    #
                    best_reward = avg_reward
                    #
                    print(f"New best reward: {best_reward:.2f}! Saving best model...")
                    #
                    torch.save(agent.policy.state_dict(), best_model_path)
                    
                    #
                    ### Save metadata. ###
                    #
                    with open(best_model_info_path, 'w') as f:
                        #
                        json.dump({'best_reward': best_reward, 'episode': i_episode}, f)
        
        #
        envs.close()
                
    #
    @staticmethod
    def play(config: SimConfig = SimConfig(), model_path: Optional[str] = None, live_vision: bool = False):

        #
        path = model_path if model_path else config.training.model_path
        print(f"Playing with model: {path}")
        
        #
        ### Setup scene. ###
        #
        #
        root_scene = RootWorldScene()
        root_scene.construct_scene(config=config, floor_type="standard", robot_height=1.0)
        
        #
        physics = Physics(root_scene.mujoco_model, root_scene.mujoco_data)
        camera = Camera()
        controls = Controls(physics, camera, render_mode=False)
        robot_track = TrackRobot(root_scene.mujoco_data)
        
        #
        ### Setup Agent. ###
        #
        ### Setup Agent. ###
        #
        # state_dim = 3612
        #
        ### Calculate state dim dynamically. ###
        #
        view_range_grid = int(config.simulation.robot_view_range / config.simulation.env_precision)
        vision_width = 2 * view_range_grid
        vision_height = 2 * view_range_grid
        vision_size = vision_width * vision_height
        state_dim = vision_size + 13
        #
        action_dim = 4
        agent = PPOAgent(state_dim, action_dim, (vision_width, vision_height), lr=config.training.learning_rate, gamma=config.training.gamma, K_epochs=config.training.k_epochs, eps_clip=config.training.eps_clip, gae_lambda=config.training.gae_lambda)
        
        #
        try:
            #
            ### Load model to cpu first then move to device. ###
            #
            agent.policy.load_state_dict(torch.load(path, map_location=agent.device))
            #
            ### Set to eval mode. ###
            #
            agent.policy.eval()
            #
            print("Model loaded successfully.")
        #
        except Exception as e:
            #
            print(f"Error loading model: {e}")
            #
            return

        #
        ### Setup Collision System for Vision. ###
        #
        collision_system = EfficientCollisionSystemBetweenEnvAndAgent(
            environment_obstacles=root_scene.environment_rects,
            env_bounds=root_scene.env_bounds,
            env_precision=config.simulation.env_precision
        )
        
        #
        ### Calculate observation space size. ###
        #
        view_range_grid = int(config.simulation.robot_view_range / config.simulation.env_precision)
        vision_width = 2 * view_range_grid
        vision_height = 2 * view_range_grid
        vision_size = vision_width * vision_height
        state_dim = vision_size + 13 # 13 for state vector
        
        #
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        #
        ### Helper to get observation. ###
        #
        def get_observation(previous_action):

            #
            robot_id = mujoco.mj_name2id(root_scene.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "robot")
            #
            pos = Vec3(root_scene.mujoco_data.xpos[robot_id][0], root_scene.mujoco_data.xpos[robot_id][1], root_scene.mujoco_data.xpos[robot_id][2])
            
            # Get rotation from quaternion
            quat = root_scene.mujoco_data.xquat[robot_id]
            rot = quaternion_to_euler(quat[1], quat[2], quat[3], quat[0]) # Mujoco quat is w, x, y, z
            
            vel = Vec3(root_scene.mujoco_data.cvel[robot_id][0], root_scene.mujoco_data.cvel[robot_id][1], root_scene.mujoco_data.cvel[robot_id][2])
            # acc = Vec3(root_scene.mujoco_data.cacc[robot_id][0], root_scene.mujoco_data.cacc[robot_id][1], root_scene.mujoco_data.cacc[robot_id][2])

            #
            return collision_system.get_robot_vision_and_state(pos, rot, vel, previous_action, robot_view_range=config.simulation.robot_view_range)

        #
        ### Launch viewer. ###
        #
        with viewer.launch_passive(root_scene.mujoco_model, root_scene.mujoco_data, key_callback=controls.key_callback) as viewer_instance:

            #   
            state_obj = State(
                mj_model=root_scene.mujoco_model,
                mj_data=root_scene.mujoco_data,
                physics=physics,
                camera=camera,
                controls=controls,
                viewer_instance=viewer_instance,
                robot_track=robot_track
            )
            
            #
            ### Initial reset. ###
            #

            ### Lift robot. ###
            #
            robot_id = mujoco.mj_name2id(root_scene.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "robot")
            #
            root_scene.mujoco_data.xpos[robot_id][2] = 0.2
            
            #
            previous_action = np.zeros(4, dtype=np.float64)
            
            #
            while viewer_instance.is_running() and not state_obj.controls.quit_requested:

                #   
                ### Get observation. ###
                #
                obs = get_observation(previous_action)
                
                #
                ### Get action from agent. ###
                #
                action, log_prob = agent.select_action(obs)
                
                #
                ### Apply action. ###
                #
                action = np.clip(action, -1.0, 1.0)
                previous_action = action # Update previous action
                max_speed = 200.0
                target_speeds = action * max_speed
                physics.robot_wheels_speed[:] = target_speeds
                
                #
                ### Step. ###
                #
                state_obj = Main.state_step(state_obj)

                if live_vision:
                    # Extract vision
                    vision_data = obs[:vision_size]
                    vision_img = vision_data.reshape((vision_width, vision_height))
                    
                    # Normalize for display
                    vision_display = cv2.normalize(vision_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    
                    # Resize
                    scale = 10
                    vision_display = cv2.resize(vision_display, (vision_height * scale, vision_width * scale), interpolation=cv2.INTER_NEAREST)
                    
                    # Convert to BGR for colored text
                    vision_display = cv2.cvtColor(vision_display, cv2.COLOR_GRAY2BGR)
                    
                    # Extract state
                    state_vec = obs[vision_size:]
                    
                    info_text = [
                        f"Pos: {state_vec[0]:.2f}, {state_vec[1]:.2f}, {state_vec[2]:.2f}",
                        f"Rot: {state_vec[3]:.2f}, {state_vec[4]:.2f}, {state_vec[5]:.2f}",
                        f"Spd: {state_vec[6]:.2f}, {state_vec[7]:.2f}, {state_vec[8]:.2f}",
                        f"Prv: {state_vec[9]:.2f}, {state_vec[10]:.2f}, {state_vec[11]:.2f}, {state_vec[12]:.2f}"
                    ]
                    
                    for idx, text in enumerate(info_text):
                        cv2.putText(vision_display, text, (10, 20 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                    cv2.imshow("Robot Vision & State", vision_display)
                    cv2.waitKey(1)

            if live_vision:
                cv2.destroyAllWindows()


#
if __name__ == "__main__":

    #
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    #
    parser.add_argument('--render_mode', action="store_true", default=False)
    parser.add_argument('--render_video', action="store_true", default=False)
    parser.add_argument('--train', action="store_true", default=False, help="Train the RL agent")
    parser.add_argument('--play', action="store_true", default=False, help="Play with trained model")
    parser.add_argument('--live_vision', action="store_true", default=False, help="Show live vision window")
    parser.add_argument('--model_path', type=str, default="ppo_robot_corridor.pth", help="Path to model file")
    parser.add_argument('--episodes', type=int, default=1000, help="Number of episodes/updates to train")
    parser.add_argument('--update_timestep', type=int, default=4000, help="Timesteps per update")
    parser.add_argument('--load_weights_from', type=str, default=None, help="Path to load initial weights from")
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to configuration file")
    #
    args: argparse.Namespace = parser.parse_args()

    #
    ### Load config. ###
    #
    config = SimConfig.load(args.config)

    #
    if args.train:

        #
        try:
            #
            mp.set_start_method('spawn', force=True)
        #
        except RuntimeError:
            #
            pass
        
        #
        Main.train(
            config=config
        )

    #
    elif args.play:
        #
        Main.play(config=config, model_path=args.model_path, live_vision=args.live_vision)
    #
    elif args.render_video:
        #
        Main.main_video_render()
    #
    else:
        #
        Main.main(config=config, render_mode=args.render_mode)

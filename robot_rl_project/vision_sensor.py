"""
Vision Sensor using Raycasting for MuJoCo
Replaces box-based collision detection with efficient ray casting
"""
import numpy as np
from numpy.typing import NDArray
import mujoco
import math


class VisionSensor:
    """
    Vision sensor using raycasting around the robot.
    Returns distance array for RL: -1 if no hit, distance if hit.
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_body_name: str = "robot",
        num_rays: int = 16,
        max_distance: float = 5.0,
        height_offset: float = 0.0,
        ray_pattern: str = "circle"
    ) -> None:
        """
        Args:
            model: MuJoCo model
            data: MuJoCo data
            robot_body_name: Name of the robot body to attach sensor to
            num_rays: Number of rays to cast around the robot
            max_distance: Maximum detection distance (meters)
            height_offset: Height offset from robot center (meters)
            ray_pattern: "circle" for horizontal rays, "hemisphere" for 3D coverage
        """
        self.model = model
        self.data = data
        self.num_rays = num_rays
        self.max_distance = max_distance
        self.height_offset = height_offset
        
        # Get robot body ID
        self.robot_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, robot_body_name
        )
        
        if self.robot_body_id == -1:
            raise ValueError(f"Robot body '{robot_body_name}' not found in model")
        
        # Find robot geom IDs to exclude from detection
        self.robot_geom_ids: set[int] = set()
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name and ('chassis' in geom_name or 'wheel' in geom_name):
                self.robot_geom_ids.add(geom_id)
        
        # Generate ray directions based on pattern
        if ray_pattern == "circle":
            self.ray_directions = self._generate_circle_pattern()
        elif ray_pattern == "hemisphere":
            self.ray_directions = self._generate_hemisphere_pattern()
        else:
            raise ValueError(f"Unknown ray pattern: {ray_pattern}")
        
        print(f"Initialized VisionSensor with {num_rays} rays, max distance {max_distance}m")
        print(f"Excluding {len(self.robot_geom_ids)} robot geoms from detection")
    
    def _generate_circle_pattern(self) -> NDArray[np.float64]:
        """
        Generate rays in a horizontal circle around the robot.
        Perfect for 2D navigation tasks.
        """
        directions = []
        angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)
        
        for angle in angles:
            # Horizontal rays (XY plane)
            x = np.cos(angle)
            y = np.sin(angle)
            z = 0.0  # Horizontal
            directions.append([x, y, z])
        
        return np.array(directions, dtype=np.float64)
    
    def _generate_hemisphere_pattern(self) -> NDArray[np.float64]:
        """
        Generate rays in a hemisphere pattern for 3D coverage.
        Useful for flying robots or complex environments.
        """
        directions = []
        
        # Golden spiral for even distribution
        phi = np.pi * (3. - np.sqrt(5.))  # Golden angle
        
        for i in range(self.num_rays):
            # Normalize to hemisphere (y >= 0)
            y_val = i / float(self.num_rays - 1)  # 0 to 1
            radius = np.sqrt(1 - y_val * y_val)
            theta = phi * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            y = y_val
            
            directions.append([x, y, z])
        
        return np.array(directions, dtype=np.float64)
    
    def get_sensor_readings(self) -> NDArray[np.float32]:
        """
        Get distance readings for all rays.
        
        Returns:
            Array of shape (num_rays,) with dtype np.float32
            - Value is -1.0 if no collision detected
            - Value is distance (meters) if collision detected
        """
        # Get robot position and orientation
        robot_pos = self.data.xpos[self.robot_body_id].copy()
        robot_mat = self.data.xmat[self.robot_body_id].reshape(3, 3)
        
        # Apply height offset
        robot_pos[2] += self.height_offset
        
        # Initialize result array
        distances = np.full(self.num_rays, -1.0, dtype=np.float32)
        
        # Cast each ray
        for i, ray_dir in enumerate(self.ray_directions):
            # Transform ray direction to world frame
            world_dir = robot_mat @ ray_dir
            world_dir = world_dir / np.linalg.norm(world_dir)  # Normalize
            
            # Cast ray
            distance = self._cast_single_ray(robot_pos, world_dir)
            
            if distance >= 0:  # Hit detected
                distances[i] = distance
        
        return distances
    
    def _cast_single_ray(
        self,
        start_pos: NDArray[np.float64],
        direction: NDArray[np.float64]
    ) -> float:
        """
        Cast a single ray and return distance to first hit.
        
        Returns:
            -1.0 if no hit, distance if hit
        """
        # Calculate ray endpoint
        end_pos = start_pos + direction * self.max_distance
        
        # MuJoCo ray casting
        # We need to check all geoms except the robot's own geoms
        min_distance = -1.0
        
        for geom_id in range(self.model.ngeom):
            # Skip robot's own geoms
            if geom_id in self.robot_geom_ids:
                continue
            
            # Get geom body
            body_id = self.model.geom_bodyid[geom_id]
            
            # Cast ray using mj_ray
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                pnt=start_pos,
                vec=direction * self.max_distance,
                geomgroup=None,
                flg_static=1,
                bodyexclude=self.robot_body_id,
                geomid=np.array([geom_id], dtype=np.int32)
            )
            
            if distance >= 0:  # Hit detected
                if min_distance < 0 or distance < min_distance:
                    min_distance = distance
        
        return min_distance
    
    def get_visualization_data(self) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
        """
        Get data for visualizing rays (useful for debugging).
        
        Returns:
            robot_pos: Robot position
            ray_endpoints: List of ray endpoint positions
        """
        robot_pos = self.data.xpos[self.robot_body_id].copy()
        robot_pos[2] += self.height_offset
        robot_mat = self.data.xmat[self.robot_body_id].reshape(3, 3)
        
        distances = self.get_sensor_readings()
        ray_endpoints = []
        
        for i, ray_dir in enumerate(self.ray_directions):
            world_dir = robot_mat @ ray_dir
            world_dir = world_dir / np.linalg.norm(world_dir)
            
            if distances[i] > 0:
                # Use actual hit distance
                endpoint = robot_pos + world_dir * distances[i]
            else:
                # Use max distance
                endpoint = robot_pos + world_dir * self.max_distance
            
            ray_endpoints.append(endpoint)
        
        return robot_pos, ray_endpoints


class ImprovedVisionSensor(VisionSensor):
    """
    Optimized version using batch ray casting.
    Significantly faster for many rays.
    """
    
    def get_sensor_readings(self) -> NDArray[np.float32]:
        """
        Optimized batch ray casting.
        """
        # Get robot position and orientation
        robot_pos = self.data.xpos[self.robot_body_id].copy()
        robot_mat = self.data.xmat[self.robot_body_id].reshape(3, 3)
        robot_pos[2] += self.height_offset
        
        # Initialize result
        distances = np.full(self.num_rays, -1.0, dtype=np.float32)
        
        # Transform all rays to world frame at once
        world_dirs = (robot_mat @ self.ray_directions.T).T
        
        # Normalize
        norms = np.linalg.norm(world_dirs, axis=1, keepdims=True)
        world_dirs = world_dirs / norms
        
        # Cast all rays
        for i in range(self.num_rays):
            # Use MuJoCo's optimized ray casting
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                pnt=robot_pos,
                vec=world_dirs[i] * self.max_distance,
                geomgroup=None,
                flg_static=1,
                bodyexclude=self.robot_body_id,
                geomid=None
            )
            
            if distance >= 0:
                distances[i] = distance
        
        return distances

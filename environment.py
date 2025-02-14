import time
import numpy as np
import os
import datetime
import transform_utils as T
import copy
import gymnasium as gym
import robomimic.utils.file_utils as FileUtils
import trimesh
import json
import open3d as o3d
import imageio
from collections import OrderedDict
import omnigibson as og

from omnigibson.utils.usd_utils import PoseAPI, mesh_prim_mesh_to_trimesh_mesh, mesh_prim_shape_to_trimesh_mesh
from omnigibson.robots.fetch import Fetch
from omnigibson.controllers import IsGraspingState
from og_utils import OGCamera
from robomimic.utils.tensor_utils import flatten_nested_dict_list
from collections import OrderedDict
from utils import (
    bcolors,
    get_clock_time,
    angle_between_rotmat,
    angle_between_quats,
    get_linear_interpolation_steps,
    linear_interpolate_poses,
    exec_safe,
    get_callable_grasping_cost_fn,
    load_functions_from_txt
)
from omnigibson.robots.manipulation_robot import ManipulationRobot
from omnigibson.controllers.controller_base import ControlType, BaseController
import torch
# Don't use GPU dynamics and use flatcache for performance boost
 
def calculate_bbox_to_point(bbox, p):
    min_x, max_x = min(bbox[0][0],bbox[1][0]), max(bbox[0][0],bbox[1][0])
    min_y, max_y = min(bbox[0][1],bbox[1][1]), max(bbox[0][1],bbox[1][1])
    min_z, max_z = min(bbox[0][2],bbox[1][2]), max(bbox[0][2],bbox[1][2]) 
    
    dx = max(min_x - p[0], 0, p[0] - max_x)
    dy = max(min_y - p[1], 0, p[1] - max_y)
    dz = max(min_z - p[2], 0, p[2] - max_z)

    return (dx*dx + dy*dy + dz*dz)**.5

def preprocess_obs(d):
    # This function recursively flattens a nested dictionary into a list of tensors
    out = flatten_nested_dict_list(d)
    return torch.cat((out[0][1], out[1][1]))

# some customization to the OG functions
def custom_clip_control(self, control):
    """
    Clips the inputted @control signal based on @control_limits.

    Args:
        control (Array[float]): control signal to clip

    Returns:
        Array[float]: Clipped control signal
    """
    clipped_control = control.clip(
        self._control_limits[self.control_type][0][self.dof_idx],
        self._control_limits[self.control_type][1][self.dof_idx],
    )
    idx = (
        self._dof_has_limits[self.dof_idx]
        if self.control_type == ControlType.POSITION
        else [True] * self.control_dim
    )
    if len(control) > 1:
        control[idx] = clipped_control[idx]
    return control

Fetch._initialize = ManipulationRobot._initialize
BaseController.clip_control = custom_clip_control

class Trajectory:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.terminals = []
        self.agent_infos = []
        self.env_infos = []
        self.next_observations = []
        self.is_grasping_releasing = []

    def __len__(self):
        return len(self.actions)
    
    def add_step(self, o,a,r,t, agent_infos, env_infos, next_o, is_grasping_releasing):
        self.observations.append(o)
        self.actions.append(a)
        self.rewards.append(r)
        self.terminals.append(t)
        self.agent_infos.append(agent_infos)
        self.env_infos.append(env_infos)
        self.next_observations.append(next_o)
        self.is_grasping_releasing.append(is_grasping_releasing)

    
    
    def add_new_traj(self, other_traj):
        other_o, other_a, other_r, other_t, other_ai, other_ei, other_next_o, other_igr = other_traj.get_paths
        self.observations.extend(other_o)
        self.actions.extend(other_a)
        self.rewards.extend(other_r)
        self.terminals.extend(other_t)
        self.agent_infos.extend(other_ai)
        self.env_infos.extend(other_ei)
        self.next_observations.extend(other_next_o)
        self.is_grasping_releasing.extend(other_igr)

    def get_paths(self):
        return (
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.agent_infos,
            self.env_infos,
            self.next_observations,
            self.is_grasping_releasing
        )

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.terminals = []
        self.agent_infos = []
        self.env_infos = []
        self.next_observations = []
        self.is_grasping_releasing = []
    


class CustomOGEnv(og.Environment):
    """
    Custom OmniGibson env that supports
      (1) custom reward functions
      (2) keypoint tracking
    """
    def __init__(self, configs, config, in_vec_env=False, randomize=False, low_dim =True, use_oracle_reward=False):
        
        self.randomize = None
        self.low_dim=low_dim
        self.objs = OrderedDict(
            {'pen_1': {'randomize' : randomize,
                       'obs' : True,
                  'min_bounds':[-0.45, -0.35, 0.775],
                  'max_bounds':[0.05, 0.05, 0.781]},
             
           #  'table_1' : { 'randomize' : False, 'obs':False}
             }
            
             )
        self.robot = None
      
        self.use_oracle_reward=use_oracle_reward
        print(f'Oracle reward {self.use_oracle_reward}')
        super().__init__(configs, in_vec_env)
        self._automatic_reset = True

        self.robot = self.robots[0]
        self.randomize=randomize
        self.in_vec_env = in_vec_env
        self.pen = self.scene.object_registry("name", 'pen_1')
        for _ in range(10):
            og.sim.step()
        self.reward_functions = None
        self.step_counter = 0        
        self.low_dim=low_dim
        self.config = config
        self.action_dim= 7
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        
        
        if not in_vec_env:
            self.scene.update_initial_state()
            obs_size = self.get_low_dim_obs().shape
            self.observation_size = obs_size
                        # Update the initial state of the scene
            self.init_pos = None
            self._initialize_cameras(self.config['camera'])
            shape = self.get_low_dim_obs().shape[0]
            self.observation_space = gym.spaces.Box(low=-np.inf, high = np.inf, shape = (shape,), dtype=np.float32)
        

        
    
    def post_play_load(self):
        super().post_play_load()
        if self.in_vec_env: 
            self.scene.update_initial_state()
            
            if self.low_dim:
                obs_size = self.get_low_dim_obs().shape
                self.observation_size = obs_size
                obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_size, dtype=np.float32)
                self.observation_space = obs_space
                shape = self.get_low_dim_obs().shape[0]
                self.observation_space = gym.spaces.Box(low=-np.inf, high = np.inf, shape = (shape,), dtype=np.float32)

            else:
                raise NotImplementedError()

            # Update the initial state of the scene
            self.action_dim= 7
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
            
            self.init_pos = None
            self._initialize_cameras(self.config['camera'])
            

       
        
    def set_rekep_program_dir(self, rekep_program_dir):
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            
            self.program_info = json.load(f)
        self.constraint_fns = dict()
        for stage in range(1, self.program_info['num_stages'] + 1):  # stage starts with 1
            stage_dict = dict()
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(rekep_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
                get_grasping_cost_fn = get_callable_grasping_cost_fn(self)  # special grasping function for VLM to call
                stage_dict[constraint_type] = load_functions_from_txt(load_path, get_grasping_cost_fn) if os.path.exists(load_path) else []
            self.constraint_fns[stage] = stage_dict

        self.register_keypoints(self.program_info['init_keypoint_positions'])

    def set_reward_function_for_stage(self, stage):
        assert hasattr(self,'constraint_fns'), 'Make sure to set rekep program dir!'
        self.stage = stage
        self.substage_constraints = self.constraint_fns[stage]
        self.grasp_point = self.program_info['grasp_keypoints'][self.stage - 1] 
        self.release_point = self.program_info['release_keypoints'][self.stage - 1] 
        

    def oracle_reward(self):
        rew = 0
        if self.get_done():
            rew += 2.25

        dist = calculate_bbox_to_point(self.get_aabb('pen_1'), self.get_ee_pos())

        reaching_reward = 1 - np.tanh(100* dist)
        reward =  rew +reaching_reward

    
    def get_reward(self):
        if self.use_oracle_reward:
            return self.oracle_reward() 

        assert self.substage_constraints is not None
        # negative subgoal constraints
        # cannot violate path constraints, so make that very negative
        ee_pos = self.get_ee_pos()
        keypoint_pos = self.get_keypoint_positions()

        reward = 0

        for fn in self.substage_constraints['subgoal']:
            reward -= 10*fn(ee_pos, keypoint_pos) 
        for fn in self.substage_constraints['path']:
            reward -= 20*fn(ee_pos, keypoint_pos)  # - constraint violations

        if self.grasp_point != -1 and self.is_grasping(
                                      self.get_object_by_keypoint(self.grasp_point)):
            reward += 2
        
        if self.release_point != -1 and not self.is_grasping(
                                       self.get_object_by_keypoint(self.release_point)):
            reward += 2

        return reward
        

    def step(self, action):
        
        if  (action.shape[-1] == 7):
            a = np.zeros(12)
            a[4:-2]=action[:-1]
            a[-2:] = action[-1]
            action = a

        next_o, reward, done, truncated, info = super().step(action)
        
        reward = self.get_reward()
        
        if self.low_dim:
            next_o  = self.get_low_dim_obs()
        done = self.get_done()
        
        if done:
            print('DONE!')
        self.step_counter +=1

        if (self.step_counter >= 40):
            truncated = not done
            done = True
        return next_o,reward,done,truncated, info

    def render(self):
        cam_obs = self.get_cam_obs()
        rgb = cam_obs[1]['rgb']
        return rgb

    def reset(self, seed=None, objs=None):
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        super().reset(get_obs=False)
        self.init_pos = dict()
        if self.randomize and objs is None:
           # print('randomizing')
            for obj in self.objs:
                if self.objs[obj]['randomize']:
                    current_object = self.scene.object_registry("name", obj)
                    pos,orn = self.get_random_position(current_object, self.objs[obj]['min_bounds'], self.objs[obj]['max_bounds'])         
                    self.init_pos[obj]=np.concatenate((pos, orn))
                    current_object.set_position_orientation(
                        position = pos,
                        orientation = orn
                    ) 
        elif objs is not None:
            for obj in objs:
                current_object = self.scene.object_registry("name", obj)
                current_object.set_position_orientation(
                        position = objs[obj][:3],
                        orientation = objs[obj][3:]
                    ) 

        for i in range (5):
            og.sim.step()
        
        if self.robot is not None:
            self.step_counter=0
            print('resetted')
            return self.get_low_dim_obs(), {}

    def get_done(self):
        return self.is_grasping(self.pen)

    def get_aabb(self, obj_name):
        obj = self.scene.object_registry("name", obj_name)
        return obj.aabb

    def get_centered_ee(self, obj_name):
        
        #gets ee wrt obj_name
        assert self.mesh_prim_path  is not None
        obj = self.scene.object_registry("name", obj_name)
        
        ee = self.get_ee_pos()
      
        curr_pose = T.pose2mat(PoseAPI.get_world_pose(self.mesh_prim_path))
        centering_transform = T.pose_inv(curr_pose)
        

        return np.dot(centering_transform, np.append(ee, 1))[:3]
       
        
  

    def get_mesh(self, obj_name):
        obj = self.scene.object_registry("name", obj_name)
        trimesh_objects = []
        for link in obj.links.values():
            for mesh in link.visual_meshes.values():
                mesh_type = mesh.prim.GetPrimTypeInfo().GetTypeName()
                if mesh_type == 'Mesh':
                    trimesh_object = mesh_prim_mesh_to_trimesh_mesh(mesh.prim)
                else:
                    trimesh_object = mesh_prim_shape_to_trimesh_mesh(mesh.prim)
                
                
                world_pose_w_scale = PoseAPI.get_world_pose_with_scale(mesh.prim_path)
                trimesh_object.apply_transform(world_pose_w_scale)

                pose = PoseAPI.get_world_pose(mesh.prim_path)
                
                matrix_pose = T.pose2mat(pose)
                inverse_pose = T.pose_inv(matrix_pose)
                self.mesh_prim_path = mesh.prim_path
                trimesh_object.apply_transform(inverse_pose)
                
                # center the object but scale it accordingly
                trimesh_objects.append(trimesh_object)
                
        scene_mesh = trimesh.util.concatenate(trimesh_objects)
        
        return scene_mesh

                


    def register_keypoints(self, keypoints):
        """
        Args:
            keypoints (np.ndarray): keypoints in the world frame of shape (N, 3)
        Returns:
            None
        Given a set of keypoints in the world frame, this function registers them so that their newest positions can be accessed later.
        """

        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)
        self.keypoints = keypoints
        self._keypoint_registry = dict()
        self._keypoint2object = dict()
        exclude_names = ['wall', 'floor', 'ceiling', 'table', 'fetch', 'robot']
        for idx, keypoint in enumerate(keypoints):
            closest_distance = np.inf
            for obj in self.scene.objects:
                if any([name in obj.name.lower() for name in exclude_names]):
                    continue
                for link in obj.links.values():
                    for mesh in link.visual_meshes.values():
                        mesh_prim_path = mesh.prim_path
                        mesh_type = mesh.prim.GetPrimTypeInfo().GetTypeName()
                        if mesh_type == 'Mesh':
                            trimesh_object = mesh_prim_mesh_to_trimesh_mesh(mesh.prim)
                        else:
                            trimesh_object = mesh_prim_shape_to_trimesh_mesh(mesh.prim)
                        world_pose_w_scale = PoseAPI.get_world_pose_with_scale(mesh.prim_path)
                        trimesh_object.apply_transform(world_pose_w_scale)
                        points_transformed = trimesh_object.sample(1000)
                        
                        # find closest point
                        dists = np.linalg.norm(points_transformed - keypoint, axis=1)
                        point = points_transformed[np.argmin(dists)]
                        distance = np.linalg.norm(point - keypoint)
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_prim_path = mesh_prim_path
                            closest_point = point
                            closest_obj = obj
            self._keypoint_registry[idx] = (closest_prim_path, PoseAPI.get_world_pose(closest_prim_path))
            self._keypoint2object[idx] = closest_obj
            # overwrite the keypoint with the closest point
            self.keypoints[idx] = closest_point

    def get_keypoint_positions(self):
        """
        Args:
            None
        Returns:
            np.ndarray: keypoints in the world frame of shape (N, 3)
        Given the registered keypoints, this function returns their current positions in the world frame.
        """
        assert hasattr(self, '_keypoint_registry') and self._keypoint_registry is not None, "Keypoints have not been registered yet."
        keypoint_positions = []
        for idx, (prim_path, init_pose) in self._keypoint_registry.items():
            init_pose = T.pose2mat(init_pose)
            centering_transform = T.pose_inv(init_pose)
            keypoint_centered = np.dot(centering_transform, np.append(self.keypoints[idx], 1))[:3]
            curr_pose = T.pose2mat(PoseAPI.get_world_pose(prim_path))
            keypoint = np.dot(curr_pose, np.append(keypoint_centered, 1))[:3]
            keypoint_positions.append(keypoint)
        return np.array(keypoint_positions)

    def get_low_dim_obs(self):
        
        raw_obs = super().get_obs()
        
        proprio = [self.robot.get_eef_position(), self.robot.get_eef_orientation()]
        
        
        #for robot in raw_obs:
        #    d=robot['Fetch']['proprio']
        #    if not isinstance(d, dict):
        #        proprio.append(d)

        proprio = torch.cat(proprio, dim=0)
        
        for obj in self.objs:
            current_object = self.scene.object_registry("name", obj)
            bbox = torch.cat(current_object.aabb)
            pose = torch.cat(current_object.get_position_orientation())
            proprio = torch.cat((proprio, pose, bbox))
       
        proprio = np.asarray(proprio)
        
        return proprio
    

    def get_object_by_keypoint(self, keypoint_idx):
        """
        Args:
            keypoint_idx (int): the index of the keypoint
        Returns:
            pointer: the object that the keypoint is associated with
        Given the keypoint index, this function returns the name of the object that the keypoint is associated with.
        """
        assert hasattr(self, '_keypoint2object') and self._keypoint2object is not None, "Keypoints have not been registered yet."
        return self._keypoint2object[keypoint_idx]

    def get_random_position(self, obj, min_bounds, max_bounds):
        
        pos, orn = obj.get_position_orientation()
        x = np.random.uniform(min_bounds[0], max_bounds[0])
        y = np.random.uniform(min_bounds[1], max_bounds[1])
        z = np.random.uniform(min_bounds[2], max_bounds[2])
        z_angle = np.random.uniform(0, 2 * np.pi)
        orn1 = T.quat_multiply(T.euler2quat(np.array([0, 0, z_angle])), np.array(orn))
        return np.asarray([x,y,z]), np.array(orn)
    

    def get_cam_obs(self):
        self.last_cam_obs = dict()
        for cam_id in self.cams:
            self.last_cam_obs[cam_id] = self.cams[cam_id].get_obs()  # each containing rgb, depth, points, seg
        return self.last_cam_obs
    
    def _initialize_cameras(self, cam_config):
        """
        ::param poses: list of tuples of (position, orientation) of the cameras
        """
        self.cams = dict()
        for cam_id in cam_config:
            cam_id = int(cam_id)
            self.cams[cam_id] = OGCamera(self, cam_config[cam_id])
        for _ in range(10): og.sim.render() 
    
    def is_grasping(self, candidate_obj=None):
        return self.robot.is_grasping(candidate_obj=candidate_obj) == IsGraspingState.TRUE

    def get_ee_pose(self):
        ee_pos, ee_xyzw = (self.robot.get_eef_position(), self.robot.get_eef_orientation())
        ee_pose = np.concatenate([ee_pos, ee_xyzw])  # [7]
        return ee_pose

    def get_ee_pos(self):
        return self.get_ee_pose()[:3]

    def get_ee_quat(self):
        return self.get_ee_pose()[3:]
    
    def get_arm_joint_postions(self):
        assert isinstance(self.robot, Fetch), "The IK solver assumes the robot is a Fetch robot"
        arm = self.robot.default_arm
        dof_idx = np.concatenate([self.robot.trunk_control_idx, self.robot.arm_control_idx[arm]])
        arm_joint_pos = self.robot.get_joint_positions()[dof_idx]
        return arm_joint_pos

    def shutdown(self):
        og.shutdown()

    def _post_step(self, action):
        """Apply the post-sim-step part of an environment step, i.e. grab observations and return the step results."""
        # Grab observations
        
        assert sum(action[:4])==0 and action.shape[0]==12

        obs = self.get_low_dim_obs()

        # Step the scene graph builder if necessary
        if self._scene_graph_builder is not None:
            self._scene_graph_builder.step(self.scene)

        # Grab reward, done, and info, and populate with internal info
        reward, done, info = self.task.step(self, action)
        self._populate_info(info)
        reward, done = self.get_reward(), self.get_done()
        info["obs_info"] = 'low dim'
     #   print(f'VEC REW: {reward}')

 

        if done and self._automatic_reset:
            # Add lost observation to our information dict, and reset
            info["last_observation"] = obs
            obs = self.reset()

        # Hacky way to check for time limit info to split terminated and truncated
        terminated = False
        truncated = False
      
        for tc, tc_data in info["done"]["termination_conditions"].items():
            if tc_data["done"]:
                if tc == "timeout":
                    truncated = True
                else:
                    terminated = True
        terminated=done
        assert (terminated or truncated) == done, "Terminated and truncated must match done!"

        # Increment step
        self._current_step += 1
        print(reward)
        return obs, reward, terminated, truncated, info






class ReKepOGEnv:
    def __init__(self, config, scene_file, og_env, verbose=False, randomize=True):
        self.video_cache = []
        self.traj = Trajectory()
        self.config = config
        self.verbose = verbose
        self.config['scene']['scene_file'] = scene_file
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.interpolate_pos_step_size = self.config['interpolate_pos_step_size']
        self.interpolate_rot_step_size = self.config['interpolate_rot_step_size']
        # create omnigibson environment
        self.step_counter = 0
        self.og_env = og_env
        self.og_env.scene.update_initial_state()
        for _ in range(10): og.sim.step()
        # robot vars
        self.robot = self.og_env.robots[0]
        dof_idx = np.concatenate([self.robot.trunk_control_idx,
                                  self.robot.arm_control_idx[self.robot.default_arm]])
        self.reset_joint_pos = self.robot.reset_joint_pos[dof_idx]
        self.world2robot_homo = T.pose_inv(T.pose2mat(self.robot.get_position_orientation()))
        # initialize cameras
        self.og_env._initialize_cameras(self.config['camera'])
        self.last_og_gripper_action = 1.0 

         
        
 
    # ======================================
    # = exposed functions
    # ======================================
    def get_sdf_voxels(self, resolution, exclude_robot=True, exclude_obj_in_hand=True):
        """
        open3d-based SDF computation
        1. recursively get all usd prim and get their vertices and faces
        2. compute SDF using open3d
        """
        start = time.time()
        exclude_names = ['wall', 'floor', 'ceiling']
        if exclude_robot:
            exclude_names += ['fetch', 'robot']
        if exclude_obj_in_hand:
            assert self.config['robot']['robot_config']['grasping_mode'] in ['assisted', 'sticky'], "Currently only supported for assisted or sticky grasping"
            in_hand_obj = self.robot._ag_obj_in_hand[self.robot.default_arm]
            if in_hand_obj is not None:
                exclude_names.append(in_hand_obj.name.lower())
        trimesh_objects = []
        for obj in self.og_env.scene.objects:
            if any([name in obj.name.lower() for name in exclude_names]):
                continue
            for link in obj.links.values():
                for mesh in link.collision_meshes.values():
                    mesh_type = mesh.prim.GetPrimTypeInfo().GetTypeName()
                    if mesh_type == 'Mesh':
                        trimesh_object = mesh_prim_mesh_to_trimesh_mesh(mesh.prim)
                    else:
                        trimesh_object = mesh_prim_shape_to_trimesh_mesh(mesh.prim)
                    world_pose_w_scale = PoseAPI.get_world_pose_with_scale(mesh.prim_path)
                    trimesh_object.apply_transform(world_pose_w_scale)
                    trimesh_objects.append(trimesh_object)
        # chain trimesh objects
        scene_mesh = trimesh.util.concatenate(trimesh_objects)
        # Create a scene and add the triangle mesh
        scene = o3d.t.geometry.RaycastingScene()
        vertex_positions = scene_mesh.vertices
        triangle_indices = scene_mesh.faces
        vertex_positions = o3d.core.Tensor(vertex_positions, dtype=o3d.core.Dtype.Float32)
        triangle_indices = o3d.core.Tensor(triangle_indices, dtype=o3d.core.Dtype.UInt32)
        _ = scene.add_triangles(vertex_positions, triangle_indices)  # we do not need the geometry ID for mesh
        # create a grid
        shape = np.ceil((self.bounds_max - self.bounds_min) / resolution).astype(int)
        steps = (self.bounds_max - self.bounds_min) / shape
        grid = np.mgrid[self.bounds_min[0]:self.bounds_max[0]:steps[0],
                        self.bounds_min[1]:self.bounds_max[1]:steps[1],
                        self.bounds_min[2]:self.bounds_max[2]:steps[2]]
        grid = grid.reshape(3, -1).T
        # compute SDF
        sdf_voxels = scene.compute_signed_distance(grid.astype(np.float32))
        # convert back to np array
        sdf_voxels = sdf_voxels.cpu().numpy()
        # open3d [has ]flipped sign from our convention
        sdf_voxels = -sdf_voxels
        sdf_voxels = sdf_voxels.reshape(shape)
        self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] SDF voxels computed in {time.time() - start:.4f} seconds{bcolors.ENDC}')
        return sdf_voxels

    def get_cam_obs(self):
        return self.og_env.get_cam_obs()
    
    def register_keypoints(self, keypoints):
        """
        Args:
            keypoints (np.ndarray): keypoints in the world frame of shape (N, 3)
        Returns:
            None
        Given a set of keypoints in the world frame, this function registers them so that their newest positions can be accessed later.
        """
        self.og_env.register_keypoints(keypoints)

    def get_keypoint_positions(self):
        """
        Args:
            None
        Returns:
            np.ndarray: keypoints in the world frame of shape (N, 3)
        Given the registered keypoints, this function returns their current positions in the world frame.
        """
        return self.og_env.get_keypoint_positions()

    def get_object_by_keypoint(self, keypoint_idx):
        """
        Args:
            keypoint_idx (int): the index of the keypoint
        Returns:
            pointer: the object that the keypoint is associated with
        Given the keypoint index, this function returns the name of the object that the keypoint is associated with.
        """
        return self.og_env.get_object_by_keypoint(keypoint_idx)
    

    def get_collision_points(self, noise=True):
        """
        Get the points of the gripper and any object in hand.
        """
        # add gripper collision points
        collision_points = []
        for obj in self.og_env.scene.objects:
            if 'fetch' in obj.name.lower():
                for name, link in obj.links.items():
                    if 'gripper' in name.lower() or 'wrist' in name.lower():  # wrist_roll and wrist_flex
                        for collision_mesh in link.collision_meshes.values():
                            mesh_prim_path = collision_mesh.prim_path
                            mesh_type = collision_mesh.prim.GetPrimTypeInfo().GetTypeName()
                            if mesh_type == 'Mesh':
                                trimesh_object = mesh_prim_mesh_to_trimesh_mesh(collision_mesh.prim)
                            else:
                                trimesh_object = mesh_prim_shape_to_trimesh_mesh(collision_mesh.prim)
                            world_pose_w_scale = PoseAPI.get_world_pose_with_scale(mesh_prim_path)
                            trimesh_object.apply_transform(world_pose_w_scale)
                            points_transformed = trimesh_object.sample(1000)
                            # add to collision points
                            collision_points.append(points_transformed)
        # add object in hand collision points
        in_hand_obj = self.robot._ag_obj_in_hand[self.robot.default_arm]
        if in_hand_obj is not None:
            for link in in_hand_obj.links.values():
                for collision_mesh in link.collision_meshes.values():
                    mesh_type = collision_mesh.prim.GetPrimTypeInfo().GetTypeName()
                    if mesh_type == 'Mesh':
                        trimesh_object = mesh_prim_mesh_to_trimesh_mesh(collision_mesh.prim)
                    else:
                        trimesh_object = mesh_prim_shape_to_trimesh_mesh(collision_mesh.prim)
                    world_pose_w_scale = PoseAPI.get_world_pose_with_scale(collision_mesh.prim_path)
                    trimesh_object.apply_transform(world_pose_w_scale)
                    points_transformed = trimesh_object.sample(1000)
                    # add to collision points
                    collision_points.append(points_transformed)
        collision_points = np.concatenate(collision_points, axis=0)
        return collision_points

    

    def reset(self, objs=None):
        
        self.og_env.reset(objs)
        self.robot.reset()
        

        for _ in range(5): self._step()
        self.open_gripper()
        # moving arm to the side to unblock view 
        
        ee_pose = self.og_env.get_ee_pose()
        ee_pose[:3] += np.array([0.0, -0.2, -0.1])
        action = np.concatenate([ee_pose, [self.get_gripper_null_action()]])
        self.execute_action(action, precise=True)
        self.video_cache = []
        
        print(f'{bcolors.HEADER}Reset done.{bcolors.ENDC}')
        self.step_counter=0
        self.traj.clear()
        

    

  
    def is_grasping(self, candidate_obj=None):
        return self.robot.is_grasping(candidate_obj=candidate_obj) == IsGraspingState.TRUE

    def get_ee_pose(self):
        return self.og_env.get_ee_pose()

    def get_ee_pos(self):
        return self.og_env.get_ee_pose()[:3]

    def get_ee_quat(self):
        return self.og_env.get_ee_pose()[3:]
    
    def get_arm_joint_postions(self):
        return self.og_env.get_arm_joint_postions()

    def close_gripper(self):
        """
        Exposed interface: 1.0 for closed, -1.0 for open, 0.0 for no change
        Internal OG interface: 1.0 for open, 0.0 for closed
        """

        if self.last_og_gripper_action == 0.0:
            return
        action = np.zeros(12)
        action[10:] = [0, 0]  # gripper: float. 0. for closed, 1. for open.
        
        for _ in range(10):
            out = self._step(action)
            self.traj.add_step(*out, is_grasping_releasing=True)

        self.last_og_gripper_action = 0.0
       
    def open_gripper(self):
        
        if self.last_og_gripper_action == 1.0:
            return
        action = np.zeros(12)
        action[10:] = [1, 1]  # gripper: float. 0. for closed, 1. for open.
        for _ in range(10):
            out = self._step(action)
            self.traj.add_step(*out, is_grasping_releasing=True)  
        self.last_og_gripper_action = 1.0
        
    def get_last_og_gripper_action(self):
        return self.last_og_gripper_action
    
    def get_gripper_open_action(self):
        return -1.0
    
    def get_gripper_close_action(self):
        return 1.0
    
    def get_gripper_null_action(self):
        return 0.0
    
    def compute_target_delta_ee(self, target_pose):
        target_pos, target_xyzw = target_pose[:3], target_pose[3:]
        ee_pose = self.get_ee_pose()
        ee_pos, ee_xyzw = ee_pose[:3], ee_pose[3:]
        pos_diff = np.linalg.norm(ee_pos - target_pos)
        rot_diff = angle_between_quats(ee_xyzw, target_xyzw)
        return pos_diff, rot_diff

    def execute_action(
            self,
            action,
            contact_rich=False,
            precise=True,
        ):
            """
            Moves the robot gripper to a target pose by specifying the absolute pose in the world frame and executes gripper action.

            Args:
                action (x, y, z, qx, qy, qz, qw, gripper_action): absolute target pose in the world frame + gripper action.
                precise (bool): whether to use small position and rotation thresholds for precise movement (robot would move slower).
            Returns:
                tuple: A tuple containing the position and rotation errors after reaching the target pose.
            """
            

            if precise:
                pos_threshold = 0.02
                rot_threshold = 3.0
            else:
                pos_threshold = 0.10
                rot_threshold = 5.0
            action = np.array(action).copy()
            assert action.shape == (8,)
            target_pose = action[:7]
            gripper_action = action[7]

            # ======================================
            # = status and safety check
            # ======================================
            if np.any(target_pose[:3] < self.bounds_min) \
                 or np.any(target_pose[:3] > self.bounds_max):
                print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Target position is out of bounds, clipping to workspace bounds{bcolors.ENDC}')
                target_pose[:3] = np.clip(target_pose[:3], self.bounds_min, self.bounds_max)

            # ======================================
            # = interpolation
            # ======================================
            current_pose = self.get_ee_pose()
            pos_diff = np.linalg.norm(current_pose[:3] - target_pose[:3])
            rot_diff = angle_between_quats(current_pose[3:7], target_pose[3:7])
            pos_is_close = pos_diff < self.interpolate_pos_step_size
            rot_is_close = rot_diff < self.interpolate_rot_step_size
            if pos_is_close and rot_is_close:
                self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Skipping interpolation{bcolors.ENDC}')
                pose_seq = np.array([target_pose])
            else:
                num_steps = get_linear_interpolation_steps(current_pose, target_pose, self.interpolate_pos_step_size, self.interpolate_rot_step_size)
                pose_seq = linear_interpolate_poses(current_pose, target_pose, num_steps)
                self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Interpolating for {num_steps} steps{bcolors.ENDC}')

            # ======================================
            # = move to target pose
            # ======================================
            # move faster for intermediate poses
            intermediate_pos_threshold = 0.10
            intermediate_rot_threshold = 5.0
             
            for pose in pose_seq[:-1]:
                waypoint_traj = self._move_to_waypoint(pose, intermediate_pos_threshold, intermediate_rot_threshold, contact_rich=contact_rich)
              
            # move to the final pose with required precision
            pose = pose_seq[-1]
            waypoint_traj = self._move_to_waypoint(pose, pos_threshold, rot_threshold, max_steps=20 if not precise else 40, contact_rich=contact_rich) 
          
            # compute error
            pos_error, rot_error = self.compute_target_delta_ee(target_pose)
            self.verbose and print(f'\n{bcolors.BOLD}[environment.py | {get_clock_time()}] Move to pose completed (pos_error: {pos_error}, rot_error: {np.rad2deg(rot_error)}){bcolors.ENDC}\n')

            # ======================================
            # = apply gripper action
            # ======================================
            if gripper_action == self.get_gripper_open_action():
                open_gripper_traj = self.open_gripper()
               

            elif gripper_action == self.get_gripper_close_action():
                close_gripper_traj = self.close_gripper()
                 

            elif gripper_action == self.get_gripper_null_action():
                pass
                
            else:
                raise ValueError(f"Invalid gripper action: {gripper_action}")
            print(f'Action done, no. steps taken: {self.step_counter}')


    def sleep(self, seconds):
        start = time.time()
        while time.time() - start < seconds:
            self._step()
    
    def save_video(self, save_path=None):
        save_dir = os.path.join(os.path.dirname(__file__), 'videos')
        os.makedirs(save_dir, exist_ok=True)
        if save_path is None:
            save_path = os.path.join(save_dir, f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.mp4')
        video_writer = imageio.get_writer(save_path, fps=30)
        for rgb in self.video_cache:
            video_writer.append_data(rgb)
        video_writer.close()

        return save_path

    # ======================================
    # = internal functions
    # ======================================
    def _check_reached_ee(self, target_pos, target_xyzw, pos_threshold, rot_threshold):
        """
        this is supposed to be for true ee pose (franka hand) in robot frame
        """
        current_pos = self.robot.get_eef_position().cpu().numpy()
        current_xyzw = self.robot.get_eef_orientation().cpu().numpy()
        current_rotmat = T.quat2mat(current_xyzw)
        target_rotmat = T.quat2mat(target_xyzw)
        # calculate position delta
        pos_diff = (target_pos - current_pos).flatten()
        pos_error = np.linalg.norm(pos_diff)
        # calculate rotation delta
        rot_error = angle_between_rotmat(current_rotmat, target_rotmat)
        # print status
        self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}]  Curr pose: {current_pos}, {current_xyzw} (pos_error: {pos_error.round(4)}, rot_error: {np.rad2deg(rot_error).round(4)}){bcolors.ENDC}')
        self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}]  Goal pose: {target_pos}, {target_xyzw} (pos_thres: {pos_threshold}, rot_thres: {rot_threshold}){bcolors.ENDC}')
        if pos_error < pos_threshold and rot_error < np.deg2rad(rot_threshold):
            self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] OSC pose reached (pos_error: {pos_error.round(4)}, rot_error: {np.rad2deg(rot_error).round(4)}){bcolors.ENDC}')
            return True, pos_error, rot_error
        return False, pos_error, rot_error

    def _move_to_waypoint(self, target_pose_world, pos_threshold=0.02, rot_threshold=3.0, max_steps=10, contact_rich=False):
        pos_errors = []
        rot_errors = []
        count = 0
       
        while count < max_steps:
            reached, pos_error, rot_error = self._check_reached_ee(target_pose_world[:3], target_pose_world[3:7], pos_threshold, rot_threshold)
            pos_errors.append(pos_error)
            rot_errors.append(rot_error)
            if reached:
                break
            # convert world pose to robot pose
            target_pose_robot = np.dot(self.world2robot_homo, T.convert_pose_quat2mat(target_pose_world))
            # convert to relative pose to be used with the underlying controller
            relative_position = target_pose_robot[:3, 3] - self.robot.get_relative_eef_position().numpy()
            relative_quat = T.quat_distance(T.mat2quat(target_pose_robot[:3, :3]), self.robot.get_relative_eef_orientation().numpy())
            assert isinstance(self.robot, Fetch), "this action space is only for fetch"
            action = np.zeros(12)  # first 4 are base, which we don't use
            action[4:7] = relative_position
            action[7:10] = T.quat2axisangle(relative_quat)
            action[10:] = [self.last_og_gripper_action, self.last_og_gripper_action]
            # step the action
            out = self._step(action=action)
            count += 1
           
            self.traj.add_step(*out, is_grasping_releasing=contact_rich)

        if count == max_steps:
           
            print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] OSC pose not reached after {max_steps} steps (pos_error: {pos_errors[-1].round(4)}, rot_error: {np.rad2deg(rot_errors[-1]).round(4)}){bcolors.ENDC}')

     

    def _step(self, action=None):
        if hasattr(self, 'disturbance_seq') and self.disturbance_seq is not None:
            next(self.disturbance_seq)
        
        should_return = False
        if action is not None:
            should_return=True
            o = self.og_env.get_low_dim_obs()
           # print(o)
            next_o, reward, terminated, truncated, info = self.og_env.step(action)

        else:
            og.sim.step()
        cam_obs = self.get_cam_obs()
        rgb = cam_obs[1]['rgb']
        if len(self.video_cache) < self.config['video_cache_size']:
            self.video_cache.append(rgb)
        else:
            self.video_cache.pop(0)
            self.video_cache.append(rgb)

        self.step_counter += 1

        if should_return:
            return (o, action[4:], reward, terminated, "", info, next_o)



class RLEnvWrapper(CustomOGEnv):
    def __init__(self, args, config, in_vec_env=False, randomize=True, low_dim=True, bc_policy=None, threshold=10, use_oracle_reward=False): # threshold in cm
        super().__init__(args, config, in_vec_env, randomize, low_dim, use_oracle_reward)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=bc_policy, device=device, verbose=False)
        self.threshold = threshold

    def reset(self, seed=None):
        self.step_counter=0
        obs = super().reset(seed=seed) 
        step = 0
      
        if self.robot is not None:
            while 100 * calculate_bbox_to_point(self.get_aabb('pen_1'), self.get_ee_pos()) > self.threshold and step < 80:
                action = self.policy({'all' : obs})
                obs, r, done, _, __ = self.step(action)
                step += 1
                 
            self.step_counter=0
            return self.get_low_dim_obs(), {}
        self.step_counter=0
    

    
    
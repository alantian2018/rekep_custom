import torch
import numpy as np
import json
import os
from scipy.spatial.transform import Rotation as R
 
import argparse
from environment import *
from keypoint_proposal import KeypointProposer
from constraint_generation import ConstraintGenerator
from ik_solver import IKSolver
from subgoal_solver import SubgoalSolver
from path_solver import PathSolver
from visualizer import Visualizer
import transform_utils as T
from omnigibson.robots.fetch import Fetch
 
import torch
from utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)

class ReKepCollector:
    def __init__(self, scene_file, og_env, rekep_program_dir=None, visualize=False, randomize=True):
        global_config = get_config(config_path="./configs/config.yaml")
        self.config = global_config['main']
        
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.visualize = visualize
        self.stage = None
        self.prev_stage=None
        # set random seed
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        # initialize keypoint proposer and constraint generator
        self.keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
        self.constraint_generator = ConstraintGenerator(global_config['constraint_generator'])
        # initialize environment
        self.env = ReKepOGEnv(global_config['env'], scene_file, og_env, verbose=False, randomize=randomize)
        print('ENV created')
      
        self.randomize=randomize
        self.rekep_program_dir = rekep_program_dir
        self.paths = []
        #min_bounds =  [-0.5, -0.5, 0.78]   
        #max_bounds = [0.0, 0.0, 0.8]
       
        

        # setup ik solver (for reachability cost)
        assert isinstance(self.env.robot, Fetch), "The IK solver assumes the robot is a Fetch robot"
        ik_solver = IKSolver(
            robot_description_path=self.env.robot.robot_arm_descriptor_yamls[self.env.robot.default_arm],
            robot_urdf_path=self.env.robot.urdf_path,
            eef_name=self.env.robot.eef_link_names[self.env.robot.default_arm],
            reset_joint_pos=self.env.reset_joint_pos,
            world2robot_homo=self.env.world2robot_homo,
        )
        # initialize solvers
        self.subgoal_solver = SubgoalSolver(global_config['subgoal_solver'], ik_solver, self.env.reset_joint_pos.cpu().numpy())
        self.path_solver = PathSolver(global_config['path_solver'], ik_solver, self.env.reset_joint_pos.cpu().numpy())
        # initialize visualizer
        print('solvers done...')
        if self.visualize:
            self.visualizer = Visualizer(global_config['visualizer'], self.env)



    def perform_task(self, instruction, disturbance_seq=None, render=False):
        self.env.reset()
        self.render=render
        self.paths = [self.env.og_env.init_pos]
        cam_obs = self.env.get_cam_obs()
        rgb = cam_obs[self.config['vlm_camera']]['rgb']
        points = cam_obs[self.config['vlm_camera']]['points']
        mask = cam_obs[self.config['vlm_camera']]['seg']
        

                     
        # ====================================
        # = keypoint proposal and constraint generation
        # ====================================
        
        if self.rekep_program_dir is None:
            keypoints, projected_img = self.keypoint_proposer.get_keypoints(rgb, points, mask)
            print(f'{bcolors.HEADER}Got {len(keypoints)} proposed keypoints{bcolors.ENDC}')
            if self.visualize:
                self.visualizer.show_img(projected_img)
            metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
            self.rekep_program_dir = self.constraint_generator.generate(projected_img, instruction, metadata)
            print(f'{bcolors.HEADER}Constraints generated{bcolors.ENDC}')

        if self.rekep_program_dir is not None and self.env.og_env.reward_functions is None:
            self.env.og_env.set_reward_functions(self.rekep_program_dir)
        # ====================================
        # = execute
        # ====================================
        self._execute(self.rekep_program_dir, disturbance_seq) 
        
        ## extract the replay buffer here.
       
        return self.paths

    def _update_disturbance_seq(self, stage, disturbance_seq):
        if disturbance_seq is not None:
            if stage in disturbance_seq and not self.applied_disturbance[stage]:
                # set the disturbance sequence, the generator will yield and instantiate one disturbance function for each env.step until it is exhausted
                self.env.disturbance_seq = disturbance_seq[stage](self.env)
                self.applied_disturbance[stage] = True

    def _execute(self, rekep_program_dir, disturbance_seq=None):
        # load metadata
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        self.applied_disturbance = {stage: False for stage in range(1, self.program_info['num_stages'] + 1)}
        # register keypoints to be tracked
        self.env.register_keypoints(self.program_info['init_keypoint_positions'])
        self.env.og_env.register_keypoints(self.program_info['init_keypoint_positions'])
        # load constraints
        self.constraint_fns = dict()
        for stage in range(1, self.program_info['num_stages'] + 1):  # stage starts with 1
            stage_dict = dict()
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(rekep_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
                get_grasping_cost_fn = get_callable_grasping_cost_fn(self.env)  # special grasping function for VLM to call
                stage_dict[constraint_type] = load_functions_from_txt(load_path, get_grasping_cost_fn) if os.path.exists(load_path) else []
            self.constraint_fns[stage] = stage_dict
        
        # bookkeeping of which keypoints can be moved in the optimization
        self.keypoint_movable_mask = np.zeros(self.program_info['num_keypoints'] + 1, dtype=bool)
        self.keypoint_movable_mask[0] = True  # first keypoint is always the ee, so it's movable

        # main loop
        self.last_sim_step_counter = -np.inf
        self._update_stage(1)
        while True:

            scene_keypoints = self.env.get_keypoint_positions()
            self.keypoints = np.concatenate([[self.env.get_ee_pos()], scene_keypoints], axis=0)  # first keypoint is always the ee
            self.curr_ee_pose = self.env.get_ee_pose()
            self.curr_joint_pos = self.env.get_arm_joint_postions()
            self.sdf_voxels = self.env.get_sdf_voxels(self.config['sdf_voxel_size'])
            self.collision_points = self.env.get_collision_points()
            # ====================================
            # = decide whether to backtrack
            # ====================================
            backtrack = False
            if self.stage > 1:
                path_constraints = self.constraint_fns[self.stage]['path']
                for constraints in path_constraints:
                    violation = constraints(self.keypoints[0], self.keypoints[1:])
                    if violation > self.config['constraint_tolerance']:
                        backtrack = True
                        break
            if backtrack:
                # determine which stage to backtrack to based on constraints
                for new_stage in range(self.stage - 1, 0, -1):
                    path_constraints = self.constraint_fns[new_stage]['path']
                    # if no constraints, we can safely backtrack
                    if len(path_constraints) == 0:
                        break
                    # otherwise, check if all constraints are satisfied
                    all_constraints_satisfied = True
                    for constraints in path_constraints:
                        violation = constraints(self.keypoints[0], self.keypoints[1:])
                        if violation > self.config['constraint_tolerance']:
                            all_constraints_satisfied = False
                            break
                    if all_constraints_satisfied:   
                        break
                print(f"{bcolors.HEADER}[stage={self.stage}] backtrack to stage {new_stage}{bcolors.ENDC}")
                self._add_path(fail=True)
                self._update_stage(new_stage, fail=True)
            else:
                # apply disturbance
                self._update_disturbance_seq(self.stage, disturbance_seq)
                # ====================================
                # = get optimized plan
                # ====================================
                if self.last_sim_step_counter == self.env.step_counter:
                    
                    print(f"{bcolors.WARNING}sim did not step forward within last iteration (HINT: adjust action_steps_per_iter to be larger or the pos_threshold to be smaller){bcolors.ENDC}")
                    raise MemoryError()
                next_subgoal = self._get_next_subgoal(from_scratch=self.first_iter)
                next_path = self._get_next_path(next_subgoal, from_scratch=self.first_iter)
                self.first_iter = False
                self.action_queue = next_path.tolist()
                self.last_sim_step_counter = self.env.step_counter

                # ====================================
                # = execute
                # ====================================
                # determine if we proceed to the next stage
                count = 0
                while len(self.action_queue) > 0 and count < self.config['action_steps_per_iter']:
                    next_action = self.action_queue.pop(0)
                    precise = len(self.action_queue) == 0
                    self.env.execute_action(next_action, precise=precise) 
                    count += 1
                if len(self.action_queue) == 0 or self.env.step_counter > 1000: # TODO REMOVE THIS
                    if self.is_grasp_stage:
                        self._execute_grasp_action()
                    elif self.is_release_stage:
                        self._execute_release_action()
                    if self.stage == self.program_info['num_stages'] or self.env.step_counter > 1000: # TODO REMOVE THIS 
                        self._add_path(fail=self.env.step_counter>1000)
                        self.env.sleep(2.0) 
                        
                        print('Done with this trajectory!')
                        if (self.render):
                            save_path = self.env.save_video()
                            print(f"{bcolors.OKGREEN}Video saved to {save_path}\n\n{bcolors.ENDC}")
                        return
                  
                    # progress to next stage
                    self._add_path(fail=self.env.step_counter>1000)
                    self._update_stage(self.stage + 1, fail=self.env.step_counter>1000)
                     

    def _get_next_subgoal(self, from_scratch):
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        subgoal_pose, debug_dict = self.subgoal_solver.solve(self.curr_ee_pose,
                                                            self.keypoints,
                                                            self.keypoint_movable_mask,
                                                            subgoal_constraints,
                                                            path_constraints,
                                                            self.sdf_voxels,
                                                            self.collision_points,
                                                            self.is_grasp_stage,
                                                            self.curr_joint_pos,
                                                            from_scratch=from_scratch)
        subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        # if grasp stage, back up a bit to leave room for grasping
        if self.is_grasp_stage:
            subgoal_pose[:3] += subgoal_pose_homo[:3, :3] @ np.array([-self.config['grasp_depth'] / 2.0, 0, 0])
        debug_dict['stage'] = self.stage
      #  print_opt_debug_dict(debug_dict)
        if self.visualize:
            self.visualizer.visualize_subgoal(subgoal_pose)
        return subgoal_pose

    def _get_next_path(self, next_subgoal, from_scratch):
        path_constraints = self.constraint_fns[self.stage]['path']
        path, debug_dict = self.path_solver.solve(self.curr_ee_pose,
                                                    next_subgoal,
                                                    self.keypoints,
                                                    self.keypoint_movable_mask,
                                                    path_constraints,
                                                    self.sdf_voxels,
                                                    self.collision_points,
                                                    self.curr_joint_pos,
                                                    from_scratch=from_scratch)
      #  print_opt_debug_dict(debug_dict)
        processed_path = self._process_path(path)
        if self.visualize:
            self.visualizer.visualize_path(processed_path)
        return processed_path

    def _process_path(self, path):
        # spline interpolate the path from the current ee pose
        full_control_points = np.concatenate([
            self.curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
                                                    self.config['interpolate_pos_step_size'],
                                                    self.config['interpolate_rot_step_size'])
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        # add gripper action
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        ee_action_seq[:, 7] = self.env.get_gripper_null_action()
        return ee_action_seq

    def _add_path(self, fail):
        if (len(self.env.traj)!=0):
            self.paths.append({'stage':self.stage, 'fail':fail, 'path':self.env.traj.get_paths()})

    def _update_stage(self, stage, fail=False):
        # update stage
        self.prev_stage = self.stage
        self.stage = stage
        self.is_grasp_stage = self.program_info['grasp_keypoints'][self.stage - 1] != -1
        self.is_release_stage = self.program_info['release_keypoints'][self.stage - 1] != -1
        # can only be grasp stage or release stage or none
        assert self.is_grasp_stage + self.is_release_stage <= 1, "Cannot be both grasp and release stage"
        if self.is_grasp_stage:  # ensure gripper is open for grasping stage
            self.env.open_gripper()
        # clear action queue
        self.action_queue = []
        # update keypoint movable mask
        self._update_keypoint_movable_mask()
         

        
        
        self.first_iter = True

    def _update_keypoint_movable_mask(self):
        for i in range(1, len(self.keypoint_movable_mask)):  # first keypoint is ee so always movable
            keypoint_object = self.env.get_object_by_keypoint(i - 1)
            self.keypoint_movable_mask[i] = self.env.is_grasping(keypoint_object)

    def _execute_grasp_action(self):
        pregrasp_pose = self.env.get_ee_pose()
        grasp_pose = pregrasp_pose.copy()
        grasp_pose[:3] += T.quat2mat(pregrasp_pose[3:]) @ np.array([self.config['grasp_depth'], 0, 0])
        grasp_action = np.concatenate([grasp_pose, [self.env.get_gripper_close_action()]])

        self.env.execute_action(grasp_action, contact_rich=True, precise=True)


    def _align_orthogonal(self, grasp_action, pen_orn):
        # Convert pen orientation to rotation matrix to get its axes
        pen_rotation = R.from_quat(pen_orn)
        pen_matrix = pen_rotation.as_matrix()
        
        # Get pen's primary axis (assuming it's along the Z-axis of pen's local frame)
        pen_axis = pen_matrix[:, 2]
        
        # Define the gripper's desired Z-axis (approach direction)
        gripper_z = np.array([0, 0, -1])  # Gripper should point down towards the table
        
        # Find rotation aligning gripper Z-axis with pen axis
        axis = np.cross(gripper_z, pen_axis)
        angle = np.arccos(np.clip(np.dot(gripper_z, pen_axis) / (np.linalg.norm(gripper_z) * np.linalg.norm(pen_axis)), -1.0, 1.0))
        
        if np.linalg.norm(axis) < 1e-6:
            # Axis is nearly aligned, no adjustment needed
            return grasp_action
        
        axis = axis / np.linalg.norm(axis)
        rotation = R.from_rotvec(axis * angle)
        
        # Apply rotation to the default gripper orientation
        default_gripper_rotation = R.from_euler('xyz', [0, 0, 0])  # Default no rotation
        aligned_rotation = rotation * default_gripper_rotation
        
        # Update grasp action with new quaternion
        grasp_action[3:7] = aligned_rotation.as_quat()
        
        return grasp_action

    def _execute_release_action(self):
        self.env.open_gripper()





import torch
import numpy as np
import json
import os
import argparse
from environment import ReKepOGEnv, CustomOGEnv
from keypoint_proposal import KeypointProposer
from constraint_generation import ConstraintGenerator
from ik_solver import IKSolver
from subgoal_solver import SubgoalSolver
from path_solver import PathSolver
from visualizer import Visualizer
import transform_utils as T
from rlkit_custom import CustomEnvReplayBuffer
from omnigibson.robots.fetch import Fetch
from rlkit_utils import *
from rlkit_custom import collect_rekep_paths
from robomimic.utils.tensor_utils import flatten_nested_dict_list
from rlkit_custom import *
from utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)

def preprocess_obs(d):
    # This function recursively flattens a nested dictionary into a list of tensors
    out = flatten_nested_dict_list(d)
    return torch.cat((out[0][1].flatten(), out[1][1].flatten()))

if __name__=='__main__':
    task_list = {
        'pen': {
            'scene_file': './configs/og_scene_file_pen.json',
            'instruction': 'Pick the white pen up with the gripper',
            'rekep_program_dir': './vlm_query/pen',
        }
    }
    

    global_config = get_config(config_path="./configs/config.yaml")
    config = global_config['env']
    
    config['scene']['scene_file'] = task_list['pen']['scene_file']

    og_env = CustomOGEnv(dict(scene=config['scene'], robots=[config['robot']['robot_config']], env=config['og_sim']),
                                     randomize=False)
  
    
    og_env.reset()
    o = og_env.get_obs()
    
    
    a = og_env.step(np.zeros(12))
    o = a[0]
    o=og_env.get_low_dim_obs()
   
    replaybuffer = CustomEnvReplayBuffer(10,10, og_env)

    paths = collect_rekep_paths(og_env, task_list, 1)
 
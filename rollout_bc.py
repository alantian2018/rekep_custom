import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch
import numpy as np
from robomimic.config import config_factory
from robomimic.algo import algo_factory
from robomimic.utils.dataset import SequenceDataset
from robomimic.config import Config
import json
import robomimic.utils.train_utils as TrainUtils
from torch.utils.data import DataLoader
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
import time
import os
import imageio
from environment import *
from rollout_based_on_dataset import *
from tqdm import tqdm
from utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)
import matplotlib.pyplot as plt




def rollout(policy, environment, video_buffer,n, ee_points,ee_distance_data):
   

    obs = environment.reset()
    key = f'demo_{n+1}'
    ee_distance_data[key] = []
    for i in tqdm(range (80), desc=f'Rollout: {n+1}'):
        action = policy({'all' : obs})
        obs, r, done, _, __ = env.step(action)
        
        ee_point = env.get_centered_ee('pen_1')
        ee_points.append(ee_point)

        bbox = environment.get_aabb('pen_1')
        distance = calculate_bbox_to_point(bbox, environment.get_ee_pos())
        ee_distance_data[key].append(distance)

        if video_buffer is not None:
            cam_obs = env.get_cam_obs()
            rgb = cam_obs[1]['rgb']
            video_buffer.append_data(rgb)
        
        if (done):
            print('Done, breaking early....')
            break
    ee_distance_data[key] = np.array(ee_distance_data[key])
    return ee_distance_data[key]
     

 
    




if __name__ == '__main__':

    path_to_policy='/nethome/atian31/flash8/repos/ReKep/pen_pickup_models/Pen_Pickup_rnn/20250120004223/models/model_epoch_2589_best_validation_14794.90048828125.pth'
    dir_path = os.path.join(
        os.path.dirname(
        os.path.dirname(path_to_policy)),
        
    )
    video_path = os.path.join(dir_path, 'videos/rollout.mp4')
    dist_path = os.path.join(dir_path, 'distances.png')

    video_buffer = None #imageio.get_writer(video_path, fps=20)
     
    task_list = {
        'pen': {
            'scene_file': 'configs/og_scene_file_pen.json',
            'instruction': '',
            'rekep_program_dir': './vlm_query/pen'
            },
    }
    
    task = task_list['pen']

    scene_file = task['scene_file']
    instruction = task['instruction']
    global_config = get_config(config_path="./configs/config.yaml")
    config = global_config['env']
     
    config['scene']['scene_file'] = scene_file

    env = CustomOGEnv(dict(scene=config['scene'], robots=[config['robot']['robot_config']], env=config['og_sim']), config,
                                     randomize=True)
    mesh = env.get_mesh('pen_1')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=path_to_policy, device=device, verbose=True)
    
    ee_points = []
    ee_distance_data=dict()
    for i in range(200):
        dist = rollout(policy, env, video_buffer, i, ee_points, ee_distance_data)
        dist *= 100
        x = range(len(dist)) 
        print(f'MINIMUM DISTANCE: {min(dist)} cm')
        plt.plot(x, dist, label=f"Rollout {i + 1}")

    plt.xlabel('step')
    plt.ylabel('distance to pen (cm)')
    plt.savefig(dist_path)


    path = os.path.join(dir_path,"trajectories/")
    os.makedirs(path, exist_ok=True)

    new_mesh = add_independent_points_to_ply(mesh, ee_points)
    new_mesh.export(path+"bc_cr.ply")

    ee_data = {'ee_points':ee_points, 'ee_distance_data':ee_distance_data}
    ee_save_path = os.path.join(dir_path, 'ee_raw.pkl')
    torch.save(ee_data, ee_save_path)
    print(f'Saved ee data to {ee_save_path}')

    breakpoint()
    if video_buffer is not None:
        video_buffer.close()
        print(f"video written to {video_path}")
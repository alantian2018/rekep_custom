ENVS_PER_GPU = 10
TOTAL_TIMESTEPS = 1_000_000
SAVE_FREQ = TOTAL_TIMESTEPS // ENVS_PER_GPU // 20
EVAL_FREQ = 10_000 // ENVS_PER_GPU
 







# train.py
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import json
import os
import argparse
from environment import ReKepOGEnv, CustomOGEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from keypoint_proposal import KeypointProposer
from constraint_generation import ConstraintGenerator
from ik_solver import IKSolver
from subgoal_solver import SubgoalSolver
from path_solver import PathSolver
from visualizer import Visualizer
import transform_utils as T
from rlkit_custom import CustomEnvReplayBuffer
from omnigibson.robots.fetch import Fetch
from omnigibson.macros import gm
from rlkit_utils import *
from rlkit_custom import collect_rekep_paths
from robomimic.utils.tensor_utils import flatten_nested_dict_list
from rlkit_custom import *
from stable_baselines3 import SAC, PPO
from sb3_contrib import TQC
import copy
import time
from sb3_vec_env import CustomSB3VectorEnvironment, make_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
import torch.distributed as dist
from argparse import ArgumentParser
from utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)

###########################################


###########################################



# assert TOTAL_TIMESTEPS % ENVS_PER_GPU == 0 and (TOTAL_TIMESTEPS // ENVS_PER_GPU) % SAVE_FREQ == 0 
gm.ENABLE_FLATCACHE=True
gm.USE_GPU_DYNAMICS=False
gm.RENDER_VIEWER_CAMERA=False

def setup_ddp():
    # DDP setup
   
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return local_rank, world_size

def main():
    # Set up DDP
    #local_rank, world_size = setup_ddp()
    parser = ArgumentParser()
    parser.add_argument("--oracle", action="store_true", help="Set this flag to use oracle rews")
    parser.add_argument("--algo", choices=["PPO",'SAC','TQC'], required=True, help="Choose SAC or PPO")
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    use_oracle_reward = args.oracle

    print(f'Using oracle? {use_oracle_reward}')
    print(f'Which Algo? {args.algo}')

   
    tensorboard_log_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "log_dir", time.strftime("%Y%m%d-%H%M%S")
         )

    os.makedirs(tensorboard_log_dir, exist_ok=True)


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
    
    rekep_program_dir = '/coc/flash8/atian31/repos/ReKep/./vlm_query/2025-01-19_02-06-17_pick_up_the_white_pen_in_the_middle._'
    model_path = '/nethome/atian31/flash8/repos/ReKep/pen_pickup_models/Pen_Pickup_rnn/20250120004223/models/model_epoch_2893_best_validation_14653.164111328126.pth'
   
    t=time.perf_counter()
   # if args.debug:
   #     TOTAL_TIMESTEPS = 1000
   #     SAVE_FREQ = 500
   #     ENVS_PER_GPU = 2
   #     EVAL_FREQ = 25
    
    eval_env = DummyVecEnv([
        lambda: make_env(config, rekep_program_dir, render_on_step=False, use_oracle_reward=use_oracle_reward, bc_policy =model_path)
    ])
    vec_env = CustomSB3VectorEnvironment(
        num_envs=ENVS_PER_GPU,
        config=config,
        rekep_program_dir=rekep_program_dir,
        render_on_step=False,
        bc_policy=model_path,
        use_oracle_reward=use_oracle_reward
    )
    print(f'ENV made in {time.perf_counter()-t} s')
    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=tensorboard_log_dir, name_prefix='')
    eval_callback = EvalCallback(eval_env,
                                best_model_save_path=tensorboard_log_dir,
                                eval_freq=EVAL_FREQ,   
                                n_eval_episodes=3,
                                deterministic=True,
                                render=False
    )

    callback = CallbackList([checkpoint_callback, eval_callback])
    if args.algo == 'SAC':
        model = SAC("MlpPolicy", vec_env, verbose=2,  tensorboard_log=tensorboard_log_dir)
    elif args.algo == 'PPO':
        model = PPO("MlpPolicy", vec_env, verbose=2, gae_lambda=.97, tensorboard_log=tensorboard_log_dir)
    elif args.algo == 'TQC':
        model = TQC('MlpPolicy', vec_env, verbose=2, tensorboard_log = tensorboard_log_dir)

    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True,  callback=callback, tb_log_name="run")
    #model.save(os.path.join(tensorboard_log_dir,'model.zip'))
    # Cleanup
 
    og.shutdown()

if __name__ == "__main__":
    main()
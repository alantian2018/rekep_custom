###########################################

ENVS_PER_GPU = 5
EVAL_ENVS = 5
TOTAL_TIMESTEPS = 10_000_000
SAVE_FREQ = TOTAL_TIMESTEPS // ENVS_PER_GPU // 40
EVAL_FREQ = 10_000 // ENVS_PER_GPU

from omnigibson.macros import gm
gm.ENABLE_FLATCACHE=True
gm.USE_GPU_DYNAMICS=False
gm.RENDER_VIEWER_CAMERA=False
gm.ENABLE_TRANSITION_RULES=False
gm.ENABLE_HQ_RENDERING=False
gm.DEFAULT_PHYSICS_FREQ = 240
gm.DEFAULT_RENDERING_FREQ = 60

print(gm)
import wandb
from wandb import AlertLevel
from wandb.integration.sb3 import WandbCallback
###########################################



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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from keypoint_proposal import KeypointProposer
from constraint_generation import ConstraintGenerator
from ik_solver import IKSolver
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, VecVideoRecorder

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
from stable_baselines3 import SAC, PPO
from sb3_contrib import TQC
import copy
import time
from sb3_vec_env import CustomSB3VectorEnvironment, make_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback
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

class AfterEvalCallback(BaseCallback):
    def __init__(self, env, eval_env, verbose=0):
        super(AfterEvalCallback, self).__init__(verbose)
        self.env = env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        self.env.reset()
        return True

class SaveReplayBufferCallback(CheckpointCallback):
    def __init__(self, save_freq, save_path, name_prefix="rl_model", verbose=0):
        super().__init__(save_freq, save_path, name_prefix, verbose)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
            self.model.save(model_path)
            
            # Save the replay buffer separately
            replay_buffer_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_replay_buffer.pkl")
            if self.model.replay_buffer is not None:
                self.model.save_replay_buffer(replay_buffer_path)

        return True


# assert TOTAL_TIMESTEPS % ENVS_PER_GPU == 0 and (TOTAL_TIMESTEPS // ENVS_PER_GPU) % SAVE_FREQ == 0 

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
    parser.add_argument("--algo", choices=["PPO",'SAC','TQC'], default='SAC' , help="Choose SAC, PPO or TQC")
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    use_oracle_reward = args.oracle

    print(f'Using oracle? {use_oracle_reward}')
    print(f'Which Algo? {args.algo}')


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
    model_path = None #'/nethome/atian31/flash8/repos/ReKep/pen_pickup_models/Pen_Pickup_rnn/20250120004223/models/model_epoch_2893_best_validation_14653.164111328126.pth'

    global ENVS_PER_GPU 
    global TOTAL_TIMESTEPS 
    global EVAL_FREQ 
    global SAVE_FREQ 
    global EVAL_ENVS
    
    if args.debug:
        TOTAL_TIMESTEPS = 2000
        SAVE_FREQ = 400
        ENVS_PER_GPU = 2
        EVAL_ENVS = 1
        EVAL_FREQ = 800
         
    t = time.perf_counter()

    tensorboard_log_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "log_dir",
             args.algo, 'ORACLE' if args.oracle else "VLM",
             'DEBUG/' if args.debug else '',
             time.strftime("%Y%m%d-%H%M%S")
         )
    

    os.makedirs(tensorboard_log_dir, exist_ok=True)


    run = wandb.init(
        entity="alantian2018",
        project="sb3_omnigibson",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    vec_env = CustomSB3VectorEnvironment(
        num_envs=ENVS_PER_GPU,
        config=config,
        rekep_program_dir=rekep_program_dir,
        render_on_step=False,
        bc_policy=model_path,
        use_oracle_reward=use_oracle_reward
    )

    vec_env = VecMonitor(vec_env,info_keywords=("is_success",))
  #  breakpoint()
    
    eval_env = CustomSB3VectorEnvironment(
        num_envs=EVAL_ENVS,
        config=config,
        rekep_program_dir=rekep_program_dir,
        render_on_step=True,
        bc_policy=model_path,
        use_oracle_reward=use_oracle_reward
    )

    eval_env = VecMonitor(eval_env,info_keywords=("is_success",))
    traj_len = eval_env.envs[0].max_traj_len
     


     
    after_eval_callback = AfterEvalCallback(vec_env, eval_env)
    print(f'ENV made in {time.perf_counter()-t} s')

    if args.algo == 'PPO':
        checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=tensorboard_log_dir, name_prefix='')
    else:
        checkpoint_callback = SaveReplayBufferCallback(save_freq = SAVE_FREQ, save_path=tensorboard_log_dir, name_prefix='')

    eval_callback = EvalCallback(eval_env,
                                best_model_save_path=tensorboard_log_dir,
                                eval_freq=EVAL_FREQ,   
                                n_eval_episodes= EVAL_ENVS * 3,
                                deterministic=True,
                                render=False,
                                callback_after_eval=after_eval_callback,
    )
    
    
    wandb_callback = WandbCallback(
        model_save_path=tensorboard_log_dir,
        verbose=2,
    )


    callback = CallbackList([checkpoint_callback, eval_callback, wandb_callback])

    train_freq=(40, "step")
    gradient_steps=40

    wandb.alert(title="Run launched", text=f"Run ID: {wandb.run.id}", level=AlertLevel.INFO)
    
    if args.algo == 'SAC':
        model = SAC("MlpPolicy", vec_env, learning_starts=5000, train_freq = train_freq, gradient_steps=gradient_steps, learning_rate=3e-4, verbose=0,  tensorboard_log=tensorboard_log_dir)
    
    elif args.algo == 'PPO':
        model = PPO("MlpPolicy", vec_env, verbose=0, gae_lambda=.97, tensorboard_log=tensorboard_log_dir)
    
    elif args.algo == 'TQC':
        model = TQC('MlpPolicy', vec_env, learning_starts=5000, train_freq = train_freq, gradient_steps=gradient_steps, learning_rate = 3e-4,  verbose=0, tensorboard_log = tensorboard_log_dir)
    
    print(f'Tensorboard log dir: {tensorboard_log_dir}')
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=args.debug,  callback=callback, tb_log_name="run",log_interval=4)
    
    if args.algo =='SAC' or args.algo == 'TQC':
        model.save_replay_buffer(os.path.join(tensorboard_log_dir, "sac_replay_buffer"))
   
    # Cleanup
 
    og.shutdown()

if __name__ == "__main__":
    main()
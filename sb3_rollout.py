import numpy as np
from stable_baselines3 import SAC,PPO  # Change to your algorithm
from sb3_contrib import TQC
from environment import RLEnvWrapper, CustomOGEnv
import os
import imageio
import time
from utils import get_config
import matplotlib.pyplot as plt
import copy
from tqdm import trange
import omnigibson as og
if __name__=='__main__':
    rl_path = "/coc/flash8/atian31/repos/ReKep/log_dir/SAC/VLM/20250221-021359/_1000000_steps.zip"
    rekep_program_dir = '/coc/flash8/atian31/repos/ReKep/./vlm_query/2025-01-19_02-06-17_pick_up_the_white_pen_in_the_middle._'
    bc_policy =None# '/nethome/atian31/flash8/repos/ReKep/pen_pickup_models/Pen_Pickup_rnn/20250120004223/models/model_epoch_2893_best_validation_14653.164111328126.pth'
    

    model = SAC.load(rl_path)
    base=time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join (os.path.dirname(rl_path), base+'.mp4')
    video_buffer = imageio.get_writer(save_path, fps=30)

    num_episodes = 5  # Adjust as needed
    episode_rewards = []
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

    
    env = CustomOGEnv(
            copy.deepcopy(dict(scene=config['scene'], robots=[config['robot']['robot_config']], env=config['og_sim'])),
            in_vec_env=False,
            config=copy.deepcopy(config),
            randomize=True,
            low_dim=True,
           # bc_policy=bc_policy

        )
    
    env.set_rekep_program_dir(rekep_program_dir)
    env.set_reward_function_for_stage(1)
    
    
    for _ in trange(num_episodes):
        step = 0
        obs , _ = env.reset()
        done = False
        total_reward = 0
        rew = []
        while not done:
            action, _states = model.predict(obs, deterministic=True)  # Set False for stochastic policies
            obs, reward, done, t,  info = env.step(action)
            total_reward += reward

            rgb = env.render()
            video_buffer.append_data(rgb)
            rew.append(reward)
            step+=1
        
        plt.plot(range(1,len(rew)+1), rew)
        episode_rewards.append(total_reward)

    # Compute average reward
    avg_reward = np.mean(episode_rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")
    video_buffer.close()
    print(f'Saved video to {save_path}')
    plt.savefig(os.path.join(os.path.dirname(rl_path), base + '.png'))

    og.shutdown()

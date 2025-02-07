from rlkit_utils import simulate_policy
 
 
import numpy as np
import torch
import imageio
import os
import json
from ReKepCollector import *
from environment import *

from signal import signal, SIGINT
from sys import exit
from argparse import ArgumentParser
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

# Add and parse arguments
psr =ArgumentParser()
psr.add_argument('--load_dir', required=True)
args = psr.parse_args()

# Define callbacks
video_writer = None


def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Closing video writer and exiting gracefully')
    video_writer.close()
    exit(0)


# Tell Python to run the handler() function when SIGINT is recieved
signal(SIGINT, handler)

if __name__ == "__main__":
    # Set random seed
    

    # Get path to saved model
    kwargs_fpath = os.path.join(args.load_dir, "variant.json")
    try:
        with open(kwargs_fpath) as f:
            kwargs = json.load(f)
    except FileNotFoundError:
        print("Error opening default controller filepath at: {}. "
              "Please check filepath and try again.".format(kwargs_fpath))

    # Grab / modify env args
    env_args = kwargs
    np.random.seed(env_args['seed'])
    torch.manual_seed (env_args['seed'])

    # Specify camera name if we're recording a video
 
    # Setup video recorder if necesssary
    
    # Grab name of this rollout combo
    video_name = "{}-{}".format(
        'pen', env_args['task_list']['pen']['instruction']).replace("_", "-")
    # Calculate appropriate fps
     
    # Define video writer
    video_path =os.path.join(args.load_dir, 'rollout.mp4')
    video_writer = imageio.get_writer(video_path, fps=20)

  

    # Create env
    global_config = get_config(config_path="./configs/config.yaml")
    config = global_config['env']
    variant = env_args
    print(f'variant: {variant}')
    
    config['scene']['scene_file'] = variant['task_list']['pen']['scene_file']

    env = CustomOGEnv(dict(scene=config['scene'], robots=[config['robot']['robot_config']], env=config['og_sim']), config,
                                     randomize=True)
    env.set_rekep_program_dir(variant['rekep_program_dir'])
    env.set_reward_function_for_stage(1)
    
    paths  = simulate_policy(
        env=env,
        model_path=os.path.join(args.load_dir, "params.pkl"),
        horizon=variant['algorithm_kwargs']['eval_max_path_length'],
        render=False,
        video_writer=video_writer,
        num_episodes=10,
        printout=True,
        use_gpu=False,
    )
    
    for c,i in enumerate(paths):
        r = i['rewards']
        x = range(len(r)) 
        plt.plot(x, r, label=f"Rollout {c + 1}")
    plt.xlabel('step')
    plt.ylabel('reward')

    reward_path =os.path.join(args.load_dir, 'rewards.png')
    plt.savefig(reward_path)
        
    if video_writer is not None:
        print('Writing video to {}'.format(video_path))
        video_writer.close()
    breakpoint()
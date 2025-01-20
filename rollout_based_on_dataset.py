import torch
import imageio
import os
from utils import *
from environment import *

import cv2
import numpy as np
from tqdm import tqdm
if __name__ == '__main__':
    path = '/coc/flash8/atian31/repos/ReKep/data/2025-01-19-01-58-13/raw.pkl'

    # Load dataset
    dataset = torch.load(path)
    paths, rekep_program_dir = dataset[0], dataset[1]

    # Determine video output path
    dir_path = os.path.dirname(path)
    video_path = os.path.join(dir_path, 'rollout.mp4')

    # Task configuration
    task_list = {
        'pen': {
            'scene_file': 'configs/og_scene_file_pen.json',
            'instruction': '',
            'rekep_program_dir': './vlm_query/pen'
        },
    }

    task = task_list['pen']
    scene_file = task['scene_file']

    # Load global config
    global_config = get_config(config_path="./configs/config.yaml")
    config = global_config['env']

    # Update scene file in config
    config['scene']['scene_file'] = scene_file

    # Initialize environment
    og_env = CustomOGEnv(
        dict(
            scene=config['scene'],
            robots=[config['robot']['robot_config']],
            env=config['og_sim']
        ),
        config,
        randomize=False
    )
    env = ReKepOGEnv(global_config['env'], scene_file, og_env, verbose=False, randomize=False)

    # Prepare video writer
    video_buffer = imageio.get_writer(video_path, fps=20)

    try:
        for rollout in tqdm(paths):
            object_info = rollout[0]  # Object state info
            env.reset(object_info)  # Reset environment to specific state

            trajs = rollout[1:]  # Remaining trajectory stages
            for stage in trajs:
                actions = stage['path'][1]
                contact_rich = stage['path'][-1]

                for action, is_contact_rich in zip(actions, contact_rich):
                    env._step(action)
                    cam_obs = env.og_env.get_cam_obs()
                    rgb = cam_obs[1]['rgb']
        
                    # Overlay text if contact-rich
                    if is_contact_rich:
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        text = "Contact-Rich Step"
                        position = (50, 50)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1
                        color = (0, 255, 0)  # Green in BGR
                        thickness = 2
                        cv2.putText(bgr, text, position, font, font_scale, color, thickness)
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    
                    video_buffer.append_data(rgb)  # Append frame to video
    finally:
        video_buffer.close()  # Safely close video buffer
        print(f'Saved to {video_path}')

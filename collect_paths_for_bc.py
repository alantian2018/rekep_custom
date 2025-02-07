import os
import argparse
from environment import *
from rlkit_custom import *
from rlkit_utils import *
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, required=True, help='number of trajs to collects')
    parser.add_argument('--task', type=str, default='pen', help='task to perform')
    parser.add_argument('--use_cached_query', action='store_true', help='instead of querying the VLM, use the cached query')
  
    args = parser.parse_args()

     
    task_list = {
        'pen': {
            'scene_file': '/coc/flash8/atian31/repos/ReKep/configs/og_scene_file_pen.json',
            'instruction':  "Pick up the white pen in the middle. ",#'Pick the white pen up with the gripper. Lift it off the table by 15 cm. This involves grasping the pen and then lifting it up off the table.',
            'rekep_program_dir': './vlm_query/pen'
            },
    }
    
    task = task_list['pen']

    scene_file = task['scene_file']
    instruction = task['instruction']
    global_config = get_config(config_path="./configs/config.yaml")
    config = global_config['env']
    
    config['scene']['scene_file'] = scene_file

    save_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(save_dir, exist_ok=True)
    folder= f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
    save_path = os.path.join(save_dir, f'{folder}/raw.pkl')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
  
    env = CustomOGEnv(dict(scene=config['scene'], robots=[config['robot']['robot_config']], env=config['og_sim']), config,
                                     randomize=True)

    paths, rekep_program_dir = collect_rekep_paths(env, task_list, args.n, save_path)

    torch.save([paths, rekep_program_dir], save_path)
    print(f'Saved {len(paths)} to {save_path}')
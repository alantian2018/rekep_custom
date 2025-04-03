 
import time
import numpy as np
from omnigibson.macros import gm
from utils import *
from environment import *
from tqdm import trange
import imageio
from sb3_vec_env import *
"""
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=False,
)

# reset the environment
env.reset()


start_time = time.perf_counter()


for i in range(1000):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)  # take action in the environment
end_time = time.perf_counter()
rs_time = (end_time - start_time) / 1000
 
"""

import omnigibson as og

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

#env = CustomOGEnv(dict(scene=config['scene'], robots=[config['robot']['robot_config']], env=config['og_sim']), config,
#                                    randomize=True)

rekep_program_dir = '/nethome/atian31/flash8/repos/ReKep/vlm_query/2025-01-19_02-06-17_pick_up_the_white_pen_in_the_middle._'
vec_env = CustomSB3VectorEnvironment(
        num_envs=2,
        config=config,
        rekep_program_dir=rekep_program_dir,
        render_on_step=False,
        bc_policy=None,
        use_oracle_reward=False,
     
    )

#env.set_rekep_program_dir(rekep_program_dir)
#env.set_reward_function_for_stage(1)
#mesh = env.get_mesh('pen_1')
breakpoint()
start_time = time.perf_counter()

 

 
video_buffer = imageio.get_writer('hah.mp4', fps=1)
 
for i in trange(6):
        vec_env.reset()
        # for e in vec_env.envs:
        
        for e in vec_env.envs:
            rgb = e.render()
            video_buffer.append_data(rgb)
            print(e.get_reward())
            print(e.pen.get_position_orientation()[0])
            print(e.table.get_position_orientation()[0])
            #print(e.get_low_dim_obs)
            print(e.get_ee_pos())
#print(env.get_ee_pos())
video_buffer.close()
 
og.shutdown()
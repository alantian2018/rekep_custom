 
import time
import numpy as np
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
from omnigibson.macros import gm
gm.HEADLESS=True
cfg = dict()

# Define scene
cfg["scene"] = {
    "type": "Scene",
    "floor_plane_visible": True,
}

# Define objects
cfg["objects"] = [
    {
        "type": "USDObject",
        "name": "ghost_stain",
        "usd_path": f"{gm.ASSET_PATH}/models/stain/stain.usd",
        "category": "stain",
        "visual_only": True,
        "scale": [1.0, 1.0, 1.0],
        "position": [1.0, 2.0, 0.001],
        "orientation": [0, 0, 0, 1.0],
    },
    {
        "type": "DatasetObject",
        "name": "delicious_apple",
        "category": "apple",
        "model": "agveuv",
        "position": [0, 0, 1.0],
    },
    {
        "type": "PrimitiveObject",
        "name": "incredible_box",
        "primitive_type": "Cube",
        "rgba": [0, 1.0, 1.0, 1.0],
        "scale": [0.5, 0.5, 0.1],
        "fixed_base": True,
        "position": [-1.0, 0, 1.0],
        "orientation": [0, 0, 0.707, 0.707],
    },
    {
        "type": "LightObject",
        "name": "brilliant_light",
        "light_type": "Sphere",
        "intensity": 50000,
        "radius": 0.1,
        "position": [3.0, 3.0, 4.0],
    },
]

# Define robots
cfg["robots"] = [
    {
        "type": "Fetch",
        "name": "skynet_robot",
        "obs_modalities": ["scan", "rgb", "depth"],
    },
]

 

# Create the environment
env = og.Environment(cfg)

 

# Step!
start_time = time.perf_counter()
for _ in range(1000):
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())

end_time = time.perf_counter()
print(f'{(end_time - start_time)/1000} s')
og.shutdown()


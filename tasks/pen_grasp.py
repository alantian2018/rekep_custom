from omnigibson.tasks.grasp_task import GraspTask
import omnigibson as og
import yaml
import json
import sys
sys.path.append("..")
from utils import get_config


termination_config = {'max_steps' : 40}
scene_config_path = "/coc/flash8/atian31/repos/ReKep/configs/og_scene_file_pen.json"
scene_config = get_config(scene_config_path)

object_registry = scene_config['state']


# make the env
global_config = yaml.load(open('/coc/flash8/atian31/repos/ReKep/configs/config.yaml', "r"), Loader=yaml.FullLoader)

cfg = global_config['env']
cfg['scene']['scene_file'] = scene_config_path
cfg['robots'] = [cfg['robot']['robot_config']]
cfg['env']= cfg['og_sim']

breakpoint()
env = og.Environment(configs=dict(scene=cfg['scene'], robots=[cfg['robot']['robot_config']], env=cfg['og_sim']))
print('env made')

# load the grasp task; load env
pen_grasp_task = GraspTask(obj_name = 'pen_1',termination_config = termination_config, objects_config = [])

pen_grasp_task.load(env)
breakpoint()
pen_grasp_task.reset(env)
print('pen loaded')


og.shutdown()
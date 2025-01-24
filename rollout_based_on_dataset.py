import torch
import imageio
import os
from utils import *
from environment import *
import datetime
import cv2
import numpy as np
from tqdm import tqdm
import trimesh
import open3d as o3d
import numpy as np

def calculate_bbox_to_point(bbox, p):
    min_x, max_x = min(bbox[0][0],bbox[1][0]), max(bbox[0][0],bbox[1][0])
    min_y, max_y = min(bbox[0][1],bbox[1][1]), max(bbox[0][1],bbox[1][1])
    min_z, max_z = min(bbox[0][2],bbox[1][2]), max(bbox[0][2],bbox[1][2]) 
    
    dx = max(min_x - p[0], 0, p[0] - max_x)
    dy = max(min_y - p[1], 0, p[1] - max_y)
    dz = max(min_z - p[2], 0, p[2] - max_z)

    return (dx*dx + dy*dy + dz*dz)**.5

def add_independent_points_to_ply(mesh, points, radius=0.0007):
    """
    Add new independent points to a PLY file that already contains a mesh.
    The points will be added as additional vertices without being part of the mesh faces.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The existing mesh
    points : np.ndarray
        New points to add, shape should be (N, 3) where N is number of points
        
    Returns:
    --------
    trimesh.Trimesh
        New mesh object containing both original mesh and new points
    """
    # Convert points to numpy array if not already
    points = np.asarray(points)

    scene = trimesh.Scene()

    
    spheres = []
    for point in points:
        # Create a sphere at each point
        sphere = trimesh.creation.icosphere(radius=radius)
        sphere.apply_translation(point)  # Move sphere to the correct position
        spheres.append(sphere)

    # Combine the original mesh with all the spheres
    

    # Combine the original mesh, spheres, and paths
    combined = trimesh.util.concatenate([mesh] + spheres)
 

    return combined

if __name__ == '__main__':
    path = '/coc/flash8/atian31/repos/ReKep/data/2025-01-19-02-05-18/raw.pkl'

    # Load dataset
    dataset = torch.load(path)
    paths, rekep_program_dir = dataset[0], dataset[1]

    # Determine video output path
    dir_path = os.path.dirname(path)
    video_path = os.path.join(dir_path, f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.mp4')

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
    #video_buffer = imageio.get_writer(video_path, fps=20)
    mesh = env.og_env.get_mesh('pen_1')
    ee_points = []
    ee_no_cr  = []
    ee_last = []
    ee_point = env.og_env.get_centered_ee('pen_1')
    for rollout in tqdm(paths):
        
        object_info = rollout[0]  # Object state info
        env.reset(object_info)  # Reset environment to specific state

        trajs = rollout[1:]  # Remaining trajectory stages
        
        for stage in trajs:
            actions = stage['path'][1]
            contact_rich = stage['path'][-1]
            should_add = True
            for action, is_contact_rich in zip(actions, contact_rich):
                
                if should_add and is_contact_rich:
                    ee_last.append(ee_point)

                env._step(action)
                

                ee_point = env.og_env.get_centered_ee('pen_1')
                ee_points.append(ee_point)
                if should_add:
                    if is_contact_rich:
                        should_add = False
                    else:
                        ee_no_cr.append(ee_point)
                

                """
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
                """

    #video_buffer.close()  # Safely close video buffer
    path = os.path.join(dir_path,"trajectories/")
    os.makedirs(path, exist_ok=True)
    new_mesh = add_independent_points_to_ply(mesh, ee_points)
    new_mesh.export(path+"cr.ply")

    new_mesh = add_independent_points_to_ply(mesh, ee_no_cr)
    new_mesh.export(path+"no_cr.ply")

    new_mesh = add_independent_points_to_ply(mesh, ee_last)
    new_mesh.export(path+"ee_last.ply")

    ee_raw = {'ee_points':ee_points, 'ee_no_cr':ee_no_cr, 'ee_last':ee_last}
    torch.save(ee_raw, path+'ee_raw.pkl')

    print(f'Saved mesh to {path}')
    #print(f'Saved to {video_path}')
  
import os
import numpy as np

import copy
import json
import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger
import datetime
from rlkit_utils import experiment
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Add necessary command line args
 
 
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
from collections import OrderedDict
task_list = OrderedDict({
                'pen': {
                    'scene_file': './configs/og_scene_file_pen.json',
                    'instruction': 'Pick the white pen up in the middle with the gripper. Then, put the pen into the pencil holder.',
                    'rekep_program_dir': './vlm_query/pen'
                    },
            })

# Objective function
def run_experiment(debug=False):
    # Define agent-specific arguments
    trainer_kwargs = None
  
    trainer_kwargs = dict(
        discount=0.99,
        soft_target_tau=0.005,
        target_update_period=5,
        policy_lr=1e-3,
        qf_lr=0.0005,
        reward_scale=1,
        use_automatic_entropy_tuning=True,
    )
    
    # Construct variant to train
    model_path = '/nethome/atian31/flash8/repos/ReKep/pen_pickup_models/Pen_Pickup_rnn/20250120004223/models/model_epoch_2893_best_validation_14653.164111328126.pth'
    rekep_program_dir='/nethome/atian31/flash8/repos/ReKep/vlm_query/2025-01-19_02-06-17_pick_up_the_white_pen_in_the_middle._'
    if not debug:
        variant = dict(
            algorithm="SAC",
            seed=1,
            version="normal",
            model=model_path,
            normal_buffer_size=int(1E6),
            qf_kwargs=dict(
                hidden_sizes=[256,256],
            ),
            policy_kwargs=dict(
                hidden_sizes=[256,256],
            ),
            rekep_program_dir=rekep_program_dir,
            task_list=task_list,
            algorithm_kwargs=dict(
                num_epochs=1000,
                num_eval_steps_per_epoch=320,
                num_trains_per_train_loop=600,
                num_expl_steps_per_train_loop=800,
                min_num_steps_before_training=1200,
                expl_max_path_length=80,
                eval_max_path_length=80,
                batch_size=128,
            ),
            trainer_kwargs=trainer_kwargs,
       
        )
    else: # debug
        variant = dict(
            
            algorithm="SAC",
            seed=1,
            version="normal",
            model = model_path,
            normal_buffer_size=int(1E6),
            qf_kwargs=dict(
                hidden_sizes=[256,256],
            ),
            policy_kwargs=dict(
                hidden_sizes=[256,256],
            ),
            task_list=task_list,
            rekep_program_dir=rekep_program_dir,
            algorithm_kwargs=dict(
                num_epochs=500,
                num_eval_steps_per_epoch=50,
                num_trains_per_train_loop=50,
                num_expl_steps_per_train_loop=50,
                min_num_steps_before_training=50,
                expl_max_path_length=10,
                eval_max_path_length=10,
                batch_size=128,
            ),
            trainer_kwargs=trainer_kwargs,

        )
    variant['randomize'] = True #not debug
    # Set logging
    tmp_file_prefix = "pen"
    if debug:
        tmp_file_prefix +='_debug' 

    # Setup logger
    abs_root_dir = THIS_DIR
    tmp_dir = setup_logger(tmp_file_prefix, variant=variant, base_log_dir=abs_root_dir)
    ptu.set_gpu_mode(torch.cuda.is_available())  # optionally set the GPU (default=False

    # Run experiment
    experiment(variant)


if __name__ == '__main__':
    # First, parse args
    print('  Params: ')
     

    print('\n\n')

    # Execute run
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=0)
    args = parser.parse_args()
    run_experiment(args.debug)

    print('Finished run!')


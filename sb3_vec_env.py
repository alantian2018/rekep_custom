import copy
import time

import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from tqdm import trange
import torch.nn as nn
import omnigibson as og
from omnigibson.envs.env_base import Environment
import os
import json
import numpy as np
# Keep track of the last used env and what time, to require that others be reset before getting used
last_stepped_env = None
last_stepped_time = None
from environment import CustomOGEnv, RLEnvWrapper
from stable_baselines3.common.monitor import Monitor
from gym import Wrapper
from gymnasium.wrappers import NormalizeReward

def make_env(config, rekep_program_dir, render_on_step, use_oracle_reward,bc_policy):
    if bc_policy is None:
        env =CustomOGEnv(
            copy.deepcopy(dict(scene=config['scene'], robots=[config['robot']['robot_config']], env=config['og_sim'])),
            in_vec_env=True,
            config=copy.deepcopy(config),
            randomize=True,
            low_dim=True,
            use_oracle_reward=use_oracle_reward,
             
        )
    else:
        env = RLEnvWrapper(
            copy.deepcopy(dict(scene=config['scene'], robots=[config['robot']['robot_config']], env=config['og_sim'])),
            in_vec_env=True,
            config=copy.deepcopy(config),
            randomize=True,
            low_dim=True,
            use_oracle_reward=use_oracle_reward,
            bc_policy  =bc_policy
    )

    with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
        program_info = json.load(f)

    env.set_rekep_program_dir(rekep_program_dir)
    env.register_keypoints(program_info['init_keypoint_positions'])
    env.set_reward_function_for_stage(1)

    og.sim.play()
    env.post_play_load()
    env = Monitor(env)  # Wrap in Monitor for logging
    return env

class CustomSB3VectorEnvironment(DummyVecEnv):
    def __init__(self, num_envs, config, rekep_program_dir, render_on_step, bc_policy=None, use_oracle_reward=False):
        self.num_envs = num_envs
        self.render_on_step = render_on_step

        if og.sim is not None:
            og.sim.stop()

        # First we create the environments. We can't let DummyVecEnv do this for us because of the play call
        # needing to happen before spaces are available for it to read things from.
        if bc_policy is None:
         
            tmp_envs = [
                CustomOGEnv(
                    copy.deepcopy(dict(scene=config['scene'], robots=[config['robot']['robot_config']], env=config['og_sim'])),
                    in_vec_env=True,
                    config=copy.deepcopy(config),
                    randomize=True,
                    low_dim=True,
                    use_oracle_reward=use_oracle_reward
                
                )
                for _ in trange(num_envs, desc="Loading environments")
            ]
        else:
            tmp_envs = [
                RLEnvWrapper(
                    copy.deepcopy(dict(scene=config['scene'], robots=[config['robot']['robot_config']], env=config['og_sim'])),
                    in_vec_env=True,
                    config=copy.deepcopy(config),
                    randomize=True,
                    low_dim=True,
                    bc_policy=bc_policy,
                    use_oracle_reward=use_oracle_reward)
                for _ in trange(num_envs, desc="Loading environments")
            ]
      
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
                program_info = json.load(f)

        for env in tmp_envs:
            env.set_rekep_program_dir(rekep_program_dir)
            env.register_keypoints(program_info['init_keypoint_positions'])
            env.set_reward_function_for_stage(1)

        

        # Play, and finish loading all the envs
        og.sim.play()
        for env in tmp_envs:
            env.post_play_load()

        # Now produce some functions that will make DummyVecEnv think it's creating these envs itself
    
        env_fns = [lambda env_=env: env_ for env in tmp_envs]
        super().__init__(env_fns)

        # Keep track of our last reset time
        self.last_reset_time = time.time()

    

    def step_async(self, actions: th.tensor) -> None:
        # We go into this context in case the pre-step tries to call step / render
        with og.sim.render_on_step(self.render_on_step):
            global last_stepped_env, last_stepped_time

            if last_stepped_env != self:
                # If another environment was used after us, we need to check that we have been reset after that.
                # Consider the common setup where you have a train env and an eval env in the same process.
                # When you step the eval env, the physics state of the train env also gets stepped,
                # despite the train env not taking new actions or outputting new observations.
                # By the time you next step the train env your state has drastically changed.
                # To avoid this from happening, we add a requirement: you can only be stepping
                # one vector env at a time - if you want to step another one, you need to reset it first.
                assert (
                    last_stepped_time is None or self.last_reset_time > last_stepped_time
                ), "You must call reset() before using a different environment."
                last_stepped_env = self
                last_stepped_time = time.time()

            self.actions = actions
            
            if self.actions.shape[-1] != 12:
                self.actions = np.pad(actions, ((0, 0), (4, 0)))
            self.actions = np.concatenate([self.actions, self.actions[:, [-1]]], axis=1)
        

            for i, action in enumerate(self.actions):
                self.envs[i]._pre_step(action)

    def step_wait(self) -> VecEnvStepReturn:
        with og.sim.render_on_step(self.render_on_step):
            # Step the entire simulation
            og.sim.step()

            for env_idx in range(self.num_envs):
                obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[
                    env_idx
                ]._post_step(self.actions[env_idx])
                # convert to SB3 VecEnv api
                self.buf_dones[env_idx] = terminated or truncated
                # See https://github.com/openai/gym/issues/3102
                # Gym 0.26 introduces a breaking change
                self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated
              
                if self.buf_dones[env_idx]:
                    # save final observation where user can get it, then reset
                    self.buf_infos[env_idx]["terminal_observation"] = obs
                    obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
                self._save_obs(env_idx, obs)
             
            return (
                self._obs_from_buf(),
                self.buf_rews,
                self.buf_dones,
                copy.deepcopy(self.buf_infos),
            )

    def reset(self):
        
        with og.sim.render_on_step(self.render_on_step):
            self.last_reset_time = time.time()

            for env_idx in range(self.num_envs):
                maybe_options = {"options": self._options[env_idx]} if self._options[env_idx] else {}
                self.envs[env_idx].reset( **maybe_options)

            # Settle the environments
            # TODO: fix this once we make the task classes etc. vectorized
            for _ in range(13):
                og.sim.step()

            # Get the new obs
            for env_idx in range(self.num_envs):
                obs =  self.envs[env_idx].get_low_dim_obs()
                self._save_obs(env_idx, obs)

            # Seeds and options are only used once
            self._reset_seeds()
            self._reset_options()
            return self._obs_from_buf()

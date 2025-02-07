"""
Set of functions and classes that are modified versions of existing ones in rlkit
"""
import abc
import torch
from rlkit.core import logger, eval_util
import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm, _get_epoch_timings
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector
from rlkit.samplers.data_collector import PathCollector
import os
from ReKepCollector import ReKepCollector
from collections import OrderedDict
import traceback
import numpy as np
import time

import rlkit.pythonplusplus as ppp
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer

def collect_rekep_paths(env, task_list, n, save_path=None):
    task = task_list['pen']
    scene_file = task['scene_file']
    instruction = task['instruction']

    rekep_program_dir = None
    if not env.randomize:
            print('Not randomizing...')
            rekep_program_dir = '/nethome/atian31/flash8/repos/ReKep/vlm_query/2024-12-04_17-31-43_pick_the_white_pen_up_with_the_gripper._lift_it_off_the_table_by_15_cm_'
    
    main = ReKepCollector(scene_file, rekep_program_dir=rekep_program_dir, visualize=False, og_env=env, randomize=env.randomize)
    paths = []
    while len(paths) < n:
        try:
            print(f"Collecting path {len(paths)+1}")
            start_time = time.time()


            buffer = main.perform_task(
                    instruction,
                    disturbance_seq=None,
                    render= len(paths) <= 3
                    )
            print(f"Collected path {len(paths)}, took {time.time() - start_time} seconds")
            if rekep_program_dir is None:
                rekep_program_dir = main.rekep_program_dir
            paths.append(buffer)
            if save_path and len(paths) % 20 == 0:
                torch.save([paths, rekep_program_dir], save_path)
                print(save_path)
                 

        except Exception as e:
            print(traceback.format_exc())
            print('Error in path collecting, retrying...')
            continue

    return paths, rekep_program_dir

class CustomEnvReplayBuffer:
    """
    Essentially 2 replay buffers in one; See https://arxiv.org/pdf/2410.21845 for more details.
    (1) buffer maintains "expert" trajectory transitions.
    (2) buffer maintains normal transitions.
    """
    def __init__(
        self,
        max_expert_buffer_size,
        max_normal_buffer_size, 
        env,
        expert_sample_split=0.5 # what proportion of batch should be sampled from expert buffer. Default 1/2
    ):
        self.expert_buffer = EnvReplayBuffer(max_expert_buffer_size, env)
        self.normal_buffer = EnvReplayBuffer(max_normal_buffer_size, env)
        self.env = env
        self.expert_sample_split = expert_sample_split
        self._ob_space = env.observation_space
        self._action_space = env.action_space
    
    def add_expert_paths(self, paths):
        self.expert_buffer.add_paths(paths)

    def add_normal_paths(self, paths):
        self.normal_buffer.add_paths(paths)

    def random_batch(self, batch_size):
        no_expert_samples = int(batch_size * self.expert_sample_split)
        no_normal_samples  = batch_size - no_expert_samples
        
        expert_samples = self.expert_buffer.random_batch(no_expert_samples)
        normal_samples = self.normal_buffer.random_batch(no_normal_samples)

        batch = dict()
        permutation = np.random.permutation(batch_size)

        for i in expert_samples:
            batch[i] = np.vstack((expert_samples[i],normal_samples[i]))
            batch[i] = batch[i][permutation]

        return batch

    def clear(self):
        self.expert_buffer.clear()
        self.normal_buffer.clear()
        

    def end_epoch(self, epoch):
        return
    
    def get_snapshot(self):
        return {}
    
    def get_diagnostics(self):
        
        return OrderedDict([
            ('size_expert', self.expert_buffer._size),
            ('size_normal', self.normal_buffer._size)
        ])


    def terminate_episode(self):
        pass


class CustomBaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    """
    Identical to normal Base RLA algo, except we don't save 'env' after each iteration (will crash since cython can't
    compress MujocoEnv into C primitives OTS)
    """
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: DataCollector,
            evaluation_data_collector: DataCollector,
            replay_buffer: ReplayBuffer,
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = 0

        self.post_epoch_funcs = []

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _end_epoch(self, epoch):
        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        gt.stamp('saving')
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            if k == 'env':
                continue
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            if k == 'env':
                continue
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix='exploration/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="exploration/",
        )
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths),
                prefix='evaluation/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/",
        )

        """
        Misc
        """
        gt.stamp('logging', )
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass


class CustomBatchDRRLAlgorithm(CustomBaseRLAlgorithm, metaclass=abc.ABCMeta):
    """
    A custom BatleRLAlgorithm class that extends the vanilla rlkit version in the following ways:
    -Exploration and evaluation environments can have different horizons
    -Max path length is now correspondingly unique to both exploration and evaluation environment
    -Logger now additionally stores the following:
        -cumulative rewards (return) of evaluation environment at timestep where
            exploration horizon ends, if horizon_expl < horizon_eval
        -normalized rewards and returns of evaluation environment
    """
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
           
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            expl_max_path_length,
            eval_max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            task_list,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            number_rekep_paths=7,
            add_rekep_every_n=100
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.expl_max_path_length = expl_max_path_length
        self.eval_max_path_length = eval_max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.number_rekep_paths = number_rekep_paths
        self.add_rekep_every_n = add_rekep_every_n
        self.task_list = task_list

    def _train(self):
        if self.min_num_steps_before_training > 0:
            # OK, this works only for the normal env, but not the rekep ones.
            # how do we want to handle sampling??? Start with a constant and then add one every n epochs?
            init_rekep_paths  = collect_rekep_paths(self.expl_env, self.task_list, self.number_rekep_paths)
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.expl_max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_normal_paths(init_expl_paths)
            self.replay_buffer.add_expert_paths(init_rekep_paths)
            
           
            
            self.expl_data_collector.end_epoch(-1)

            #probably want to add a bunch of rekep ones 

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.eval_max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=False,
            )
            gt.stamp('evaluation sampling', unique=False)

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.expl_max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_normal_paths(new_expl_paths)

                if ((epoch + 1) % self.add_rekep_every_n == 0):
                    path = collect_rekep_paths(self.expl_env,self.task_list, 1)
                    self.replay_buffer.add_expert_paths(path)

                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix='exploration/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="exploration/",
        )
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths),
                prefix='evaluation/',
            )
        logger.record_dict(
            get_custom_generic_path_information(eval_paths, self.expl_max_path_length, self.trainer.reward_scale),
            prefix="evaluation/",
        )

        """
        Misc
        """
        gt.stamp('logging', )
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

        
class CustomBatchRLAlgorithm(CustomBaseRLAlgorithm, metaclass=abc.ABCMeta):
    """
    A custom BatleRLAlgorithm class that extends the vanilla rlkit version in the following ways:
    -Exploration and evaluation environments can have different horizons
    -Max path length is now correspondingly unique to both exploration and evaluation environment
    -Logger now additionally stores the following:
        -cumulative rewards (return) of evaluation environment at timestep where
            exploration horizon ends, if horizon_expl < horizon_eval
        -normalized rewards and returns of evaluation environment
    """
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            expl_max_path_length,
            eval_max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.expl_max_path_length = expl_max_path_length
        self.eval_max_path_length = eval_max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.expl_max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.eval_max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling', unique=False)

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.expl_max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix='exploration/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="exploration/",
        )
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths),
                prefix='evaluation/',
            )
        logger.record_dict(
            get_custom_generic_path_information(eval_paths, self.expl_max_path_length, self.trainer.reward_scale),
            prefix="evaluation/",
        )

        """
        Misc
        """
        gt.stamp('logging', )
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)




class CustomTorchBatchRLAlgorithm(CustomBatchRLAlgorithm):
    """Identical to normal TBRLA, but simply extends from our custom BatchRLAlgorithm instead"""
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


def get_custom_generic_path_information(paths, path_length, reward_scale, stat_prefix=''):
    """
    Get an OrderedDict with a bunch of statistic names and values.

    Differs from normal rlkit utility function in the following ways:
    Grabs normalized reward / return values where reward is normalized to 1.0
    Grabs cumulative reward specified accumulated at @path_length timestep
    """
    statistics = OrderedDict()
    returns = [sum(path["rewards"]) for path in paths]

    # Grab returns accumulated up to specified timestep
    expl_returns = [sum(path["rewards"][:path_length]) for path in paths]

    rewards = np.vstack([path["rewards"] for path in paths])
    # norm_rewards = [path["rewards"] / reward_scale for path in paths]
    statistics.update(eval_util.create_stats_ordered_dict('Rewards', rewards,
                                                stat_prefix=stat_prefix))
    statistics.update(eval_util.create_stats_ordered_dict('Returns', returns,
                                                stat_prefix=stat_prefix))

    # Add extra stats
    statistics.update(eval_util.create_stats_ordered_dict('ExplReturns', expl_returns,
                                                          stat_prefix=stat_prefix))

    actions = [path["actions"] for path in paths]
    if len(actions[0].shape) == 1:
        actions = np.hstack([path["actions"] for path in paths])
    else:
        actions = np.vstack([path["actions"] for path in paths])
    statistics.update(eval_util.create_stats_ordered_dict(
        'Actions', actions, stat_prefix=stat_prefix
    ))
    statistics['Num Paths'] = len(paths)
    statistics[stat_prefix + 'Average Returns'] = eval_util.get_average_returns(paths)
    
    for info_key in ['env_infos', 'agent_infos']:
        if info_key in paths[0]:
            all_env_infos = [
                ppp.list_of_dicts__to__dict_of_lists(p[info_key])
                for p in paths
            ]
         #   print(paths)
         #   print(f'all_env_infos {all_env_infos}')
            for k in all_env_infos[0].keys():
                final_ks = np.array([info[k][-1] for info in all_env_infos])
                first_ks = np.array([info[k][0] for info in all_env_infos])
                all_ks = np.concatenate([info[k] for info in all_env_infos])
         #       print(final_ks, first_ks, all_ks)
                statistics.update(eval_util.create_stats_ordered_dict(
                    stat_prefix + k,
                    final_ks,
                    stat_prefix='{}/final/'.format(info_key),
                ))
                statistics.update(eval_util.create_stats_ordered_dict(
                    stat_prefix + k,
                    first_ks,
                    stat_prefix='{}/initial/'.format(info_key),
                ))
                statistics.update(eval_util.create_stats_ordered_dict(
                    stat_prefix + k,
                    all_ks,
                    stat_prefix='{}/'.format(info_key),
                ))
    
    return statistics


def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        video_writer=None,
):
    """
    Custom rollout function that extends the basic rlkit functionality in the following ways:
    - Allows for automatic video writing if @video_writer is specified

    Added args:
        video_writer (imageio.get_writer): If specified, will write image frames to this writer

    The following is pulled directly from the rlkit rollout(...) function docstring:

    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    env.reset()
    o = env.get_low_dim_obs()
    agent.reset()
    next_o = None
    path_length = 0

    # Only render if specified AND there's no video writer
    if render and video_writer is None:
        env.render(**render_kwargs)

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d,t, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a[4:])

        # Grab image data to write to video writer if specified
        if video_writer is not None:
            cam_obs = env.get_cam_obs()
            rgb = cam_obs[1]['rgb']
            video_writer.append_data(rgb)

        agent_infos.append({})
        env_infos.append({})
        path_length += 1
        if d:
            break
        o = next_o
        
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.asarray(next_o)
        next_o = np.array([next_o])

    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
 
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
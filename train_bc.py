import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch
import numpy as np
from robomimic.config import config_factory
from robomimic.algo import algo_factory
from robomimic.utils.dataset import SequenceDataset
from robomimic.config import Config
import json
import robomimic.utils.train_utils as TrainUtils
from torch.utils.data import DataLoader
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
import time
import os
import psutil
import h5py

def train_offline(dataset_path, config_path, algo_name="bc"):
    """
    Train a model offline using RoboMimic.
    
    Args:
        dataset_path: Path to saved dataset
        model_dir: Directory to save model checkpoints
        algo_name: Algorithm to use (e.g., 'bc' for behavioral cloning)
    """
    # Create config
    with open (config_path) as f:
        config = Config(json.load(f))
    assert config
    if dataset_path:
        config.train.data = dataset_path

    log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)
    f= h5py.File(config.train.data)
    with open (
        os.path.join(
            os.path.dirname(log_dir),
            'rekep_program_dir.txt'
        ), 'w') as wf:  
        wf.write(f.attrs['rekep_program_dir'])

    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
  
    
    ObsUtils.initialize_obs_utils_with_config(config)
    # Load dataset
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=None,
        verbose=True
    )

    
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])

    
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True
    )


    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True
        )
    else:
        valid_loader = None
 
   

    #   make algo
    model = algo_factory(
        algo_name,
        config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device='cuda' if config.train.cuda else 'cpu',
    )

    best_valid_loss = None
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1. for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    for epoch in range(1, config.train.num_epochs + 1): # epoch numbers start at 1
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
            
        )
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and \
                (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
            epoch_check = (config.experiment.save.every_n_epochs is not None) and \
                (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
            epoch_list_check = (epoch in config.experiment.save.epochs)
            should_save_ckpt = (time_check or epoch_check or epoch_list_check)
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print("Train Epoch {}".format(epoch))
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Train/{}".format(k), v, epoch)

        # Evaluate the model on validation set
        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps)
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Valid/{}".format(k), v, epoch)

            print("Validation Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        if should_save_ckpt:
            TrainUtils.save_model(
                    model=model,
                    config=config,
                    env_meta = None,
                    shape_meta=shape_meta,
                    ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                    obs_normalization_stats=obs_normalization_stats,
            )

            # Finally, log memory usage in MB
            process = psutil.Process(os.getpid())
            mem_usage = int(process.memory_info().rss / 1000000)
            data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
            print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # terminate logging
    data_logger.close()

# Example usage
if __name__ == "__main__":
    
    config_path = '/nethome/atian31/flash8/repos/ReKep/configs/bc_rnn_gmm_pen_pick_500_with_contact_rich.json'
    dataset_path=None
    train_offline(dataset_path, config_path)
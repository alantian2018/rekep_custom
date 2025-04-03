#!/bin/bash
#SBATCH --job-name=train_rl
#SBATCH --partition=overcap
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17
#SBATCH --gres=gpu:a40:1
#SBATCH --qos=long
#SBATCH --output=/nethome/atian31/flash8/repos/ReKep/slurm_logs/TQC/%j.out



source ~/.bashrc
cd /nethome/atian31/flash8/repos/ReKep/
source activate omnigibson

python train_sb3.py --oracle --algo TQC

# export MASTER_PORT=12355
# export WORLD_SIZE=1
# export MASTER_ADDR=$(hostname)
 


# srun torchrun \
#    --nproc_per_node=4 \
#    --nnodes=1 \
#    --node_rank=0 \
#    --master_addr="$MASTER_ADDR" \
#    --master_port="$MASTER_PORT" \
#    test.py
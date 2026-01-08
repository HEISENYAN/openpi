#!/bin/bash


#SBATCH --job-name=yanzhengyang
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=48:00:00
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --nice=0
#SBATCH --account=peilab
#SBATCH --output=train_aloha_test_4.out


nvidia-smi

module purge
#module load slurm
#module avail cuda
module load slurm
#module load cuda12.2/toolkit/12.2.2
#module avail tensorrt
source /home/congcong/miniconda3/bin/activate RoboTwin

#nvidia-smi
#nvcc -V
export PYTHONUNBUFFERED=1
#/project/peilab/yanzhengyang/.cache/huggingface/lerobot/folding_clothes/data/chunk-000/episode_000028.parquet
#python -u pipeline.py
#python -u pipeline-scenario-2.py #world_model_config.py
#bash vla-server.bash
export XDG_CACHE_HOME=/project/peilab/yanzhengyang/.cache
#bash eval_distributed.sh adjust_bottle demo_clean demo_clean 50 0 0 
bash finetune.sh pi0_base_aloha_robotwin_full folding_clothes_test_4 0,1
#bash quick_eval.bash
#uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_base_torch_full --exp_name pytorch_handover_block
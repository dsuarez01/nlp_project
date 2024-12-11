#!/bin/bash
#SBATCH --job-name=<insert name of run here>
#SBATCH --partition=xeon-g6-volta
#SBATCH --output=<insert log directory here>
#SBATCH --error=<insert error directory here>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=1-00:00:00 # training for 1 day

# load conda environment and set library path
module load anaconda/Python-ML-2024a
export LD_LIBRARY_PATH=/state/partition1/llgrid/pkg/anaconda/python-ML-2024a/lib:${LD_LIBRARY_PATH}
export PATH=$HOME/.local/bin:$PATH
export WANDB_MODE=offline # have to sync results manually

date
echo "Running my job on $HOSTNAME"

python -u main.py

echo "Job completed on $(date) on $(hostname)"
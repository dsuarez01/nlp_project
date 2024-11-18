#!/bin/bash
#SBATCH --job-name=t64s
#SBATCH --partition=xeon-g6-volta
#SBATCH --output=/home/gridsan/dsuarez/nlp/output_%j.log
#SBATCH --error=/home/gridsan/dsuarez/nlp/error_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=1-00:00:00 # training for 1 day

# load conda environment and set library path
module load anaconda/Python-ML-2024a
export LD_LIBRARY_PATH=/state/partition1/llgrid/pkg/anaconda/python-ML-2024a/lib:${LD_LIBRARY_PATH}

date
echo "Running my job on $HOSTNAME"

python -u main.py

echo "Job completed on $(date) on $(hostname)"
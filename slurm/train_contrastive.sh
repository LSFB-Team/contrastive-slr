#!/bin/bash
# Submission script for Lucia
#SBATCH --job-name=backbone_lsfb
#SBATCH --time=30:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=6144 
#SBATCH --partition=ia
#
#SBATCH --mail-user=jerome.fink@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb
#
#SBATCH --output=./outputs/backbone_multi_LS.out

module purge
module load Python/3.10.4-GCCcore-11.3.0

source ./venv/bin/activate
pip install -r requirements.txt
nvidia-smi


python ./src/cslr/launch_training.py
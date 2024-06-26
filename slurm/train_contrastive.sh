#!/bin/bash
# Submission script for Lucia
#SBATCH --job-name=backbone_lsfb
#SBATCH --time=00:10:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=6144 
#SBATCH --partition=gpu
#
#SBATCH --mail-user=jerome.fink@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb
#
#SBATCH --output=./outputs/test.out

module purge
module load EasyBuild/2023a
module load Python/3.11.3-GCCcore-12.3.0

source ./venv/bin/activate
pip install -r requirements.txt
nvidia-smi


python ./src/cslr/launch_training.py
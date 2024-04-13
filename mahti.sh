#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH -p gpusmall
#SBATCH -t 1-12:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --account=project_2001659

module purge
module load python-data cuda openblas

source venv/bin/activate

set -euo pipefail

python3 main.py

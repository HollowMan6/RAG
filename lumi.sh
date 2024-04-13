#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -p dev-g
#SBATCH -t 3:00:00
#SBATCH --gpus-per-node=mi250:1
#SBATCH --account=project_462000007
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

module --quiet purge
module load cray-python CrayEnv rocm/5.6.1

source venv/bin/activate

set -euo pipefail

python3 main.py

#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J test_apu
#
#SBATCH --ntasks=1
#SBATCH --constraint="apu"
#
# --- default case: use a single APU on a shared node ---
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110000
#
# --- uncomment to use 2 APUs on a full node ---
# #SBATCH --gres=gpu:2
# #SBATCH --cpus-per-task=48
# #SBATCH --mem=220000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=david.carreto.fidalgo@mpcdf.mpg.de
#SBATCH --time=12:00:00

module purge
module load apptainer/1.4.1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export CUPIDO_TOML=cupido_8B.toml

export HF_HOME=/ptmp/dcfidalgo/huggingface
export TMPDIR=$JOB_SHMTMPDIR
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1

srun apptainer exec -B /ptmp/dcfidalgo/huggingface -B .:$HOME containers/viper.sif python main.py


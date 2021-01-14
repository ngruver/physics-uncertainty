#!/usr/bin/env bash

#SBATCH --job-name=CHNN-UNC
##SBATCH --output=
##SBATCH --error=
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_12gb

set -e

if [[ -z "${WANDB_SWEEP_ID}" ]]; then
  echo "Missing WANDB_SWEEP_ID"
  exit 1
fi

# source "${HOME}/.bash_profile"

export WANDB_MODE=run
export WANDB_DIR="${LOGDIR}"
export WANDB_PROJECT="physics-uncertainty-exps"
export WANDB_NAME="${SLURM_JOB_NAME}--${SLURM_JOB_ID}"

cd "${WORKDIR}/physics-uncertainty"

export PYTHONPATH="$(pwd):${PYTHONPATH}"

source $(conda info --base)/bin/deactivate
source $(conda info --base)/bin/activate phy-unc

wandb agent --count=1 ${WANDB_SWEEP_ID}

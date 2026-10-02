#!/bin/bash
#SBATCH --job-name=POST_v30_EnKF_full
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --account=lp_ees_swm_ls_001
#SBATCH --cluster=wice
#SBATCH --partition=batch

# Transforms the per-cell outputs of the whole submit_v30_array.slurm array into
# RESULTS_*/resultsSpatialGrid.zarr. Submit with --dependency=afterok:<array job id>.

ROOTDIR=/data/leuven/378/vsc37876/MuSAsummerjob
RUNDIR=/staging/leuven/stg_00024/MSC_TMP/Snowshop/MuSAruns_Elisabeth/v30/y035x200

# "ml load GCC" fails on wice (Lmod: needs a version) and was always ignored by
# runMuSAtile.sh; the conda python below is called by absolute path, so keep it non-fatal
ml load GCC || true
ml load Miniconda3 || true
set -euo pipefail
pythonpath="/data/leuven/378/vsc37876/conda/envs/MuSAenv/bin/python"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd $ROOTDIR

export MUSA_CONFIG=$(cat $RUNDIR/MUSA_CONFIG_PATH)
$pythonpath postprocessMuSArunTile.py

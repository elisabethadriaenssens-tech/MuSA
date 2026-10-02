#!/bin/bash
#SBATCH --job-name=PRE_v32_EnKF_full_fix
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=lp_ees_swm_ls_001
#SBATCH --cluster=wice
#SBATCH --partition=batch

# Preprocessing only (run dirs, forcings, Obs, adjusted config), run once before the
# submit_v32_array.slurm job array so the array tasks don't race on FORCINGS/ and Obs/.

ROOTDIR=/data/leuven/378/vsc37876/MuSAsummerjob
EXPERIMENT=$ROOTDIR/experiments/v32_enkf_full_fix.yml
IDX_TILE=1
RUNDIR=/staging/leuven/stg_00024/MSC_TMP/Snowshop/MuSAruns_Elisabeth/v32/y035x200

# "ml load GCC" fails on wice (Lmod: needs a version) and was always ignored by
# runMuSAtile.sh; the conda python below is called by absolute path, so keep it non-fatal
ml load GCC || true
ml load Miniconda3 || true
set -euo pipefail
pythonpath="/data/leuven/378/vsc37876/conda/envs/MuSAenv/bin/python"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd $ROOTDIR

# run dirs + DEM/mask symlinks, then forcings/Obs/config for the tile
$pythonpath helpers/setup_run_dirs.py --experiment "$EXPERIMENT"
cfg_path=$($pythonpath preprocessMuSArunTile.py --experiment "$EXPERIMENT" --idx_tile $IDX_TILE | tail -n 1)
echo "$cfg_path" > $RUNDIR/MUSA_CONFIG_PATH
echo "Adjusted config: $cfg_path"

# fail here (so an afterok-dependent array never starts) if the config or Obs are off
$pythonpath $ROOTDIR/submission/check_preprocessing.py "$cfg_path" EnKF "[0.09]"

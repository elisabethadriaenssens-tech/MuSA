#!/bin/bash
#SBATCH --job-name=DA_v31_PBS_full
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=1-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=lp_ees_swm_ls_001
#SBATCH --cluster=wice
#SBATCH --partition=batch

# HPC.array run of MuSA: task i simulates the cells with index % 4 == i-1.
# Requires preprocess_v31.slurm to have finished (submit with --dependency=afterok).

ROOTDIR=/data/leuven/378/vsc37876/MuSAsummerjob
RUNDIR=/staging/leuven/stg_00024/MSC_TMP/Snowshop/MuSAruns_Elisabeth/v31/y035x200
NTASKS=4

# "ml load GCC" fails on wice (Lmod: needs a version) and was always ignored by
# runMuSAtile.sh; the conda python below is called by absolute path, so keep it non-fatal
ml load GCC || true
ml load Miniconda3 || true
set -euo pipefail
pythonpath="/data/leuven/378/vsc37876/conda/envs/MuSAenv/bin/python"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd $ROOTDIR  # fsm_src_path (./FSM2/) is relative to the repo root

# FSM runs every ensemble member in a directory under tmp_path; with hundreds of processes
# doing that on staging the run is I/O-bound (~2.6x slower per cell), so each task uses
# node-local /dev/shm through its own copy of the adjusted config
LOCAL_TMP=/dev/shm/musa_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $LOCAL_TMP/fsm
trap 'rm -rf $LOCAL_TMP' EXIT
sed "s#^tmp_path = .*#tmp_path = '$LOCAL_TMP/fsm'#" $(cat $RUNDIR/MUSA_CONFIG_PATH) > $LOCAL_TMP/config.py
export MUSA_CONFIG=$LOCAL_TMP/config.py
grep "^tmp_path" $MUSA_CONFIG
echo "Task ${SLURM_ARRAY_TASK_ID}/${NTASKS} with config $MUSA_CONFIG"

$pythonpath runMuSAtile.py $NTASKS 36 ${SLURM_ARRAY_TASK_ID}

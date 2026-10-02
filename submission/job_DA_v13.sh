#!/bin/bash
#SBATCH --job-name=DA_v13_EnKF_QR_const03
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --account=lp_ees_swm_ls_001
#SBATCH --cluster=wice
#SBATCH --partition=batch

ROOTDIR=/data/leuven/378/vsc37876/MuSAsummerjob

bash $ROOTDIR/submission/runMuSAtile.sh 1 $ROOTDIR/experiments/v13_enkf_qr_const.yml

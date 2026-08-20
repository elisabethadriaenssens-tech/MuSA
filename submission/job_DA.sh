#!/bin/bash
#SBATCH --job-name=DA_v2_IP
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=lp_ees_swm_ls_001
#SBATCH --cluster=wice
#SBATCH --partition=batch

ROOTDIR=/data/leuven/378/vsc37876/MuSAsummerjob

bash $ROOTDIR/submission/runMuSAtile.sh 1 $ROOTDIR/experiments/v2_ip.yml

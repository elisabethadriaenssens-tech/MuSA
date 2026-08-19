#!/bin/bash
: '
Shell script to run MuSA on a single tile.
The shell script first calls the preprocessing script to generate the forcing zarr files
and adjust the config_template py-file to the setups of the current run. Next, it runs
MuSA usign the adjusted config file and forcings.

NOTE: in the MuSA run script, the final results are transformed from the pickle files to
an xr dataset/zarr store!

author: Lucas Boeykens - lucas.boeykens@ugent.be lucas.boeykens@kuleuven.be
'

#--- load modules ---
ml load GCC
ml load Miniconda3

#--- hard coded paths ---
ROOTDIR=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)

#--- inputs ---
idx_run=${1:-19} # which tile handeled
experiment=${2:-"$ROOTDIR/experiments/v1_const.yml"} # experiment config yml (merged on top of base.yml in the same directory)
#pythonpath="/kyukon/data/gent/vo/000/gvo00090/\
#vsc44965/Conda/envs/MuSAenv/bin/python"

pythonpath="/data/leuven/378/vsc37876/conda/envs/MuSAenv/bin/python"

# --- preprocess ---
cfg_path=$($pythonpath $ROOTDIR/preprocessMuSArunTile.py --experiment "$experiment" \
                                            --idx_tile "$idx_run"
                                            )

# #--- run MuSA --- -> also exports the results to another format
export MUSA_CONFIG=$cfg_path

${pythonpath} ${ROOTDIR}/runMuSAtile.py # ADDED: ROOTDIR

#--- remove the config file after the run ---
#rm -f $cfg_path  # keep for debugging
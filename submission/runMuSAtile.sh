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

#--- inputs ---
idx_run=${1:-19}
rootdirMuSAruns=${2:-"/kyukon/data/gent/vo/000/gvo00090/vsc44965/Doctoraat/Python/Machine_learning/Traditional_MLA/sd/MuSA/test2/"}
date_ini=${3:-"2015-09-01 00:00"}
date_end=${4:-"2015-09-30 23:00"}
model_only_sites=${5:-True}
remove_output_cells=${6:-False}
store_measurements=${7:-"/kyukon/data/gent/vo/000/gvo00090/SNOWSHOP/measurements/insitu/Alps_dataset_SD.nc"}
tilefile=${8:-"/kyukon/data/gent/vo/000/gvo00090/SNOWSHOP/auxdata/mountain_tiles/Alps_tiles.txt"}

#--- hard coded paths ---
ROOTDIR=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
#pythonpath="/kyukon/data/gent/vo/000/gvo00090/\
#vsc44965/Conda/envs/MuSAenv/bin/python"

pythonpath="/data/leuven/378/vsc37876/conda/envs/MuSAenv/bin/python"

# --- preprocess ---
cfg_path=$($pythonpath $ROOTDIR/preprocessMuSArunTile.py --date_ini "$date_ini" \
                                            --date_end "$date_end" \
                                            --rootdirMuSAruns "$rootdirMuSAruns" \
                                            --model_only_sites "$model_only_sites" \
                                            --tilefile "$tilefile" \
                                            --idx_tile "$idx_run" \
                                            --remove_output_cells "$remove_output_cells" \
                                            --store_measurements "$store_measurements"
                                            )

# #--- run MuSA --- -> also exports the results to another format
export MUSA_CONFIG=$cfg_path

${pythonpath} ${ROOTDIR}/runMuSAtile.py # ADDED: ROOTDIR

#--- remove the config file after the run ---
rm -f $cfg_path
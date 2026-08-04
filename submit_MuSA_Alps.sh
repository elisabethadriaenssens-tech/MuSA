#!/bin/bash
: '
Shell script to submit the MuSA runs for the Alps on a specific cluster.
The shell script performs parallel job submission over processing tiles.
Each tile is processed in a separate job, which calls the runMuSAtile.sh script!

author: Lucas Boeykens - lucas.boeykens@ugent.be lucas.boeykens@kuleuven.be
adapted for VSC Leuven by Elisabeth Adriaenssens
'

#--- inputs ---
cluster=${1:-"wice"}                       # CHANGED: was "skiddo"
modelOnlySites=${2:-True}
date_ini=${3:-"2015-08-01 00:00"}          # CHANGED: matches existing forcings zarr
date_end=${4:-"2024-08-31 23:00"}
rootdirMuSAruns=${5:-"PATH_TBD/DATA/"}     # TODO: fill in once access to data
remove_output_cells=${6:-True}

# -- hard coded ---
store_measurements="PATH_TBD/Alps_dataset_SD.nc"   # TODO: once access data
tiledir="PATH_TBD/Tile_lists/"                     # TODO: once access data

region="Alps"

ROOTDIR=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
submitJobsScript="${ROOTDIR}/submission/runMuSAtile.sh"

mkdir -p "${ROOTDIR}/logs"                 # ADDED: logs/ must exist or sbatch fails

cpus=24
if [ "$modelOnlySites" == "True" ]; then
    time="01:00:00"
else
    time="12:00:00"
fi

# find the file with the tiles
tilefile=$(find $tiledir -name "${region}_*.txt" | head -n 1)

#--- submit the jobs ---
file=$(find $tiledir -name "${region}_*.txt" | head -n 1)
ntiles="$(wc -l <"$file")"

# submit
job=$(sbatch --job-name=MuSArun_Alps \
  --output=${ROOTDIR}/logs/output_MuSArun_Alps_%A_%a.log \
  --error=${ROOTDIR}/logs/error_MuSArun_Alps_%A_%a.log \
  --nodes=1 \
  --cpus-per-task=$cpus \
  --time=$time \
  --clusters=${cluster} \
  --partition=batch_icelake \
  --account=lp_ees_swm_ls_001 \
  --array=0-$((ntiles-1)) \
  --wrap="$submitJobsScript \${SLURM_ARRAY_TASK_ID} \
                        $rootdirMuSAruns \
                        \"$date_ini\" \
                        \"$date_end\" \
                        $modelOnlySites \
                        $remove_output_cells \
                        $store_measurements \
                        $tilefile"
                        )
echo "$job"
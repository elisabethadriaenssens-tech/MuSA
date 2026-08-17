'''
Script that performs the preprocessing of the data and config file before running MuSA!
Within the script, the following steps are performed:

1) checks if the forcings are available. 
    Currently, it checks if the forcings are available for the specified time period (start_date, end_date)!

2) if no forcing file found, generate a zarr store with the forcings!

3) Regrids a DEM file to the resolution and grid of the forcings.
    Note that the DEM files are stored at a specific directory, whcih you might have to adjust

4) Adjusts the config file for the specific tile (tx,ty) and time period (start_date, end_date)

contact: Lucas Boeykens -lucas.boeyykens@ugent.be lucas.boeykens@kuleuven.be
'''

#---modules---
from glob import glob
import os, sys, argparse, yaml
import pandas as pd
project_root=os.getcwd()
sys.path.append(project_root)
from modules.dem_tools import RegridDEMtoForcings
import modules.prepareForcingsZarr as prepForcing_tools
import modules.prepareRunTile_tools as prepRuntile_tools
import modules.internal_fns as ifn

#---custom functions---
def load_config(
        ConfigFile:str=None,
        ) -> dict:
    '''
    Function to load a YAML configuration file.
    '''
    with open(ConfigFile, "r") as f:
        return yaml.safe_load(f)


#keys read from the merged base.yml + experiment yml, mapped to the kwarg
#names expected by PrepareRunTile / pd.read_csv (tilefile)
EXPERIMENT_CONFIG_KEYS = {
    "date_ini": "date_ini",
    "date_end": "date_end",
    "runs_root": "rootdirMuSAruns",
    "model_only_sites": "model_only_sites",
    "tilefile": "tilefile",
    "remove_output_cells": "remove_output_cells",
    "store_measurements": "store_measurements",
    "implementation": "implementation",
}


def load_experiment_config(
        experiment_path:str=None,
        ) -> dict:
    '''
    Function to load an experiment YAML config file, merged on top of the
    shared base.yml config (expected alongside the experiment file in the
    same directory). Experiment-file keys take precedence over base.yml.

    Returns a dict keyed by the kwarg names expected by PrepareRunTile
    (plus "tilefile", used only for the tile lookup in main()).
    '''
    base_path = os.path.join(os.path.dirname(experiment_path), "base.yml")
    merged_cfg = {**load_config(base_path), **load_config(experiment_path)}

    missing = [key for key in EXPERIMENT_CONFIG_KEYS if key not in merged_cfg]
    if missing:
        raise KeyError(
            f"Missing required key(s) {missing} in {base_path} and/or {experiment_path}"
        )

    return {arg_name: merged_cfg[key] for key, arg_name in EXPERIMENT_CONFIG_KEYS.items()}


def parse_arguments() -> argparse.Namespace:
    '''
    Function to parse command line arguments for the script.
    '''
    parser = argparse.ArgumentParser(description="Regrid DEM to Forcings")
    parser.add_argument("--experiment",
                        type=str,
                        required=True,
                        help="Path to the experiment YAML config file (e.g. experiments/v1_const.yml). "
                             "Merged on top of base.yml found in the same directory.")

    parser.add_argument("--snow_model",
                        type=str,
                        default="FSM2",
                        help="Snow model to use (default: FSM2)")

    parser.add_argument("--idx_tile",
                        type=int,
                        default=0,
                        help="Index of the tile to process (default: 0)")

    args = parser.parse_args()

    return args


class PrepareRunTile:
    """
    Class that prepares the run of MuSA for a specific tile (tx,ty) and time period (date_ini, date_end).

    Args:
        tx: int tile x-coordinate.
        ty: int tile y-coordinate.
        rootdirMuSAruns: str, root directory for MuSA runs
        date_ini: str, initial date for the simulation.
        date_end: str, end date for the simulation.
        snow_model: str, snow model to use.
        implementation: str, implementation type (e.g., "open_loop").
        model_only_sites: bool, flag to indicate if only model sites should be considered.
        remove_output_cells: bool, flag to indicate if output cells should be removed after the run
        store_measurements: str, path to xr dataset containing the in situ measurements (default: "/kyukon/data/gent/vo/000/gvo00090/SNOWSHOP/measurements/insitu/Alps_dataset_SD.nc")

    Returns:
        str, path to the adjusted config file
    """
    def __init__(self, 
                 tx:int, 
                 ty:int, 
                 rootdirMuSAruns:str, 
                 date_ini:str, 
                 date_end:str, 
                 snow_model:str, 
                 implementation:str, 
                 model_only_sites:bool,
                 remove_output_cells:bool,
                 store_measurements:str
                 ):
        self.tx=tx
        self.ty=ty
        self.rootdirMuSAruns=rootdirMuSAruns
        self.date_ini=date_ini
        self.date_end=date_end
        self.snow_model=snow_model
        self.implementation=implementation
        self.model_only_sites=model_only_sites
        self.remove_output_cells=remove_output_cells
        self.store_measurements=store_measurements

    def runPreprocessing(self) -> str:
        ''' 
        Function that prepares the run of MuSA for a specific tile (tx,ty) and time period (date_ini, date_end).
        '''
        #generate specific directories for the run of MuSA for the specific tile (tx,ty)
        rootdirRun, forcing_dir, dem_dir=prepRuntile_tools.CreateDirectoriesMuSArunTile(
                                                        tx=self.tx,
                                                        ty=self.ty,
                                                        rootdirMuSAruns=self.rootdirMuSAruns
                                                        )
        #check if forcings are availble for the specified time period
        check_forcings_store=ifn.check_forcings_timerange(
            date_ini=self.date_ini,
            date_end=self.date_end,
            forcing_dir=forcing_dir,
            verbose=True
            )
        
        if check_forcings_store is None:
            prepForcing_tools.CreateZarrTransformedForcings(
                tx=self.tx,
                ty=self.ty,
                date_ini=self.date_ini,
                date_end=self.date_end,
                savedir=forcing_dir,
                filename="forcings.zarr"
                )
            print("Forcings zarr file created successfully.", file=sys.stderr)
        
        #check the DEM -> generate for tx,ty if required
        dem_var, dem_res=RegridDEMtoForcings(
                datadir_forcings=forcing_dir, 
                savedir=dem_dir
            )

        #adjust the config-file
        out_path=prepRuntile_tools.adjust_config_file(
                snowmodel=self.snow_model,
                rootdirRun=rootdirRun,
                dem_varname=dem_var,
                dem_res=dem_res,
                date_ini=self.date_ini,
                date_end=self.date_end,
                implementation=self.implementation,
                model_only_sites=self.model_only_sites,
                remove_output_cells=self.remove_output_cells,
                store_measurements=self.store_measurements
            )
        print("Preprocessing complete!", file=sys.stderr)

        return out_path

#---main function---
def main():
    #parse the command line arguments
    args = parse_arguments()

    #load and merge the experiment config (base.yml + the --experiment override file)
    experiment_cfg = load_experiment_config(args.experiment)

    #get the tiles
    tiles=pd.read_csv(experiment_cfg.pop("tilefile"),header=0)
    tx=tiles.iloc[args.idx_tile]["tx"]
    ty=tiles.iloc[args.idx_tile]["ty"]

    #create an instance of the PrepareRunTile class and run the preprocessing
    prepclass=PrepareRunTile(tx=tx, ty=ty, snow_model=args.snow_model, **experiment_cfg)
    #run the preprocessing and get the path to the adjusted config file
    config_file=prepclass.runPreprocessing()
    print(config_file)

if __name__ == "__main__":
    main()
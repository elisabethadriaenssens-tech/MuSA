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
from modules.dem_tools import ReadRegriddedDEM
import modules.prepareForcingsZarr as prepForcing_tools
import modules.prepareRunTile_tools as prepRuntile_tools
import modules.prepareObs_tools as prepObs_tools
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
#names expected by PrepareRunTile / pd.read_csv (tilefile).
#Doubles as the REQUIRED_KEYS check in load_experiment_config below.
#NOTE: "tilefile" is not accepted by PrepareRunTile and must be popped from
#the merged dict before it is passed there (see main()). "tiles_source" IS
#accepted by PrepareRunTile (needed by SubsetForcingsZarr).
EXPERIMENT_CONFIG_KEYS = {
    "date_ini": "date_ini",
    "date_end": "date_end",
    "runs_root": "rootdirMuSAruns",
    "model_only_sites": "model_only_sites",
    "tilefile": "tilefile",
    "tiles_source": "tiles_source",
    "remove_output_cells": "remove_output_cells",
    "store_measurements": "store_measurements",
    "implementation": "implementation",
    "parallelization": "parallelization",
    "tmp_path": "tmp_path",
    "save_ensemble": "save_ensemble",
    "write_stat_daily": "write_stat_daily",
    "da_algorithm": "da_algorithm",
    "obs_source": "obs_source",
    "uq_method": "uq_method",
    "obs_var_names": "obs_var_names",
    "obs_error_var_names": "obs_error_var_names",
    "r_cov": "r_cov",
    "lat_obs_var_name": "lat_obs_var_name",
    "lon_obs_var_name": "lon_obs_var_name",
    "obs_sd_floor": "floor",
    "sigma_divisor": "divisor",
    "cqr_adjustment": "cqr_adjustment",
    "obs_error_aggregation": "obs_error_aggregation",
    "obs_date_ini": "obs_date_ini",
    "obs_date_end": "obs_date_end",
}
# NOTE: "dates_obs" is deliberately absent here -- it is derived at runtime by
# PrepareRunTile.runPreprocessing() from the observation files PrepareObsTile actually
# writes, never taken from the yml (see PrepareObsTile in modules/prepareObs_tools.py).


def load_experiment_config(
        experiment_path:str=None,
        ) -> dict:
    '''
    Function to load an experiment YAML config file, merged on top of the
    shared base.yml config (expected alongside the experiment file in the
    same directory). Experiment-file keys take precedence over base.yml.

    Returns a dict keyed by the kwarg names expected by PrepareRunTile,
    plus "tilefile" which is not accepted by PrepareRunTile and must be
    popped by the caller before use (see main() below). "tiles_source" IS
    accepted by PrepareRunTile and is also read directly by
    helpers/setup_run_dirs.py.

    "rootdirMuSAruns" is returned version-scoped ({runs_root}/v{version}) so that
    this script and helpers/setup_run_dirs.py always agree on the same run root.
    '''
    base_path = os.path.join(os.path.dirname(experiment_path), "base.yml")
    merged_cfg = {**load_config(base_path), **load_config(experiment_path)}

    missing = [key for key in EXPERIMENT_CONFIG_KEYS if key not in merged_cfg]
    if missing:
        raise KeyError(
            f"Missing required key(s) {missing} in {base_path} and/or {experiment_path}"
        )

    if "version" not in merged_cfg:
        raise KeyError(f"Missing required key 'version' in {base_path} and/or {experiment_path}")

    cfg = {arg_name: merged_cfg[key] for key, arg_name in EXPERIMENT_CONFIG_KEYS.items()}
    cfg["rootdirMuSAruns"] = os.path.join(cfg["rootdirMuSAruns"], f"v{merged_cfg['version']}")

    return cfg


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
        parallelization: str, MuSA parallelization scheme ("multiprocessing" or "HPC.array").
        model_only_sites: bool, flag to indicate if only model sites should be considered.
        remove_output_cells: bool, flag to indicate if output cells should be removed after the run
        store_measurements: str, path to xr dataset containing the in situ measurements (default: "/kyukon/data/gent/vo/000/gvo00090/SNOWSHOP/measurements/insitu/Alps_dataset_SD.nc")
        tiles_source: str, root directory on staging containing the per-tile source forcings zarr stores
        tmp_path: str, path to the temporary directory used by MuSA during the run
        save_ensemble: bool, flag to indicate if the ensemble should be saved as a pkl object
        write_stat_daily: bool, flag to indicate if the outputs should be averaged at a daily time step
        da_algorithm: str, DA algorithm to use (e.g. "EnKF", "PBS", "deterministic_OL"). Forced
            to "deterministic_OL" when implementation=="open_loop", regardless of this value.
        obs_source: str, root directory on staging containing the per-tile prediction/obs files
            (expects {obs_source}/y{ty:03d}x{tx:03d}/sd_*.nc). Unused when uq_method=="none".
        uq_method: str, one of "none" (open_loop, no obs needed), "constant" (obs files
            written as-is, fixed r_cov), "IP"/"QR"/"CQR" (dynamic per-cell obs error variance
            computed and attached before writing -- see PrepareObsTile).
        obs_var_names: list[str], names of the observed variable(s) as they appear in the
            (regridded) observation files.
        obs_error_var_names: list[str], names under which the dynamic obs error variance is
            stored in the observation files. Used when r_cov=="dynamic_error".
        r_cov: list[float] | str, fixed observation error variance(s), or "dynamic_error" to
            read the per-cell variance from obs_error_var_names instead.
        lat_obs_var_name / lon_obs_var_name: str, names of the lat/lon coordinates in the
            observation files.
        floor: float, minimum sigma (in the same units as the observation) enforced when
            computing the dynamic obs error variance.
        cqr_adjustment: float | None, required when uq_method=="CQR".
        divisor: float, divisor applied when computing sigma from quantiles (QR/CQR).
        obs_error_aggregation: str, one of "correlated" (default; sigma_week = mean(sigma_i),
            no error reduction from averaging retrievals within a weekly window) or
            "independent" (sigma_week = sqrt(sum(sigma_i**2)) / n). See PrepareObsTile.
        obs_date_ini / obs_date_end: str | None, optional narrower range within
            [date_ini, date_end] that a window's midpoint must fall in to be assimilated.
            The model still runs the full [date_ini, date_end] period; this only controls
            which weekly windows PrepareObsTile writes observation files for. None (the
            default) imposes no additional restriction. Unused when uq_method=="none".

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
                 parallelization:str,
                 model_only_sites:bool,
                 remove_output_cells:bool,
                 store_measurements:str,
                 tiles_source:str,
                 tmp_path:str,
                 save_ensemble:bool,
                 write_stat_daily:bool,
                 da_algorithm:str,
                 obs_source:str,
                 uq_method:str,
                 obs_var_names:list,
                 obs_error_var_names:list,
                 r_cov,
                 lat_obs_var_name:str,
                 lon_obs_var_name:str,
                 floor:float,
                 cqr_adjustment,
                 divisor:float,
                 obs_error_aggregation:str="correlated",
                 obs_date_ini:str=None,
                 obs_date_end:str=None,
                 ):
        self.tx=tx
        self.ty=ty
        self.rootdirMuSAruns=rootdirMuSAruns
        self.date_ini=date_ini
        self.date_end=date_end
        self.snow_model=snow_model
        self.implementation=implementation
        self.parallelization=parallelization
        self.model_only_sites=model_only_sites
        self.remove_output_cells=remove_output_cells
        self.store_measurements=store_measurements
        self.tiles_source=tiles_source
        self.tmp_path=tmp_path
        self.save_ensemble=save_ensemble
        self.write_stat_daily=write_stat_daily
        self.da_algorithm=da_algorithm
        self.obs_source=obs_source
        self.uq_method=uq_method
        self.obs_var_names=obs_var_names
        self.obs_error_var_names=obs_error_var_names
        self.r_cov=r_cov
        self.lat_obs_var_name=lat_obs_var_name
        self.lon_obs_var_name=lon_obs_var_name
        self.floor=floor
        self.cqr_adjustment=cqr_adjustment
        self.divisor=divisor
        self.obs_error_aggregation=obs_error_aggregation
        self.obs_date_ini=obs_date_ini
        self.obs_date_end=obs_date_end

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
            prepForcing_tools.SubsetForcingsZarr(
                tiles_source=self.tiles_source,
                tx=self.tx,
                ty=self.ty,
                date_ini=self.date_ini,
                date_end=self.date_end,
                savedir=forcing_dir,
                )
            print("Forcings zarr file created successfully.", file=sys.stderr)
        
        #DEM is already regridded and symlinked into dem_dir by helpers/setup_run_dirs.py
        dem_var, dem_res=ReadRegriddedDEM(dem_dir=dem_dir)

        #prepare the observation files for this tile/period and derive dates_obs from
        #the files actually written (obs_array assumes file n <-> dates_obs[n], so the
        #dates must come from this loop, never from the yml). Skipped for uq_method=="none"
        #(open_loop), which never reads observations at all.
        #NOTE: obs_dir must match cfg.nc_obs_path, set from rootdirRun the same way in
        #modules/prepareRunTile_tools.py::_UpdateConfigPaths.
        obs_dir=os.path.join(rootdirRun, "Obs")
        if self.uq_method == "none":
            dates_obs=[]
        else:
            dates_obs=prepObs_tools.PrepareObsTile(
                obs_source=self.obs_source,
                tx=self.tx,
                ty=self.ty,
                date_ini=self.date_ini,
                date_end=self.date_end,
                dem_dir=dem_dir,
                obs_dir=obs_dir,
                uq_method=self.uq_method,
                obs_var_names=self.obs_var_names,
                obs_error_var_names=self.obs_error_var_names,
                floor=self.floor,
                cqr_adjustment=self.cqr_adjustment,
                divisor=self.divisor,
                obs_error_aggregation=self.obs_error_aggregation,
                obs_date_ini=self.obs_date_ini,
                obs_date_end=self.obs_date_end,
                )
            print(f"Observation files prepared: {len(dates_obs)} dates.", file=sys.stderr)

        #adjust the config-file
        out_path=prepRuntile_tools.adjust_config_file(
                snowmodel=self.snow_model,
                rootdirRun=rootdirRun,
                dem_varname=dem_var,
                dem_res=dem_res,
                date_ini=self.date_ini,
                date_end=self.date_end,
                implementation=self.implementation,
                parallelization=self.parallelization,
                model_only_sites=self.model_only_sites,
                remove_output_cells=self.remove_output_cells,
                store_measurements=self.store_measurements,
                tmp_path=self.tmp_path,
                save_ensemble=self.save_ensemble,
                write_stat_daily=self.write_stat_daily,
                da_algorithm=self.da_algorithm,
                dates_obs=dates_obs,
                obs_var_names=self.obs_var_names,
                obs_error_var_names=self.obs_error_var_names,
                r_cov=self.r_cov,
                lat_obs_var_name=self.lat_obs_var_name,
                lon_obs_var_name=self.lon_obs_var_name,
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
    #tiles_source is now also consumed by PrepareRunTile (SubsetForcingsZarr needs it
    #to locate the source store); helpers/setup_run_dirs.py still reads it separately
    prepclass=PrepareRunTile(tx=tx, ty=ty, snow_model=args.snow_model, **experiment_cfg)
    #run the preprocessing and get the path to the adjusted config file
    config_file=prepclass.runPreprocessing()
    print(config_file)

if __name__ == "__main__":
    main()
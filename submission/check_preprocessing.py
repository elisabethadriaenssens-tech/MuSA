'''
Sanity checks on a preprocessed MuSA tile run, used by the preprocess_v3*.slurm scripts
before a job array is allowed to start. Exits non-zero on any failure.

Checks that the adjusted config is set up for a full-tile HPC.array DA run, and that the
Obs/*.nc files share the DEM's and forcings' lat/lon exactly (lat descending): obs_array()
indexes observations positionally with the forcing lat_idx, so any mismatch in order
means every cell reads its N-S mirrored counterpart.

usage: python check_preprocessing.py <adjusted_config.py> <expected da_algorithm> [expected r_cov]
       (r_cov as a python literal, e.g. "[0.09]"; default "dynamic_error")
'''
import ast, glob, importlib.util, os, sys
import numpy as np
import xarray as xr

cfg_path, expected_algo = sys.argv[1], sys.argv[2]
expected_r_cov = ast.literal_eval(sys.argv[3]) if len(sys.argv) > 3 else "dynamic_error"
spec = importlib.util.spec_from_file_location("config", cfg_path)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

errors = []
def check(ok, msg):
    print(("OK   " if ok else "FAIL ") + msg)
    if not ok:
        errors.append(msg)

#---config---
check(cfg.parallelization == "HPC.array", f"parallelization = {cfg.parallelization!r}")
check(cfg.da_algorithm == expected_algo, f"da_algorithm = {cfg.da_algorithm!r}")
check(cfg.nc_maks_path is None, f"nc_maks_path = {cfg.nc_maks_path!r} (None -> saveFinalOutputToZarr)")
check(cfg.save_ensemble is False, f"save_ensemble = {cfg.save_ensemble!r}")
check(cfg.ensemble_members == 100, f"ensemble_members = {cfg.ensemble_members!r}")
check(cfg.r_cov == expected_r_cov, f"r_cov = {cfg.r_cov!r}")
check(cfg.obs_var_names == ["SD_Q50"], f"obs_var_names = {cfg.obs_var_names!r}")
# date_end is snapped to the last forcing time step (21:00 for 3-hourly forcings)
check(cfg.date_ini == "2016-09-01 00:00" and cfg.date_end.startswith("2017-08-31"),
      f"date_ini/date_end = {cfg.date_ini} / {cfg.date_end}")
check(len(cfg.dates_obs) > 0 and cfg.dates_obs[0] >= "2016-11-01" and cfg.dates_obs[-1] <= "2017-04-30 23:00",
      f"dates_obs: {len(cfg.dates_obs)} dates, {cfg.dates_obs[0]} .. {cfg.dates_obs[-1]}")

#---grid orientation: Obs vs DEM vs forcings---
dem = xr.open_dataset(cfg.dem_path)
forcing = xr.open_zarr(glob.glob(os.path.join(cfg.nc_forcing_path, "*.zarr"))[0], consolidated=True)
dem_lat, dem_lon = dem["lat"].values, dem["lon"].values
check(bool(np.all(np.diff(dem_lat) < 0)), "DEM lat descending")
check(np.array_equal(forcing["lat"].values, dem_lat) and np.array_equal(forcing["lon"].values, dem_lon),
      "forcing lat/lon == DEM lat/lon")

obs_files = sorted(glob.glob(os.path.join(cfg.nc_obs_path, "*.nc")))
check(len(obs_files) == len(cfg.dates_obs), f"{len(obs_files)} Obs files for {len(cfg.dates_obs)} dates_obs")
bad = [os.path.basename(f) for f in obs_files
       if not (np.array_equal(xr.open_dataset(f)["lat"].values, dem_lat)
               and np.array_equal(xr.open_dataset(f)["lon"].values, dem_lon))]
check(not bad, f"Obs lat/lon == DEM lat/lon (lat descending) in all files; mismatched: {bad}")

if errors:
    sys.exit(f"{len(errors)} preprocessing check(s) failed")
print("All preprocessing checks passed")

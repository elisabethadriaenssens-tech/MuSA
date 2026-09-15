#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script containing functions to prepare observation/prediction data for use as
MuSA input: regridding it onto the DEM grid of the tile it will be assimilated into,
and computing the observation error variance MuSA's r_cov expects.

# author: Elisabeth Adriaenssens - elisabethadriaenssens@gmail.com
"""

#---modules---
import glob
import os
import re
import sys
import numpy as np
import pandas as pd
import xarray as xr
sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.OperationsXrDatasets import saveXrtoNetCDF

#---functions---
def regrid_obs_to_dem(
        obs:xr.Dataset,
        dem:xr.Dataset,
    ) -> tuple[xr.Dataset, dict, dict]:
    '''
    Function that regrids an observation/prediction dataset (e.g. 960x960 at 1/1008 deg)
    onto the DEM grid of a tile (e.g. 192x192 at 0.004960 deg) using linear interpolation.

    Both datasets are sorted ascending on lat and lon before interpolation, and this must
    happen first: the prediction files store lat ascending, but dem_regridded.nc stores it
    descending, and assign_coords() below is positional -- it overwrites the interpolated
    lat/lon values with dem's index-for-index, with no check that the two run in the same
    direction. Without sorting both first, the field comes back mirrored north-south with
    no error raised.

    After interpolating onto dem's lat/lon, the coordinates are reassigned to dem's exact
    values (rather than the interpolated ones interp() produces, which can differ from
    dem's by floating-point noise) so downstream code can rely on obs and dem sharing an
    identical grid.

    Returns the regridded dataset, plus a dict of valid (non-NaN) cell counts per data
    variable before and after regridding, so the caller can log any data loss.
    '''
    dem_lat_original=dem["lat"].values.copy()
    dem_lon_original=dem["lon"].values.copy()

    obs=obs.sortby("lat").sortby("lon")
    dem=dem.sortby("lat").sortby("lon")

    valid_before={var: int(obs[var].notnull().sum()) for var in obs.data_vars}

    obs_regridded=obs.interp(lat=dem["lat"], lon=dem["lon"], method="linear")
    obs_regridded=obs_regridded.assign_coords(lat=dem["lat"], lon=dem["lon"])

    # obs_array() in internal_fns.py indexes observations positionally with the same
    # lat_idx used for the forcings/DEM/mask, which are all descending. Restore the
    # DEM's original axis order here or every cell reads its N-S mirrored counterpart.
    obs_regridded=obs_regridded.reindex(lat=dem_lat_original, lon=dem_lon_original)

    valid_after={var: int(obs_regridded[var].notnull().sum()) for var in obs_regridded.data_vars}

    #linear interpolation needs points on both sides of each target cell, so the outermost
    #row/column of the regridded grid has nothing to interpolate from and comes back NaN --
    #log it so silent edge data loss is at least visible
    edge_nan_counts={}
    for var in obs_regridded.data_vars:
        isnull=obs_regridded[var].isnull()
        edge_nan_counts[var]=(
            int(isnull.isel(lat=0).sum()) + int(isnull.isel(lat=-1).sum())
            + int(isnull.isel(lon=0).sum()) + int(isnull.isel(lon=-1).sum())
        )
    print(
        f"[regrid_obs_to_dem] NaN cells on regridded grid's outermost row/column: "
        f"{edge_nan_counts}",
        file=sys.stderr,
    )

    return obs_regridded, valid_before, valid_after

def compute_obs_error(
        ds:xr.Dataset,
        uq_method:str,
        cqr_adjustment:float=None,
        divisor:float=2.0,
    ) -> xr.DataArray:
    '''
    Function that computes the observation error as a sigma (standard deviation) field,
    from a single retrieval.

    Supported uq_method values:
        - "IP":  sigma = SD_Ensemble_Std
        - "QR":  sigma = (SD_Q75 - SD_Q25) / divisor
        - "CQR": lower = maximum(0, SD_Q25 - cqr_adjustment)
                 upper = SD_Q75 + cqr_adjustment
                 sigma = (upper - lower) / divisor

    Deliberately returns raw sigma, not variance, and does not apply the floor: within
    PrepareObsTile several retrievals in the same weekly window get aggregated together
    (see aggregate_weekly_sigma), and that aggregation must happen in sigma-space before
    the floor is applied -- flooring per-retrieval first would floor away exactly the
    small-sigma retrievals that should pull the weekly average down.

    Returns the sigma field.
    '''
    if uq_method == "IP":
        sigma=ds["SD_Ensemble_Std"]
    elif uq_method == "QR":
        sigma=(ds["SD_Q75"] - ds["SD_Q25"]) / divisor
    elif uq_method == "CQR":
        if cqr_adjustment is None:
            raise ValueError("uq_method 'CQR' requires cqr_adjustment to be set.")
        lower=np.maximum(0, ds["SD_Q25"] - cqr_adjustment)
        upper=ds["SD_Q75"] + cqr_adjustment
        sigma=(upper - lower) / divisor
    else:
        raise ValueError(f"Unknown uq_method '{uq_method}'. Expected one of 'IP', 'QR', 'CQR'.")

    return sigma

def aggregate_weekly_sigma(
        sigmas:list,
        obs_error_aggregation:str,
    ) -> xr.DataArray:
    '''
    Function that aggregates the per-retrieval sigma fields of one weekly window into a
    single sigma field, still in sigma-space (before the floor is applied -- see
    apply_floor_and_square).

    Supported obs_error_aggregation values:
        - "correlated":  sigma_week = mean(sigma_i)
        - "independent": sigma_week = sqrt(sum(sigma_i**2)) / n

    "correlated" is the default because the retrievals in one window come from the same
    XGBoost model run over the same terrain: their errors share a systematic component, so
    averaging them shouldn't shrink the uncertainty by sqrt(n) the way independent draws
    would.

    Returns the weekly sigma field.
    '''
    stacked=xr.concat(sigmas, dim="retrieval")
    n=len(sigmas)

    if obs_error_aggregation == "correlated":
        sigma_week=stacked.mean(dim="retrieval", skipna=True)
    elif obs_error_aggregation == "independent":
        sigma_week=np.sqrt((stacked**2).sum(dim="retrieval", skipna=True)) / n
    else:
        raise ValueError(
            f"Unknown obs_error_aggregation '{obs_error_aggregation}'. "
            "Expected 'correlated' or 'independent'."
        )

    return sigma_week

def apply_floor_and_square(
        sigma:xr.DataArray,
        floor:float=0.05,
    ) -> tuple[xr.DataArray, float]:
    '''
    Function that turns a sigma field into the variance field MuSA's r_cov expects.

    variance = maximum(sigma, floor) ** 2 -- the floor is applied in sigma-space, before
    squaring, so "floor" always means a minimum standard deviation in the same units as
    SD, regardless of how sigma was derived or aggregated.

    Returns the variance field, plus the fraction of valid (non-NaN) cells where the
    floor was active (sigma < floor before flooring), so the caller can log how much of
    the error field is floor-dominated.
    '''
    #---floor in sigma-space, then square to get variance---
    #why: keep the two steps explicit rather than flooring the variance directly --
    #squaring only commutes with the floor because sigma and floor are both non-negative
    sigma_floored=np.maximum(sigma, floor)
    variance=sigma_floored**2

    #---fraction of valid cells where the floor was active---
    valid=sigma.notnull()
    n_valid=int(valid.sum())
    frac_floor_active=float(((sigma < floor) & valid).sum()) / n_valid if n_valid > 0 else float("nan")

    return variance, frac_floor_active

def PrepareObsTile(
        obs_source:str,
        tx:int,
        ty:int,
        date_ini:str,
        date_end:str,
        dem_dir:str,
        obs_dir:str,
        uq_method:str,
        obs_var_names:list,
        obs_error_var_names:list,
        floor:float=0.05,
        cqr_adjustment:float=None,
        divisor:float=2.0,
        obs_error_aggregation:str="correlated",
        obs_date_ini:str=None,
        obs_date_end:str=None,
    ) -> list:
    '''
    Function that prepares observation/prediction files for a specific tile (tx, ty) and
    time period (date_ini, date_end) for assimilation with MuSA.

    Globs {obs_source}/y{ty:03d}x{tx:03d}/sd_*.nc, keeps files within [date_ini, date_end],
    regrids each onto the tile's DEM grid, and bins the regridded retrievals into
    consecutive 7-day windows starting at date_ini. A window is also dropped (with a
    message explaining why) if its midpoint falls outside [date_ini, date_end] -- the
    midpoint is never clamped into range, since that would assimilate the observation on
    the wrong date.

    obs_date_ini / obs_date_end optionally narrow the range a window's midpoint must fall
    in to be assimilated, independently of [date_ini, date_end] (which still controls the
    model run period and which retrieval files are read at all). This is how a full-year
    run can assimilate only part of the year: the model still runs the whole period, but
    windows whose midpoint falls outside [obs_date_ini, obs_date_end] are dropped the same
    way windows outside the run period are. Either bound may be given alone; a bound left
    as None (the default) behaves exactly as if obs_date_ini/obs_date_end were absent from
    the config, i.e. it imposes no additional restriction beyond [date_ini, date_end].

    One output file is written per remaining non-empty window (windows
    with no retrievals are skipped, never written empty):
        - obs_var_names[0] is averaged across the retrievals present in the window
          (skipping NaNs), and timestamped at the window's midpoint (e.g. a 1-7 Sep
          window is timestamped 4 Sep).
        - if uq_method is "IP"/"QR"/"CQR", the per-retrieval sigma (see compute_obs_error)
          is aggregated across the window (see aggregate_weekly_sigma, controlled by
          obs_error_aggregation), and only then is the floor applied and the result
          squared into the variance stored under obs_error_var_names[0] (see
          apply_floor_and_square). Aggregating in sigma-space before flooring/squaring
          means the floor reflects the aggregated week, not any one retrieval.
        - "n_retrievals" records how many retrievals contributed to the window, so a
          3-retrieval week can be told apart from a 1-retrieval week downstream.
    uq_method "constant" skips the error computation entirely (obs files carry no dynamic
    error variance; MuSA uses a fixed r_cov instead) but still averages obs_var_names[0]
    per window.

    Why order matters: obs_array (modules/internal_fns.py) globs and sorts .nc files in
    nc_obs_path alphabetically and assumes file n corresponds to date n, with no date
    matching. Filenames are therefore derived from each window's midpoint date (e.g.
    sd_20170904.nc), which sorts chronologically the same way the windows themselves are
    processed -- in increasing window order, since dated_files is sorted chronologically
    and window index is monotonic in retrieval date. The returned date list is built in
    that same order, so it stays paired 1:1 with the files on disk and can become
    cfg.dates_obs verbatim.

    Returns the list of observation dates written (one per window, at its midpoint),
    formatted as "%Y-%m-%d %H:%M" strings.
    '''
    date_pattern=re.compile(r"\d{8}")
    date_ini_ts=pd.Timestamp(date_ini).normalize()
    date_end_ts=pd.Timestamp(date_end)

    #optional narrower assimilation range: a bound left as None imposes no restriction
    #beyond [date_ini_ts, date_end_ts], so behaviour is unchanged when both are absent
    obs_date_ini_ts=pd.Timestamp(obs_date_ini).normalize() if obs_date_ini is not None else None
    obs_date_end_ts=pd.Timestamp(obs_date_end) if obs_date_end is not None else None

    tile_dir=os.path.join(obs_source, f"y{ty:03d}x{tx:03d}")
    files=glob.glob(os.path.join(tile_dir, "sd_*.nc"))
    if not files:
        raise FileNotFoundError(f"No observation files found in {tile_dir}")

    #parse the YYYYMMDD date out of each filename up front: used to filter/sort, and as
    #the fallback if a file's own time coordinate decodes to an unexpected calendar day
    dated_files=[]
    for f in files:
        match=date_pattern.search(os.path.basename(f))
        if match is None:
            raise ValueError(f"Could not find a YYYYMMDD date in observation filename: {f}")
        filename_date=pd.Timestamp(match.group())
        if date_ini_ts <= filename_date <= date_end_ts:
            dated_files.append((filename_date, f))
    dated_files.sort(key=lambda x: x[0])

    if not dated_files:
        raise FileNotFoundError(
            f"No observation files in {tile_dir} fall within [{date_ini}, {date_end}]."
        )

    os.makedirs(obs_dir, exist_ok=True)

    #open the DEM once and reuse it for every file
    dem=xr.open_dataset(os.path.join(dem_dir, "dem_regridded.nc")).load()

    obs_var=obs_var_names[0]
    compute_error=uq_method in ("IP", "QR", "CQR")

    #---regrid every retrieval, compute its sigma (from the regridded quantiles, not the
    #native-grid ones), and bucket it into its 7-day window---
    windows={}
    for filename_date, f in dated_files:
        ds=xr.open_dataset(f).load()

        #the file's own time coordinate is the authoritative observation date; fall back
        #to the filename-parsed date (midnight) if it disagrees, so a CF-decoding surprise
        #can't silently misdate an assimilated observation
        obs_time=pd.Timestamp(ds["time"].values[0])
        if obs_time.normalize() != filename_date.normalize():
            print(
                f"WARNING: time coordinate in {f} decodes to {obs_time}, which disagrees "
                f"with the filename date {filename_date.date()}. Falling back to the "
                "filename-parsed date.",
                file=sys.stderr,
            )
            obs_time=filename_date

        ds_regridded, valid_before, valid_after=regrid_obs_to_dem(ds, dem)
        print(
            f"[PrepareObsTile] {os.path.basename(f)}: valid cells before/after regrid: "
            f"{valid_before} -> {valid_after}",
            file=sys.stderr,
        )

        sigma=None
        if compute_error:
            sigma=compute_obs_error(
                ds_regridded, uq_method, cqr_adjustment=cqr_adjustment, divisor=divisor
            )
            if "time" in sigma.dims:
                sigma=sigma.isel(time=0, drop=True)

        obs_da=ds_regridded[obs_var]
        if "time" in obs_da.dims:
            obs_da=obs_da.isel(time=0, drop=True)

        window_idx=(obs_time.normalize() - date_ini_ts).days // 7
        windows.setdefault(window_idx, []).append((obs_da, sigma))

    #---average each window and write one file per non-empty window, in chronological order---
    dates_written=[]
    n_skipped_obs_range=0
    for window_idx in sorted(windows):
        entries=windows[window_idx]
        n_retrievals=len(entries)

        window_start=date_ini_ts + pd.Timedelta(days=7 * window_idx)
        midpoint=window_start + pd.Timedelta(days=3)

        if not (date_ini_ts <= midpoint <= date_end_ts):
            print(
                f"[PrepareObsTile] window {window_start.date()}-"
                f"{(window_start + pd.Timedelta(days=6)).date()} dropped: its midpoint "
                f"{midpoint} falls outside [{date_ini}, {date_end}].",
                file=sys.stderr,
            )
            continue

        if (obs_date_ini_ts is not None and midpoint < obs_date_ini_ts) or (
            obs_date_end_ts is not None and midpoint > obs_date_end_ts
        ):
            n_skipped_obs_range+=1
            print(
                f"[PrepareObsTile] window {window_start.date()}-"
                f"{(window_start + pd.Timedelta(days=6)).date()} dropped: its midpoint "
                f"{midpoint} falls outside the obs assimilation range "
                f"[{obs_date_ini}, {obs_date_end}].",
                file=sys.stderr,
            )
            continue

        obs_das=[e[0] for e in entries]
        obs_week=xr.concat(obs_das, dim="retrieval").mean(dim="retrieval", skipna=True)

        data_vars={
            obs_var: obs_week.expand_dims(time=[midpoint]).transpose("time", "lat", "lon"),
            "n_retrievals": xr.DataArray(np.int32(n_retrievals)),
        }

        if compute_error:
            sigmas=[e[1] for e in entries]
            sigma_week=aggregate_weekly_sigma(sigmas, obs_error_aggregation)
            variance, frac_floor_active=apply_floor_and_square(sigma_week, floor=floor)
            data_vars[obs_error_var_names[0]]=(
                variance.expand_dims(time=[midpoint]).transpose("time", "lat", "lon")
            )
            print(
                f"[PrepareObsTile] window {window_start.date()}-"
                f"{(window_start + pd.Timedelta(days=6)).date()} ({n_retrievals} "
                f"retrieval(s), {obs_error_aggregation}): floor active on "
                f"{frac_floor_active:.1%} of valid cells",
                file=sys.stderr,
            )

        ds_week=xr.Dataset(data_vars)

        filename=f"sd_{midpoint:%Y%m%d}.nc"
        saveXrtoNetCDF(ds=ds_week, savedir=obs_dir, filename=filename)

        dates_written.append(midpoint.strftime("%Y-%m-%d %H:%M"))

    if obs_date_ini_ts is not None or obs_date_end_ts is not None:
        print(
            f"[PrepareObsTile] {n_skipped_obs_range} window(s) skipped for falling "
            f"outside the obs assimilation range [{obs_date_ini}, {obs_date_end}].",
            file=sys.stderr,
        )

    return dates_written

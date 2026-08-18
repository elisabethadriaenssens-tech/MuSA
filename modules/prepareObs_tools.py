#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script containing functions to prepare observation/prediction data for use as
MuSA input: regridding it onto the DEM grid of the tile it will be assimilated into,
and computing the observation error variance MuSA's r_cov expects.

# author: Elisabeth Adriaenssens - elisabethadriaenssens@gmail.com
"""

#---modules---
import numpy as np
import xarray as xr

#---functions---
def regrid_obs_to_dem(
        obs:xr.Dataset,
        dem:xr.Dataset,
    ) -> tuple[xr.Dataset, dict, dict]:
    '''
    Function that regrids an observation/prediction dataset (e.g. 960x960 at 1/1008 deg)
    onto the DEM grid of a tile (e.g. 192x192 at 0.004960 deg) using nearest-neighbour
    interpolation.

    Both datasets are sorted ascending on lat and lon before interpolation: the prediction
    files run ascending in lat while dem_regridded.nc runs descending, and without aligning
    them first, interp() would silently flip the field north-south.

    Returns the regridded dataset, plus a dict of valid (non-NaN) cell counts per data
    variable before and after regridding, so the caller can log any data loss.
    '''
    obs=obs.sortby("lat").sortby("lon")
    dem=dem.sortby("lat").sortby("lon")

    valid_before={var: int(obs[var].notnull().sum()) for var in obs.data_vars}

    obs_regridded=obs.interp(lat=dem["lat"], lon=dem["lon"], method="nearest")

    valid_after={var: int(obs_regridded[var].notnull().sum()) for var in obs_regridded.data_vars}

    return obs_regridded, valid_before, valid_after

def compute_obs_error(
        ds:xr.Dataset,
        uq_method:str,
        floor:float=0.05,
        cqr_adjustment:float=None,
        divisor:float=2.0,
    ) -> tuple[xr.DataArray, float]:
    '''
    Function that computes the observation error as a variance field for use as MuSA's
    r_cov (which expects variance, not sigma).

    Supported uq_method values:
        - "IP":  sigma = SD_Ensemble_Std
        - "QR":  sigma = (SD_Q75 - SD_Q25) / divisor
        - "CQR": lower = maximum(0, SD_Q25 - cqr_adjustment)
                 upper = SD_Q75 + cqr_adjustment
                 sigma = (upper - lower) / divisor

    variance = maximum(sigma, floor) ** 2 -- the floor is applied in sigma-space, before
    squaring, so "floor" always means a minimum standard deviation in the same units as
    SD, regardless of how sigma was derived.

    Returns the variance field, plus the fraction of valid (non-NaN) cells where the
    floor was active (sigma < floor before flooring), so the caller can log how much of
    the error field is floor-dominated.
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

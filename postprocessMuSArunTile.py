'''
Script that transforms the per-cell MuSA outputs into the final zarr store / netcdf file.

Used for HPC.array runs, where runMuSAtile.py skips transform_results() because each
array task only simulates its own share of the cells. Run this once, after every array
task has finished, with MUSA_CONFIG pointing to the same adjusted config file.
'''
from runMuSAtile import transform_results

if __name__ == "__main__":
    transform_results()

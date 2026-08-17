'''
Script that creates the per-tile run directory tree ({rootdirMuSAruns}/{tile}/
with DEM/, FORCINGS/ and Obs/ subfolders) for every tile in an experiment's
tilefile, and symlinks in the dem_regridded.nc and mask_msites_tile.nc files
shared across experiment versions from {tiles_source}/{tile}/.

Symlinks (rather than copies) are used because these files are identical
across experiment versions, so there is no need to duplicate them per run.

contact: Lucas Boeykens -lucas.boeykens@ugent.be lucas.boeykens@kuleuven.be
'''

#---modules---
import os, sys, argparse
import pandas as pd
project_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from preprocessMuSArunTile import load_experiment_config, load_config

#---custom functions---
def parse_arguments() -> argparse.Namespace:
    '''
    Function to parse command line arguments for the script.
    '''
    parser = argparse.ArgumentParser(
        description="Create per-tile MuSA run directories and symlink in the shared DEM/mask files."
        )
    parser.add_argument("--experiment",
                        type=str,
                        required=True,
                        help="Path to the experiment YAML config file (e.g. experiments/v1_const.yml). "
                             "Merged on top of base.yml found in the same directory.")

    args = parser.parse_args()

    return args


def _symlink(source_file:str, link_path:str) -> None:
    '''
    Helper function to (re)create a symlink at link_path pointing to source_file.
    Warns and skips (without crashing) if source_file does not exist.
    '''
    if not os.path.isfile(source_file):
        print(f"WARNING: {source_file} not found -- skipping symlink for {link_path}", file=sys.stderr)
        return

    if os.path.lexists(link_path):
        os.remove(link_path)

    os.symlink(source_file, link_path)


def setup_tile_dirs(
        tile_str:str,
        rootdirMuSAruns:str,
        tiles_source:str,
        ) -> None:
    '''
    Function that creates the run-directory tree for a single tile
    ({rootdirMuSAruns}/{tile_str}/ with DEM/, FORCINGS/ and Obs/ subfolders)
    and symlinks in dem_regridded.nc and mask_msites_tile.nc from
    {tiles_source}/{tile_str}/.
    '''
    rootdirRun=os.path.join(rootdirMuSAruns, tile_str)
    dem_dir=os.path.join(rootdirRun, "DEM")
    forcing_dir=os.path.join(rootdirRun, "FORCINGS")
    obs_dir=os.path.join(rootdirRun, "Obs")

    for directory in (dem_dir, forcing_dir, obs_dir):
        os.makedirs(directory, exist_ok=True)

    source_dir=os.path.join(tiles_source, tile_str)
    if not os.path.isdir(source_dir):
        print(f"WARNING: source directory {source_dir} not found -- skipping symlinks for tile {tile_str}", file=sys.stderr)
        return

    _symlink(os.path.join(source_dir, "DEM", "dem_regridded.nc"), os.path.join(dem_dir, "dem_regridded.nc"))
    _symlink(os.path.join(source_dir, "mask_msites_tile.nc"), os.path.join(rootdirRun, "mask_msites_tile.nc"))


def _derive_rootdirMuSAruns(experiment_path:str, runs_root:str) -> str:
    '''
    Helper function that derives the version-scoped run root ({runs_root}/v{version})
    from the "version" key in the merged base.yml + experiment yml config.
    '''
    base_path=os.path.join(os.path.dirname(experiment_path), "base.yml")
    merged_cfg={**load_config(base_path), **load_config(experiment_path)}

    if "version" not in merged_cfg:
        raise KeyError(f"Missing required key 'version' in {base_path} and/or {experiment_path}")

    return os.path.join(runs_root, f"v{merged_cfg['version']}")


#---main function---
def main():
    #parse the command line arguments
    args = parse_arguments()

    #load and merge the experiment config (base.yml + the --experiment override file)
    experiment_cfg = load_experiment_config(args.experiment)

    rootdirMuSAruns = _derive_rootdirMuSAruns(args.experiment, experiment_cfg["rootdirMuSAruns"])
    tiles_source = experiment_cfg["tiles_source"]

    #get the tiles
    tiles=pd.read_csv(experiment_cfg["tilefile"], header=0)

    for _, row in tiles.iterrows():
        tx, ty = int(row["tx"]), int(row["ty"])
        tile_str=f"y{ty:03d}x{tx:03d}"
        setup_tile_dirs(tile_str=tile_str, rootdirMuSAruns=rootdirMuSAruns, tiles_source=tiles_source)

    print("Run directory setup complete!", file=sys.stderr)

if __name__ == "__main__":
    main()

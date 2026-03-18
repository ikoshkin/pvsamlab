"""
parallel_string_sizing.py

Runs batch PV string sizing simulations across multiple modules, years, and
string lengths using parallel worker processes. Outputs summary and hourly CSVs.

Fix notes
---------
Fix 1 – OND file lookup: find_ond_file() returns the path *string*, not the
         result of os.path.exists() (which would return True/1 and cause
         "OND file not found: 1" downstream).
Fix 2 – File descriptor safety: run_simulation() accepts only serialisable
         arguments (strings, ints, floats). All file I/O — PAN, OND, weather —
         happens inside the worker after the pool forks, never before.
"""

import os
import time
import pathlib
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from pvsamlab.system import System, OND_DEFAULT

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = pathlib.Path(__file__).parent.resolve()
_REPO_ROOT = (_THIS_DIR / ".." / "..").resolve()

# ---------------------------------------------------------------------------
# Defaults  (all overridable via main() keyword arguments)
# ---------------------------------------------------------------------------
DEFAULT_PAN_FOLDERS = [
    str(_REPO_ROOT / "pvsamlab" / "data" / "modules" / "ja29mps"),
]
DEFAULT_OND_FILE = OND_DEFAULT        # re-exported for notebook import
DEFAULT_LAT = 33.027755
DEFAULT_LON = -100.081376
DEFAULT_YEAR_RANGE = range(1998, 2024)
DEFAULT_STRING_RANGE = range(28, 30)
DEFAULT_NUM_WORKERS = 8
DEFAULT_OUTPUT_DIR = str(_THIS_DIR)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_pan_files(folder_path):
    """Return a sorted list of .PAN file paths found in folder_path."""
    folder = pathlib.Path(folder_path)
    return sorted(str(p) for p in folder.iterdir() if p.suffix.lower() == ".pan")


def find_ond_file(folder_path):
    """Return the first .OND file path found in folder_path, or None.

    Fix 1: returns the path *string*, not os.path.exists(path) which is a
    boolean (True == 1) and would cause "OND file not found: 1" errors when
    passed to Inverter.from_ond().
    """
    folder = pathlib.Path(folder_path)
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() == ".ond":
            return str(p)        # <-- path string, not os.path.exists(p)
    return None


# ---------------------------------------------------------------------------
# Simulation worker
# ---------------------------------------------------------------------------

def run_simulation(pan_file, ond_file, year, modules_per_string, lat, lon):
    """Run a single PySAM simulation inside a worker process.

    Fix 2: all arguments are plain strings/ints/floats — no file handles are
    opened or passed from the parent. All file I/O (PAN, OND, weather) happens
    here, after the pool has forked, so there are no inherited file descriptors
    to go stale.

    Returns
    -------
    (summary_dict, df_hourly) on success
    (error_dict,   None)      on failure
    """
    try:
        plant = System(
            met_year=str(year),
            lat=lat,
            lon=lon,
            pan_file=pan_file,
            ond_file=ond_file,
            modules_per_string=modules_per_string,
        )
        plant.run()

        summary = {
            "Module": plant.module.model,
            "Year": year,
            "ModulesPerString": modules_per_string,
            "VocMax": round(max(plant.model.Outputs.subarray1_voc), 2),
            "MPPTLoss": round(plant.model.Outputs.annual_dc_invmppt_loss, 2),
        }

        voc = plant.model.Outputs.subarray1_voc
        poa = plant.model.Outputs.subarray1_poa_nom
        tamb = plant.model.Outputs.tdry
        vmp = plant.model.Outputs.subarray1_dc_voltage
        isc = plant.model.Outputs.subarray1_isc

        dt_index = pd.date_range(
            start=f"{year}-01-01 00:30", end=f"{year}-12-31 23:30", freq="h"
        )
        dt_index = dt_index[~((dt_index.month == 2) & (dt_index.day == 29))]

        df_hourly = pd.DataFrame({
            "Year": dt_index.year,
            "Month": dt_index.month,
            "Day": dt_index.day,
            "Hour": dt_index.hour,
            "Minute": dt_index.minute,
            "Module": plant.module.model,
            "YearSimulated": year,
            "ModulesPerString": modules_per_string,
            "POA": poa,
            "Tamb": tamb,
            "Voc": voc,
            "Vmp": vmp,
            "Isc": isc,
        })

        return summary, df_hourly

    except Exception as exc:
        return {
            "Error": str(exc),
            "Module": pathlib.Path(pan_file).stem,
            "Year": year,
            "ModulesPerString": modules_per_string,
        }, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    pan_folders=None,
    ond_file=None,
    year_range=None,
    string_range=None,
    lat=None,
    lon=None,
    num_workers=None,
    output_dir=None,
    tqdm_class=None,
):
    """Run batch string sizing simulations.

    Parameters
    ----------
    pan_folders : list of str, optional
        Folders containing .PAN module files.
    ond_file : str, optional
        Path to the .OND inverter file. Defaults to the bundled Sungrow SG4400.
    year_range : iterable of int, optional
        Meteorological years to simulate.
    string_range : iterable of int, optional
        Modules-per-string values to sweep.
    lat, lon : float, optional
        Site coordinates.
    num_workers : int, optional
        Number of parallel worker processes.
    output_dir : str or Path, optional
        Directory for CSV outputs. Defaults to the script directory.
    tqdm_class : class, optional
        tqdm class for the progress bar. Pass ``tqdm.notebook.tqdm`` when
        calling from Jupyter so the bar renders as a widget.

    Returns
    -------
    dict
        Keys: summary_path, hourly_path, n_total, n_failed, elapsed_s.
    """
    pan_folders  = pan_folders  if pan_folders  is not None else DEFAULT_PAN_FOLDERS
    ond_file     = ond_file     if ond_file     is not None else DEFAULT_OND_FILE
    year_range   = year_range   if year_range   is not None else DEFAULT_YEAR_RANGE
    string_range = string_range if string_range is not None else DEFAULT_STRING_RANGE
    lat          = lat          if lat          is not None else DEFAULT_LAT
    lon          = lon          if lon          is not None else DEFAULT_LON
    num_workers  = num_workers  if num_workers  is not None else DEFAULT_NUM_WORKERS
    output_dir   = pathlib.Path(output_dir) if output_dir is not None \
                   else pathlib.Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect PAN files
    pan_files = []
    for folder in pan_folders:
        pan_files.extend(get_pan_files(folder))
    if not pan_files:
        raise ValueError(f"No .PAN files found in: {pan_folders}")

    tasks = [
        (pan, ond_file, year, mps, lat, lon)
        for pan in pan_files
        for year in year_range
        for mps in string_range
    ]

    summary_rows = []
    hourly_rows = []
    n_failed = 0
    t_start = time.time()

    print(f"Running {len(tasks)} simulations with {num_workers} parallel workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(run_simulation, *t): t for t in tasks}
        iterator = as_completed(futures)

        if tqdm_class is not None:
            iterator = tqdm_class(iterator, total=len(futures), desc="Simulations")

        for future in iterator:
            summary, hourly = future.result()
            summary_rows.append(summary)
            if "Error" not in summary:
                if hourly is not None:
                    hourly_rows.append(hourly)
            else:
                n_failed += 1

    elapsed = time.time() - t_start
    n_ok = len(tasks) - n_failed

    summary_path = output_dir / "string_sizing_results_summary.csv"
    hourly_path  = output_dir / "string_sizing_results_hourly.csv"

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    if hourly_rows:
        pd.concat(hourly_rows, ignore_index=True).to_csv(hourly_path, index=False)

    print(f"Done: {n_ok} OK, {n_failed} failed, {elapsed:.1f}s elapsed")
    print(f"  Summary CSV : {summary_path}")
    if hourly_rows:
        print(f"  Hourly CSV  : {hourly_path}")

    return {
        "summary_path": str(summary_path),
        "hourly_path":  str(hourly_path) if hourly_rows else None,
        "n_total":      len(tasks),
        "n_failed":     n_failed,
        "elapsed_s":    elapsed,
    }


# ---------------------------------------------------------------------------
# Required for multiprocessing on Windows (spawn start method)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

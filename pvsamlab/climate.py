"""
This module contains functions to work with NSRDB GOES v4 resources using PySAM
"""

import os
import glob
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import requests
import pandas as pd
import numpy as np
from scipy import spatial
from dotenv import load_dotenv
from PySAM.ResourceTools import FetchResourceFiles

# Load secrets
_dir = os.path.dirname(__file__)
load_dotenv(os.path.join(_dir, 'secrets.env'))

api_key = os.getenv('NSRDB_API_KEY')
email = os.getenv('NSRDB_API_EMAIL')
your_name = os.getenv('NSRDB_API_NAME')

attrs = 'dhi,dni,ghi,air_temperature,surface_pressure,wind_direction,wind_speed'

# Centralized logging functions
logger = logging.getLogger(__name__)

def log_info(message):
    logger.info(message)

def log_error(message):
    logger.error(message)


def download_nsrdb_csv(coords, year='tmy', interval=60):
    """
    Downloads a single NSRDB GOES v4 resource CSV using PySAM into a per-location/year-type folder.

    Args:
        coords (tuple): (lat, lon)
        year (str): 'tmy' or a specific calendar year like '2017'
        interval (int): 60 or 30 (only 60 for 'tmy')

    Returns:
        str or None: path to downloaded CSV file
    """
    lat, lon = coords

    location_folder = f"{lat:.4f}_{lon:.4f}"

    if not isinstance(year, str):
        raise ValueError("`year` must be a string like 'tmy' or '2017'.")

    if year.lower() == 'tmy':
        resource_type = 'nsrdb-GOES-tmy-v4-0-0'
        year_type_folder = 'tmy'
    elif year.isdigit():
        resource_type = 'nsrdb-GOES-aggregated-v4-0-0'
        year_type_folder = 'time_series'
    else:
        raise ValueError("`year` must be 'tmy' or a numeric calendar year string like '2017'.")

    download_dir = os.path.join(_dir, 'data', 'weather_files', location_folder, year_type_folder)
    os.makedirs(download_dir, exist_ok=True)

    def _run_fetcher():
        fetcher = FetchResourceFiles(
            tech='solar',
            nrel_api_key=api_key,
            nrel_api_email=email,
            resource_type=resource_type,
            resource_year=year,
            resource_interval_min=interval,
            resource_dir=download_dir,
            verbose=False
        )
        # PySAM's FetchResourceFiles has developer.nrel.gov hardcoded internally.
        # That domain 301-redirects to developer.nlr.gov during the brownout period
        # (deadline April 30 2026). Wrap with a timeout so hung downloads fail fast.
        with ThreadPoolExecutor(max_workers=1) as _executor:
            _future = _executor.submit(fetcher.fetch, [(lon, lat)])
            try:
                _future.result(timeout=60)
            except FuturesTimeoutError:
                raise RuntimeError(
                    "NSRDB download timed out. Check that developer.nlr.gov "
                    "is reachable and API credentials are valid."
                )
        return fetcher.resource_file_paths

    try:
        paths = _run_fetcher()
        cleanup_query_json(download_dir, lat, lon)

        if not paths:
            log_error("❌ PySAM returned no resource files.")
            return None

        final_path = paths[0]

        result = validate_weather_file(final_path)

        if not result['valid'] and result['issue'] and 'nan' in result['issue'].lower():
            log_error(f"⚠️ Weather file has NaN issues ({result['issue']}), retrying download...")
            try:
                os.remove(final_path)
            except OSError:
                pass
            try:
                paths2 = _run_fetcher()
                cleanup_query_json(download_dir, lat, lon)
                if not paths2:
                    log_error("⚠️ Retry returned no resource files.")
                    return None
                final_path = paths2[0]
                result2 = validate_weather_file(final_path)
                if not result2['valid']:
                    log_error(
                        f"⚠️ Re-downloaded file also failed validation "
                        f"({result2['issue']}). Returning path anyway."
                    )
            except Exception as retry_exc:
                log_error(f"⚠️ Retry download failed: {retry_exc}")
                return None
        elif not result['valid']:
            log_error(f"⚠️ Weather file validation failed: {result['issue']}")

        return final_path

    except Exception as e:
        log_error(f"❌ PySAM download failed: {e}")
        return None


def cleanup_query_json(folder, lat, lon):
    """
    Removes PySAM metadata json files from the folder for the given location.
    """
    pattern = f"nsrdb_data_query_response_{lat:.4f}_{lon:.4f}.json"
    full_path = os.path.join(folder, pattern)
    matches = glob.glob(full_path)
    for f in matches:
        try:
            os.remove(f)
        except Exception:
            pass  # Silent cleanup, logging not needed per user instruction


def validate_weather_file(filepath) -> dict:
    """
    Validate an NSRDB GOES v4 CSV weather file.

    GOES v4 headers have no 'Interval' field (unlike older PSM3 format).
    Interval is detected from row count: 8760 = 60-min, 17520 = 30-min.
    Minute=30 in every data row is normal center-of-hour convention and
    does NOT indicate a 30-minute file.

    Returns:
        dict with keys: valid (bool), interval (int or None),
        nan_fraction (float or None), row_count (int or None),
        issue (str or None).
    """
    try:
        meta_keys = pd.read_csv(filepath, nrows=1, header=None).iloc[0].tolist()
        meta_vals = pd.read_csv(filepath, skiprows=1, nrows=1, header=None).iloc[0].tolist()
        meta = dict(zip(meta_keys, meta_vals))

        lat = meta.get('Latitude') or meta.get('latitude')
        lon = meta.get('Longitude') or meta.get('longitude')

        # Skip 2 metadata rows + 1 column-header row = skiprows=2 reads data
        df = pd.read_csv(filepath, skiprows=2)
        row_count = len(df)

        # Detect interval from row count only; Minute=30 is center-of-hour, not 30-min data
        if row_count == 8760:
            interval = 60
        elif row_count == 17520:
            interval = 30
        else:
            interval = None

        required = ['GHI', 'DNI', 'DHI']
        missing_cols = [c for c in required if c not in df.columns]

        nan_frac = 0.0
        if not missing_cols:
            nan_frac = float(df[required].isna().mean().mean())

        valid = (
            not missing_cols
            and nan_frac < 0.05
            and interval in (60, 30)
            and lat is not None
            and lon is not None
        )

        issue = None
        if missing_cols:
            issue = f"Missing columns: {missing_cols}"
        elif nan_frac >= 0.05:
            issue = f"NaN fraction {nan_frac:.1%} exceeds 5%"
        elif interval is None:
            issue = f"Unexpected row count: {row_count}"
        elif lat is None:
            issue = "Latitude missing from header"

        return {
            'valid': valid,
            'interval': interval,
            'nan_fraction': round(nan_frac, 4),
            'row_count': row_count,
            'issue': issue,
        }

    except Exception as e:
        return {
            'valid': False,
            'interval': None,
            'nan_fraction': None,
            'row_count': None,
            'issue': str(e),
        }


def resample_to_hourly(filepath):
    """
    Resample a 30-minute NSRDB CSV to 60-minute in place.

    Pairs of consecutive rows are collapsed to one hourly row:
    - Irradiance (GHI, DNI, DHI), Temperature, Pressure, Wind Speed: mean
    - Wind Direction and time columns: first value of each pair
    Updates 'Interval' in the metadata header from 30 to 60.
    """
    import pathlib

    # Preserve the raw header lines so the file structure stays intact
    with open(filepath, 'r') as fh:
        lines = fh.readlines()

    header_line0 = lines[0]   # metadata keys
    header_line1 = lines[1]   # metadata values
    col_line = lines[2]       # column names

    # Update Interval value in metadata
    keys = [k.strip() for k in header_line0.split(',')]
    vals = [v.strip() for v in header_line1.split(',')]
    try:
        idx = keys.index('Interval')
        vals[idx] = '60'
        header_line1 = ','.join(vals) + '\n'
    except ValueError:
        pass

    df = pd.read_csv(filepath, skiprows=2)

    irr_cols  = [c for c in ['GHI', 'DNI', 'DHI'] if c in df.columns]
    mean_cols = [c for c in ['Temperature', 'Pressure', 'Wind Speed'] if c in df.columns]
    first_cols = [c for c in ['Wind Direction'] if c in df.columns]
    time_cols = [c for c in ['Year', 'Month', 'Day', 'Hour', 'Minute'] if c in df.columns]

    agg_funcs = {}
    for c in df.columns:
        if c in irr_cols or c in mean_cols:
            agg_funcs[c] = 'mean'
        else:
            agg_funcs[c] = 'first'

    df['_group'] = df.index // 2
    df_hourly = df.groupby('_group').agg(agg_funcs).reset_index(drop=True)
    df_hourly = df_hourly[df.columns.drop('_group')]

    filename = pathlib.Path(filepath).name
    log_info(f"Resampled 30-min → 60-min: {filename}")

    with open(filepath, 'w') as fh:
        fh.write(header_line0)
        fh.write(header_line1)
        fh.write(col_line)
        df_hourly.to_csv(fh, index=False, header=False)


def find_nearest(point, set_of_points):
    """Find index of nearest point from a set using KDTree."""
    tree = spatial.KDTree(set_of_points)
    distance, index_of_nearest = tree.query(point)
    return index_of_nearest


def get_ashrae_design_low(lat, lon):
    """Returns the ASHRAE extreme low temperature for the nearest station."""
    csv_path = os.path.join(_dir, 'data', '_ashraeDB.csv')

    if not os.path.exists(csv_path):
        log_error(f"❌ ASHRAE database not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    stations = df.loc[:, ['Lat', 'Lon']].values
    nearest_station_index = find_nearest((lat, lon), stations)
    return df.loc[nearest_station_index, 'ExtrLow']


def check_nsrdb_connectivity() -> bool:
    """Ping the NLR API to verify connectivity and credentials."""
    api_key = os.getenv('NSRDB_API_KEY')
    url = (
        "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download"
        f"?api_key={api_key}&wkt=POINT(-100+33)&names=2017"
        "&interval=60&attributes=ghi&email="
        + os.getenv('NSRDB_API_EMAIL', '')
    )
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            print("NSRDB API connection OK (developer.nlr.gov)")
            return True
        else:
            print(f"NSRDB API returned {r.status_code}: {r.text[:200]}")
            return False
    except requests.exceptions.Timeout:
        print("NSRDB API timed out — check developer.nlr.gov is reachable")
        return False
    except Exception as e:
        print(f"NSRDB API error: {e}")
        return False


if __name__ == '__main__':
    download_nsrdb_csv((30.9759, -97.2465), 'tmy')      # TMY file
    download_nsrdb_csv((30.9759, -97.2465), '2017')     # Aggregated year file

    for year in range(1998, 2024):
        log_info(f"🔄 Downloading year {year}...")
        download_nsrdb_csv((30.9759, -97.2465), str(year))

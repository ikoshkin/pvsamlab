"""
This module contains functions to work with NSRDB GOES v4 resources using PySAM
"""

import os
import glob
import logging
from pathlib import Path

import requests
import pandas as pd
import numpy as np
from scipy import spatial
from dotenv import load_dotenv

# Load secrets
_dir = os.path.dirname(__file__)
load_dotenv(os.path.join(_dir, 'secrets.env'))

api_key = os.getenv('NSRDB_API_KEY')
email = os.getenv('NSRDB_API_EMAIL')
your_name = os.getenv('NSRDB_API_NAME')

attrs = 'dhi,dni,ghi,air_temperature,surface_pressure,wind_direction,wind_speed'

_SAM_UI_DIRS = [
    Path.home() / "SAM Downloaded Weather Files",
    Path.home() / "Documents" / "SAM Downloaded Weather Files",
]

# Centralized logging functions
logger = logging.getLogger(__name__)

def log_info(message):
    logger.info(message)

def log_error(message):
    logger.error(message)


def _download_direct(lat, lon, year, download_dir):
    """Direct NSRDB CSV download via the synchronous streaming endpoint.

    Returns (filepath, is_interpolated).  is_interpolated is True when the
    CSV doesn't cover a full Jan 1 – Dec 31 calendar year, indicating the
    server filled a source gap with modelled data.
    """
    import io
    import time

    if year.lower() == 'tmy':
        endpoint = (
            "https://developer.nlr.gov/api/nsrdb/v2/solar/"
            "nsrdb-GOES-tmy-v4-0-0-download.csv"
        )
    else:
        endpoint = (
            "https://developer.nlr.gov/api/nsrdb/v2/solar/"
            "nsrdb-GOES-aggregated-v4-0-0-download.csv"
        )

    params = {
        "api_key":      api_key,
        "full_name":    your_name,
        "email":        email,
        "affiliation":  "pvsamlab",
        "reason":       "research",
        "mailing_list": "false",
        "wkt":          f"POINT({lon} {lat})",
        "names":        year,
        "attributes":   attrs,
        "leap_day":     "false",
        "utc":          "false",
        "interval":     "60",
    }

    response = requests.get(endpoint, params=params, timeout=120)

    if response.status_code == 429:
        time.sleep(30)
        response = requests.get(endpoint, params=params, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(
            f"NLR API error {response.status_code}: {response.text[:300]}"
        )

    ct = response.headers.get('content-type', '')
    is_csv = 'text/csv' in ct or response.text.lstrip().startswith('Source,')
    if not is_csv:
        raise RuntimeError(
            f"NLR API returned non-CSV response: {response.text[:300]}"
        )

    csv_text = response.text

    import io as _io
    df_check = pd.read_csv(_io.StringIO(csv_text), skiprows=2)
    if len(df_check) not in (8760, 17520):
        raise RuntimeError(
            f"Incomplete data: got {len(df_check)} rows, expected 8760 or 17520"
        )

    # Flag years whose date range doesn't span Jan 1 – Dec 31: the server
    # filled a source gap and some rows may be modelled rather than observed.
    first_month = int(df_check.iloc[0]['Month'])
    first_day   = int(df_check.iloc[0]['Day'])
    last_month  = int(df_check.iloc[-1]['Month'])
    last_day    = int(df_check.iloc[-1]['Day'])
    is_interpolated = not (
        first_month == 1 and first_day == 1 and
        last_month == 12 and last_day == 31
    )

    resource = (
        'nsrdb-GOES-tmy-v4-0-0' if year == 'tmy'
        else 'nsrdb-GOES-aggregated-v4-0-0'
    )
    filename = f"nsrdb_{lat:.6f}_{lon:.6f}_{resource}_60_{year}.csv"
    filepath = os.path.join(download_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(csv_text)

    return filepath, is_interpolated


def download_nsrdb_csv(coords, year='tmy', interval=60, _meta_out=None, search_dirs=None):
    """
    Downloads a single NSRDB GOES v4 resource CSV into a per-location/year-type folder.

    Checks local files (pvsamlab cache, SAM UI folders, search_dirs) before
    attempting any API download.

    Args:
        coords (tuple): (lat, lon)
        year (str): 'tmy' or a specific calendar year like '2017'
        interval (int): 60 or 30 (only 60 for 'tmy')
        _meta_out (list, optional): internal — if provided, a dict with key
            'interpolated' (bool) is appended after a successful download.
        search_dirs (list of str/Path, optional): additional directories to
            search for local weather files before downloading.

    Returns:
        str or None: path to downloaded CSV file
    """
    lat, lon = coords

    # Check local files before attempting any API download
    local = find_local_weather_file(lat, lon, year, search_dirs)
    if local is not None:
        log_info(f"Using local weather file: {local}")
        if _meta_out is not None:
            _meta_out.append({'interpolated': False})
        return local

    location_folder = f"{lat:.4f}_{lon:.4f}"

    if not isinstance(year, str):
        raise ValueError("`year` must be a string like 'tmy' or '2017'.")

    if year.lower() == 'tmy':
        year_type_folder = 'tmy'
    elif year.isdigit():
        year_type_folder = 'time_series'
    else:
        raise ValueError("`year` must be 'tmy' or a numeric calendar year string like '2017'.")

    download_dir = os.path.join(_dir, 'data', 'weather_files', location_folder, year_type_folder)
    os.makedirs(download_dir, exist_ok=True)

    try:
        final_path, is_interpolated = _download_direct(lat, lon, year, download_dir)
        if is_interpolated:
            log_info(
                f"Source gap filled with modelled data: "
                f"year {year} at ({lat:.4f}, {lon:.4f})"
            )
        if _meta_out is not None:
            _meta_out.append({'interpolated': is_interpolated})
    except Exception as e:
        log_error(f"Download failed: {e}")
        return None

    result = validate_weather_file(final_path)
    if not result['valid']:
        log_error(f"⚠️ Weather file validation failed: {result['issue']}")

    return final_path


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


def find_local_weather_file(lat, lon, year, search_dirs=None):
    """
    Search for a cached or user-provided weather file.

    Searches:
    1. pvsamlab cache (exact naming)
    2. SAM Downloaded Weather Files folder
    3. User-specified search_dirs

    Matches files by lat/lon proximity (4 decimal places) and year.
    Returns first match or None.
    """
    # 1. pvsamlab cache — exact naming convention
    resource = 'nsrdb-GOES-aggregated-v4-0-0'
    exact = (
        Path(_dir) / 'data' / 'weather_files' /
        f"{lat:.4f}_{lon:.4f}" / 'time_series' /
        f"nsrdb_{lat:.6f}_{lon:.6f}_{resource}_60_{year}.csv"
    )
    if exact.exists():
        return str(exact)

    # 2 & 3. External directories — flexible glob patterns
    external_dirs = list(search_dirs or []) + _SAM_UI_DIRS
    patterns = [
        f"nsrdb_{lat:.4f}*_{lon:.4f}*_{year}*.csv",
        f"*{lat:.2f}_{lon:.2f}*{year}*.csv",
        f"*{lat:.2f}_{lon:.2f}*{year}*.epw",
    ]
    for dir_path in external_dirs:
        dp = Path(dir_path)
        if not dp.exists():
            continue
        for pattern in patterns:
            matches = list(dp.rglob(pattern))
            if matches:
                return str(matches[0])

    return None


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


def download_weather_files(
    lat: float,
    lon: float,
    year_range,
    delay_seconds: float = 3.0,
    progress_callback=None,
    search_dirs=None,
) -> dict:
    """Download NSRDB GOES v4 weather files for a location and range of years.

    Checks pvsamlab cache, then SAM UI folders / search_dirs, then downloads
    from the API. Calls::

        progress_callback(year, status, filepath, elapsed)

    at each event, where *status* is one of:

    * ``'cached'``        — file already in pvsamlab cache, no download needed
    * ``'local_sam_ui'``  — file found in SAM Downloaded Weather Files folder
    * ``'local_user'``    — file found in a user-specified search_dirs path
    * ``'downloading'``   — download about to start (filepath/elapsed are None)
    * ``'ok'``            — download succeeded, full calendar year data
    * ``'interpolated'``  — download succeeded but server filled a source gap
    * ``'failed'``        — download failed (filepath is None)

    Parameters
    ----------
    lat, lon : float
        Site coordinates (decimal degrees).
    year_range : iterable of int
        Calendar years to download (e.g. ``range(1998, 2024)``).
    delay_seconds : float
        Seconds to sleep between downloads to avoid API rate limits.
    progress_callback : callable, optional
        Called after each event as described above.
    search_dirs : list of str or Path, optional
        Additional directories to search for local weather files before
        downloading. Searched before the default SAM UI directories.

    Returns
    -------
    dict
        Mapping ``{year (int): filepath str or None}``.
    """
    import time as _time

    location_folder = f"{lat:.4f}_{lon:.4f}"
    cache_dir = os.path.join(_dir, 'data', 'weather_files', location_folder, 'time_series')
    resource  = 'nsrdb-GOES-aggregated-v4-0-0'

    years = sorted(set(int(y) for y in year_range))
    results = {}

    for i, year in enumerate(years):
        cached_path = os.path.join(
            cache_dir,
            f"nsrdb_{lat:.6f}_{lon:.6f}_{resource}_60_{year}.csv",
        )
        if os.path.exists(cached_path):
            if progress_callback:
                progress_callback(year, 'cached', cached_path, 0.0)
            results[year] = cached_path
            continue

        # Check external local files (SAM UI dirs and user-specified dirs)
        local_path = find_local_weather_file(lat, lon, str(year), search_dirs=search_dirs)
        if local_path is not None:
            local_p = Path(local_path)
            is_sam_ui = any(local_p.is_relative_to(d) for d in _SAM_UI_DIRS)
            status = 'local_sam_ui' if is_sam_ui else 'local_user'
            if progress_callback:
                progress_callback(year, status, local_path, 0.0)
            results[year] = local_path
            continue

        if progress_callback:
            progress_callback(year, 'downloading', None, None)

        t0 = _time.monotonic()
        try:
            _meta = []
            path = download_nsrdb_csv((lat, lon), str(year), _meta_out=_meta)
            elapsed = _time.monotonic() - t0
            if path:
                is_interpolated = _meta[0]['interpolated'] if _meta else False
                status = 'interpolated' if is_interpolated else 'ok'
                if progress_callback:
                    progress_callback(year, status, path, elapsed)
                results[year] = path
            else:
                if progress_callback:
                    progress_callback(year, 'failed', None, elapsed)
                results[year] = None
        except Exception:
            elapsed = _time.monotonic() - t0
            if progress_callback:
                progress_callback(year, 'failed', None, elapsed)
            results[year] = None

        if i < len(years) - 1:
            _time.sleep(delay_seconds)

    return results


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

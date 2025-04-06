"""
This module contains functions to work with NSRDB GOES v4 resources using PySAM
"""

import os
import glob
import logging

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

    # log_info(f"📡 Starting NSRDB download: year={year}, location=({lat}, {lon}), type={resource_type}")
    # log_info(f"📁 Target folder: {download_dir}")

    try:
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

        fetcher.fetch([(lon, lat)])
        paths = fetcher.resource_file_paths

        cleanup_query_json(download_dir, lat, lon)

        if not paths:
            log_error("❌ PySAM returned no resource files.")
            return None

        final_path = paths[0]
        # log_info(f"✅ Weather file saved to: {final_path}")
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


if __name__ == '__main__':
    download_nsrdb_csv((30.9759, -97.2465), 'tmy')      # TMY file
    download_nsrdb_csv((30.9759, -97.2465), '2017')     # Aggregated year file

    for year in range(1998, 2024):
        log_info(f"🔄 Downloading year {year}...")
        download_nsrdb_csv((30.9759, -97.2465), str(year))

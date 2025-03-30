"""
This module contains functions to work with NSRDB GOES v4 resources using PySAM
"""

import os
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


def download_nsrdb_csv(coords, year='tmy', interval=60):
    """
    Downloads NSRDB GOES resource CSV using PySAM's FetchResourceFiles.

    Args:
        coords (tuple): (lat, lon)
        year (str): 'tmy' or specific year like '2017'
        interval (int): data interval in minutes (60 or 30)

    Returns:
        str or None: path to downloaded CSV file
    """
    lat, lon = coords
    ftag = f'{year}_nsrdb_goes_v4'
    fname = os.path.join(_dir, 'data', 'tmp', f'{coords}_{ftag}.csv')
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    if os.path.isfile(fname):
        return fname

    resource_type = (
        'nsrdb-GOES-tmy-v4-0-0' if str(year).lower() == 'tmy'
        else 'nsrdb-GOES-aggregated-v4-0-0'
    )

    try:
        fetcher = FetchResourceFiles(
            tech='solar',
            nrel_api_key=api_key,
            nrel_api_email=email,
            resource_type=resource_type,
            resource_year=year,
            resource_interval_min=interval,
            resource_dir=os.path.dirname(fname),
            verbose=False
        )

        fetcher.fetch([(lon, lat)])
        if fetcher.resource_file_paths:
            return fetcher.resource_file_paths[0]
        else:
            print("[ERROR] PySAM downloaded, but no file path found.")
            return None

    except Exception as e:
        print(f"[ERROR] PySAM download failed: {e}")
        return None



def find_nearest(point, set_of_points):
    """Find index of nearest point from a set using KDTree."""
    tree = spatial.KDTree(set_of_points)
    distance, index_of_nearest = tree.query(point)
    return index_of_nearest


def get_ashrae_design_low(lat, lon):
    """Returns the ASHRAE extreme low temperature for the nearest station."""
    df = pd.read_csv(os.path.join(_dir, 'data', '_ashraeDB.csv'))
    stations = df.loc[:, ['Lat', 'Lon']].values
    nearest_station_index = find_nearest((lat, lon), stations)
    return df.loc[nearest_station_index, 'ExtrLow']


if __name__ == '__main__':
    tmy = download_nsrdb_csv((30.9759, -97.2465), 'tmy')
    print(f"TMY file: {tmy}")

    annual = download_nsrdb_csv((30.9759, -97.2465), '2017')
    print(f"2017 file: {annual}")

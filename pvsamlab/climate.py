"""
This module contains functions to work with NSRDB resources
"""
import os

import pandas as pd
import numpy as np
from scipy import spatial

_dir = os.path.dirname(__file__)

api_key = 'DEMO_KEY'
your_name = 'Lolo+Pepe'
email = 'lolo@pepe.com'
attrs = 'dhi,dni,ghi,air_temperature,surface_pressure,wind_direction,wind_speed'

def download_nsrdb_csv(coords, year='tmy', attributes=attrs):
    """
    Downloads solar resource file in csv format from NSRDB to a local folder.

    Args:
        lat (float): latitude
        lon (float): longitude
        year (str): 'tmy' or '1998' through '2016'
        attribute (str): list of values to be included
    Returns:
        wfname (str): path to downloaded file
    """

    lat, lon = coords
    wfname = os.path.join(_dir, 'data', 'tmp', f'{coords}_nsrdb_{year}.csv')
    if os.path.isfile(wfname):
        return wfname

    url = (f'http://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?'
           f'api_key={api_key}&'
           f'full_name={your_name}'
           f'&email={email}'
           f'&affiliation=lolopepe'
           f'&reason=Prospecting'
           f'&mailing_list=true'
           f'&wkt=POINT({lon}+{lat})'
           f'&names={year}'
           f'&attributes={attributes}'
           f'&leap_day=false'
           f'&utc=false'
           f'&interval=60')
    try:
        wdf = pd.read_csv(url)   
        wdf.to_csv(wfname, index=False)
        return wfname
    except:
        pass


def find_nearest(point, set_of_points):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html

    tree = spatial.KDTree(set_of_points)
    distance, index_of_nearest = tree.query(point)
    return index_of_nearest


def get_ashrae_design_low(lat, lon):

    df = pd.read_csv(os.path.join(_dir, 'data', '_ashraeDB.csv'))
    stations = df.loc[:, ['Lat', 'Lon']].values
    nearest_station_index = find_nearest((lat, lon), stations)
    

    return df.loc[nearest_station_index, 'ExtrLow']


if __name__ == '__main__':
    # tmy = get_nsrdb_data(41.7566, -111.0480, 'tmy')
    # print(tmy.head())
    nf = download_nsrdb_csv((41.7566, -101.0480), '2016')
    print(nf)

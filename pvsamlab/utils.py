import os
import logging
import pandas as pd
from scipy import spatial
from pvlib.iotools import read_panond
import PySAM.Wfreader as wf  # Corrected import for weather file reader
from pkg_resources import resource_filename

# Configure logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FILE = "app.log"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[
    logging.FileHandler(LOG_FILE),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

def log_info(message):
    """Logs informational messages."""
    logger.info(message)

def log_error(message):
    """Logs error messages."""
    logger.error(message)

def fetch_weather_file(lat, lon, dataset_type="TMY", api_key="your_nsrdb_api_key"):
    """
    Fetches a weather file from NSRDB if it doesn't exist locally.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        dataset_type (str, optional): Type of weather data to fetch (default: "TMY").
        api_key (str, optional): NSRDB API key for downloading weather files.

    Returns:
        str: Path to the downloaded or existing weather file.
    """
    weather_folder = resource_filename(__name__, 'data/weather_files')
    os.makedirs(weather_folder, exist_ok=True)

    file_name = f"weather_{lat}_{lon}_{dataset_type}.csv"
    file_path = os.path.join(weather_folder, file_name)

    if os.path.exists(file_path):
        log_info(f"✅ Using existing weather file: {file_path}")
        return file_path

    log_info(f"🌍 Downloading {dataset_type} weather data for lat: {lat}, lon: {lon}...")
    try:
        weather_data = wf.SRWdownload(lat, lon, api_key=api_key)
        if weather_data:
            with open(file_path, "w") as f:
                f.write(weather_data)
            log_info(f"✅ Weather file saved to: {file_path}")
            return file_path
    except Exception as e:
        log_error(f"❌ Error fetching weather data: {e}")

    return os.path.join(weather_folder, "default_weather.csv")

def mw_to_kw(mw):
    """Convert megawatts (MW) to kilowatts (kW)."""
    return mw * 1000

def w_to_kw(watts):
    """Convert watts (W) to kilowatts (kW)."""
    return watts / 1000

def calculate_capacity_factor(mwh_per_year, kwac):
    """Calculates capacity factor as a percentage based on kWac."""
    hours_per_year = 8760
    return (mwh_per_year / (kwac * hours_per_year)) * 100

def calculate_string_size(module_voc, module_tc_voc, design_low_temp, system_voltage):
    """
    Calculates the optimal number of modules per string.

    Args:
        module_voc (float): Open-circuit voltage of the module (V).
        module_tc_voc (float): Temperature coefficient of voltage (%/°C).
        design_low_temp (float): Lowest design temperature (°C).
        system_voltage (int): System DC voltage (V).

    Returns:
        int: Optimal number of modules per string.
    """
    correction = 1 + module_tc_voc / 100 * (design_low_temp - 25)
    return max(1, int(system_voltage / (module_voc * correction)))

def parse_pan_file(pan_file):
    """
    Parses a .PAN file using the latest pvlib library and extracts relevant module parameters.

    Args:
        pan_file (str): Path to the .PAN file.

    Returns:
        dict: Extracted module parameters.
    """
    if not os.path.exists(pan_file):
        log_error(f"❌ PAN file not found: {pan_file}")
        return {}

    try:
        # Read .PAN file using pvlib.iotools.read_panond()
        pan_data = read_panond(pan_file)

        # Detect the correct PVObject key (with or without BOM)
        pv_object_key = next((k for k in pan_data.keys() if "PVObject_" in k), None)
        if not pv_object_key:
            log_error(f"⚠️ PAN file format issue: Missing 'PVObject_' key in {pan_file}")
            return {}

        pan_object = pan_data.get(pv_object_key, {})

        # Extract module parameters
        extracted_params = {
            "manufacturer": pan_object.get("PVObject_Commercial", {}).get("Manufacturer", "Unknown"),
            "model": pan_object.get("PVObject_Commercial", {}).get("Model", "Unknown"),
            "width": pan_object.get("PVObject_Commercial", {}).get("Width", 1.0),
            "height": pan_object.get("PVObject_Commercial", {}).get("Height", 2.0),
            "technology": pan_object.get("Technol", "Unknown"),
            "n_series": pan_object.get("NCelS", 60),
            "n_parallel": pan_object.get("NCelP", 1),
            "n_diodes": pan_object.get("NDiode", 3),
            "submodule_layout": pan_object.get("SubModuleLayout", "Standard"),
            "v_oc": pan_object.get("Voc", 0),
            "i_sc": pan_object.get("Isc", 0),
            "v_mp": pan_object.get("Vmp", 0),
            "i_mp": pan_object.get("Imp", 0),
            "area": pan_object.get("Width", 1.134) * pan_object.get("Height", 2.382),  # Compute module area
            "muIsc": pan_object.get("muISC", 0),
            "muVoc": pan_object.get("muVocSpec", 0),
            "muPmp": pan_object.get("muPmpReq", 0),
            "r_shunt": pan_object.get("RShunt", 700),
            "r_series": pan_object.get("RSerie", 0.152),
            "gamma": pan_object.get("Gamma", 1.065),
            "muGamma": pan_object.get("muGamma", -0.0004),
            "max_voltage_iec": pan_object.get("VMaxIEC", 1500),
            "max_voltage_ul": pan_object.get("VMaxUL", 1500),
            "bifaciality": pan_object.get("BifacialityFactor", 0.8),
            "iam_profile": pan_object.get("PVObject_IAM", {}).get("IAMProfile", {}).get("Point_1", [])[1:],  # Extract IAM curve if available
        }

        log_info(f"✅ Successfully parsed PAN file: {pan_file}")
        return extracted_params
    except Exception as e:
        log_error(f"⚠️ Error parsing PAN file: {e}")
        return {}

def parse_ond_file(ond_file):
    """
    Parses an .OND file and extracts relevant inverter parameters.

    Args:
        ond_file (str): Path to the .OND file.

    Returns:
        dict: Extracted inverter parameters.
    """
    if not os.path.exists(ond_file):
        log_error(f"❌ OND file not found: {ond_file}")
        return {}

    try:
        ond_data = pd.read_csv(ond_file, delimiter=";")

        extracted_params = {
            "inv_ds_paco": ond_data["Max_AC_Power(W)"].values[0],
            "inv_ds_eff": ond_data["Efficiency(%)"].values[0],
            "mppt_low_inverter": ond_data["MPPT_Low(V)"].values[0],
            "mppt_hi_inverter": ond_data["MPPT_High(V)"].values[0]
        }

        log_info(f"✅ Successfully parsed OND file: {ond_file}")
        return extracted_params
    except Exception as e:
        log_error(f"⚠️ Error parsing OND file: {e}")
        return {}

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
    weather_folder = resource_filename(__name__, 'data/tmp')
    os.makedirs(weather_folder, exist_ok=True)
    wfname = os.path.join(weather_folder, f'{coords}_nsrdb_{year}.csv')
    
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
    except Exception as e:
        log_error(f"❌ Error downloading NSRDB data: {e}")
        return None

def find_nearest(point, set_of_points):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html

    tree = spatial.KDTree(set_of_points)
    distance, index_of_nearest = tree.query(point)
    return index_of_nearest

def get_ashrae_design_low(lat, lon):
    ashrae_db_path = resource_filename(__name__, 'data/_ashraeDB.csv')
    
    if not os.path.exists(ashrae_db_path):
        log_error(f"❌ ASHRAE database file not found: {ashrae_db_path}")
        return None

    df = pd.read_csv(ashrae_db_path)
    stations = df.loc[:, ['Lat', 'Lon']].values
    nearest_station_index = find_nearest((lat, lon), stations)
    
    return df.loc[nearest_station_index, 'ExtrLow']
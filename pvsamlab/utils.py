import os
import logging
import pandas as pd
import pvlib
import PySAM.Wfreader as wf  # ✅ Corrected import for weather file reader

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
    weather_folder = "data/weather_files"
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
    Parses a .PAN file using pvlib and extracts relevant module parameters.

    Args:
        pan_file (str): Path to the .PAN file.

    Returns:
        dict: Extracted module parameters.
    """
    if not os.path.exists(pan_file):
        log_error(f"❌ PAN file not found: {pan_file}")
        return {}

    try:
        pan_data = pvlib.pvsystem.read_pan(pan_file)

        extracted_params = {
            "v_oc": pan_data["Voc_ref"],
            "i_sc": pan_data["Isc_ref"],
            "v_mp": pan_data["Vmp_ref"],
            "i_mp": pan_data["Imp_ref"],
            "area": pan_data.get("Area", 1.6),
            "n_series": pan_data["Cells_in_Series"],
            "t_noct": pan_data.get("T_NOCT", 45),
            "standoff": pan_data.get("Standoff", 6),
            "mounting": pan_data.get("Mounting", 1),
            "is_bifacial": int(pan_data.get("Bifacial", 0)),
            "bifacial_transmission_factor": pan_data.get("Bifacial_Trans_Factor", 0.95),
            "bifaciality": pan_data.get("Bifaciality", 0.7)
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

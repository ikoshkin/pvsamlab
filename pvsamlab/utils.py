import os
import logging
import PySAM.WeatherFile as wf

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
    weather_folder = "weather_files"
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

def calculate_capacity_factor(mwh_per_year, system_capacity_kw):
    """Calculates capacity factor as a percentage."""
    hours_per_year = 8760
    return (mwh_per_year / (system_capacity_kw * hours_per_year)) * 100

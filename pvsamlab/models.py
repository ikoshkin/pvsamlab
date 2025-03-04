import os
import requests
import PySAM.Pvsamv1 as pv
import PySAM.WeatherFile as wf
from pvsamlab.utils import log_info, log_error, fetch_weather_file, mw_to_kw, w_to_kw, calculate_capacity_factor, parse_pan_file

# 🔹 Define constants at the top
PAN_FILE_DIR = os.path.join(os.path.dirname(__file__), "data/modules")
DEFAULT_PAN_FILE = os.path.join(PAN_FILE_DIR, "default_module.pan")

class Location:
    def __init__(self, lat=None, lon=None, elev=None, tz=None):
        self.lat = lat if lat is not None else 0.0
        self.lon = lon if lon is not None else 0.0
        self.elev = elev if elev is not None else 0.0
        self.tz = tz if tz is not None else 0

class SolarResource:
    def __init__(self, solar_resource_file=None, solar_resource_data=None, lat=None, lon=None, use_wf_albedo=0, dataset_type="TMY"):
        self.use_wf_albedo = use_wf_albedo
        self.irrad_mode = 0
        self.sky_model = 2

        if solar_resource_file:
            self.solar_resource_file = solar_resource_file
        elif lat is not None and lon is not None:
            self.solar_resource_file = fetch_weather_file(lat, lon, dataset_type)
        else:
            self.solar_resource_file = "data/weather_files/default_weather.csv"

        self.solar_resource_data = solar_resource_data

class Module:
    def __init__(self, pan_file=None):
        self.module_model = 2
        pan_file = pan_file or DEFAULT_PAN_FILE
        self.params = parse_pan_file(pan_file)

class Inverter:
    def __init__(self, paco=4000.0, efficiency=96.0):
        self.paco = paco
        self.efficiency = efficiency

class SystemDesign:
    def __init__(self, system_capacity=None, inverter_count=None, subarray1_gcr=None):
        self.system_capacity = system_capacity
        self.inverter_count = inverter_count
        self.subarray1_gcr = subarray1_gcr

class Losses:
    def __init__(self, acwiring_loss=1.0, subarray1_dcwiring_loss=2.0, subarray1_soiling=5.0):
        self.acwiring_loss = acwiring_loss
        self.subarray1_dcwiring_loss = subarray1_dcwiring_loss
        self.subarray1_soiling = subarray1_soiling

class PVSystem:
    def __init__(self, lat=None, lon=None, elev=None, tz=None, mwac=None, dcac_ratio=None, module_model="Default Module", 
                 gcr=None, inverter_efficiency=97.5, solar_resource=None, system_design=None, losses=None, module=None, inverter=None):
        self.location = Location(lat, lon, elev, tz)
        self.mwac = mwac
        self.dcac_ratio = dcac_ratio
        self.module_model = module_model
        self.gcr = gcr
        self.inverter_efficiency = inverter_efficiency

        self.solar_resource = solar_resource if solar_resource else SolarResource(lat=lat, lon=lon)
        self.module = module if module else Module()
        self.inverter = inverter if inverter else Inverter()
        self.system_design = system_design if system_design else SystemDesign()
        self.losses = losses if losses else Losses()

        self.compute_pysam_inputs()
        self.assign_inputs()
        self.run_simulation()

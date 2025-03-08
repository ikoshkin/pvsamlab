import os
import json
import requests
import PySAM.Pvsamv1 as pv
import PySAM.Wfreader as wf  # ✅ Corrected weather file reader import
from pvsamlab.utils import log_info, log_error, fetch_weather_file, w_to_kw, calculate_capacity_factor, parse_pan_file, parse_ond_file

# Define default file paths
PAN_FILE_DIR = os.path.join(os.path.dirname(__file__), "data/modules")
DEFAULT_PAN_FILE = os.path.join(PAN_FILE_DIR, "JAM66D45-640LB(3.2+2.0mm).PAN")

OND_FILE_DIR = os.path.join(os.path.dirname(__file__), "data/inverters")
DEFAULT_OND_FILE = os.path.join(OND_FILE_DIR, "Sungrow_SG4400UD-MV-US_20230817_V14_PVsyst.6.8.6（New Version).OND")

class Location:
    def __init__(self, lat=None, lon=None, elev=None, tz=None):
        self.lat = lat if lat is not None else 0.0
        self.lon = lon if lon is not None else 0.0
        self.elev = elev if elev is not None else 0.0
        self.tz = tz if tz is not None else 0

    def __repr__(self):
        return json.dumps(vars(self), indent=4)

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

    def __repr__(self):
        return json.dumps(vars(self), indent=4)

class Module:
    def __init__(self, pan_file=None):
        self.module_model = 2
        pan_file = pan_file or DEFAULT_PAN_FILE
        self.params = parse_pan_file(pan_file)

    def __repr__(self):
        return json.dumps(vars(self), indent=4)

class Inverter:
    def __init__(self, ond_file=None):
        self.inverter_model = 1
        ond_file = ond_file or DEFAULT_OND_FILE
        self.params = parse_ond_file(ond_file)

    def __repr__(self):
        return json.dumps(vars(self), indent=4)

class SystemDesign:
    def __init__(self, kwac=None, target_dcac_ratio=1.35, inverter_count=None, subarray1_nstrings=None, subarray1_modules_per_string=None, 
                 subarray1_track_mode=1, subarray1_tilt=0.0, subarray1_azimuth=180.0, subarray1_backtrack=1, subarray1_gcr=0.33, 
                 mppt_low_inverter=250.0, mppt_hi_inverter=800.0, system_voltage=1500):
        self.kwac = kwac if kwac is not None else 100000  # Default to 100,000 kW (100 MW)
        self.target_dcac_ratio = target_dcac_ratio if target_dcac_ratio is not None else 1.35
        self.system_voltage = system_voltage

        self.system_capacity = self.kwac * self.target_dcac_ratio

        self.inverter_count = inverter_count
        self.subarray1_nstrings = subarray1_nstrings
        self.subarray1_modules_per_string = subarray1_modules_per_string
        self.subarray1_track_mode = subarray1_track_mode
        self.subarray1_tilt = subarray1_tilt
        self.subarray1_azimuth = subarray1_azimuth
        self.subarray1_backtrack = subarray1_backtrack
        self.subarray1_gcr = subarray1_gcr
        self.mppt_low_inverter = mppt_low_inverter
        self.mppt_hi_inverter = mppt_hi_inverter

        self.dcac_ratio = None  # Placeholder for computed value

    def compute_actual_dcac_ratio(self, module_power_watts=400):
        if self.subarray1_nstrings and self.subarray1_modules_per_string:
            total_modules = self.subarray1_nstrings * self.subarray1_modules_per_string
            total_dc_capacity = total_modules * module_power_watts / 1000  # Convert to kW
            self.dcac_ratio = total_dc_capacity / self.kwac
        else:
            self.dcac_ratio = self.target_dcac_ratio

        return self.dcac_ratio

    def __repr__(self):
        return json.dumps(vars(self), indent=4)

class Losses:
    def __init__(self, acwiring_loss=1.0, subarray1_dcwiring_loss=2.0, subarray1_soiling=5.0):
        self.acwiring_loss = acwiring_loss
        self.subarray1_dcwiring_loss = subarray1_dcwiring_loss
        self.subarray1_soiling = subarray1_soiling

    def __repr__(self):
        return json.dumps(vars(self), indent=4)

class PVSystem:
    def __init__(self, kwac=None, target_dcac_ratio=None, module_model="Default Module", 
                 solar_resource=None, system_design=None, losses=None, module=None, inverter=None):
        self.kwac = kwac
        self.target_dcac_ratio = target_dcac_ratio
        self.module_model = module_model

        self.solar_resource = solar_resource if solar_resource else SolarResource()
        self.module = module if module else Module()
        self.inverter = inverter if inverter else Inverter()
        self.system_design = system_design if system_design else SystemDesign()
        self.losses = losses if losses else Losses()

        self.compute_pysam_inputs()
        self.assign_inputs()
        self.run_simulation()

    def compute_pysam_inputs(self):
        self.system_capacity = self.system_design.system_capacity
        self.pysam_inputs = {
            "system_capacity": self.system_capacity,
            "subarray1_nstrings": self.system_design.subarray1_nstrings,
            "subarray1_modules_per_string": self.system_design.subarray1_modules_per_string,
            "subarray1_gcr": self.system_design.subarray1_gcr,
            "inverter_count": self.system_design.inverter_count,
            "inv_ds_eff": self.inverter.params.get("inv_ds_eff", 97.5)
        }

    def assign_inputs(self):
        self.model = pv.default("FlatPlatePVNone")
        self.model.SystemDesign.assign(self.pysam_inputs)

    def run_simulation(self):
        self.model.execute()
        self.process_outputs()

    def process_outputs(self):
        self.outputs = {
            "max_dc_voltage": self.model.Outputs.dc_voltage_max,
            "mwh_per_year": self.model.Outputs.annual_energy / 1000,
            "capacity_factor": calculate_capacity_factor(self.model.Outputs.annual_energy / 1000, self.system_capacity),
            "energy_losses_summary": {
                "inverter_efficiency_loss": self.model.Outputs.annual_ac_inv_eff_loss_percent,
                "soiling_loss": self.model.Outputs.annual_dc_soiling_loss_percent
            },
            "hourly_energy_data": self.model.Outputs.ac_gross
        }

    def __repr__(self):
        return json.dumps(vars(self), indent=4)

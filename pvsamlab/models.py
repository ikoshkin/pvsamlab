import os
import requests
import PySAM.Pvsamv1 as pv
import PySAM.WeatherFile as wf
from pvsamlab.utils import log_info, log_error, fetch_weather_file, mw_to_kw, w_to_kw, calculate_capacity_factor, parse_pan_file, parse_ond_file

# Define default file paths
PAN_FILE_DIR = os.path.join(os.path.dirname(__file__), "data/modules")
DEFAULT_PAN_FILE = os.path.join(PAN_FILE_DIR, "default_module.pan")

OND_FILE_DIR = os.path.join(os.path.dirname(__file__), "data/inverters")
DEFAULT_OND_FILE = os.path.join(OND_FILE_DIR, "default_inverter.ond")

class Location:
    def __init__(self, lat=None, lon=None, elev=None, tz=None):
        self.lat = lat if lat is not None else 0.0
        self.lon = lon if lon is not None else 0.0
        self.eleadd v = elev if elev is not None else 0.0
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
        """
        Represents a photovoltaic module using the 6-parameter user-defined model.

        Args:
            pan_file (str, optional): Path to a .PAN file to extract module parameters.
                                      Defaults to the default module file.
        """
        self.module_model = 2
        pan_file = pan_file or DEFAULT_PAN_FILE
        self.params = parse_pan_file(pan_file)

    def __repr__(self):
        return f"Module({self.params})"

class Inverter:
    def __init__(self, ond_file=None):
        """
        Represents an inverter using the datasheet model.

        Args:
            ond_file (str, optional): Path to an .OND file to extract inverter parameters.
                                      Defaults to the default inverter file.
        """
        self.inverter_model = 1  # Always use datasheet-based model
        ond_file = ond_file or DEFAULT_OND_FILE
        self.params = parse_ond_file(ond_file)

    def __repr__(self):
        return f"Inverter({self.params})"

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

    def compute_pysam_inputs(self):
        DEFAULT_MWAC = 100
        DEFAULT_DCAC_RATIO = 1.35
        DEFAULT_GCR = 0.3

        self.mwac = self.mwac if self.mwac is not None else DEFAULT_MWAC
        self.dcac_ratio = self.dcac_ratio if self.dcac_ratio is not None else DEFAULT_DCAC_RATIO
        self.gcr = self.gcr if self.gcr is not None else DEFAULT_GCR

        self.system_capacity = mw_to_kw(self.mwac) * self.dcac_ratio
        
        module_power_watts = 400
        total_modules = w_to_kw(self.system_capacity * 1000) / module_power_watts
        modules_per_string = 20
        n_strings = total_modules // modules_per_string

        self.pysam_inputs = {
            "system_capacity": self.system_capacity,
            "subarray1_nstrings": n_strings,
            "subarray1_modules_per_string": modules_per_string,
            "subarray1_gcr": self.gcr,
            "inverter_count": round(mw_to_kw(self.mwac) / (self.dcac_ratio * 4000)),
            "inv_ds_eff": self.inverter_efficiency
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

    def update_parameter(self, param: str, value):
        if hasattr(self, param):
            setattr(self, param, value)
            self.compute_pysam_inputs()
            self.assign_inputs()
            self.run_simulation()
        else:
            log_error(f"Parameter '{param}' not found in PVSystem")

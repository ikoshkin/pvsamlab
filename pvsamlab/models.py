import os
import json
import requests
import PySAM.Pvsamv1 as pv
import PySAM.Wfreader as wf
from pvsamlab.utils import log_info, log_error, fetch_weather_file, w_to_kw, calculate_capacity_factor, parse_pan_file, parse_ond_file

# Define paths dynamically based on the current working directory (repo root)
DATA_DIR = os.path.join(os.getcwd(), "data")
PAN_FILE_DIR = os.path.join(DATA_DIR, "modules")
OND_FILE_DIR = os.path.join(DATA_DIR, "inverters")

# Default file paths
DEFAULT_PAN_FILE = os.path.join(PAN_FILE_DIR, "JAM66D45-640LB(3.2+2.0mm).PAN")
DEFAULT_OND_FILE = os.path.join(OND_FILE_DIR, "Sungrow_SG4400UD-MV-US_20230817_V14_PVsyst.6.8.6（New Version).OND")

# Ensure directories exist to prevent errors
os.makedirs(PAN_FILE_DIR, exist_ok=True)
os.makedirs(OND_FILE_DIR, exist_ok=True)

class Location:
    def __init__(self, lat=None, lon=None, elev=None, tz=None):
        self.lat = lat if lat is not None else 0.0
        self.lon = lon if lon is not None else 0.0
        self.elev = elev if elev is not None else 0.0
        self.tz = tz if tz is not None else 0

    def __repr__(self):
        return json.dumps(vars(self), indent=4)

class SolarResource:
    def __init__(self, solar_resource_file=None, lat=None, lon=None, use_wf_albedo=0, dataset_type="TMY"):
        self.use_wf_albedo = use_wf_albedo
        self.irrad_mode = 0
        self.sky_model = 2

        if solar_resource_file:
            self.solar_resource_file = solar_resource_file
        elif lat is not None and lon is not None:
            self.solar_resource_file = fetch_weather_file(lat, lon, dataset_type)
        else:
            self.solar_resource_file = "data/weather_files/default_weather.csv"

    def assign_inputs(self, model):
        model.SolarResource.assign(vars(self))

    def __repr__(self):
        return json.dumps(vars(self), indent=4)

class Module:
    def __init__(self, pan_file=None):
        self.module_model = 2
        self.pan_file = pan_file or DEFAULT_PAN_FILE

        # Check if the file exists, log error if missing
        if not os.path.exists(self.pan_file):
            log_error(f"❌ PAN file not found: {self.pan_file}")

        # Parse full PAN file
        self.params = parse_pan_file(self.pan_file)

    def assign_inputs(self, model):
        """Filters and assigns only PySAM-compatible module parameters."""
        pysam_params = {
            "module_model": 2,  # Always `6par_user`
            "sixpar_voc": self.params.get("v_oc"),
            "sixpar_isc": self.params.get("i_sc"),
            "sixpar_vmp": self.params.get("v_mp"),
            "sixpar_imp": self.params.get("i_mp"),
            "sixpar_area": self.params.get("area"),
            "sixpar_nser": self.params.get("n_series"),
            "sixpar_tnoct": self.params.get("t_noct", 45),
            "sixpar_standoff": self.params.get("standoff", 6),
            "sixpar_mounting": self.params.get("mounting", 1),
            "sixpar_is_bifacial": int(self.params.get("bifaciality", 0) > 0),
            "sixpar_bifacial_transmission_factor": self.params.get("bifacial_transmission_factor", 0.95),
            "sixpar_bifaciality": self.params.get("bifaciality", 0.7),
        }

        model.Module.assign(pysam_params)  # ✅ Assign only PySAM-compatible values

    def __repr__(self):
        return json.dumps(vars(self), indent=4)


class Inverter:
    def __init__(self, ond_file=None):
        self.inverter_model = 1
        ond_file = ond_file or DEFAULT_OND_FILE
        self.params = parse_ond_file(ond_file)

    def assign_inputs(self, model):
        model.Inverter.assign(self.params)

    def __repr__(self):
        return json.dumps(vars(self), indent=4)

class SystemDesign:
    def __init__(self, kwac=None, target_dcac_ratio=1.35, inverter_count=None, subarray1_nstrings=None, subarray1_modules_per_string=None, 
                 subarray1_track_mode=1, subarray1_tilt=0.0, subarray1_azimuth=180.0, subarray1_backtrack=1, subarray1_gcr=0.33, 
                 mppt_low_inverter=250.0, mppt_hi_inverter=800.0, system_voltage=1500):
        self.kwac = kwac if kwac is not None else 100000
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

    def assign_inputs(self, model):
        model.SystemDesign.assign(vars(self))

    def __repr__(self):
        return json.dumps(vars(self), indent=4)

class Losses:
    def __init__(self, acwiring_loss=1.0, dcoptimizer_loss=0.5, subarray1_dcwiring_loss=2.0, subarray1_diodeconn_loss=0.5,
                 subarray1_mismatch_loss=1.0, subarray1_nameplate_loss=1.0, subarray1_rack_shading=0.5, subarray1_rear_soiling_loss=1.0,
                 subarray1_soiling=None, subarray1_tracking_loss=0.5, transmission_loss=1.0, transformer_no_load_loss=0.0,
                 transformer_load_loss=0.0):
        self.acwiring_loss = acwiring_loss
        self.dcoptimizer_loss = dcoptimizer_loss
        self.subarray1_dcwiring_loss = subarray1_dcwiring_loss
        self.subarray1_diodeconn_loss = subarray1_diodeconn_loss
        self.subarray1_mismatch_loss = subarray1_mismatch_loss
        self.subarray1_nameplate_loss = subarray1_nameplate_loss
        self.subarray1_rack_shading = subarray1_rack_shading
        self.subarray1_rear_soiling_loss = subarray1_rear_soiling_loss
        self.subarray1_soiling = subarray1_soiling if subarray1_soiling else [5.0] * 12
        self.subarray1_tracking_loss = subarray1_tracking_loss
        self.transmission_loss = transmission_loss
        self.transformer_no_load_loss = transformer_no_load_loss
        self.transformer_load_loss = transformer_load_loss

    def assign_inputs(self, model):
        model.Losses.assign(vars(self))

    def __repr__(self):
        return json.dumps(vars(self), indent=4)

class PVSystem:
    def __init__(self, kwac=None, target_dcac_ratio=None, module_model="Default Module", 
                 solar_resource=None, system_design=None, losses=None, module=None, inverter=None, location=None):
        self.kwac = kwac
        self.target_dcac_ratio = target_dcac_ratio
        self.module_model = module_model

        self.solar_resource = solar_resource if solar_resource else SolarResource()
        self.module = module if module else Module()
        self.inverter = inverter if inverter else Inverter()
        self.system_design = system_design if system_design else SystemDesign()
        self.losses = losses if losses else Losses()
        self.location = location if location else Location()

        self.assign_inputs()
        self.run_simulation()

    def assign_inputs(self):
        self.model = pv.default("FlatPlatePVNone")

        self.solar_resource.assign_inputs(self.model)
        self.module.assign_inputs(self.model)
        self.inverter.assign_inputs(self.model)
        self.system_design.assign_inputs(self.model)
        self.losses.assign_inputs(self.model)

    def run_simulation(self):
        self.model.execute()
        self.process_outputs()

    def __repr__(self):
        return json.dumps(vars(self), indent=4)
    
if __name__ == "__main__":
    # Define location
    location = Location(lat=37.7749, lon=-122.4194, elev=61.0, tz=-8)

    # Define PV system
    pv_system = PVSystem(kwac=1000, target_dcac_ratio=1.35, module_model="JAM66D45-640LB(3.2+2.0mm).PAN", location=location)

    # Print the PV system object
    print(pv_system)

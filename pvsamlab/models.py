import PySAM.Pvsamv1 as pv

class Location:
    def __init__(self, lat=None, lon=None, elev=None, tz=None):
        """
        Represents the location of the PV system.
        
        Args:
            lat (float, optional): Latitude in decimal degrees.
            lon (float, optional): Longitude in decimal degrees.
            elev (float, optional): Elevation above sea level (meters).
            tz (float, optional): Time zone offset from UTC (hours).
        """
        self.lat = lat if lat is not None else 0.0
        self.lon = lon if lon is not None else 0.0
        self.elev = elev if elev is not None else 0.0
        self.tz = tz if tz is not None else 0

    def __repr__(self):
        return f"Location(lat={self.lat}, lon={self.lon}, elev={self.elev}, tz={self.tz})"

class SolarResource:
    def __init__(self, albedo=0.2, irrad_mode=1, sky_model=2):
        self.albedo = albedo
        self.irrad_mode = irrad_mode
        self.sky_model = sky_model

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

class Module:
    def __init__(self, v_oc=64.4, i_sc=6.05, v_mp=54.7, i_mp=5.67, area=1.631):
        self.v_oc = v_oc
        self.i_sc = i_sc
        self.v_mp = v_mp
        self.i_mp = i_mp
        self.area = area

class Inverter:
    def __init__(self, paco=4000.0, efficiency=96.0):
        self.paco = paco
        self.efficiency = efficiency

class PVSystem:
    def __init__(self, lat=None, lon=None, elev=None, tz=None, mwac=None, dcac_ratio=None, module_model="Default Module", 
                 gcr=None, inverter_efficiency=97.5, solar_resource=None, system_design=None, losses=None, module=None, inverter=None):
        """Initialize with user-friendly inputs and compute necessary values."""
        
        # Automatically create a Location instance if lat/lon are provided
        self.location = Location(lat, lon, elev, tz)

        # Store user inputs
        self.mwac = mwac
        self.dcac_ratio = dcac_ratio
        self.module_model = module_model
        self.gcr = gcr
        self.inverter_efficiency = inverter_efficiency

        # Store optional detailed inputs
        self.solar_resource = solar_resource if solar_resource else SolarResource()
        self.system_design = system_design if system_design else SystemDesign()
        self.losses = losses if losses else Losses()
        self.module = module if module else Module()
        self.inverter = inverter if inverter else Inverter()

        self.compute_pysam_inputs()
        self.assign_inputs()
        self.run_simulation()

    def compute_pysam_inputs(self):
        """Bridges user inputs to PySAM-compatible values, handling defaults dynamically."""
        DEFAULT_MWAC = 100
        DEFAULT_DCAC_RATIO = 1.35
        DEFAULT_GCR = 0.3

        self.mwac = self.mwac if self.mwac is not None else DEFAULT_MWAC
        self.dcac_ratio = self.dcac_ratio if self.dcac_ratio is not None else DEFAULT_DCAC_RATIO
        self.gcr = self.gcr if self.gcr is not None else DEFAULT_GCR

        self.system_capacity = self.mwac * 1000 * self.dcac_ratio
        
        module_power_watts = 400
        total_modules = self.system_capacity * 1000 / module_power_watts
        modules_per_string = 20
        n_strings = total_modules // modules_per_string

        self.pysam_inputs = {
            "system_capacity": self.system_capacity,
            "subarray1_nstrings": n_strings,
            "subarray1_modules_per_string": modules_per_string,
            "subarray1_gcr": self.gcr,
            "inverter_count": round(self.mwac * 1000 / (self.dcac_ratio * 4000)),
            "inv_ds_eff": self.inverter_efficiency
        }

    def assign_inputs(self):
        """Assigns computed values to PySAM."""
        self.model = pv.default("FlatPlatePVNone")
        self.model.SystemDesign.assign(self.pysam_inputs)

    def run_simulation(self):
        """Runs PySAM with assigned inputs and processes outputs."""
        self.model.execute()
        self.process_outputs()

    def process_outputs(self):
        """Transforms PySAM outputs into user-friendly results."""
        self.outputs = {
            "max_dc_voltage": self.model.Outputs.dc_voltage_max,
            "mwh_per_year": self.model.Outputs.annual_energy / 1000,
            "energy_losses_summary": {
                "inverter_efficiency_loss": self.model.Outputs.annual_ac_inv_eff_loss_percent,
                "soiling_loss": self.model.Outputs.annual_dc_soiling_loss_percent
            },
            "hourly_energy_data": self.model.Outputs.ac_gross
        }

    def update_parameter(self, param: str, value):
        """Allows users to update key high-level parameters and re-run the simulation."""
        if hasattr(self, param):
            setattr(self, param, value)
            self.compute_pysam_inputs()
            self.assign_inputs()
            self.run_simulation()
        else:
            raise ValueError(f"Parameter '{param}' not found in PVSystem")

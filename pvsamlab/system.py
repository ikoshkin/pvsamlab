'''
This module contains classes to build PV system'
'''

import os
import csv
from dataclasses import dataclass, field, asdict, InitVar
from typing import List
from math import floor

import pandas as pd

import sys
from pathlib import Path

from .components import Module, Inverter
from .model import SSCmodel
from .climate import download_nsrdb_csv, get_ashrae_design_low


class TrackingMode:
    FT = 0
    SAT = 1
    DAT = 2


class Orientation:
    PORTRAIT = 0
    LANDSCAPE = 1


class Shading:
    NONE = 0
    STANDARD = 1
    LINEAR = 2


@dataclass
class System:
    # Target design parameters
    target_kwac: InitVar[float] = 100e3
    target_dcac: InitVar[float] = 1.30
    met_year: InitVar[str] = 'tmy'

    # Location and meteo
    lat: float = 32.71595
    lon: float = -101.91084
    weather_file: str = field(init=False)
    design_low_temp: str = field(init=False)

    # Modules
    module: Module = Module()
    module_model: int = 2  # {2:6par, 5:mlm}

    # String Sizing
    system_voltage: int = 1500
    n_series: int = field(init=False)

    # Racking/Orientation
    tracking_mode: int = TrackingMode.SAT
    module_orientation: int = Orientation.PORTRAIT
    n_modules_x: int = 1
    n_modules_y: int = field(init=False)
    tilt: float = 0
    azimuth: float = 180
    rotation_limit: float = 52
    backtracking: bool = True
    gcr: float = 0.33
    bifacial_ground_clearance: float = 1.0

    # Inverters
    inverter: Inverter = Inverter()
    inverter_derate: float = 2000/2200
    inverter_pac_derated: float = field(init=False)
    n_inverters: int = field(init=False)
    inverter_mppt_input: int = 1
    inv_tdc_ds: List[float] = field(default_factory=list)

    # Array
    n_strings: int = field(init=False)
    dc_ac_ratio: float = field(init=False)

    # System
    kwac: float = field(init=False)
    kwdc: float = field(init=False)

    # Model
    model: SSCmodel = field(init=False)

    # Builder (kind of)
    def __post_init__(self, target_kwac, target_dcac, met_year):

        # Location and meteo

        if os.path.exists(met_year):
            self.weather_file = met_year
            # print(self.weather_file)
            # location = pd.read_csv(met_year, nrows=1)
            # try:
            #     self.lat = location.Latitude.values[0]
            #     self.lon = location.Longitude.values[0]
            # except:
            with open(self.weather_file, newline='') as f:
                reader = csv.reader(f)
                tmy_header = next(reader)
                self.lat = float(tmy_header[4])
                self.lon = float(tmy_header[5])
                
        else:
            self.weather_file = download_nsrdb_csv(
                coords=(self.lat, self.lon), year=met_year)

        self.design_low_temp = get_ashrae_design_low(self.lat, self.lon)

        # String Sizing
        self.n_series = calculate_string_size(self.module,
                                              self.design_low_temp,
                                              self.system_voltage)
        # Racking/Orientation
        self.n_modules_y = self.n_series

        # Inverterts
        self.inverter_pac_derated = self.inverter.pac_max * self.inverter_derate
        self.n_inverters = floor(target_kwac*1000 / self.inverter_pac_derated)
        self.kwac = round(self.inverter_pac_derated/1000 * self.n_inverters, 3)
        self.inv_tdc_ds = [[1, 52.8, -0.021]]

        # Array
        string_kwdc = self.n_series * self.module.pmax/1000
        self.n_strings = round(self.kwac * target_dcac / string_kwdc)
        self.kwdc = round(self.n_strings * string_kwdc, 2)

        self.dc_ac_ratio = round(self.kwdc / self.kwac, 3)

        # Model
        self.model = SSCmodel(generate_ssc_input(self))
        self.model.run()


def calculate_string_size(module: Module, design_low_temp, system_voltage):

    correction = 1 + module.tc_voc / 100 * (design_low_temp - 25)
    return floor(system_voltage / (module.voc * correction))


def generate_pysam_inputs(plant: System):
    pysam_inputs = {
        'SolarResource': {
            "albedo": [0.2] * 12,
            "albedo_spatial": [],
            "irrad_mode": 1,
            "sky_model": 2,
            'solar_resource_data':{},
            'solar_resource_file': plant.weather_file,
            "use_spatial_albedos": 0,
            "use_wf_albedo": 0,
            },

        'Losses': {
            "acwiring_loss": 0.80,
            "calculate_bifacial_electrical_mismatch": 1,
            "calculate_rack_shading": 1,
            "dcoptimizer_loss": 0.0,
            "en_snow_model": 0,
            "snow_slide_coefficient": 1.97,

            "subarray1_dcwiring_loss": 1.5,
            "subarray1_diodeconn_loss": 0.5,
            "subarray1_electrical_mismatch": 0.0,
            "subarray1_mismatch_loss": 1.0,
            "subarray1_nameplate_loss": -0.4,
            "subarray1_rack_shading": 0.0,
            "subarray1_rear_soiling_loss": 0.0,
            "subarray1_soiling": [2.5] * 12,
            "subarray1_tracking_loss": 0.5,

            "subarray2_dcwiring_loss": 1.5,
            "subarray2_diodeconn_loss": 0.5,
            "subarray2_electrical_mismatch": 0.0,
            "subarray2_mismatch_loss": 1.0,
            "subarray2_nameplate_loss": -0.4,
            "subarray2_rack_shading": 0.0,
            "subarray2_rear_soiling_loss": 0.0,
            "subarray2_soiling": [2.5] * 12,
            "subarray2_tracking_loss": 0.5,

            "subarray3_dcwiring_loss": 1.5,
            "subarray3_diodeconn_loss": 0.5,
            "subarray3_electrical_mismatch": 0.0,
            "subarray3_mismatch_loss": 1.0,
            "subarray3_nameplate_loss": -0.4,
            "subarray3_rack_shading": 0.0,
            "subarray3_rear_soiling_loss": 0.0,
            "subarray3_soiling": [2.5] * 12,
            "subarray3_tracking_loss": 0.5,

            "subarray4_dcwiring_loss": 1.5,
            "subarray4_diodeconn_loss": 0.5,
            "subarray4_electrical_mismatch": 0.0,
            "subarray4_mismatch_loss": 1.0,
            "subarray4_nameplate_loss": -0.4,
            "subarray4_rack_shading": 0.0,
            "subarray4_rear_soiling_loss": 0.0,
            "subarray4_soiling": [2.5] * 12,
            "subarray4_tracking_loss": 0.5,

            "transformer_load_loss": 0.6,
            "transformer_no_load_loss": 0.1,
            "transmission_loss": 0.5
            },

        'Lifetime': {},

        'SystemDesign': {
            'enable_mismatch_vmax_calc': 1,
            'inverter_count': plant.n_inverters,

            'subarray1_azimuth': plant.azimuth,
            'subarray1_backtrack': plant.backtracking,
            'subarray1_gcr': plant.gcr,
            'subarray1_modules_per_string': plant.n_series,
            'subarray1_monthly_tilt': (),
            'subarray1_mppt_input': 1,
            'subarray1_nstrings': plant.n_strings,
            'subarray1_rotlim': plant.rotation_limit,
            'subarray1_slope_azm': 0.0,
            'subarray1_slope_tilt': 0.0,
            'subarray1_tilt': plant.tilt,
            'subarray1_tilt_eq_lat': 0,
            'subarray1_track_mode': TrackingMode.SAT,

            'subarray2_azimuth': plant.azimuth,
            'subarray2_backtrack': plant.backtracking,
            'subarray2_gcr': plant.gcr,
            'subarray2_modules_per_string': plant.n_series,
            'subarray2_monthly_tilt': (),
            'subarray2_mppt_input': 1,
            'subarray2_nstrings': plant.n_strings,
            'subarray2_rotlim': plant.rotation_limit,
            'subarray2_slope_azm': 0.0,
            'subarray2_slope_tilt': 0.0,
            'subarray2_tilt': plant.tilt,
            'subarray2_tilt_eq_lat': 0,
            'subarray2_track_mode': TrackingMode.SAT,

            'subarray3_azimuth': plant.azimuth,
            'subarray3_backtrack': plant.backtracking,
            'subarray3_gcr': plant.gcr,
            'subarray3_modules_per_string': plant.n_series,
            'subarray3_monthly_tilt': (),
            'subarray3_mppt_input': 1,
            'subarray3_nstrings': plant.n_strings,
            'subarray3_rotlim': plant.rotation_limit,
            'subarray3_slope_azm': 0.0,
            'subarray3_slope_tilt': 0.0,
            'subarray3_tilt': plant.tilt,
            'subarray3_tilt_eq_lat': 0,
            'subarray3_track_mode': TrackingMode.SAT,

            'subarray4_azimuth': plant.azimuth,
            'subarray4_backtrack': plant.backtracking,
            'subarray4_gcr': plant.gcr,
            'subarray4_modules_per_string': plant.n_series,
            'subarray4_monthly_tilt': (),
            'subarray4_mppt_input': 1,
            'subarray4_nstrings': plant.n_strings,
            'subarray4_rotlim': plant.rotation_limit,
            'subarray4_slope_azm': 0.0,
            'subarray4_slope_tilt': 0.0,
            'subarray4_tilt': plant.tilt,
            'subarray4_tilt_eq_lat': 0,
            'subarray4_track_mode': TrackingMode.SAT,

            'system_capacity': plant.kwdc
            },

        'Shading': {
            'subarray1_shade_mode': 1,
            'subarray1_shading_azal': ((0.0,),),
            'subarray1_shading_diff': 0.0,
            'subarray1_shading_en_azal': 0.0,
            'subarray1_shading_en_diff': 0.0,
            'subarray1_shading_en_mxh': 0.0,
            'subarray1_shading_en_string_option': 1,
            'subarray1_shading_en_timestep': 0.0,
            'subarray1_shading_mxh': ((0.0,),),
            'subarray1_shading_string_option': 0.0,
            'subarray1_shading_timestep': ((0.0,),),

            'subarray2_shade_mode': 0.0,
            'subarray2_shading_azal': ((0.0,),),
            'subarray2_shading_diff': 0.0,
            'subarray2_shading_en_azal': 0.0,
            'subarray2_shading_en_diff': 0.0,
            'subarray2_shading_en_mxh': 0.0,
            'subarray2_shading_en_string_option': 0.0,
            'subarray2_shading_en_timestep': 0.0,
            'subarray2_shading_mxh': ((0.0,),),
            'subarray2_shading_string_option': 0.0,
            'subarray2_shading_timestep': ((0.0,),),

            'subarray3_shade_mode': 0.0,
            'subarray3_shading_azal': ((0.0,),),
            'subarray3_shading_diff': 0.0,
            'subarray3_shading_en_azal': 0.0,
            'subarray3_shading_en_diff': 0.0,
            'subarray3_shading_en_mxh': 0.0,
            'subarray3_shading_en_string_option': 0.0,
            'subarray3_shading_en_timestep': 0.0,
            'subarray3_shading_mxh': ((0.0,),),
            'subarray3_shading_string_option': 0.0,
            'subarray3_shading_timestep': ((0.0,),),

            'subarray4_shade_mode': 0.0,
            'subarray4_shading_azal': ((0.0,),),
            'subarray4_shading_diff': 0.0,
            'subarray4_shading_en_azal': 0.0,
            'subarray4_shading_en_diff': 0.0,
            'subarray4_shading_en_mxh': 0.0,
            'subarray4_shading_en_string_option': 0.0,
            'subarray4_shading_en_timestep': 0.0,
            'subarray4_shading_mxh': ((0.0,),),
            'subarray4_shading_string_option': 0.0,
            'subarray4_shading_timestep': ((0.0,),)
            },

        'Layout': {
            'module_aspect_ratio': round(plant.module.length/plant.module.width, 2),

            'subarray1_mod_orient': plant.module_orientation,
            'subarray1_nmodx': plant.n_modules_x,
            'subarray1_nmody': plant.n_modules_y,

            'subarray2_mod_orient': plant.module_orientation,
            'subarray2_nmodx': plant.n_modules_x,
            'subarray2_nmody': plant.n_modules_y,

            'subarray3_mod_orient': plant.module_orientation,
            'subarray3_nmodx': plant.n_modules_x,
            'subarray3_nmody': plant.n_modules_y,
            
            'subarray4_mod_orient': plant.module_orientation,
            'subarray4_nmodx': plant.n_modules_x,
            'subarray4_nmody': plant.n_modules_y
            },

        'Module': {
            'module_model': plant.module_model
            },

        'CECPerformanceModelWithUserEnteredSpecifications': {
            'sixpar_aisc': plant.module.tc_isc * plant.module.isc/100,
            'sixpar_area': round(plant.module.length * plant.module.width, 2),
            'sixpar_bifacial_ground_clearance_height': plant.bifacial_ground_clearance,
            'sixpar_bifacial_transmission_factor': plant.module.bifacial_transmission_factor,
            'sixpar_bifaciality': plant.module.bifaciality,
            'sixpar_bvoc': plant.module.tc_voc * plant.module.voc/100,
            'sixpar_celltech': int(plant.module.cell),
            'sixpar_gpmp': plant.module.tc_pmax,
            'sixpar_imp': plant.module.imp,
            'sixpar_is_bifacial': int(plant.module.is_bifacial),
            'sixpar_isc': plant.module.isc,
            'sixpar_mounting': 1,
            'sixpar_nser': plant.module.n_series,
            'sixpar_standoff': 6.0,
            'sixpar_tnoct': plant.module.noct,
            'sixpar_transient_thermal_model_unit_mass': 11.0919,
            'sixpar_vmp': plant.module.vmp,
            'sixpar_voc': plant.module.voc
            },

        'SimpleEfficiencyModuleModel': {},
        'CECPerformanceModelWithModuleDatabase': {},
        'SandiaPVArrayPerformanceModelWithModuleDatabase': {},
        'IEC61853SingleDiodeModel': {},
        'MermoudLejeuneSingleDiodeModel': {},
        
        'Inverter': {
            'inv_ds_eff': plant.inverter.eff_max,
            'inv_ds_paco': plant.inverter.pac_max,

            'mppt_hi_inverter': plant.inverter.vmp_max,
            'mppt_low_inverter': plant.inverter.vmp_min,
            'inv_num_mppt': plant.inverter_mppt_input,

            'inverter_count': plant.n_inverters,
            'inverter_model': 1,
            },

        'InverterDatasheet': {
            'inv_ds_eff': plant.inverter.eff_max,
            'inv_ds_paco': plant.inverter.pac_max,
            'inv_ds_pnt': 1.0,
            'inv_ds_pso': plant.inverter.pdc_min,
            'inv_ds_vdcmax': plant.inverter.vmp_max,
            'inv_ds_vdco': plant.inverter.vmp_min,
            'inv_tdc_ds': plant.inv_tdc_ds
            },

        'InverterPartLoadCurve': {},
        'InverterMermoudLejeuneModel': {},
        'InverterCECDatabase': {},
        'InverterCECCoefficientGenerator': {},

        'BatterySystem': {'en_batt': 0.0},
        'Load': {},

        'PVLosses': {
            'enable_subhourly_clipping': 0.0,
            'enable_subinterval_distribution': 0.0
            },

        'AdjustmentFactors': {
            'adjust_constant': 0.0,
            'adjust_en_periods': 0.0,
            'adjust_en_timeindex': 0.0,
            'adjust_periods': ((0.0, 0.0, 0.0),),
            'adjust_timeindex': (0.0,)
            },

        'BatteryCell': {},
        'BatteryDispatch': {},
        'SystemCosts': {},
        'FuelCell': {},
        'PriceSignal': {},
        'Revenue': {},
        'ElectricityRates': {},

        'GridLimits': {
            'enable_interconnection_limit': 0.0,
            'grid_curtailment': (),
            'grid_interconnection_limit_kwac': 100000.0
            },

        'HybridCosts': {},
        }
    return pysam_inputs



def generate_ssc_input(plant: System):
    ssc_input = {}

    ssc_input['solar_resource_file'] = plant.weather_file

    ssc_input['transformer_no_load_loss'] = 0
    ssc_input['transformer_load_loss'] = 0

    ssc_input['system_use_lifetime_output'] = 0
    ssc_input['analysis_period'] = 1
    ssc_input['dc_degradation'] = 0.7
    ssc_input['dc_degrade_factor'] = 1.0
    ssc_input['en_dc_lifetime_losses'] = 0
    ssc_input['dc_lifetime_losses'] = 0
    ssc_input['en_ac_lifetime_losses'] = 0
    ssc_input['ac_lifetime_losses'] = 0

    ssc_input['en_snow_model'] = 0

    ssc_input['system_capacity'] = plant.kwdc

    ssc_input['use_wf_albedo'] = 0
    ssc_input['albedo'] = [0.20] * 12

    ssc_input['irrad_mode'] = 1
    ssc_input['sky_model'] = 2

    ssc_input['enable_mismatch_vmax_calc'] = 0

    ssc_input['subarray1_nstrings'] = plant.n_strings
    ssc_input['subarray1_modules_per_string'] = plant.n_series
    ssc_input['subarray1_mppt_input'] = plant.inverter_mppt_input
    ssc_input['subarray1_tilt'] = plant.tilt
    ssc_input['subarray1_tilt_eq_lat'] = 0
    ssc_input['subarray1_azimuth'] = plant.azimuth
    ssc_input['subarray1_track_mode'] = plant.tracking_mode
    ssc_input['subarray1_rotlim'] = plant.rotation_limit
    ssc_input['subarray1_shade_mode'] = Shading.STANDARD
    ssc_input['subarray1_gcr'] = plant.gcr
    ssc_input['subarray1_monthly_tilt'] = []
    ssc_input['subarray1_shading:string_option'] = -1
    ssc_input['subarray1_soiling'] = [3.00] * 12
    ssc_input['subarray1_rear_irradiance_loss'] = 0
    ssc_input['subarray1_mismatch_loss'] = 0.50
    ssc_input['subarray1_diodeconn_loss'] = 0.50
    ssc_input['subarray1_dcwiring_loss'] = 1.50
    ssc_input['subarray1_tracking_loss'] = 0.50
    ssc_input['subarray1_nameplate_loss'] = 0.50
    ssc_input['subarray1_mod_orient'] = plant.module_orientation
    ssc_input['subarray1_nmodx'] = plant.n_modules_x
    ssc_input['subarray1_nmody'] = plant.n_modules_y
    ssc_input['subarray1_backtrack'] = plant.backtracking

    ssc_input['subarray2_enable'] = 0
    ssc_input['subarray2_tilt'] = 0
    ssc_input['subarray2_track_mode'] = 0
    ssc_input['subarray2_shade_mode'] = 0

    ssc_input['subarray3_enable'] = 0
    ssc_input['subarray3_tilt'] = 0
    ssc_input['subarray3_track_mode'] = 0
    ssc_input['subarray3_shade_mode'] = 0

    ssc_input['subarray4_enable'] = 0
    ssc_input['subarray4_tilt'] = 0
    ssc_input['subarray4_track_mode'] = 0
    ssc_input['subarray4_shade_mode'] = 0

    ssc_input['dcoptimizer_loss'] = 0
    ssc_input['acwiring_loss'] = 1.0
    ssc_input['transmission_loss'] = 0

    ssc_input['module_model'] = plant.module_model
    ssc_input['module_aspect_ratio'] = round(
        plant.module.length/plant.module.width, 2)

    ssc_input['6par_celltech'] = int(plant.module.cell)
    ssc_input['6par_vmp'] = plant.module.vmp
    ssc_input['6par_imp'] = plant.module.imp
    ssc_input['6par_voc'] = plant.module.voc
    ssc_input['6par_isc'] = plant.module.isc
    ssc_input['6par_bvoc'] = plant.module.tc_voc * plant.module.voc/100
    ssc_input['6par_aisc'] = plant.module.tc_isc * plant.module.isc/100
    ssc_input['6par_gpmp'] = plant.module.tc_pmax
    ssc_input['6par_nser'] = plant.module.n_series
    ssc_input['6par_area'] = round(plant.module.length * plant.module.width, 2)
    ssc_input['6par_tnoct'] = plant.module.noct
    ssc_input['6par_standoff'] = 6
    ssc_input['6par_mounting'] = 1
    ssc_input['6par_is_bifacial'] = int(plant.module.is_bifacial)
    ssc_input['6par_bifacial_transmission_factor'] = plant.module.bifacial_transmission_factor
    ssc_input['6par_bifaciality'] = plant.module.bifaciality
    ssc_input['6par_bifacial_ground_clearance_height'] = plant.bifacial_ground_clearance

    ssc_input['mlm_N_series'] = plant.module.model_params.n_series
    ssc_input['mlm_N_parallel'] = plant.module.model_params.n_parallel
    ssc_input['mlm_N_diodes'] = plant.module.model_params.n_diodes
    ssc_input['mlm_Width'] = plant.module.model_params.width
    ssc_input['mlm_Length'] = plant.module.model_params.length
    ssc_input['mlm_V_mp_ref'] = plant.module.model_params.vmp_ref
    ssc_input['mlm_I_mp_ref'] = plant.module.model_params.imp_ref
    ssc_input['mlm_V_oc_ref'] = plant.module.model_params.voc_ref
    ssc_input['mlm_I_sc_ref'] = plant.module.model_params.isc_ref
    ssc_input['mlm_S_ref'] = plant.module.model_params.irr_ref
    ssc_input['mlm_T_ref'] = plant.module.model_params.t_ref
    ssc_input['mlm_R_shref'] = plant.module.model_params.R_shref
    ssc_input['mlm_R_sh0'] = plant.module.model_params.R_sh0
    ssc_input['mlm_R_shexp'] = plant.module.model_params.R_shexp
    ssc_input['mlm_R_s'] = plant.module.model_params.R_s
    ssc_input['mlm_alpha_isc'] = plant.module.model_params.alpha_isc
    ssc_input['mlm_beta_voc_spec'] = plant.module.model_params.beta_voc_spec
    ssc_input['mlm_E_g'] = plant.module.model_params.E_g
    ssc_input['mlm_n_0'] = plant.module.model_params.n_0
    ssc_input['mlm_mu_n'] = plant.module.model_params.mu_n
    ssc_input['mlm_D2MuTau'] = plant.module.model_params.D2MuTau
    ssc_input['mlm_T_mode'] = plant.module.model_params.T_mode
    ssc_input['mlm_T_c_no_tnoct'] = plant.module.model_params.T_c_no_tnoct
    ssc_input['mlm_T_c_no_mounting'] = plant.module.model_params.T_c_no_mounting
    ssc_input['mlm_T_c_no_standoff'] = plant.module.model_params.T_c_no_standoff
    ssc_input['mlm_T_c_fa_alpha'] = plant.module.model_params.T_c_fa_alpha
    ssc_input['mlm_T_c_fa_U0'] = plant.module.model_params.T_c_fa_U0
    ssc_input['mlm_T_c_fa_U1'] = plant.module.model_params.T_c_fa_U1
    ssc_input['mlm_AM_mode'] = plant.module.model_params.AM_mode
    ssc_input['mlm_AM_c_sa0'] = 0
    ssc_input['mlm_AM_c_sa1'] = 0
    ssc_input['mlm_AM_c_sa2'] = 0
    ssc_input['mlm_AM_c_sa3'] = 0
    ssc_input['mlm_AM_c_sa4'] = 0
    ssc_input['mlm_AM_c_lp0'] = 0
    ssc_input['mlm_AM_c_lp1'] = 0
    ssc_input['mlm_AM_c_lp2'] = 0
    ssc_input['mlm_AM_c_lp3'] = 0
    ssc_input['mlm_AM_c_lp4'] = 0
    ssc_input['mlm_AM_c_lp5'] = 0
    ssc_input['mlm_IAM_mode'] = plant.module.model_params.IAM_mode
    ssc_input['mlm_IAM_c_as'] = plant.module.model_params.IAM_c_as
    ssc_input['mlm_IAM_c_sa0'] = 0
    ssc_input['mlm_IAM_c_sa1'] = 0
    ssc_input['mlm_IAM_c_sa2'] = 0
    ssc_input['mlm_IAM_c_sa3'] = 0
    ssc_input['mlm_IAM_c_sa4'] = 0
    ssc_input['mlm_IAM_c_sa5'] = 0
    ssc_input['mlm_groundRelfectionFraction'] = plant.module.model_params.ground_relfection_fraction
    ssc_input['mlm_IAM_c_cs_incAngle'] = plant.module.model_params.IAM_c_cs_incAngle
    ssc_input['mlm_IAM_c_cs_iamValue'] = plant.module.model_params.IAM_c_cs_iamValue

    ssc_input['inverter_model'] = 1
    ssc_input['inverter_count'] = plant.n_inverters

    ssc_input['mppt_low_inverter'] = plant.inverter.vmp_min
    ssc_input['mppt_hi_inverter'] = plant.inverter.vmp_max
    ssc_input['inv_num_mppt'] = plant.inverter_mppt_input
    ssc_input['inv_ds_paco'] = plant.inverter.pac_max
    ssc_input['inv_ds_eff'] = plant.inverter.eff_max
    ssc_input['inv_ds_pnt'] = plant.inverter.night_loss
    ssc_input['inv_ds_pso'] = plant.inverter.pdc_min
    ssc_input['inv_ds_vdco'] = plant.inverter.vmp_min
    ssc_input['inv_ds_vdcmax'] = plant.inverter.vmp_max
    ssc_input['inv_tdc_ds'] = plant.inv_tdc_ds

    ssc_input['en_batt'] = 0

    ssc_input['adjust:constant'] = 1.0

    return ssc_input

if __name__ == '__main__':
    plant = System()
    print(plant.model.ac_monthly)
    print(plant.model.ac_annual)
    print(plant.model.dc_monthly)
    print(plant.model.dc_annual)
    print(plant.model.loss_monthly)
    print(plant.model.loss_annual)§
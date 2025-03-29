import ast
import re
from dataclasses import dataclass, field, asdict, InitVar, fields
from typing import List, Tuple
from pathlib import Path


class CellType:
    MONO = 0
    POLY = 1
    CDTE = 2
    CIS = 3
    CIGS = 4
    AMORPH = 5


class Units:
    CURRENT = "A"
    VOLTAGE = "V"
    POWER = "W"
    POWER_K = "kW"
    ENERGY = "kWh"
    PERCENT = "%"
    LENGTH = "m"
    WEIGHT = "kg"
    TEMPERATURE = "degC"


def map_cell_type():
    pass


"""
number = "[-+]?\d*\.\d+|[-+]\d+"
latitude = float(re.search(f"Latitude:*?\s?({number, pfid=""))", content).group(1))
longitude = float(
    re.search(f"Longitude:*?\s?({number, pfid=""))", content).group(1))
elevation = int(
    float(re.search(f"Elevation:*?\s?({number, pfid=""))", content).group(1)))
tz = int(re.search(f"UTC*?\s?({number, pfid=""))", content).group(1))
"""

"""
name: float = field(
        init=False,
        metadata=dict(
            units="",
            pan_file="",
            ssc="",
            ))
# ===
    init=False,
    repr=True,
    metadata=asdict(Metadata()))

    units=unitss.CURRENT,
    required=True,
    pan_file="Manufacturer"))



        init=False,
        metadata=dict(
            units="",
            pan_file="",
            ssc="",
            ))
# ===
"""


def tryeval(s):
    """Safely evaluate a string that consistof Python literal structure:
        strings, bytes, numbers, tuples, dicts, sets, booleans and None
    """
    try:
        return ast.literal_eval(s)
    except ValueError:
        pass
    except SyntaxError:
        pass

    return s


def extract_value(key, content):
    return tryeval(re.search(f"{key}=(.+)", content).group(1))


@dataclass(frozen=True)
class Module:

    source: InitVar[str]

    """
    General
    """

    manufacturer: str = field(
        init=False,
        metadata=dict(
            pan_file="Manufacturer",
        ))

    model: str = field(
        init=False,
        metadata=dict(
            pan_file="Model",
        ))

    """
    Electrical
    """

    pmpp: float = field(
        init=False,
        metadata=dict(
            units="W",
            pan_file="PNom",
        ))

    vmp: float = field(
        init=False,
        metadata=dict(
            units="V",
            pan_file="Vmp",
            ssc="mlm_V_mp_ref",
        ))

    imp: float = field(
        init=False,
        metadata=dict(
            units="A",
            pan_file="Imp",
            ssc="mlm_I_mp_ref",
        ))

    voc: float = field(
        init=False,
        metadata=dict(
            units="V",
            pan_file="Voc",
            ssc="mlm_V_oc_ref",
        ))

    isc: float = field(
        init=False,
        metadata=dict(
            units="A",
            pan_file="Isc",
            ssc="mlm_I_sc_ref",
        ))

    efficiency: float = field(
        init=False,
        metadata=dict(
            units="%",
        ))

    voc_max: float = field(
        init=False,
        metadata=dict(
            units="V",
            pan_file="VMaxIEC",
            details="System Voltage Class",
        ))

    isc_max: float = field(
        init=False,
        metadata=dict(
            units="A",
            details="Fuse Rating",
        ))

    pmpp_tolerance_lower: float = field(
        init=False,
        metadata=dict(
            units="%",
            pan_file="PNomTolLow",
        ))

    pmpp_tolerance_upper: float = field(
        init=False,
        metadata=dict(
            units="%",
            pan_file="PNomTolUp",
        ))

    """
    Mechanical
    """

    cell_type: str = field(
        init=False,
        metadata=dict(
            pan_file="Technol",
            ssc="6par_celltech",
            apply=map_cell_type,
        ))

    num_of_cells_series: int = field(
        init=False,
        metadata=dict(
            pan_file="NCelS",
            ssc="mlm_N_series",
        ))

    num_of_cells_parallel: int = field(
        init=False,
        metadata=dict(
            pan_file="NCelP",
            ssc="mlm_N_parallel",
        ))

    num_of_diodes: int = field(
        init=False,
        metadata=dict(
            pan_file="NDiode",
            ssc="mlm_N_diodes",
        ))

    length: float = field(
        init=False,
        metadata=dict(
            units="m",
            pan_file="Height",
            ssc="mlm_Length",
        ))

    width: float = field(
        init=False,
        metadata=dict(
            units="m",
            pan_file="Width",
            ssc="mlm_Width",
        ))

    thickness: float = field(
        init=False,
        metadata=dict(
            units="m",
            pan_file="Depth",
        ))

    weight: float = field(
        init=False,
        metadata=dict(
            units="kg",
            pan_file="Weight",
        ))

    """
    Thermal
    """

    temp_coeff_pmpp: float = field(
        init=False,
        metadata=dict(
            units="%/K",
            pan_file="muPmpReq",
        ))

    temp_coeff_voc: float = field(
        init=False,
        metadata=dict(
            units="mV/K or %/K",
            pan_file="muVocSpec",
            ssc="mlm_beta_voc_spec",
        ))

    temp_coeff_isc: float = field(
        init=False,
        metadata=dict(
            units="mA/K or %/K",
            pan_file="muISC",
            ssc="mlm_alpha_isc",
        ))

    noct: float = field(
        init=False,
        metadata=dict(
            units="degC",
            details="Nominal Module Operating Temperature",
        ))

    absorbtivity: float = field(
        init=False,
        metadata=dict(
            ssc="mlm_T_c_fa_alpha",
            details="Faiman's Absorbtivity Factor"
        ))

    u_c: float = field(
        init=False,
        metadata=dict(
            pan_file="",
            ssc="U_0",
            details="Constant Heat Transfer Component (Uc)",
        ))

    u_v: float = field(
        init=False,
        metadata=dict(
            ssc="U_1",
            details="Convective Heat Transfer Component (Uv)"
        ))

    """
    Single-Diode Model
    """

    r_shunt_ref: float = field(
        init=False,
        metadata=dict(
            units="Ohm",
            pan_file="RShunt",
            ssc="mlm_R_shref",
            details="Shunt Resistance at Reference Irradiation",
        ))

    r_shunt_0: float = field(
        init=False,
        metadata=dict(
            units="Ohm",
            pan_file="Rp_0",
            ssc="mlm_R_sh0",
            details="Shunt Resistance at 0 W/m2",
        ))

    r_shunt_exp: float = field(
        init=False,
        metadata=dict(
            units="Ohm",
            pan_file="Rp_Exp",
            ssc="mlm_R_shexp",
            details="Shunt Resistance Exponential Factor",
        ))

    r_series: float = field(
        init=False,
        metadata=dict(
            units="Ohm",             pan_file="RSerie",
            ssc="mlm_R_s",
            details="Series Resistance",
        ))

    energy_bandgap: float = field(
        init=False,
        metadata=dict(
            # apply=calc_bandgap,
            details="Reference Bandgap Energy"
        ))

    diode_ideality: float = field(
        init=False,
        metadata=dict(
            pan_file="Gamma",
            ssc="n_0",
            details="Diode Ideality Factor"
        ))

    temp_coeff_diode_ideality: float = field(
        init=False,
        metadata=dict(
            pan_file="muGamma",
            ssc="mu_n",
            details="Temperature Coefficient for Diode Ideality Factor"
        ))

    recombination_loss: float = field(
        init=False,
        metadata=dict(
            ssc="D2MuTau",
        ))

    # Air Mass Modifier

    # Incidence Angle Modifier

    # Bifaciality
    bifaciality_factor: float = field(
        init=False,
        metadata=dict(
            units="Ohm",
            pan_file="BifacialityFactor",
            ssc="mlm_R_shexp",
            details="Shunt Resistance Exponential Factor",
        ))

    bifacial_transmission_factor: float = field(
        init=False,
        default=0.013,
        metadata=dict(
            units="Ohm",
            pan_file="Rp_Exp",
            ssc="mlm_R_shexp",
            details="Shunt Resistance Exponential Factor",
        ))

    def __post_init__(self):

        try:
            with open(self.pan_file, "r", errors="ignore") as f:
                _content = f.read()
        except EnvironmentError:
            print("Error occured while reading .PAN file")

        print(fields(self))


if __name__ == "__main__":

    tsv_file = "/Users/ikoshkin/Repos/SolarSimulationSuite/resolar/pvmodule.tsv"
    p_file = "/Users/ikoshkin/Repos/SolarSimulationSuite/resolar/data/pan_files/180629.Longi_LR6_72HBD_405M_frame_IEC_Draft.PAN"

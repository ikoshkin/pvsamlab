import os
import json
from dataclasses import dataclass, field
from typing import List

import pandas as pd

_dir = os.path.dirname(__file__)


class CellTech:
    MONO = 0
    POLY = 1
    CDTE = 2
    CIS = 3
    CIGS = 4
    AMORPH = 5


@dataclass
class MlModelParameters:
    """
    Parameters for Mermoud/Lejeune single-diode model
    """
    n_series: int = 72  # number of cells in series: NCelS
    n_parallel: int = 2  # number of cells in parallel: NCelP
    n_diodes: int = 3  # number of bypass diodes: NDiode

    width: float = 0.992
    length: float = 1.956

    vmp_ref: float = 39.6  # Vmp
    imp_ref: float = 9.6  # Imp
    voc_ref: float = 48  # Voc
    isc_ref: float = 9.99  # Isc

    irr_ref: float = 1000  # GRef
    t_ref: float = 25  # TRef

    R_shref: float = 1450  # Shunt Resistance at GRef: RShunt
    R_sh0: float = 8000  # Shunt Resistance at 0[W/m2]: Rp_0
    R_shexp: float = 10  # Exponential factor of Shunt Resistance: Rp_Exp
    R_s: float = 0.218  # Series Resistance: RSerie
    alpha_isc: float = 0.00492  # Temp.coeff for Isc: muISC/1000 [A/K]
    # Temp.coeff for Voc: muVocSpec/1000 [V/K]
    beta_voc_spec: float = -0.116
    E_g: float = 1.12  # Gap Energy: 1.12/1.03/1.7/1.5 for Si/CIS/aSi/CdTe
    n_0: float = 1.030  # Diode Quality Factor: Gamma
    mu_n: float = -0.0006  # Temperature coeff for Gamma
    D2MuTau: float = 0  # Coefficient for recombination losses

    T_mode: int = 2  # 1: NOCT, 2: Faiman Model(PVSyst)
    T_c_no_tnoct: float = 0
    T_c_no_mounting: int = 0
    T_c_no_standoff: int = 0
    T_c_fa_alpha: float = 0.9  # Faiman's absorbtivity
    T_c_fa_U0: float = 29  # Uc
    T_c_fa_U1: float = 0  # Uv

    AM_mode: int = 1  # Air-Mass Modifier Mode (Specrtral Correction)
    AM_c_sa: List[float] = field(default_factory=list)
    AM_c_lp: List[float] = field(default_factory=list)

    IAM_mode: int = 1  # 1/2/3:ASHRAE/Sandia/User
    IAM_c_as: float = 0.04  # ASHRAE b_0
    IAM_c_sa: List[float] = field(default_factory=list)
    IAM_c_cs_incAngle:  List[float] = field(default_factory=list)
    IAM_c_cs_iamValue:  List[float] = field(default_factory=list)

    ground_relfection_fraction: float = 0.2

    def __post_init__(self):
        AM_c_sa = [0] * 5
        AM_c_lp = [0] * 6
        c = [0.0, 30.0, 40.0, 50.0, 60.0, 70.0, 75.0, 80.0, 85.0, 90.0, 100.0]
        IAM_c_cs_iamValue = [1.0, 0.999, 0.995, 0.987,
                             0.962, 0.892, 0.816, 0.681, 0.44, 0.0, 0.0]


@dataclass
class Module:

    manufacturer: str = 'Jinko Solar'
    model: str = 'Eagle Bifacial 72M G2 JKM400M-72HL-TV'
    cell: int = CellTech.MONO
    pmax: float = 400.15999999999997
    vmp: float = 41
    imp: float = 9.76
    voc: float = 48.8
    isc: float = 10.24
    tc_pmax: float = -0.36
    tc_voc: float = -0.3
    tc_isc: float = 0.05
    n_series: int = 144
    length: float = 2.031
    width: float = 1.008
    noct: float = 45
    is_bifacial: bool = True
    bifacial_transmission_factor: float = 0.05
    bifaciality: float = 0.70

    model_params: MlModelParameters = field(init=False)

    def __post_init__(self):
        self.model_params = MlModelParameters()

    @classmethod
    def from_db(cls, lookup_model):
        df = pd.read_csv(os.path.join(_dir, 'data', '_modulesDB.csv'))
        results = df[df.model == lookup_model]
        if results.empty:
            raise ValueError(f'No modules found for "{lookup_model}".')
        else:
            record_as_dict = json.loads(results.head(1).squeeze().to_json())
            return cls(**record_as_dict)


@dataclass
class Inverter:

    manufacturer: str = 'Power Electronics'
    model: str = 'V1500 HEC FS2100CH15'
    pac_max: float = 2510000
    eff_max: float = 98.5
    night_loss: float = 300
    pdc_min: float = 2000
    vmp_min: float = 800
    vmp_max: float = 1250
    abs_max: float = 1500

    @classmethod
    def from_db(cls, lookup_model):
        df = pd.read_csv(os.path.join(_dir, 'data', '_invertersDB.csv'))
        results = df[df.model == lookup_model]
        if results.empty:
            raise ValueError(f'No inverters found for "{lookup_model}".')
        else:
            record_as_dict = json.loads(results.head(1).squeeze().to_json())
            return cls(**record_as_dict)

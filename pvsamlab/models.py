import PySAM.Pvsamv1 as pv
from dataclasses import dataclass, field, asdict
from typing import Tuple

@dataclass
class SolarResource:
    albedo: Tuple[float, ...] = field(default_factory=lambda: (0.2,) * 12)
    albedo_spatial: Tuple[Tuple[float, ...], ...] = field(default_factory=lambda: ((0.2,) * 10,) * 12)
    irrad_mode: float = 0.0
    sky_model: float = 2.0
    use_spatial_albedos: float = 0.0
    use_wf_albedo: float = 1.0

@dataclass
class SystemDesign:
    system_capacity: float = 100000.0  # kW
    inverter_count: int = 30
    subarray1_azimuth: float = 180.0
    subarray1_tilt: float = 20.0
    subarray1_nstrings: int = 15000
    subarray1_modules_per_string: int = 20
    subarray1_track_mode: int = 0  # 0 = Fixed, 1 = 1-Axis, 2 = 2-Axis
    subarray1_backtrack: bool = False
    subarray1_gcr: float = 0.3

@dataclass
class Losses:
    acwiring_loss: float = 1.0
    subarray1_dcwiring_loss: float = 2.0
    subarray1_soiling: Tuple[float, ...] = field(default_factory=lambda: (5.0,) * 12)
    transformer_load_loss: float = 0.0

@dataclass
class Module:
    model_type: int = 2  # 6-parameter module model
    v_oc: float = 64.4
    i_sc: float = 6.05
    v_mp: float = 54.7
    i_mp: float = 5.67
    area: float = 1.631
    n_series: int = 96
    t_noct: float = 46.0
    standoff: float = 6.0
    mounting: int = 0  # 0 = Open rack, 1 = Roof mount
    is_bifacial: bool = False
    bifacial_transmission_factor: float = 0.013
    bifaciality: float = 0.7

@dataclass
class Inverter:
    model_type: int = 1  # Datasheet model
    paco: float = 4000.0  # Max AC power (W)
    efficiency: float = 96.0
    pnt: float = 1.0  # Night-time power loss (W)
    pso: float = 0.0  # Standby power loss (W)
    vdcmax: float = 600.0  # Max DC voltage (V)
    vdco: float = 310.0  # DC operating voltage (V)

@dataclass
class PVSystem:
    solar_resource: SolarResource = field(default_factory=SolarResource)
    system_design: SystemDesign = field(default_factory=SystemDesign)
    losses: Losses = field(default_factory=Losses)
    module: Module = field(default_factory=Module)
    inverter: Inverter = field(default_factory=Inverter)

    model: pv.Pvsamv1 = field(init=False)
    outputs: dict = field(init=False)

    def __post_init__(self):
        """Initializes and runs the PySAM model."""
        self.model = pv.default("FlatPlatePVNone")
        self.assign_inputs()
        self.run_simulation()

    def assign_inputs(self):
        """Assigns input values to PySAM groups using dataclass dictionaries."""
        self.model.SolarResource.assign(asdict(self.solar_resource))
        self.model.SystemDesign.assign(asdict(self.system_design))
        self.model.Losses.assign(asdict(self.losses))
        self.model.Module.assign(asdict(self.module))
        self.model.Inverter.assign(asdict(self.inverter))

    def run_simulation(self):
        """Executes the PySAM simulation and retrieves outputs."""
        self.model.execute()
        self.outputs = self.model.Outputs.export()

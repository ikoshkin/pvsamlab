"""
pv_bess_sizing_worker.py

Worker function for the PV+BESS parametric sizing study.

Placed in a separate module so that ProcessPoolExecutor can pickle it
correctly under both 'fork' and 'spawn' multiprocessing start methods.
All arguments must be plain Python objects (no PySAM handles).
"""


def run_bess_case(task: dict) -> dict:
    """Run one PV+BESS case inside a worker process.

    Parameters
    ----------
    task : dict
        Keys:
            power_kw, duration_hr, lat, lon, met_year,
            pv_target_kwac, pv_dcac,
            financial_kwargs, batt_kwargs, revenue_kwargs

    Returns
    -------
    dict
        Result with 'status': 'ok' or 'status': 'error'.
    """
    try:
        from pvsamlab import (
            PvBessSystem, Battery, BessDispatch,
            Financial, RevenueStack,
            compute_lcoe, compute_lcos,
            TrackingMode,
        )

        power_kw    = task["power_kw"]
        duration_hr = task["duration_hr"]
        energy_kwh  = power_kw * duration_hr

        batt = Battery(
            energy_kwh=float(energy_kwh),
            power_kw=float(power_kw),
            **task["batt_kwargs"],
        )

        # self_consumption dispatch with a flat 200 MW load.
        #
        # price_signal (choice=4) was tested and produces zero discharge in
        # PvBessSystem: without a proper ElectricityRates TOU rate matrix AND
        # an assigned load profile, the battery charges from clip/excess PV but
        # has no price signal to dispatch against. self_consumption (choice=3)
        # with a load profile is the correct approach for this PV+BESS study.
        disp = BessDispatch(
            strategy="self_consumption",
            can_gridcharge=[0, 0, 0, 0, 0, 0],  # PV-only charging
        )

        fin = Financial(**task["financial_kwargs"])
        rev = RevenueStack(**task["revenue_kwargs"])

        plant = PvBessSystem(
            lat=task["lat"],
            lon=task["lon"],
            target_kwac=task["pv_target_kwac"],
            target_dcac=task["pv_dcac"],
            met_year=task["met_year"],
            tracking_mode=TrackingMode.SAT,
            battery=batt,
            dispatch=disp,
            load_profile=task["load_profile"],
        )

        sim   = plant.run()
        fin_r = compute_lcoe(plant, fin, revenue_stack=rev)

        # LCOS: project analysis_period years using calendar degradation
        y1_kwh   = sim["batt_annual_discharge_energy_kwh"]
        deg      = batt.calendar_degradation / 100.0           # fraction / yr
        n        = fin.analysis_period
        ann_dis  = [y1_kwh * (1.0 - deg) ** yr for yr in range(n)]
        ann_opex = batt.energy_kwh * batt.opex_per_kwh_year    # $/yr
        dr       = fin.discount_rate / 100.0                   # decimal
        lcos     = compute_lcos(batt, ann_dis, ann_opex, [], dr)

        return {
            "power_mw":                          power_kw / 1000,
            "duration_hr":                       duration_hr,
            "energy_mwh":                        energy_kwh / 1000,
            "annual_energy_kwh":                 sim.get("annual_energy", 0.0),
            "batt_annual_discharge_energy_kwh":  sim["batt_annual_discharge_energy_kwh"],
            "batt_roundtrip_efficiency_pct":     sim["batt_roundtrip_efficiency_pct"],
            "batt_capacity_end_of_life_pct":     sim["batt_capacity_end_of_life_pct"],
            "lcoe_real_cents_per_kwh":           fin_r["lcoe_real_cents_per_kwh"],
            "lcoe_nom_cents_per_kwh":            fin_r["lcoe_nom_cents_per_kwh"],
            "npv_usd":                           fin_r["npv_usd"],
            "irr_pct":                           fin_r["irr_pct"],
            "lcos_usd_per_kwh":                  round(lcos, 4),
            "total_installed_cost_usd":          fin_r["total_installed_cost_usd"],
            "status": "ok",
        }

    except Exception as exc:
        return {
            "power_mw":    task["power_kw"] / 1000,
            "duration_hr": task["duration_hr"],
            "energy_mwh":  task["power_kw"] * task["duration_hr"] / 1000,
            "status": "error",
            "error": str(exc),
        }

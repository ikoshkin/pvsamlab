"""
bess_surplus_worker.py

Multiprocessing worker for brownfield/surplus interconnection BESS optimization.
Each task runs one (power_mw × duration_hr × charging_mode) case.

Price-signal dispatch uses PySAM PriceSignal.dispatch_factors_ts.
POI limit enforced via GridLimits.grid_interconnection_limit_kwac.
"""
from __future__ import annotations


def run_surplus_case(task: dict) -> dict:
    """Run one PV+BESS simulation and compute revenue metrics.

    Parameters
    ----------
    task : dict
        power_kw, duration_hr, mode, hourly_prices, poi_limit_kw,
        load_kw, can_gridcharge, can_solar_charge, lat, lon,
        pv_target_kwac, pv_dcac, met_year, batt_kwargs,
        capacity_payment_kw_yr, ancillary_kw_yr, financial_kwargs

    Returns
    -------
    dict  with status='ok' or status='error'
    """
    try:
        import numpy as np
        from pvsamlab import (
            Battery, BessDispatch, PvBessSystem, TrackingMode,
            compute_lcos,
        )
        from pvsamlab.system import generate_pysam_inputs, process_outputs
        from pvsamlab.battery import process_bess_outputs
        from pvsamlab.financial import Financial, compute_irr, compute_npv

        power_kw    = task["power_kw"]
        duration_hr = task["duration_hr"]
        mode        = task["mode"]
        prices      = task["hourly_prices"]          # $/MWh, 8760 elements
        poi_kw      = task["poi_limit_kw"]
        load_kw     = task["load_kw"]
        can_gc      = bool(task["can_gridcharge"])
        can_sc      = bool(task["can_solar_charge"])

        energy_kwh  = power_kw * duration_hr

        # ── Battery hardware ──────────────────────────────────────────────────
        batt = Battery(
            energy_kwh=float(energy_kwh),
            power_kw=float(power_kw),
            **task["batt_kwargs"],
        )

        # ── Dispatch strategy: price_signal for all modes ─────────────────────
        # Pass hourly_prices so _generate_battery_inputs() builds the PriceSignal
        # group (ppa_multiplier_model=1 + dispatch_factors_ts) automatically.
        disp = BessDispatch(
            strategy="price_signal",
            energy_arbitrage_prices=prices,
            can_gridcharge=[int(can_gc)] * 6,
        )

        # ── PvBessSystem ──────────────────────────────────────────────────────
        plant = PvBessSystem(
            lat=task["lat"],
            lon=task["lon"],
            target_kwac=task["pv_target_kwac"],
            target_dcac=task["pv_dcac"],
            met_year=task["met_year"],
            tracking_mode=TrackingMode.SAT,
            battery=batt,
            dispatch=disp,
            load_profile=[float(load_kw)] * 8760,
        )

        # ── Assign inputs manually so we can inject PriceSignal + GridLimits ──
        pv_inputs   = generate_pysam_inputs(plant)
        batt_inputs = plant._generate_battery_inputs()

        # Override auto-dispatch flags for grid_only mode
        if not can_sc:
            batt_inputs["BatteryDispatch"]["batt_dispatch_auto_can_charge"]     = 0
            batt_inputs["BatteryDispatch"]["batt_dispatch_auto_can_clipcharge"] = 0
            batt_inputs["BatteryDispatch"]["batt_dispatch_auto_can_curtailcharge"] = 0

        plant.model.assign(pv_inputs)
        plant.model.assign(batt_inputs)  # includes PriceSignal from _generate_battery_inputs()

        plant.model.assign({
            "Load": {
                "load": [float(load_kw)] * 8760,
            },
            "GridLimits": {
                "enable_interconnection_limit":  1,
                "grid_interconnection_limit_kwac": float(poi_kw),
            },
        })

        plant.model.execute()
        out = plant.model.Outputs

        # ── Hourly arrays ─────────────────────────────────────────────────────
        gen_kw   = np.array(list(out.gen))
        batt_kw  = np.array(list(out.batt_power))
        soc_pct  = np.array(list(out.batt_SOC))
        grid_kw  = np.array(list(out.grid_power))

        gen_mw    = gen_kw   / 1000.0
        batt_mw   = batt_kw  / 1000.0
        export_mw = grid_kw  / 1000.0   # net grid exchange (positive = export)

        prices_arr = np.array(prices)

        # ── Revenue computation (incremental BESS contribution only) ──────────
        # Subtract PV-only baseline so we credit only the BESS's incremental
        # export, not the $11-13M/yr of existing PV revenue.
        pv_base        = np.array(task["pv_baseline_export_mw"])
        incremental_mw = export_mw - pv_base
        energy_rev_usd = float(np.sum(incremental_mw * prices_arr))

        # Charging cost: grid-charged energy × price (only when can_gridcharge)
        if can_gc:
            charge_mw  = np.maximum(-batt_mw, 0)   # positive = charging
            charge_cost_usd = float(
                np.sum(charge_mw * np.maximum(prices_arr, 0))
            )
        else:
            charge_cost_usd = 0.0

        # Capacity + ancillary (annual, flat)
        cap_rev_usd = task["capacity_payment_kw_yr"] * power_kw
        anc_rev_usd = task["ancillary_kw_yr"]        * power_kw

        net_annual_rev = (
            energy_rev_usd - charge_cost_usd + cap_rev_usd + anc_rev_usd
        )

        # ── BESS simulation results ───────────────────────────────────────────
        bess_r = process_bess_outputs(plant.model)
        pv_r   = process_outputs(plant)

        # ── Financial metrics (simple DCF, no SAM Singleowner for speed) ──────
        fin      = Financial(**task["financial_kwargs"])
        r_dec    = fin.discount_rate / 100.0
        n_yr     = fin.analysis_period
        opex_yr  = batt.energy_kwh * batt.opex_per_kwh_year
        epc_mult = task.get("bess_epc_multiplier", 1.35)
        capex    = (batt.energy_kwh * batt.capex_per_kwh + batt.power_kw * batt.capex_per_kw) * epc_mult
        annual_cf = net_annual_rev - opex_yr

        npv_usd = -capex + sum(
            annual_cf / (1.0 + r_dec) ** t for t in range(1, n_yr + 1)
        )
        cashflows = [-capex] + [annual_cf] * n_yr
        irr_raw  = compute_irr(cashflows)
        irr_pct  = float(irr_raw) * 100 if not (
            irr_raw != irr_raw or irr_raw == float("inf")
        ) else float("nan")

        # ── LCOS ─────────────────────────────────────────────────────────────
        y1_dis_kwh = bess_r["batt_annual_discharge_energy_kwh"]
        deg        = batt.calendar_degradation / 100.0
        ann_dis    = [y1_dis_kwh * (1.0 - deg) ** yr for yr in range(n_yr)]
        lcos       = compute_lcos(batt, ann_dis, opex_yr, [], r_dec)

        # ── POI utilization (energy-weighted) ─────────────────────────────────
        poi_util = (
            float(np.sum(np.maximum(export_mw, 0))) /
            (poi_kw / 1000.0 * 8760.0) * 100.0
        )

        return {
            "power_mw":   power_kw / 1000.0,
            "duration_hr": duration_hr,
            "charging_mode": mode,
            "annual_energy_kwh": float(np.sum(export_mw)) * 1000,  # MWh×1h×1000 = kWh
            "batt_annual_discharge_energy_kwh": bess_r["batt_annual_discharge_energy_kwh"],
            "batt_annual_charge_energy_kwh":    bess_r["batt_annual_charge_energy_kwh"],
            "batt_roundtrip_efficiency_pct":    bess_r["batt_roundtrip_efficiency_pct"],
            "energy_revenue_usd":   energy_rev_usd,
            "charging_cost_usd":    charge_cost_usd,
            "capacity_revenue_usd": cap_rev_usd,
            "ancillary_revenue_usd": anc_rev_usd,
            "net_annual_revenue_usd": net_annual_rev,
            "npv_usd":    npv_usd,
            "irr_pct":    irr_pct,
            "lcos_usd_per_kwh": lcos,
            "poi_utilization_pct": poi_util,
            "total_capex_usd": capex,
            # Hourly arrays stored separately in hourly_data dict
            "_gen_mw":    gen_mw.tolist(),
            "_batt_mw":   batt_mw.tolist(),
            "_soc_pct":   soc_pct.tolist(),
            "_export_mw": export_mw.tolist(),
            "status": "ok",
        }

    except Exception as exc:
        return {
            "power_mw":    task.get("power_kw", 0) / 1000.0,
            "duration_hr": task.get("duration_hr", 0),
            "charging_mode": task.get("mode", ""),
            "status": "error",
            "error_msg": str(exc),
        }

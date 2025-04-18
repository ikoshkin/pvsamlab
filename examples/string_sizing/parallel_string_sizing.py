import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pvsamlab.system import System
from tqdm import tqdm

# -----------------------------
# CONFIGURATION
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAN_FILES_FOLDER = os.path.join(BASE_DIR, "pvsamlab", "data", "modules", "LRI-294 v4.0 LR5-72HBD V4 Pan")
YEAR_RANGE = range(1998, 2024)
STRING_RANGE = range(25, 29)

american_glory = 37.833712, -117.701832
kentucky_moc = 37.10778,-85.08639

LAT, LON = kentucky_moc
NUM_WORKERS = 8  # Matches SAM UI parallelism

# -----------------------------
# HELPERS
# -----------------------------

def get_pan_files(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.pan')]

def run_simulation(pan_file, year, modules_per_string):
    try:
        plant = System(met_year=str(year),
                        lat=LAT, lon=LON,
                        pan_file=pan_file,
                        modules_per_string=modules_per_string)
        voc_max = round(max(plant.model.Outputs.subarray1_voc), 2)
        return {
            "Module": plant.module.model,
            "Year": year,
            "ModulesPerString": modules_per_string,
            "VocMax": round(max(plant.model.Outputs.subarray1_voc), 2),
            "MPPTLoss": round(plant.model.Outputs.annual_dc_invmppt_loss, 2)
        }

        # ----- Monthly outputs from model
        df_monthly = pd.DataFrame({
            # "Module": plant.module.model,
            # "Year": year,
            # "ModulesPerString": modules_per_string,
            # "Month": list(range(1, 13)),
            # "VocMax": plant.model.Outputs.monthly_subarray1_voc,  # ← if available
            # # Add more monthly outputs if desired
        })

        # ----- Hourly outputs from model
        voc = plant.model.Outputs.subarray1_voc
        poa = plant.model.Outputs.subarray1_poa_nom
        tamb = plant.model.Outputs.tdry
        vmp = plant.model.Outputs.subarray1_dc_voltage
        isc = plant.model.Outputs.subarray1_isc

        dt_index = pd.date_range(start=f'{year}-01-01 00:30', end=f'{year}-12-31 23:30', freq='H')
        # Remove Feb 29 from the index if it exists
        dt_index = dt_index[~((dt_index.month == 2) & (dt_index.day == 29))]

        df_hourly = pd.DataFrame({
            "Year": dt_index.year,
            "Month": dt_index.month,
            "Day": dt_index.day,
            "Hour": dt_index.hour,
            "Minute": dt_index.minute,
            "Module": plant.module.model,
            "YearSimulated": year,
            "ModulesPerString": modules_per_string,
            "POA": poa,
            "Tamb": tamb,
            "Voc": voc,
            "Vmp": vmp,
            "Isc": isc,
        })

        return summary, df_monthly, df_hourly

    except Exception as e:
        return {
            "Error": str(e),
            "Module": os.path.basename(pan_file),
            "Year": year,
            "ModulesPerString": modules_per_string
        }, None, None

# -----------------------------
# MAIN
# -----------------------------

def main():
    pan_files = get_pan_files(PAN_FILES_FOLDER)
    tasks = [(pan, year, mps)
             for pan in pan_files
             for year in YEAR_RANGE
             for mps in STRING_RANGE]

    summary_rows = []
    monthly_rows = []
    hourly_rows = []

    print(f"Running {len(tasks)} simulations with {NUM_WORKERS} parallel workers...\n")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(run_simulation, *t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Simulations"):
            summary, monthly, hourly = future.result()
            if summary:
                summary_rows.append(summary)
            if monthly is not None:
                monthly_rows.append(monthly)
            if hourly is not None:
                hourly_rows.append(hourly)
            tqdm.write(f"✓ {summary.get('Module')} | {summary.get('Year')} | Strings={summary.get('ModulesPerString')}")

    # Save all outputs
    pd.DataFrame(summary_rows).to_csv("string_sizing_results_summary.csv", index=False)
    pd.concat(monthly_rows, ignore_index=True).to_csv("string_sizing_results_monthly.csv", index=False)
    pd.concat(hourly_rows, ignore_index=True).to_csv("string_sizing_results_hourly.csv", index=False)

    print("\n✅ All results saved:")
    print(" - string_sizing_results_summary.csv")
    print(" - string_sizing_results_monthly.csv")
    print(" - string_sizing_results_hourly.csv")

# -----------------------------
# REQUIRED FOR MULTIPROCESSING
# -----------------------------
if __name__ == '__main__':
    main()

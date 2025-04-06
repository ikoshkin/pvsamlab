import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pvsamlab.system import System
from tqdm import tqdm  # pip install tqdm

# -----------------------------
# CONFIGURATION
# -----------------------------
PAN_FILES_FOLDER = '/Users/ihorkoshkin/Library/Mobile Documents/com~apple~CloudDocs/Documents/jupyter/pvsamlab/pvsamlab/data/modules/ja'
YEAR_RANGE = range(1998, 2024)
STRING_RANGE = range(27, 33)
NUM_WORKERS = 8  # Matches SAM UI parallelism

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def get_pan_files(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.pan')]

def run_simulation(pan_file, year, modules_per_string):
    try:
        plant = System(met_year=str(year),
                       pan_file=pan_file,
                       modules_per_string=modules_per_string)
        voc_max = round(max(plant.model.Outputs.subarray1_voc), 2)
        return {
            "Module": plant.module.model,
            "Year": year,
            "ModulesPerString": modules_per_string,
            "VocMax": voc_max,
            # "MPPTLosses": plant.model.Outputs.annual_dc_mppt_clip_loss_percent,
        }
    except Exception as e:
        return {
            "Error": str(e),
            "Module": os.path.basename(pan_file),
            "Year": year,
            "ModulesPerString": modules_per_string
        }

# -----------------------------
# MAIN EXECUTION
# -----------------------------
def main():
    pan_files = get_pan_files(PAN_FILES_FOLDER)
    tasks = [(pan, year, mps)
             for pan in pan_files
             for year in YEAR_RANGE
             for mps in STRING_RANGE]

    results = []

    print(f"Running {len(tasks)} simulations with {NUM_WORKERS} parallel workers...\n")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(run_simulation, *t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Simulations"):
            result = future.result()
            results.append(result)
            tqdm.write(f"✓ {result.get('Module')} | Year={result.get('Year')} | Strings={result.get('ModulesPerString')}")

    df = pd.DataFrame(results)
    output_path = "string_sizing_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved results to {output_path}")
    print(df.head())

# -----------------------------
# REQUIRED FOR MULTIPROCESSING
# -----------------------------
if __name__ == '__main__':
    main()

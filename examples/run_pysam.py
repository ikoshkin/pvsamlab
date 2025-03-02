from pvsamlab.utils import run_pvsamv1_from_json

json_path = "examples/inputs_sample.json"
results = run_pvsamv1_from_json(json_path)
print(f"Annual Energy: {results['annual_energy']} kWh")
import json
import PySAM.Pvsamv1 as pv

def run_pvsamv1_from_json(json_file_path):
    """Loads a PySAM JSON file, assigns parameters to pvsamv1, and runs a simulation."""
    with open(json_file_path, 'r') as f:
        config = json.load(f)

    pv_model = pv.new()

    # Print all required PySAM inputs before assignment
    required_inputs = ["system_capacity", "inverter_count"]
    
    for key in required_inputs:
        if key not in config.get("SystemDesign", {}):
            print(f"⚠️ Missing required input: {key}")

    # Assign inputs to PySAM model
    for group, values in config.items():
        if hasattr(pv_model, group):  # Ensure it's a valid PySAM group
            getattr(pv_model, group).assign(values)

    # Run PySAM
    pv_model.execute()
    return pv_model.Outputs.export()

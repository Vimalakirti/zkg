"""
Run the full ezkl proving pipeline on GraphSAGE Cora.
Steps: gen-settings -> calibrate -> compile -> get-srs -> setup -> gen-witness -> prove -> verify
"""

import ezkl
import json
import os
import time

WORK_DIR = "/scratch/bjchen4_icgpu/zkgnn/ezkl_graphsage"

model_path = os.path.join(WORK_DIR, "graphsage_cora.onnx")
input_path = os.path.join(WORK_DIR, "input.json")
settings_path = os.path.join(WORK_DIR, "settings.json")
compiled_path = os.path.join(WORK_DIR, "model.compiled")
srs_path = os.path.join(WORK_DIR, "kzg.srs")
vk_path = os.path.join(WORK_DIR, "vk.key")
pk_path = os.path.join(WORK_DIR, "pk.key")
witness_path = os.path.join(WORK_DIR, "witness.json")
proof_path = os.path.join(WORK_DIR, "proof.json")

timings = {}


def timed(name, func, *args, **kwargs):
    print(f"\n=== {name} ===")
    t0 = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - t0
    timings[name] = elapsed
    print(f"  Done in {elapsed:.2f}s")
    return result


# Step 1: Generate settings
py_run_args = ezkl.PyRunArgs()
py_run_args.input_visibility = "public"
py_run_args.output_visibility = "public"
py_run_args.param_visibility = "fixed"

timed("gen_settings",
      ezkl.gen_settings, model_path, settings_path, py_run_args=py_run_args)

# Print initial settings
with open(settings_path) as f:
    settings = json.load(f)
print(f"  Initial logrows: {settings['run_args']['logrows']}")
print(f"  Model: {settings.get('model_output_scales', 'N/A')}")

# Step 2: Calibrate settings
timed("calibrate_settings",
      ezkl.calibrate_settings, input_path, model_path, settings_path, "resources")

with open(settings_path) as f:
    settings = json.load(f)
print(f"  Calibrated logrows: {settings['run_args']['logrows']}")
print(f"  lookup_range: {settings['run_args'].get('lookup_range', 'N/A')}")
print(f"  num_inner_cols: {settings['run_args'].get('num_inner_cols', 'N/A')}")

# Step 3: Compile circuit
timed("compile_circuit",
      ezkl.compile_circuit, model_path, compiled_path, settings_path)

# Step 4: Get SRS
timed("get_srs",
      ezkl.get_srs, settings_path, srs_path)

# Step 5: Setup (generate pk/vk)
timed("setup",
      ezkl.setup, compiled_path, vk_path, pk_path, srs_path)

# Step 6: Generate witness
timed("gen_witness",
      ezkl.gen_witness, input_path, compiled_path, witness_path)

# Step 7: Prove
timed("prove",
      ezkl.prove, witness_path, compiled_path, pk_path, proof_path, "single", srs_path)

# Step 8: Verify
result = timed("verify",
               ezkl.verify, proof_path, settings_path, vk_path, srs_path)
print(f"  Verified: {result}")

# Get proof size
proof_size = os.path.getsize(proof_path)
print(f"\n=== Proof size: {proof_size / 1024:.2f} KB ===")

# Summary
print("\n" + "=" * 50)
print("  EZKL GraphSAGE Cora Summary")
print("=" * 50)
for name, t in timings.items():
    print(f"  {name:25s}: {t:10.2f}s")
print(f"  {'proof_size':25s}: {proof_size / 1024:.2f} KB")
print("=" * 50)

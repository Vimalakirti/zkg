"""
Run ezkl pipeline on GraphSAGE Cora, skipping calibration.
Manually set logrows to accommodate the 28M-row circuit.
"""

import asyncio
import ezkl
import json
import math
import os
import time

WORK_DIR = "/home/bingjyuechen/ezkl_graphsage"

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


def timed_sync(name, func, *args, **kwargs):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - t0
    timings[name] = elapsed
    print(f"  Done in {elapsed:.2f}s", flush=True)
    return result


async def timed_async(name, coro):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    result = await coro
    elapsed = time.time() - t0
    timings[name] = elapsed
    print(f"  Done in {elapsed:.2f}s", flush=True)
    return result


async def main():
    # Step 1: Generate settings (sync)
    print("Step 1: gen_settings", flush=True)
    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "public"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "fixed"

    timed_sync("gen_settings",
               ezkl.gen_settings, model_path, settings_path, py_run_args=py_run_args)

    # Step 2: Manually adjust logrows (skip calibration)
    print("\nStep 2: Adjusting settings (skip calibration)", flush=True)
    with open(settings_path) as f:
        settings = json.load(f)

    num_rows = settings["num_rows"]
    needed_logrows = max(math.ceil(math.log2(num_rows + 1)), 25)
    settings["run_args"]["logrows"] = needed_logrows
    print(f"  num_rows: {num_rows}, logrows: {needed_logrows} (2^{needed_logrows} = {2**needed_logrows})", flush=True)

    with open(settings_path, "w") as f:
        json.dump(settings, f)

    # Step 3: Compile circuit (sync)
    timed_sync("compile_circuit",
               ezkl.compile_circuit, model_path, compiled_path, settings_path)
    compiled_size = os.path.getsize(compiled_path)
    print(f"  Compiled circuit: {compiled_size / 1024 / 1024:.1f} MB", flush=True)

    # Step 4: Get SRS (async - downloads from network)
    await timed_async("get_srs",
                      ezkl.get_srs(settings_path=settings_path, logrows=needed_logrows, srs_path=srs_path))
    srs_size = os.path.getsize(srs_path)
    print(f"  SRS size: {srs_size / 1024 / 1024:.1f} MB", flush=True)

    # Step 5: Setup (generate pk/vk)
    timed_sync("setup",
               ezkl.setup, compiled_path, vk_path, pk_path, srs_path)
    pk_size = os.path.getsize(pk_path)
    vk_size = os.path.getsize(vk_path)
    print(f"  PK: {pk_size / 1024 / 1024:.1f} MB, VK: {vk_size / 1024:.1f} KB", flush=True)

    # Step 6: Generate witness (sync)
    timed_sync("gen_witness",
               ezkl.gen_witness, input_path, compiled_path, witness_path)

    # Step 7: Prove (sync)
    timed_sync("prove",
               ezkl.prove, witness_path, compiled_path, pk_path, proof_path, "single", srs_path)

    # Step 8: Verify (sync)
    result = timed_sync("verify",
                        ezkl.verify, proof_path, settings_path, vk_path, srs_path)
    print(f"  Verified: {result}", flush=True)

    # Proof size
    proof_size = os.path.getsize(proof_path)

    # Summary
    print("\n" + "=" * 50, flush=True)
    print("  EZKL GraphSAGE Cora Summary", flush=True)
    print("=" * 50, flush=True)
    for name, t in timings.items():
        print(f"  {name:25s}: {t:10.2f}s")
    print(f"  {'proof_size':25s}: {proof_size / 1024:.2f} KB")
    print(f"  {'logrows':25s}: {needed_logrows}")
    print(f"  {'num_rows':25s}: {num_rows}")
    print("=" * 50, flush=True)


asyncio.run(main())

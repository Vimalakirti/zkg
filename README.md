# zkGNN: Artifact for Reproducing Paper Results

This artifact contains all source code and scripts needed to reproduce the experimental results in the zkGNN paper. All training data, model weights, and SRS files are generated from scratch by the provided scripts.

## Requirements

### Hardware
- **CPU**: Multi-core x86_64 (experiments were run on 96-core AMD EPYC Milan @ 2.65 GHz)
- **RAM**: At least 16 GB for citation networks; 142 GB for PubMed; 8 GB for subgraph experiments
- **Disk**: ~2 GB for generated data + up to ~16 GB for SRS files (generated on first use)

### Software
- **Rust** 1.75+ (with `cargo`)
- **Python** 3.10+ with:
  - `torch`, `torch_geometric` (for training and data generation)
  - `numpy`, `scikit-learn` (for evaluation scripts)
  - `ezkl==23.0.5` (for EZKL comparison only; `pip install ezkl`)

## Directory Structure

```
zkg/
├── src/                    # Rust source code
│   ├── bin/                # Binary entry points (gcn, graphsage, gat, kernel, setup)
│   ├── basicblock/         # Arithmetic circuit primitives (SpMV, ReLU, etc.)
│   ├── crypto/             # Cryptographic primitives (sumcheck, KZH3 PCS)
│   ├── dag/                # Computation DAG: commit, prove, verify pipeline
│   └── util/               # Utilities (config, data loading, polynomials)
├── tests/                  # Rust unit tests
├── benches/                # Rust benchmarks
├── pyg/                    # Python training & data pipeline
│   ├── train_citation.py   # Train GCN/GraphSAGE/GAT on Cora/CiteSeer/PubMed
│   ├── train_kernel.py     # Train GCN/GraphSAGE on MUTAG/PROTEINS
│   ├── train_elliptic.py   # Train GraphSAGE on Elliptic Bitcoin
│   ├── train_dgraphfin.py  # Train GraphSAGE on DGraphFin
│   ├── export_raw.py       # Export trained weights to binary format
│   ├── export_kernel.py    # Export kernel weights to binary format
│   ├── gen_fake.py         # Generate synthetic Erdős–Rényi graphs
│   ├── extract_subgraph.py # Extract k-hop subgraphs for large graphs
│   └── eval_subgraph_auc.py# Evaluate subgraph prediction quality (AUC)
├── ezkl_graphsage/         # EZKL baseline comparison
│   └── run_ezkl_subgraph.py# Run EZKL on subgraph-extracted GraphSAGE
├── config.yaml             # Runtime config (scale_factor_log=10, table_size_log=20)
├── Cargo.toml              # Rust dependencies
├── setup_data.sh           # Generate all training data and weights
├── run_table3_repro.sh     # Table 1: End-to-end proving (citation networks)
├── run_kernel_repro.sh     # Table 1: End-to-end proving (TU benchmarks)
├── run_breakdown_repro.sh  # Table 2: Prover time breakdown
├── run_subgraph_repro.sh   # Table 3 + Table 4 (zkGNN column): Subgraph extraction
├── run_ezkl_repro.sh       # Table 4 (ezkl column): EZKL comparison
├── run_ablation.sh         # Table 5: SpMM ablation study
├── run_zk_overhead_repro.sh# Table 6: Zero-knowledge overhead
└── scripts/                # Helper scripts (breakdown parser, etc.)
```

## Quick Start

### 1. Build

```bash
cd /path/to/zkg
cargo build --release
```

This compiles all binaries (`gcn`, `graphsage`, `gat`, `kernel`, `setup`).

### 2. Generate Data

All training data, model weights, synthetic graphs, and subgraph extractions are generated from the Python scripts:

```bash
./setup_data.sh
# Runs all training + export + generation scripts
# Time: ~1-2 hours (DGraphFin training dominates)
# Output: populates pyg/weights/raw/ with all binary data
```

This runs the following pipeline:
1. Train GCN/GraphSAGE/GAT on citation networks → export to binary
2. Train GCN/GraphSAGE on TU benchmarks → export to binary
3. Train GraphSAGE on Elliptic and DGraphFin → export to binary
4. Generate synthetic Erdős–Rényi graphs for ablation/ZK experiments
5. Extract 2-hop subgraphs for Elliptic and DGraphFin (1000 each)

### 3. Run a single experiment

```bash
# Prove GCN on Cora
./target/release/gcn config.yaml pyg/weights cora

# Prove GraphSAGE on a subgraph
./target/release/graphsage config.yaml pyg/weights elliptic_sub_188332 elliptic

# Prove with zero-knowledge mode
./target/release/gcn config.yaml pyg/weights cora --zk
```

SRS files are generated automatically on first use and cached to the working directory as `*.srs` files.

## Reproducing Paper Tables

All reproduction scripts write results to CSV files and print summaries to stdout. Logs for each individual run are saved to `repro_logs/`.

### Table 1: End-to-End Proving Performance (`tab:proving`)

**Citation networks** (Cora, CiteSeer, PubMed × GCN, GraphSAGE, GAT):
```bash
./run_table3_repro.sh
# Output: repro_table3.csv
# Time: ~2 hours (PubMed GAT dominates at ~20 min)
```

**TU benchmarks** (MUTAG, PROTEINS × GCN, GraphSAGE):
```bash
./run_kernel_repro.sh
# Output: repro_kernel.csv
# Time: ~5 minutes
```

### Table 2: Prover Time Breakdown (`tab:breakdown`)

```bash
./run_breakdown_repro.sh
# Runs GCN and GAT on Cora and PubMed, parses BREAKDOWN lines
# Time: ~30 minutes (PubMed GAT dominates)
```

### Table 3: Subgraph Extraction (`tab:subgraph`)

```bash
./run_subgraph_repro.sh
# Part 1: Average prove/verify/proof over 20 random subgraphs per dataset
# Part 2: zkGNN on specific subgraphs for EZKL comparison (Table 4)
# Output: repro_ezkl_zkgnn.csv
# Time: ~10 minutes
```

### Table 4: EZKL Comparison (`tab:ezkl`)

The zkGNN column is produced by `run_subgraph_repro.sh` above. For the EZKL column:

```bash
# Requires: pip install ezkl
./run_ezkl_repro.sh
# Output: ezkl_graphsage/ezkl_elliptic_results.csv, ezkl_graphsage/ezkl_dgraphfin_results.csv
# Time: ~30 minutes (calibration dominates)
# Note: Requires ~5 GB disk for proving keys; cleanup is automatic between runs
```

### Table 5: SpMM Ablation Study (`tab:ablation-spmv`)

```bash
./run_ablation.sh
# Runs GraphSAGE with proposed O(M) vs classical O(N^2) SpMM on synthetic graphs
# Uses env var CLASSICAL_SPMV=1 for the dense baseline
# Output: ablation_results.csv
# Time: ~3 hours (classical mode on 2^15 nodes is slow)
```

### Table 6: Zero-Knowledge Overhead (`tab:zk-overhead`)

```bash
./run_zk_overhead_repro.sh
# Runs GCN/GraphSAGE (N=2^12..2^16) and GAT (N=2^12..2^15) with ZK off/on
# Output: repro_summary.csv
# Time: ~8 hours (GAT 2^15 ZK dominates at ~90 min)
```

### Appendix: Quantization Accuracy (`tab:accuracy`)

Quantized accuracy numbers are printed during training (Step 2 above). The Rust prover also prints accuracy when run on each dataset.

## Running All Experiments

To reproduce all paper tables end-to-end:

```bash
# Step 1: Build (~2 min)
cargo build --release

# Step 2: Generate all data (~1-2 hours)
./setup_data.sh

# Step 3: Run experiments

# Fast (~15 min)
./run_kernel_repro.sh
./run_subgraph_repro.sh

# Medium (~2.5 hours)
./run_table3_repro.sh
./run_breakdown_repro.sh

# Long (~11 hours)
./run_ablation.sh
./run_zk_overhead_repro.sh

# EZKL comparison (~30 min, requires pip install ezkl)
./run_ezkl_repro.sh
```

**Total estimated time: ~16 hours** on a 96-core machine (including data generation). Individual experiments can be run independently once data is generated.

## Environment Variables

- `CLASSICAL_SPMV=1` — Use the O(N^2) dense SpMM prover (for ablation baseline)
- `RUST_LOG=debug` — Enable debug logging

## Notes

- SRS files (`*.srs`) are generated automatically on first use and cached to the working directory. They can be safely deleted and will be regenerated.
- Training uses default PyTorch random seeds; exact weights may differ across runs, but accuracy and proving performance numbers will be comparable.
- All experiments use the BN254 elliptic curve with the KZH3 polynomial commitment scheme.
- Proof verification is deterministic; timing may vary by ±10% across runs depending on system load.

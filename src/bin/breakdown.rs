/// Prover time breakdown for paper experiments.
/// Measures time spent in each component: commit, node proving (by type),
/// lookup proving, opening proofs, and verification.
///
/// Usage: cargo run --bin breakdown --release -- config.yaml pyg/weights [dataset]

#[cfg(all(feature = "arkworks", feature = "bn254"))]
use ark_bn254::Fr as F;
#[cfg(all(feature = "arkworks", feature = "bls12_381"))]
use ark_bls12_381::Fr as F;

use plonky2::util::timing::TimingTree;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use zk_torch_2::{
  basicblock::BasicBlockType,
  crypto::polycommit::kzh3::{setup_kzh3_srs, KZH3Commit, KZH3CommitKey, KZH3Commitment, KZH3VerifierKey},
  crypto::polycommit::sparse_kzh3::{SparseKZH3Commit, SparseKZH3CommitKey, SparseKZH3VerifierKey},
  crypto::srs_storage::{load_kzh3_srs, store_kzh3_srs},
  crypto::MaskCommitter,
  dag::{gnn::gcn, DagBuilder, DataType, Role, Witness},
  util::arith::log2_ceil,
  util::data_loader::*,
  util::poly::{CryptoField, SelectionPolynomial},
  util::transcript::Transcript,
  SF_LOG,
};

#[cfg(all(feature = "arkworks", feature = "bn254"))]
use zk_torch_2::crypto::polycommit::ArkBn254 as PairingType;

fn main() {
  let mut timing = TimingTree::default();
  env_logger::init();

  let args: Vec<String> = std::env::args().collect();
  let data_dir = if args.len() >= 3 { &args[2] } else { "pyg/weights" };
  let dataset_name = if args.len() >= 4 { &args[3] } else { "cora" };

  println!("=== Prover Time Breakdown (GCN on {}) ===\n", dataset_name);

  let dataset_path = Path::new(data_dir).join("raw").join(dataset_name);
  let weights_path = Path::new(data_dir).join("raw").join(format!("gcn_{}", dataset_name));

  // Load dataset
  let meta_str = std::fs::read_to_string(dataset_path.join("meta.json")).expect("Failed to read meta.json");
  let meta: serde_json::Value = serde_json::from_str(&meta_str).unwrap();
  let num_nodes = meta["num_nodes"].as_u64().unwrap() as usize;
  let num_features = meta["num_features"].as_u64().unwrap() as usize;
  let num_classes = meta["num_classes"].as_u64().unwrap() as usize;
  let num_nodes_pad = 1usize << (log2_ceil(num_nodes) as usize);

  println!("  Nodes: {}, Features: {}, Classes: {}", num_nodes, num_features, num_classes);

  let mut input = load_node_features::<F>(&dataset_path.join("x.bin"), num_nodes, num_features);
  input.shape[0] = num_nodes_pad;

  let edge_src = read_i32_bin(&dataset_path.join("edge_src.bin"));
  let edge_dst = read_i32_bin(&dataset_path.join("edge_dst.bin"));
  let num_edges = edge_src.len();
  println!("  Edges: {}", num_edges);

  // Build adjacency
  let sf_a = *SF_LOG;
  let mut adjacency = zk_torch_2::util::data_loader::build_gcn_adjacency::<F>(
    &edge_src, &edge_dst, num_nodes, sf_a,
  );
  adjacency.additive_factorize(2).expect("Adjacency factorization failed");

  // Build DAG
  let w1 = load_weight_matrix::<F>(&weights_path.join("conv1.lin.weight.bin"), 16, num_features);
  let b1 = load_bias::<F>(&weights_path.join("conv1.bias.bin"), 16);
  let w2 = load_weight_matrix::<F>(&weights_path.join("conv2.lin.weight.bin"), num_classes, 16);
  let b2 = load_bias::<F>(&weights_path.join("conv2.bias.bin"), num_classes);

  let mut g = DagBuilder::new();
  let x = g.input(vec![num_nodes_pad, num_features], DataType::Float);
  let output = g.pipe(&[x], gcn(adjacency, vec![w1, w2], vec![b1, b2]))[0];

  println!("\nCompiling DAG...");
  let (dag, mut init) = g.compile();

  println!("Running forward pass...");
  let presplit_sparse = dag.run(&mut init, &vec![(x, input)]);

  // Count nodes by type
  let mut type_counts: HashMap<String, usize> = HashMap::new();
  for node in &dag.nodes {
    let type_name = match &node.kind {
      BasicBlockType::Einsum(_) => "MatMul",
      BasicBlockType::SpMV(_) => "SpMM",
      BasicBlockType::Add(_) => "Add",
      BasicBlockType::Sub(_) => "Sub",
      BasicBlockType::SignBitHelper(_) => "SignBit",
      BasicBlockType::ScaleDown(_) => "ScaleDown",
      BasicBlockType::ScaleUp(_) => "ScaleUp",
      BasicBlockType::NonNegative(_) => "RangeCheck",
      BasicBlockType::ChangeShape(_) => "ChangeShape",
      _ => "Other",
    };
    *type_counts.entry(type_name.to_string()).or_insert(0) += 1;
  }
  println!("\nDAG nodes by type:");
  for (t, c) in type_counts.iter() {
    println!("  {}: {}", t, c);
  }

  // Setup SRS
  let polynomial_sizes = dag.collect_polynomial_sizes(&init);
  let mut srs_map = HashMap::new();
  for &size in &polynomial_sizes {
    let size_srs = if std::fs::metadata(&format!("{}.srs", size)).is_ok() {
      load_kzh3_srs(size).expect("Failed to load SRS")
    } else {
      println!("  Generating SRS for size {}", size);
      use rand::thread_rng;
      let size_srs = setup_kzh3_srs::<PairingType, _>(size, &mut thread_rng());
      store_kzh3_srs(&size_srs, size).expect("Failed to store SRS");
      size_srs
    };
    srs_map.insert(size, Arc::new(size_srs));
  }
  let srs_map = Arc::new(srs_map);
  let kzh3 = KZH3CommitKey::<PairingType> { srs_map: srs_map.clone(), h: None };
  let sparse_kzh3 = SparseKZH3CommitKey::<PairingType> { srs_map: srs_map.clone(), h: None };
  let dense_vk = KZH3VerifierKey { srs_map: srs_map.clone(), h: None };
  let sparse_vk = SparseKZH3VerifierKey { srs_map: srs_map.clone(), h: None };

  let mut dense_commitments: Vec<Option<KZH3Commitment<PairingType>>> = vec![None; dag.num_edges()];
  let mut sparse_commitments: Vec<Option<Vec<KZH3Commitment<PairingType>>>> = vec![None; dag.num_edges()];
  let mut factored_commitments: Vec<Option<Vec<KZH3Commitment<PairingType>>>> = vec![None; dag.num_edges()];

  // === COMMIT PHASE ===
  println!("\n=== COMMIT PHASE ===");
  let t0 = std::time::Instant::now();
  dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &kzh3, &sparse_kzh3, &init, &mut dense_commitments, &mut sparse_commitments, &mut factored_commitments, &mut timing,
  );
  let commit_time = t0.elapsed();
  println!("  Total commit: {:.3?}", commit_time);

  // === PROVE PHASE ===
  println!("\n=== PROVE PHASE ===");
  let mut transcript = Transcript::new(b"zkml");
  let t0 = std::time::Instant::now();
  let (sc_proofs, op_proofs, range_proof, two_pow_proof, reducer_proofs, _zk_proof) =
    dag.prove_zk::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &kzh3, &sparse_kzh3, &init, &presplit_sparse,
      &dense_commitments, &sparse_commitments, &factored_commitments, &mut transcript, &mut timing,
      false, None,
    );
  let prove_time = t0.elapsed();
  println!("  Total prove: {:.3?}", prove_time);

  // Proof size
  zk_torch_2::util::serialization::measure_total_proof_size(
    &sc_proofs, &op_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
  );

  // === VERIFY PHASE ===
  println!("\n=== VERIFY PHASE ===");
  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));
  let mut vt = Transcript::new(b"zkml");
  let t0 = std::time::Instant::now();
  let verified = dag.verify_zk::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &sc_proofs, &op_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
    &init, &dense_vk, &sparse_vk, &dense_commitments, &sparse_commitments, &factored_commitments, &mut vt,
    false, &Default::default(), None,
  );
  let verify_time = t0.elapsed();
  println!("  Total verify: {:.3?}", verify_time);
  println!("  Verified: {}", verified);

  // === SUMMARY ===
  println!("\n=== BREAKDOWN SUMMARY ===");
  println!("  Commit:  {:.3?}", commit_time);
  println!("  Prove:   {:.3?}", prove_time);
  println!("  Verify:  {:.3?}", verify_time);
  println!("  Total:   {:.3?}", commit_time + prove_time + verify_time);

  // Print timing tree for detailed breakdown
  println!("\n=== DETAILED TIMING TREE ===");
  println!("{}", timing);
}

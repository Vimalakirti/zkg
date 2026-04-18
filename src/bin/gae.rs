// Field type selection based on backend and curve
#[cfg(all(feature = "arkworks", feature = "bls12_381"))]
use ark_bls12_381::Fr as F;
#[cfg(all(feature = "arkworks", feature = "bn254"))]
use ark_bn254::Fr as F;

#[cfg(all(feature = "icicle", feature = "bls12_381"))]
use icicle_bls12_381::curve::ScalarField as F;
#[cfg(all(feature = "icicle", feature = "bn254"))]
use icicle_bn254::curve::ScalarField as F;
#[cfg(all(feature = "icicle", feature = "goldilocks"))]
use icicle_goldilocks::field::ScalarField as F;

use plonky2::util::timing::TimingTree;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use zk_torch_2::{
  crypto::polycommit::kzh3::{setup_kzh3_srs, KZH3Commit, KZH3CommitKey, KZH3Commitment, KZH3MaskCommitter, KZH3VerifierKey},
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
#[cfg(all(feature = "icicle", feature = "bn254"))]
use zk_torch_2::crypto::polycommit::IcicleBn254 as PairingType;

/// Build symmetric-normalized adjacency (Â = A + I, D̂^{-1/2} Â D̂^{-1/2}) from an edge list.
/// Matches PyG's GCNConv normalization.
fn build_adjacency_from_edges(
  edge_src: &[i32],
  edge_dst: &[i32],
  num_nodes: usize,
  sf_a: usize,
) -> Witness<F> {
  let num_nodes_log = log2_ceil(num_nodes) as usize;

  // Collect edges + self-loops
  let mut edge_set = HashSet::new();
  for i in 0..num_nodes {
    edge_set.insert((i, i)); // self-loops
  }
  for i in 0..edge_src.len() {
    edge_set.insert((edge_src[i] as usize, edge_dst[i] as usize));
  }
  let edges: Vec<(usize, usize)> = edge_set.into_iter().collect();

  // Compute degree
  let mut degree = vec![0usize; num_nodes];
  for &(src, _dst) in &edges {
    degree[src] += 1;
  }

  // Symmetric normalization: weight(i,j) = 1 / sqrt(deg(i) * deg(j))
  let scale = (1u64 << sf_a) as f64;
  let weighted_edges: Vec<(usize, usize, u32)> = edges
    .iter()
    .map(|&(src, dst)| {
      let w = (scale / (degree[src] as f64 * degree[dst] as f64).sqrt()).round() as u32;
      (src, dst, w)
    })
    .collect();

  // Build SelectionPolynomial
  let binary_edges: Vec<(usize, usize)> = weighted_edges.iter().map(|&(s, d, _)| (s, d)).collect();
  let selection_poly = SelectionPolynomial::<F>::new(num_nodes_log, num_nodes_log, binary_edges);
  let mut sparse_poly = selection_poly.to_sparse();
  for &(src, dst, weight) in &weighted_edges {
    let index = src + dst * (1 << num_nodes_log);
    sparse_poly.evaluations.insert(index, <F as CryptoField>::from_u32(weight));
  }

  let num_nodes_pad = 1 << num_nodes_log;
  Witness::new_sparse(
    vec![num_nodes_pad, num_nodes_pad],
    sparse_poly,
    DataType::Float,
    sf_a,
    Role::Constant,
  )
}

fn main() {
  let mut timing = TimingTree::default();
  env_logger::init();

  // Parse arguments: <config.yaml> <data_dir> [dataset_name]
  // data_dir should contain: raw/gae_{dataset}/ with x.bin, edge_src/dst.bin, weights/
  let args: Vec<String> = std::env::args().collect();
  let data_dir = if args.len() >= 3 { Some(args[2].clone()) } else { None };
  let dataset_name = if args.len() >= 4 { args[3].clone() } else { "cora".to_string() };
  let zk_mode = args.iter().any(|a| a == "--zk");

  if let Some(ref data_dir) = data_dir {
    println!("=== GAE (Graph Autoencoder) on {} ===", dataset_name);

    let dataset_path = Path::new(data_dir).join("raw").join(format!("gae_{}", dataset_name));
    let weights_path = dataset_path.join("weights");

    // Load dataset metadata
    let meta_str = std::fs::read_to_string(dataset_path.join("meta.json"))
      .expect("Failed to read dataset meta.json");
    let meta: serde_json::Value = serde_json::from_str(&meta_str).unwrap();
    let num_nodes = meta["num_nodes"].as_u64().unwrap() as usize;
    let num_features = meta["num_features"].as_u64().unwrap() as usize;
    let hidden = meta["hidden"].as_u64().unwrap_or(32) as usize;
    let out_channels = meta["out_channels"].as_u64().unwrap_or(16) as usize;
    let num_test_pos = meta["num_test_pos"].as_u64().unwrap() as usize;
    let num_test_neg = meta["num_test_neg"].as_u64().unwrap() as usize;

    println!("  Nodes: {}, Features: {}", num_nodes, num_features);
    println!("  Encoder: GCN({} → {} → {})", num_features, hidden, out_channels);
    println!("  Test edges: {} pos, {} neg", num_test_pos, num_test_neg);

    let num_nodes_log = log2_ceil(num_nodes) as usize;
    let num_nodes_pad = 1usize << num_nodes_log;

    // Load node features
    let mut input = load_node_features::<F>(
      &dataset_path.join("x.bin"), num_nodes, num_features,
    );
    input.shape[0] = num_nodes_pad;

    // Load edge list and build symmetric-normalized adjacency (with self-loops)
    let edge_src = read_i32_bin(&dataset_path.join("edge_src.bin"));
    let edge_dst = read_i32_bin(&dataset_path.join("edge_dst.bin"));
    let sf_a = *SF_LOG;
    let mut adjacency = build_adjacency_from_edges(&edge_src, &edge_dst, num_nodes, sf_a);
    println!("  Edges (train): {} raw edges", edge_src.len());

    let num_shares = 2;
    adjacency.additive_factorize(num_shares).expect("Adjacency factorization failed");
    let af = adjacency.additive_factored.as_ref().unwrap();
    println!("  Adjacency decomposed: {} terms, {} shares, chunk_sizes={:?}",
      af.terms.len(), af.chunk_sizes.len(), af.chunk_sizes);

    // Load test edges for evaluation
    let test_pos_src = read_i32_bin(&dataset_path.join("test_pos_src.bin"));
    let test_pos_dst = read_i32_bin(&dataset_path.join("test_pos_dst.bin"));
    let test_neg_src = read_i32_bin(&dataset_path.join("test_neg_src.bin"));
    let test_neg_dst = read_i32_bin(&dataset_path.join("test_neg_dst.bin"));

    // Load weight matrices (PyTorch stores as (out, in), we transpose to (in, out))
    // Layer 1: (num_features, hidden)
    let w1 = load_weight_matrix::<F>(
      &weights_path.join("encoder.conv1.lin.weight.bin"), hidden, num_features,
    );
    // Layer 2: (hidden, out_channels)
    let w2 = load_weight_matrix::<F>(
      &weights_path.join("encoder.conv2.lin.weight.bin"), out_channels, hidden,
    );

    let weights = vec![w1, w2];
    println!("  Loaded weights: layer1({},{}), layer2({},{})",
      num_features, hidden, hidden, out_channels);

    // --- Build and run DAG ---
    // Prove only the GCN encoder: X → Z (node embeddings).
    // The decoder (inner product z_i·z_j) is trivial and can be verified separately.
    // This avoids computing the full N×N matrix Z·Z^T → sigmoid.
    let mut g = DagBuilder::new();
    let x = g.input(vec![num_nodes_pad, num_features], DataType::Float);
    let output = g.pipe(
      &[x],
      gcn(adjacency, weights),
    )[0];

    println!("Compiling DAG...");
    let (dag, mut init) = g.compile();

    println!("Running forward pass...");
    let presplit_sparse = dag.run(&mut init, &vec![(x, input)]);
    println!("  Embedding Z shape: {:?}", init[output][0].shape);

    // --- Evaluate link prediction (AUC) ---
    // Output Z has shape (num_nodes_pad, out_channels).
    // Edge score = z_i · z_j (inner product). Higher = more likely edge exists.
    let z_witness = &init[output][0];

    let mut scores = Vec::new();
    let mut labels = Vec::new();

    // Compute inner product z_i · z_j for test edges
    let compute_inner_product = |i: usize, j: usize| -> i128 {
      let mut dot = 0i128;
      for d in 0..out_channels {
        let zi_d = zk_torch_2::util::arith::f_to_int(z_witness.get(&[i, d]));
        let zj_d = zk_torch_2::util::arith::f_to_int(z_witness.get(&[j, d]));
        dot += zi_d * zj_d;
      }
      dot
    };

    // Positive test edges (label = 1)
    for k in 0..num_test_pos {
      let i = test_pos_src[k] as usize;
      let j = test_pos_dst[k] as usize;
      let score = compute_inner_product(i, j);
      scores.push(score as f64);
      labels.push(1);
      println!("EDGE|{}|{}|1|{}", i, j, score);
    }

    // Negative test edges (label = 0)
    for k in 0..num_test_neg {
      let i = test_neg_src[k] as usize;
      let j = test_neg_dst[k] as usize;
      let score = compute_inner_product(i, j);
      scores.push(score as f64);
      labels.push(0);
      println!("EDGE|{}|{}|0|{}", i, j, score);
    }

    // Compute AUC manually (count concordant pairs)
    let n_pos = num_test_pos;
    let n_neg = num_test_neg;
    let mut concordant = 0u64;
    let mut tied = 0u64;
    for p in 0..n_pos {
      for n in 0..n_neg {
        let pos_score = scores[p];
        let neg_score = scores[n_pos + n];
        if pos_score > neg_score {
          concordant += 1;
        } else if pos_score == neg_score {
          tied += 1;
        }
      }
    }
    let auc = (concordant as f64 + 0.5 * tied as f64) / (n_pos as f64 * n_neg as f64);

    println!("\n=== Link Prediction Results (GAE on {}) ===", dataset_name);
    println!("  Test edges: {} pos, {} neg", n_pos, n_neg);
    println!("  AUC: {:.4}", auc);

    // --- Proving and Verification ---
    let mut transcript = Transcript::new(b"zkml");
    let mut dense_commitments: Vec<Option<KZH3Commitment<PairingType>>> = vec![None; dag.num_edges()];
    let mut sparse_commitments: Vec<Option<Vec<KZH3Commitment<PairingType>>>> = vec![None; dag.num_edges()];
    let mut factored_commitments: Vec<Option<Vec<KZH3Commitment<PairingType>>>> = vec![None; dag.num_edges()];

    let polynomial_sizes = dag.collect_polynomial_sizes(&init);
    println!("\nPolynomial sizes: {:?}", polynomial_sizes);

    let mut srs_map = HashMap::new();
    for &size in &polynomial_sizes {
      let size_srs = if std::fs::metadata(&format!("{}.srs", size)).is_ok() {
        println!("Loading SRS for size {}", size);
        load_kzh3_srs(size).expect("Failed to load SRS")
      } else {
        println!("Generating SRS for size {}", size);
        #[cfg(feature = "arkworks")]
        let size_srs = { use rand::thread_rng; setup_kzh3_srs::<PairingType, _>(size, &mut thread_rng()) };
        #[cfg(feature = "icicle")]
        let size_srs = setup_kzh3_srs::<PairingType, _>(size, &mut ());
        if let Err(e) = store_kzh3_srs(&size_srs, size) {
          println!("  Warning: could not store SRS for size {}: {}", size, e);
        }
        size_srs
      };
      srs_map.insert(size, Arc::new(size_srs));
    }
    if zk_mode {
      for mask_n in 1..=8usize {
        if !srs_map.contains_key(&mask_n) {
          let mask_srs = if std::fs::metadata(&format!("{}.srs", mask_n)).is_ok() {
            println!("Loading SRS for mask size {}", mask_n);
            load_kzh3_srs(mask_n).expect("Failed to load mask SRS")
          } else {
            println!("Generating SRS for mask size {}", mask_n);
            #[cfg(feature = "arkworks")]
            let mask_srs = { use rand::thread_rng; setup_kzh3_srs::<PairingType, _>(mask_n, &mut thread_rng()) };
            #[cfg(feature = "icicle")]
            let mask_srs = setup_kzh3_srs::<PairingType, _>(mask_n, &mut ());
            if let Err(e) = store_kzh3_srs(&mask_srs, mask_n) {
              println!("  Warning: could not store mask SRS for size {}: {}", mask_n, e);
            }
            mask_srs
          };
          srs_map.insert(mask_n, Arc::new(mask_srs));
        }
      }
    }
    let srs_map = Arc::new(srs_map);

    let mask_committer: Option<Arc<dyn MaskCommitter<F>>> = if zk_mode {
      Some(Arc::new(KZH3MaskCommitter::<PairingType>::new(srs_map.clone())))
    } else {
      None
    };

    let h_point = if zk_mode {
      println!("ZK mode enabled: generating hiding generator h");
      use zk_torch_2::crypto::polycommit::PairingTrait;
      use ark_std::UniformRand;
      let mut rng = ark_std::rand::thread_rng();
      Some(<PairingType as PairingTrait>::G1Affine::rand(&mut rng))
    } else {
      None
    };

    let kzh3 = KZH3CommitKey::<PairingType> { srs_map: srs_map.clone(), h: h_point };
    let sparse_kzh3 = SparseKZH3CommitKey::<PairingType> { srs_map: srs_map.clone(), h: h_point };
    let dense_vk = KZH3VerifierKey { srs_map: srs_map.clone(), h: h_point };
    let sparse_vk = SparseKZH3VerifierKey { srs_map: srs_map.clone(), h: h_point };

    println!("Committing...");
    let t0 = std::time::Instant::now();
    dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &kzh3, &sparse_kzh3, &init, &mut dense_commitments, &mut sparse_commitments, &mut factored_commitments, &mut timing,
    );
    println!("  commit time: {:.3?}", t0.elapsed());

    println!("Proving{}...", if zk_mode { " (ZK)" } else { "" });
    let t0 = std::time::Instant::now();
    let (sc_proofs, op_proofs, range_proof, two_pow_proof, reducer_proofs, zk_proof) =
      dag.prove_zk::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
        &kzh3, &sparse_kzh3, &init, &presplit_sparse,
        &dense_commitments, &sparse_commitments, &factored_commitments, &mut transcript, &mut timing,
        zk_mode, mask_committer.clone(),
      );
    let prove_time = t0.elapsed();
    println!("  prove time: {:.3?}", prove_time);

    init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));

    println!("Verifying{}...", if zk_mode { " (ZK)" } else { "" });
    let mut vt = Transcript::new(b"zkml");
    let t0 = std::time::Instant::now();
    let verified = dag.verify_zk::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &sc_proofs, &op_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
      &init, &dense_vk, &sparse_vk, &dense_commitments, &sparse_commitments, &factored_commitments, &mut vt,
      zk_mode, &zk_proof, mask_committer.clone(),
    );
    let verify_time = t0.elapsed();
    println!("  verify time: {:.3?}", verify_time);
    println!("verified: {:?}", verified);
  } else {
    println!("Usage: gae <config.yaml> <data_dir> [dataset_name]");
    println!("  data_dir should contain raw/gae_{{dataset}}/ with x.bin, edges, weights/");
    println!("  dataset_name defaults to 'cora'");
  }
}

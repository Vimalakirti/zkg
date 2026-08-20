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

use plonky2::{timed, util::timing::TimingTree};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use zk_torch_2::{
  crypto::polycommit::kzh3::{setup_kzh3_srs, KZH3Commit, KZH3CommitKey, KZH3Commitment, KZH3MaskCommitter, KZH3VerifierKey},
  crypto::polycommit::sparse_kzh3::{SparseKZH3Commit, SparseKZH3CommitKey, SparseKZH3VerifierKey},
  crypto::srs_storage::{load_kzh3_srs, store_kzh3_srs},
  crypto::MaskCommitter,
  dag::{gnn::graph_sage, DagBuilder, DataType, Role, Witness},
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

/// Build mean-normalized adjacency from an edge list.
/// No self-loops are added (SAGEConv handles self-connection via separate W_self).
///
/// Edge direction is transposed so the SpMV z[i] = sum_j A[i,j]*Y[j] aggregates
/// incoming neighbors, matching PyTorch SAGEConv's message-passing convention:
///   For original edge (src→dst), we store A[dst, src] = round(2^sf / in_degree(dst)).
///   Then z[dst] = sum_src A[dst,src]*Y[src] = mean(Y[src]) over incoming edges.
fn build_adjacency_from_edges(
  edge_src: &[i32],
  edge_dst: &[i32],
  num_nodes: usize,
  sf_a: usize,
) -> Witness<F> {
  let num_nodes_log = log2_ceil(num_nodes) as usize;

  // Collect unique edges and transpose: (src→dst) becomes (dst, src) in the matrix
  let mut edge_set = HashSet::new();
  for i in 0..edge_src.len() {
    edge_set.insert((edge_dst[i] as usize, edge_src[i] as usize));
  }
  let edges: Vec<(usize, usize)> = edge_set.into_iter().collect();

  // Compute row-degree (= in-degree in original graph)
  let mut row_degree = vec![0usize; num_nodes];
  for &(row, _col) in &edges {
    row_degree[row] += 1;
  }

  // Mean normalization: weight(row, col) = round(2^sf / row_degree(row))
  let scale = (1u64 << sf_a) as f64;
  let weighted_edges: Vec<(usize, usize, u32)> = edges
    .iter()
    .filter(|&&(row, _)| row_degree[row] > 0)
    .map(|&(row, col)| {
      let w = (scale / row_degree[row] as f64).round() as u32;
      (row, col, w)
    })
    .collect();

  // Build SelectionPolynomial
  let binary_edges: Vec<(usize, usize)> = weighted_edges.iter().map(|&(r, c, _)| (r, c)).collect();
  let selection_poly = SelectionPolynomial::<F>::new(num_nodes_log, num_nodes_log, binary_edges);
  let mut sparse_poly = selection_poly.to_sparse();
  for &(row, col, weight) in &weighted_edges {
    let index = row + col * (1 << num_nodes_log);
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

  // Parse arguments: <config.yaml> <data_dir> [dataset_name] [weights_name]
  // data_dir should contain: raw/{dataset}/ (data) and raw/graphsage_{weights_name}/ (weights)
  // weights_name defaults to dataset_name (for subgraphs, use the original dataset name)
  let args: Vec<String> = std::env::args().collect();
  let data_dir = if args.len() >= 3 { Some(args[2].clone()) } else { None };
  let dataset_name = if args.len() >= 4 { args[3].clone() } else { "cora".to_string() };
  let weights_name = if args.len() >= 5 && !args[4].starts_with("--") { args[4].clone() } else { dataset_name.clone() };
  let zk_mode = args.iter().any(|a| a == "--zk");
  let factor_capacity = args
    .iter()
    .position(|a| a == "--factor-capacity")
    .map(|i| {
      args
        .get(i + 1)
        .expect("--factor-capacity requires an integer")
        .parse::<usize>()
        .expect("invalid --factor-capacity value")
    });

  if let Some(ref data_dir) = data_dir {
    println!("=== GraphSage Inference on {} (weights: {}) ===", dataset_name, weights_name);

    let dataset_path = Path::new(data_dir).join("raw").join(&dataset_name);
    let weights_path = Path::new(data_dir).join("raw").join(format!("graphsage_{}", weights_name));

    // Load dataset metadata
    let meta_str = std::fs::read_to_string(dataset_path.join("meta.json"))
      .expect("Failed to read dataset meta.json");
    let meta: serde_json::Value = serde_json::from_str(&meta_str).unwrap();
    let num_nodes = meta["num_nodes"].as_u64().unwrap() as usize;
    let num_features = meta["num_features"].as_u64().unwrap() as usize;
    let num_classes = meta["num_classes"].as_u64().unwrap() as usize;

    println!("  Nodes: {}, Features: {}, Classes: {}", num_nodes, num_features, num_classes);

    let num_nodes_log = log2_ceil(num_nodes) as usize;
    let num_nodes_pad = 1usize << num_nodes_log;

    // Load node features (set shape to padded so all ops see consistent dimensions)
    let mut input = load_node_features::<F>(
      &dataset_path.join("x.bin"), num_nodes, num_features,
    );
    input.shape[0] = num_nodes_pad;

    // Load edge list and build mean-normalized adjacency (D^{-1}A, no self-loops)
    let edge_src = read_i32_bin(&dataset_path.join("edge_src.bin"));
    let edge_dst = read_i32_bin(&dataset_path.join("edge_dst.bin"));
    let sf_a = *SF_LOG;
    let mut adjacency = build_adjacency_from_edges(&edge_src, &edge_dst, num_nodes, sf_a);
    println!("  Edges (no self-loops): loaded {} raw edges", edge_src.len());

    let num_shares = 2;
    if let Some(capacity) = factor_capacity {
      adjacency
        .additive_factorize_with_capacity(num_shares, capacity)
        .expect("Adjacency factorization or public-capacity padding failed");
    } else {
      adjacency.additive_factorize(num_shares).expect("Adjacency factorization failed");
    }
    let af = adjacency.additive_factored.as_ref().unwrap();
    println!("  Adjacency decomposed: {} terms, {} shares, chunk_sizes={:?}, public_capacity={:?}",
      af.terms.len(), af.chunk_sizes.len(), af.chunk_sizes, factor_capacity);

    // Load labels and masks
    let labels = read_i32_bin(&dataset_path.join("y.bin"));
    let train_mask = read_u8_bin(&dataset_path.join("train_mask.bin"));
    let val_mask = read_u8_bin(&dataset_path.join("val_mask.bin"));
    let test_mask = read_u8_bin(&dataset_path.join("test_mask.bin"));

    // Load weight matrices (PyTorch stores as (out, in), we transpose to (in, out))
    // Read hidden size from weights meta.json if available, otherwise default to 16
    let hidden = {
      let wmeta_path = weights_path.join("meta.json");
      if wmeta_path.exists() {
        let wmeta_str = std::fs::read_to_string(&wmeta_path).unwrap();
        let wmeta: serde_json::Value = serde_json::from_str(&wmeta_str).unwrap();
        wmeta["hidden"].as_u64().unwrap_or(16) as usize
      } else {
        16
      }
    };

    // Layer 1: W_neighbor (lin_l): (hidden, num_features) → (num_features, hidden)
    let w1_neighbor = load_weight_matrix::<F>(
      &weights_path.join("conv1.lin_l.weight.bin"), hidden, num_features,
    );
    // Layer 1: W_self (lin_r): (hidden, num_features) → (num_features, hidden)
    let w1_self = load_weight_matrix::<F>(
      &weights_path.join("conv1.lin_r.weight.bin"), hidden, num_features,
    );
    // Layer 1: bias
    let b1 = load_bias::<F>(
      &weights_path.join("conv1.lin_l.bias.bin"), hidden,
    );

    // Layer 2: W_neighbor (lin_l): (num_classes, hidden) → (hidden, num_classes)
    let w2_neighbor = load_weight_matrix::<F>(
      &weights_path.join("conv2.lin_l.weight.bin"), num_classes, hidden,
    );
    // Layer 2: W_self (lin_r): (num_classes, hidden) → (hidden, num_classes)
    let w2_self = load_weight_matrix::<F>(
      &weights_path.join("conv2.lin_r.weight.bin"), num_classes, hidden,
    );
    // Layer 2: bias
    let b2 = load_bias::<F>(
      &weights_path.join("conv2.lin_l.bias.bin"), num_classes,
    );

    let weights_self = vec![w1_self, w2_self];
    let weights_neighbor = vec![w1_neighbor, w2_neighbor];
    let biases = vec![b1, b2];

    println!("  Loaded weights: layer1 self({},{}) neighbor({},{}), layer2 self({},{}) neighbor({},{})",
      num_features, hidden, num_features, hidden,
      hidden, num_classes, hidden, num_classes);

    // --- Build and run DAG ---
    let mut g = DagBuilder::new();
    let x = g.input(vec![num_nodes_pad, num_features], DataType::Float);
    let output = g.pipe(
      &[x],
      graph_sage(adjacency, weights_self, weights_neighbor, biases),
    )[0];

    println!("Compiling DAG...");
    let (dag, mut init) = g.compile();

    println!("Running forward pass...");
    let presplit_sparse = dag.run(&mut init, &vec![(x, input)]);
    println!("  Output shape: {:?}", init[output][0].shape);

    // --- Compute accuracy ---
    let output_witness = &init[output][0];
    let (predictions, train_acc, val_acc, test_acc) = compute_accuracy(
      output_witness, &labels, num_nodes,
      Some((&train_mask, &val_mask, &test_mask)),
    );

    println!("\n=== Accuracy Results (GraphSage on {}) ===", dataset_name);
    println!("  Train: {:.4} ({}/{})",
      train_acc,
      train_mask.iter().zip(predictions.iter().zip(labels.iter()))
        .filter(|(&m, (&p, &l))| m != 0 && p == l as usize).count(),
      train_mask.iter().filter(|&&m| m != 0).count(),
    );
    println!("  Val:   {:.4} ({}/{})",
      val_acc,
      val_mask.iter().zip(predictions.iter().zip(labels.iter()))
        .filter(|(&m, (&p, &l))| m != 0 && p == l as usize).count(),
      val_mask.iter().filter(|&&m| m != 0).count(),
    );
    println!("  Test:  {:.4} ({}/{})",
      test_acc,
      test_mask.iter().zip(predictions.iter().zip(labels.iter()))
        .filter(|(&m, (&p, &l))| m != 0 && p == l as usize).count(),
      test_mask.iter().filter(|&&m| m != 0).count(),
    );

    // Print logits for test nodes (for AUC computation)
    // Format: LOGIT|node_idx|label|logit_0|logit_1
    for i in 0..num_nodes {
      if test_mask[i] != 0 {
        let logit_0 = zk_torch_2::util::arith::f_to_int(output_witness.get(&[i, 0]));
        let logit_1 = zk_torch_2::util::arith::f_to_int(output_witness.get(&[i, 1]));
        println!("LOGIT|{}|{}|{}|{}", i, labels[i], logit_0, logit_1);
      }
    }

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
        // Try to store, but don't fail if disk is full
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
    println!("  prove time: {:.3?}", t0.elapsed());

    zk_torch_2::util::serialization::measure_total_proof_size(
      &sc_proofs, &op_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
    );

    let mut zk_extra: usize = 0;
    for opt in sc_proofs.iter() {
      if let Some((proofs, _)) = opt {
        for p in proofs { zk_extra += p.zk_extra_bytes(); }
      }
    }
    for opt in reducer_proofs.iter() {
      if let Some(proofs) = opt {
        for p in proofs { zk_extra += p.zk_extra_bytes(); }
      }
    }
    if zk_extra > 0 {
      println!("ZK extra proof size: {}", zk_torch_2::util::serialization::format_file_size(zk_extra as u64));
    }

    init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));

    println!("Verifying{}...", if zk_mode { " (ZK)" } else { "" });
    let mut vt = Transcript::new(b"zkml");
    let t0 = std::time::Instant::now();
    let verified = dag.verify_zk::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &sc_proofs, &op_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
      &init, &dense_vk, &sparse_vk, &dense_commitments, &sparse_commitments, &factored_commitments, &mut vt,
      zk_mode, &zk_proof, mask_committer.clone(),
    );
    println!("  verify time: {:.3?}", t0.elapsed());
    println!("verified: {:?}", verified);
  } else {
    println!("Usage: graphsage <config.yaml> <data_dir> [dataset_name] [weights_name] [--zk] [--factor-capacity T]");
    println!("  data_dir should contain raw/{{dataset}}/ and raw/graphsage_{{dataset}}/");
    println!("  dataset_name defaults to 'cora'");
  }
}

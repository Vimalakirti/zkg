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
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use zk_torch_2::{
  crypto::polycommit::kzh3::{setup_kzh3_srs, KZH3Commit, KZH3CommitKey, KZH3Commitment, KZH3MaskCommitter, KZH3VerifierKey},
  crypto::polycommit::sparse_kzh3::{SparseKZH3Commit, SparseKZH3CommitKey, SparseKZH3VerifierKey},
  crypto::srs_storage::{load_kzh3_srs, store_kzh3_srs},
  crypto::MaskCommitter,
  dag::{gnn::gat_with_offset, DagBuilder, DataType, Role, Witness},
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

/// Build incidence matrices S_src and S_dst from an edge list.
///  - S_src: (|V|, |E|) -- S_src[v, e] = 1 if v is source of edge e
///  - S_dst: (|V|, |E|) -- S_dst[v, e] = 1 if v is destination of edge e
///
/// Padded edges (beyond num_edges) are mapped to a dummy node (num_nodes)
/// to avoid division-by-zero in softmax normalization.
fn build_incidence_matrices(
  edge_src: &[i32],
  edge_dst: &[i32],
  num_nodes: usize,
  num_edges: usize,
) -> (Witness<F>, Witness<F>) {
  let num_nodes_log = log2_ceil(num_nodes) as usize;
  let num_edges_log = log2_ceil(num_edges) as usize;
  let num_edges_pad = 1 << num_edges_log;
  let num_nodes_pad = 1 << num_nodes_log;

  // Use a dummy node for padded edges (any index in [num_nodes, num_nodes_pad))
  let dummy_node = num_nodes;
  assert!(dummy_node < num_nodes_pad, "No room for dummy node in padded node space");

  // S_src: (|V|, |E|) -- S_src[v, e] = 1 if v is source of edge e
  let mut s_src_sel: Vec<(usize, usize)> = (0..num_edges)
    .map(|e| (edge_src[e] as usize, e))
    .collect();
  // Map padded edges to dummy node
  for e in num_edges..num_edges_pad {
    s_src_sel.push((dummy_node, e));
  }
  let s_src_poly = SelectionPolynomial::<F>::new(num_nodes_log, num_edges_log, s_src_sel);
  let s_src_witness = Witness::new_sparse(
    vec![num_nodes_pad, num_edges_pad],
    s_src_poly.to_sparse(),
    DataType::Float,
    0,
    Role::Constant,
  );

  // S_dst: (|V|, |E|) -- S_dst[v, e] = 1 if v is destination of edge e
  let mut s_dst_sel: Vec<(usize, usize)> = (0..num_edges)
    .map(|e| (edge_dst[e] as usize, e))
    .collect();
  // Map padded edges to dummy node
  for e in num_edges..num_edges_pad {
    s_dst_sel.push((dummy_node, e));
  }
  let s_dst_poly = SelectionPolynomial::<F>::new(num_nodes_log, num_edges_log, s_dst_sel);
  let s_dst_witness = Witness::new_sparse(
    vec![num_nodes_pad, num_edges_pad],
    s_dst_poly.to_sparse(),
    DataType::Float,
    0,
    Role::Constant,
  );

  (s_src_witness, s_dst_witness)
}

/// Load a combined weight file and split into per-head weight matrices.
/// The file contains (total_out, in_features) in PyTorch row-major format,
/// where total_out = num_heads * head_dim.
/// Returns num_heads Witness matrices each of shape (in_features, head_dim).
fn load_per_head_weights(
  path: &Path,
  num_heads: usize,
  head_dim: usize,
  in_features: usize,
) -> Vec<Witness<F>> {
  let data_f32 = read_f32_bin(path);
  let total_out = num_heads * head_dim;
  assert_eq!(
    data_f32.len(),
    total_out * in_features,
    "Expected {} values for weight ({}x{}), got {}",
    total_out * in_features,
    total_out,
    in_features,
    data_f32.len()
  );

  let sf = *zk_torch_2::SF_FLOAT;
  let row_pad = in_features.next_power_of_two();
  let col_pad = head_dim.next_power_of_two();

  let mut heads = Vec::with_capacity(num_heads);
  for h in 0..num_heads {
    let mut field_data = vec![<F as CryptoField>::zero(); row_pad * col_pad];
    // PyTorch layout: (total_out, in_features) row-major
    // Row index in PyTorch = h * head_dim + d, col index = i
    // Witness: (in_features, head_dim) column-major: index = row + col * row_pad
    for d in 0..head_dim {
      for i in 0..in_features {
        let pt_idx = (h * head_dim + d) * in_features + i;
        let val = data_f32[pt_idx];
        let y = (val * sf).round();
        let y = y.clamp(-(1i64 << 30) as f32, (1i64 << 30) as f32);
        let field_val = if y < 0.0 {
          <F as CryptoField>::zero() - F::from((-y) as u64)
        } else {
          F::from(y as u64)
        };
        field_data[i + d * row_pad] = field_val;
      }
    }
    heads.push(Witness::new(
      vec![in_features, head_dim],
      field_data,
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
  }
  heads
}

/// Load attention vector file and split into per-head vectors.
/// The file contains (1, num_heads, head_dim) = num_heads * head_dim floats.
/// Returns num_heads Witness vectors each of shape (head_dim, 1).
fn load_per_head_attention(
  path: &Path,
  num_heads: usize,
  head_dim: usize,
) -> Vec<Witness<F>> {
  let data_f32 = read_f32_bin(path);
  assert_eq!(
    data_f32.len(),
    num_heads * head_dim,
    "Expected {} attention values, got {}",
    num_heads * head_dim,
    data_f32.len()
  );

  let mut heads = Vec::with_capacity(num_heads);
  for h in 0..num_heads {
    let slice = &data_f32[h * head_dim..(h + 1) * head_dim];
    heads.push(load_attention_vector::<F>(slice));
  }
  heads
}

fn main() {
  let mut timing = TimingTree::default();
  env_logger::init();

  // Parse arguments: <config.yaml> <data_dir> [dataset_name]
  let args: Vec<String> = std::env::args().collect();
  let data_dir = if args.len() >= 3 { Some(args[2].clone()) } else { None };
  let dataset_name = if args.len() >= 4 { args[3].clone() } else { "cora".to_string() };
  let zk_mode = args.iter().any(|a| a == "--zk");
  let r_offset_log = args.iter().position(|a| a == "--offset")
    .map(|i| args[i + 1].parse::<usize>().expect("--offset requires a number"))
    .unwrap_or(42);

  if let Some(ref data_dir) = data_dir {
    println!("=== GAT Inference on {} ===", dataset_name);

    let dataset_path = Path::new(data_dir).join("raw").join(&dataset_name);
    let weights_path = Path::new(data_dir).join("raw").join(format!("gat_{}", dataset_name));

    // Load dataset metadata
    let meta_str = std::fs::read_to_string(dataset_path.join("meta.json"))
      .expect("Failed to read dataset meta.json");
    let meta: serde_json::Value = serde_json::from_str(&meta_str).unwrap();
    let num_nodes = meta["num_nodes"].as_u64().unwrap() as usize;
    let num_features = meta["num_features"].as_u64().unwrap() as usize;
    let num_classes = meta["num_classes"].as_u64().unwrap() as usize;

    println!("  Nodes: {}, Features: {}, Classes: {}", num_nodes, num_features, num_classes);

    let num_nodes_pad = 1usize << (log2_ceil(num_nodes) as usize);

    // Load node features (set shape to padded so all ops see consistent dimensions)
    let mut input = load_node_features::<F>(
      &dataset_path.join("x.bin"), num_nodes, num_features,
    );
    input.shape[0] = num_nodes_pad;

    // Load edge list (no self-loops needed for GAT)
    let edge_src = read_i32_bin(&dataset_path.join("edge_src.bin"));
    let edge_dst = read_i32_bin(&dataset_path.join("edge_dst.bin"));
    let num_edges = edge_src.len();
    println!("  Edges: {} directed edges", num_edges);

    // Build incidence matrices
    let (mut s_src_witness, mut s_dst_witness) = build_incidence_matrices(
      &edge_src, &edge_dst, num_nodes, num_edges,
    );

    // Twist and Shout: decompose one-hot-per-column incidence matrices by
    // transposing (making them one-hot-per-row, giving t=1) and factoring the
    // node variables into num_shares groups. Each factor polynomial has
    // (log_E + chunk_size) variables — much smaller than the full (log_N + log_E).
    let num_shares = 2;
    s_src_witness.selection_factorize(num_shares).expect("S_src selection_decompose failed");
    s_dst_witness.selection_factorize(num_shares).expect("S_dst selection_decompose failed");
    let s_src_af = s_src_witness.additive_factored.as_ref().unwrap();
    let s_dst_af = s_dst_witness.additive_factored.as_ref().unwrap();
    println!("  S_src: t={} terms, {} factors (chunk sizes {:?})",
      s_src_af.terms.len(), s_src_af.chunk_sizes.len(), s_src_af.chunk_sizes);
    println!("  S_dst: t={} terms, {} factors (chunk sizes {:?})",
      s_dst_af.terms.len(), s_dst_af.chunk_sizes.len(), s_dst_af.chunk_sizes);

    // Load labels and masks
    let labels = read_i32_bin(&dataset_path.join("y.bin"));
    let train_mask = read_u8_bin(&dataset_path.join("train_mask.bin"));
    let val_mask = read_u8_bin(&dataset_path.join("val_mask.bin"));
    let test_mask = read_u8_bin(&dataset_path.join("test_mask.bin"));

    // GAT hyperparameters
    let head_dim_1 = 8;
    let num_heads_1 = 4;
    let hidden = num_heads_1 * head_dim_1; // 32
    let num_heads_2 = 1;
    let head_dim_2 = num_classes;

    // Layer 1 weights: conv1.lin.weight (32, num_features) -> 4 heads of (num_features, 8)
    let w1_heads = load_per_head_weights(
      &weights_path.join("conv1.lin.weight.bin"),
      num_heads_1, head_dim_1, num_features,
    );
    // Layer 1 attention: conv1.att_src (1, 4, 8) -> 4 heads of (8, 1)
    let attn_src_1 = load_per_head_attention(
      &weights_path.join("conv1.att_src.bin"),
      num_heads_1, head_dim_1,
    );
    let attn_dst_1 = load_per_head_attention(
      &weights_path.join("conv1.att_dst.bin"),
      num_heads_1, head_dim_1,
    );

    // Layer 2 weights: conv2.lin.weight (num_classes, 32) -> 1 head of (32, num_classes)
    let w2_heads = load_per_head_weights(
      &weights_path.join("conv2.lin.weight.bin"),
      num_heads_2, head_dim_2, hidden,
    );
    // Layer 2 attention: conv2.att_src (1, 1, num_classes) -> 1 head of (num_classes, 1)
    let attn_src_2 = load_per_head_attention(
      &weights_path.join("conv2.att_src.bin"),
      num_heads_2, head_dim_2,
    );
    let attn_dst_2 = load_per_head_attention(
      &weights_path.join("conv2.att_dst.bin"),
      num_heads_2, head_dim_2,
    );

    let weights = vec![w1_heads, w2_heads];
    let attn_src_vecs = vec![attn_src_1, attn_src_2];
    let attn_dst_vecs = vec![attn_dst_1, attn_dst_2];

    println!("  Loaded weights: layer1 ({},{})x{} heads, layer2 ({},{})x{} head",
      num_features, head_dim_1, num_heads_1,
      hidden, head_dim_2, num_heads_2,
    );

    // --- Build and run DAG ---
    let mut g = DagBuilder::new();
    let x = g.input(vec![num_nodes_pad, num_features], DataType::Float);
    let output = g.pipe(
      &[x],
      gat_with_offset(s_src_witness, s_dst_witness, weights, attn_src_vecs, attn_dst_vecs, r_offset_log),
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

    println!("\n=== Accuracy Results (GAT on {}) ===", dataset_name);
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
        store_kzh3_srs(&size_srs, size).expect("Failed to store SRS");
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
            store_kzh3_srs(&mask_srs, mask_n).expect("Failed to store mask SRS");
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

    // Measure ZK overhead (extra bytes from mask fields)
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
    println!("Usage: gat <config.yaml> <data_dir> [dataset_name]");
    println!("  data_dir should contain raw/{{dataset}}/ and raw/gat_{{dataset}}/");
    println!("  dataset_name defaults to 'cora'");
  }
}

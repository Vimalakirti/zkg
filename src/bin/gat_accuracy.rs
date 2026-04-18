/// Lightweight GAT accuracy test — forward pass only, no proving.
/// Tests quantized inference accuracy across different scale factors.
///
/// Usage: cargo run --bin gat_accuracy --release -- config.yaml pyg/weights [dataset] [sf_log]

#[cfg(all(feature = "arkworks", feature = "bn254"))]
use ark_bn254::Fr as F;
#[cfg(all(feature = "arkworks", feature = "bls12_381"))]
use ark_bls12_381::Fr as F;

use std::path::Path;
use zk_torch_2::{
  dag::{gnn::gat, DagBuilder, DataType, Role, Witness},
  util::arith::log2_ceil,
  util::data_loader::*,
  util::poly::CryptoField,
  SF_LOG, SF_FLOAT,
};

/// Build incidence matrices (same as in gat.rs)
fn build_incidence_matrices(
  edge_src: &[i32],
  edge_dst: &[i32],
  num_nodes: usize,
  num_edges: usize,
) -> (Witness<F>, Witness<F>) {
  use zk_torch_2::util::poly::SelectionPolynomial;

  let num_nodes_log = log2_ceil(num_nodes) as usize;
  let num_edges_log = log2_ceil(num_edges) as usize;
  let num_edges_pad = 1 << num_edges_log;
  let num_nodes_pad = 1 << num_nodes_log;
  let dummy_node = num_nodes;
  assert!(dummy_node < num_nodes_pad);

  let mut s_src_sel: Vec<(usize, usize)> = (0..num_edges)
    .map(|e| (edge_src[e] as usize, e))
    .collect();
  for e in num_edges..num_edges_pad {
    s_src_sel.push((dummy_node, e));
  }
  let s_src_poly = SelectionPolynomial::new(num_nodes_log, num_edges_log, s_src_sel);
  let s_src = Witness::new_sparse(
    vec![num_nodes_pad, num_edges_pad],
    s_src_poly.to_sparse(),
    DataType::Float, 0, Role::Constant,
  );

  let mut s_dst_sel: Vec<(usize, usize)> = (0..num_edges)
    .map(|e| (edge_dst[e] as usize, e))
    .collect();
  for e in num_edges..num_edges_pad {
    s_dst_sel.push((dummy_node, e));
  }
  let s_dst_poly = SelectionPolynomial::new(num_nodes_log, num_edges_log, s_dst_sel);
  let s_dst = Witness::new_sparse(
    vec![num_nodes_pad, num_edges_pad],
    s_dst_poly.to_sparse(),
    DataType::Float, 0, Role::Constant,
  );

  (s_src, s_dst)
}

fn load_per_head_weights(
  path: &Path, num_heads: usize, head_dim: usize, in_features: usize,
) -> Vec<Witness<F>> {
  let data_f32 = read_f32_bin(path);
  let total_out = num_heads * head_dim;
  assert_eq!(data_f32.len(), total_out * in_features);

  let sf = *SF_FLOAT;
  let row_pad = in_features.next_power_of_two();
  let col_pad = head_dim.next_power_of_two();

  let mut heads = Vec::with_capacity(num_heads);
  for h in 0..num_heads {
    let mut field_data = vec![<F as CryptoField>::zero(); row_pad * col_pad];
    for d in 0..head_dim {
      for i in 0..in_features {
        let pt_idx = (h * head_dim + d) * in_features + i;
        let val = data_f32[pt_idx];
        let y = (val * sf).round().clamp(-(1i64 << 30) as f32, (1i64 << 30) as f32);
        field_data[i + d * row_pad] = if y < 0.0 {
          <F as CryptoField>::zero() - F::from((-y) as u64)
        } else {
          F::from(y as u64)
        };
      }
    }
    heads.push(Witness::new(
      vec![in_features, head_dim], field_data,
      DataType::Float, *SF_LOG as usize, Role::Constant,
    ));
  }
  heads
}

fn load_per_head_attention(path: &Path, num_heads: usize, head_dim: usize) -> Vec<Witness<F>> {
  let data_f32 = read_f32_bin(path);
  assert_eq!(data_f32.len(), num_heads * head_dim);
  let mut heads = Vec::with_capacity(num_heads);
  for h in 0..num_heads {
    let slice = &data_f32[h * head_dim..(h + 1) * head_dim];
    heads.push(load_attention_vector::<F>(slice));
  }
  heads
}

fn main() {
  let args: Vec<String> = std::env::args().collect();
  if args.len() < 3 {
    eprintln!("Usage: {} <config.yaml> <data_dir> [dataset] [sf_log_override]", args[0]);
    eprintln!("  dataset: cora (default), citeseer, pubmed");
    eprintln!("  sf_log_override: if provided, prints what sf is being used (config still controls)");
    std::process::exit(1);
  }
  let data_dir = &args[2];
  let dataset_name = if args.len() >= 4 { args[3].clone() } else { "cora".to_string() };

  println!("=== GAT Accuracy Test (forward pass only) ===");
  println!("  Scale factor: 2^{} = {}", *SF_LOG, *SF_FLOAT);
  println!("  Dataset: {}", dataset_name);

  let dataset_path = Path::new(data_dir).join("raw").join(&dataset_name);
  let weights_path = Path::new(data_dir).join("raw").join(format!("gat_{}", dataset_name));

  // Load metadata
  let meta_str = std::fs::read_to_string(dataset_path.join("meta.json"))
    .expect("Failed to read meta.json");
  let meta: serde_json::Value = serde_json::from_str(&meta_str).unwrap();
  let num_nodes = meta["num_nodes"].as_u64().unwrap() as usize;
  let num_features = meta["num_features"].as_u64().unwrap() as usize;
  let num_classes = meta["num_classes"].as_u64().unwrap() as usize;

  println!("  Nodes: {}, Features: {}, Classes: {}", num_nodes, num_features, num_classes);

  let num_nodes_pad = 1usize << (log2_ceil(num_nodes) as usize);

  // Load data
  let mut input = load_node_features::<F>(&dataset_path.join("x.bin"), num_nodes, num_features);
  input.shape[0] = num_nodes_pad;

  let edge_src = read_i32_bin(&dataset_path.join("edge_src.bin"));
  let edge_dst = read_i32_bin(&dataset_path.join("edge_dst.bin"));
  let num_edges = edge_src.len();
  println!("  Edges: {}", num_edges);

  let (mut s_src, mut s_dst) = build_incidence_matrices(&edge_src, &edge_dst, num_nodes, num_edges);
  let num_shares = 2;
  s_src.additive_factorize(num_shares).expect("S_src factorization failed");
  s_dst.additive_factorize(num_shares).expect("S_dst factorization failed");

  let labels = read_i32_bin(&dataset_path.join("y.bin"));
  let train_mask = read_u8_bin(&dataset_path.join("train_mask.bin"));
  let val_mask = read_u8_bin(&dataset_path.join("val_mask.bin"));
  let test_mask = read_u8_bin(&dataset_path.join("test_mask.bin"));

  // GAT hyperparameters (same as gat.rs)
  let head_dim_1 = 8;
  let num_heads_1 = 4;
  let hidden = num_heads_1 * head_dim_1;
  let num_heads_2 = 1;
  let head_dim_2 = num_classes;

  let w1_heads = load_per_head_weights(
    &weights_path.join("conv1.lin.weight.bin"), num_heads_1, head_dim_1, num_features);
  let attn_src_1 = load_per_head_attention(
    &weights_path.join("conv1.att_src.bin"), num_heads_1, head_dim_1);
  let attn_dst_1 = load_per_head_attention(
    &weights_path.join("conv1.att_dst.bin"), num_heads_1, head_dim_1);
  let w2_heads = load_per_head_weights(
    &weights_path.join("conv2.lin.weight.bin"), num_heads_2, head_dim_2, hidden);
  let attn_src_2 = load_per_head_attention(
    &weights_path.join("conv2.att_src.bin"), num_heads_2, head_dim_2);
  let attn_dst_2 = load_per_head_attention(
    &weights_path.join("conv2.att_dst.bin"), num_heads_2, head_dim_2);

  let weights = vec![w1_heads, w2_heads];
  let attn_src_vecs = vec![attn_src_1, attn_src_2];
  let attn_dst_vecs = vec![attn_dst_1, attn_dst_2];

  // Build DAG, run forward pass only
  let mut g = DagBuilder::new();
  let x = g.input(vec![num_nodes_pad, num_features], DataType::Float);
  let output = g.pipe(
    &[x],
    gat(s_src, s_dst, weights, attn_src_vecs, attn_dst_vecs),
  )[0];

  println!("Compiling DAG...");
  let (dag, mut init) = g.compile();

  println!("Running forward pass...");
  let _presplit = dag.run(&mut init, &vec![(x, input)]);

  // Compute accuracy
  let output_witness = &init[output][0];
  let (_predictions, train_acc, val_acc, test_acc) = compute_accuracy(
    output_witness, &labels, num_nodes,
    Some((&train_mask, &val_mask, &test_mask)),
  );

  println!("\n=== Results (GAT on {}, sf=2^{}) ===", dataset_name, *SF_LOG);
  println!("  Train accuracy: {:.4}", train_acc);
  println!("  Val accuracy:   {:.4}", val_acc);
  println!("  Test accuracy:  {:.4}", test_acc);

  // Reference: PyTorch float accuracy from results.json
  let results_path = Path::new(data_dir).join("results.json");
  if results_path.exists() {
    let results_str = std::fs::read_to_string(&results_path).unwrap();
    let results: serde_json::Value = serde_json::from_str(&results_str).unwrap();
    let key = format!("gat_{}", dataset_name.replace("-", "_"));
    if let Some(entry) = results.get(&key) {
      if let Some(ref_acc) = entry.get("mean_acc") {
        println!("\n  PyTorch float test acc: {:.4}", ref_acc.as_f64().unwrap());
        println!("  Accuracy drop:         {:.4}", ref_acc.as_f64().unwrap() - test_acc);
      }
    }
  }
}

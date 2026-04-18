// Graph-level classification binary for kernel benchmark datasets.
// Supports GCN, GraphSAGE, GIN on MUTAG, PROTEINS, IMDB-BINARY, REDDIT-BINARY.
//
// Usage: kernel <config.yaml> <data_dir> <model> <dataset> [graph_index]
//   model: gcn | graphsage | gin
//   data_dir: path to kernel_raw/

// Field type selection
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
  dag::{DagBuilder, DataType, Role, Witness},
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

/// Build symmetric-normalized adjacency for GCN (with self-loops).
fn build_gcn_adjacency(
  edge_src: &[i32], edge_dst: &[i32], num_nodes: usize, sf_a: usize,
) -> Witness<F> {
  let num_nodes_log = log2_ceil(num_nodes) as usize;
  let mut edge_set = HashSet::new();
  for i in 0..num_nodes { edge_set.insert((i, i)); }
  for i in 0..edge_src.len() {
    edge_set.insert((edge_src[i] as usize, edge_dst[i] as usize));
  }
  let edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
  let mut degree = vec![0usize; num_nodes];
  for &(src, _) in &edges { degree[src] += 1; }
  let scale = (1u64 << sf_a) as f64;
  let weighted_edges: Vec<(usize, usize, u32)> = edges.iter().map(|&(src, dst)| {
    let w = (scale / (degree[src] as f64 * degree[dst] as f64).sqrt()).round() as u32;
    (src, dst, w)
  }).collect();
  let binary_edges: Vec<(usize, usize)> = weighted_edges.iter().map(|&(s, d, _)| (s, d)).collect();
  let selection_poly = SelectionPolynomial::<F>::new(num_nodes_log, num_nodes_log, binary_edges);
  let mut sparse_poly = selection_poly.to_sparse();
  for &(src, dst, weight) in &weighted_edges {
    let index = src + dst * (1 << num_nodes_log);
    sparse_poly.evaluations.insert(index, <F as CryptoField>::from_u32(weight));
  }
  let num_nodes_pad = 1 << num_nodes_log;
  Witness::new_sparse(
    vec![num_nodes_pad, num_nodes_pad], sparse_poly,
    DataType::Float, sf_a, Role::Constant,
  )
}

/// Build mean-normalized adjacency for GraphSAGE (no self-loops).
fn build_sage_adjacency(
  edge_src: &[i32], edge_dst: &[i32], num_nodes: usize, sf_a: usize,
) -> Witness<F> {
  let num_nodes_log = log2_ceil(num_nodes) as usize;
  let mut edge_set = HashSet::new();
  for i in 0..edge_src.len() {
    edge_set.insert((edge_src[i] as usize, edge_dst[i] as usize));
  }
  let edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
  let mut out_degree = vec![0usize; num_nodes];
  for &(src, _) in &edges { out_degree[src] += 1; }
  let scale = (1u64 << sf_a) as f64;
  let weighted_edges: Vec<(usize, usize, u32)> = edges.iter()
    .filter(|&&(src, _)| out_degree[src] > 0)
    .map(|&(src, dst)| {
      let w = (scale / out_degree[src] as f64).round() as u32;
      (src, dst, w)
    }).collect();
  let binary_edges: Vec<(usize, usize)> = weighted_edges.iter().map(|&(s, d, _)| (s, d)).collect();
  let selection_poly = SelectionPolynomial::<F>::new(num_nodes_log, num_nodes_log, binary_edges);
  let mut sparse_poly = selection_poly.to_sparse();
  for &(src, dst, weight) in &weighted_edges {
    let index = src + dst * (1 << num_nodes_log);
    sparse_poly.evaluations.insert(index, <F as CryptoField>::from_u32(weight));
  }
  let num_nodes_pad = 1 << num_nodes_log;
  Witness::new_sparse(
    vec![num_nodes_pad, num_nodes_pad], sparse_poly,
    DataType::Float, sf_a, Role::Constant,
  )
}

/// Build binary adjacency for GIN (with self-loops, sf=0).
fn build_gin_adjacency(
  edge_src: &[i32], edge_dst: &[i32], num_nodes: usize,
) -> Witness<F> {
  let num_nodes_log = log2_ceil(num_nodes) as usize;
  let mut edge_set = HashSet::new();
  for i in 0..num_nodes { edge_set.insert((i, i)); }
  for i in 0..edge_src.len() {
    edge_set.insert((edge_src[i] as usize, edge_dst[i] as usize));
  }
  let edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
  let selection_poly = SelectionPolynomial::<F>::new(num_nodes_log, num_nodes_log, edges);
  let sparse_poly = selection_poly.to_sparse();
  let num_nodes_pad = 1 << num_nodes_log;
  Witness::new_sparse(
    vec![num_nodes_pad, num_nodes_pad], sparse_poly,
    DataType::Int, 0, Role::Constant,
  )
}

/// Build a mean-pooling vector: (1, N_pad) with entries 1/N for first N nodes.
fn build_mean_pool_vec(num_nodes: usize) -> Witness<F> {
  let num_nodes_log = log2_ceil(num_nodes) as usize;
  let num_nodes_pad = 1 << num_nodes_log;
  let sf = *SF_LOG;
  let scale = (1u64 << sf) as f64;
  let val = (scale / num_nodes as f64).round() as u64;
  // Shape (1, N_pad), column-major: index = row + col * 1_pad = col
  let mut data = vec![<F as CryptoField>::zero(); num_nodes_pad];
  for i in 0..num_nodes {
    data[i] = F::from(val);
  }
  Witness::new(
    vec![1, num_nodes_pad], data,
    DataType::Float, sf, Role::Constant,
  )
}

/// Load attention vectors from a flat file (1, num_heads, head_dim) = num_heads*head_dim floats
/// and build the block-diagonal attention matrix (num_heads*head_dim_pad, num_heads) expected by gat_layer.
fn load_attention_block(path: &Path, num_heads: usize, head_dim: usize) -> Witness<F> {
  let data_f32 = read_f32_bin(path);
  assert_eq!(data_f32.len(), num_heads * head_dim);
  let sf = *zk_torch_2::SF_FLOAT;
  let head_dim_pad = head_dim.next_power_of_two();
  let total_rows = num_heads * head_dim_pad;
  let total_cols = num_heads;
  // Block-diagonal: entry at (h*head_dim_pad + d, h) = attn[h*head_dim + d]
  let mut field_data = vec![<F as CryptoField>::zero(); total_rows * total_cols];
  for h in 0..num_heads {
    for d in 0..head_dim {
      let val = data_f32[h * head_dim + d];
      let y = (val * sf).round().clamp(-(1i64 << 30) as f32, (1i64 << 30) as f32);
      let field_val = if y < 0.0 {
        <F as CryptoField>::zero() - F::from((-y) as u64)
      } else {
        F::from(y as u64)
      };
      // Column-major: index = row + col * total_rows
      let row = h * head_dim_pad + d;
      let col = h;
      field_data[row + col * total_rows] = field_val;
    }
  }
  Witness::new(
    vec![num_heads * head_dim, num_heads], field_data,
    DataType::Float, *SF_LOG as usize, Role::Constant,
  )
}

/// Build a sum-pooling vector: (1, N_pad) with entries 1.0 for first N nodes.
fn build_sum_pool_vec(num_nodes: usize) -> Witness<F> {
  let num_nodes_log = log2_ceil(num_nodes) as usize;
  let num_nodes_pad = 1 << num_nodes_log;
  let sf = *SF_LOG;
  let val = 1u64 << sf;
  let mut data = vec![<F as CryptoField>::zero(); num_nodes_pad];
  for i in 0..num_nodes {
    data[i] = F::from(val);
  }
  Witness::new(
    vec![1, num_nodes_pad], data,
    DataType::Float, sf, Role::Constant,
  )
}

/// Run inference (and optionally proving) on a single graph. Returns (correct, prove_ms, verify_ms).
fn run_graph(
  data_dir: &str, model_name: &str, dataset_name: &str, graph_index: usize,
  hidden: usize, num_classes: usize, weights_path: &Path, do_prove: bool, zk_mode: bool,
) -> (bool, f64, f64) {
  let mut timing = TimingTree::default();

  // Load graph
  let graph_path = Path::new(data_dir).join(dataset_name).join(format!("graph_{}", graph_index));
  let meta_str = std::fs::read_to_string(graph_path.join("meta.json"))
    .expect("Failed to read graph meta.json");
  let meta: serde_json::Value = serde_json::from_str(&meta_str).unwrap();
  let num_nodes = meta["num_nodes"].as_u64().unwrap() as usize;
  let num_features = meta["num_features"].as_u64().unwrap() as usize;
  let _num_edges = meta["num_edges"].as_u64().unwrap() as usize;
  let label = meta["label"].as_i64().unwrap() as i32;

  let num_nodes_log = log2_ceil(num_nodes) as usize;
  let num_nodes_pad = 1usize << num_nodes_log;

  // Load node features
  let mut input = load_node_features::<F>(
    &graph_path.join("x.bin"), num_nodes, num_features,
  );
  input.shape[0] = num_nodes_pad;

  // Load edges
  let edge_src = read_i32_bin(&graph_path.join("edge_src.bin"));
  let edge_dst = read_i32_bin(&graph_path.join("edge_dst.bin"));

  // Build DAG based on model
  let mut g = DagBuilder::new();
  let x = g.input(vec![num_nodes_pad, num_features], DataType::Float);

  let output = match model_name {
    "gcn" => {
      let sf_a = *SF_LOG;
      let mut adjacency = build_gcn_adjacency(&edge_src, &edge_dst, num_nodes, sf_a);
      adjacency.additive_factorize(2).expect("GCN adjacency factorization failed");
      let adj_id = g.param(adjacency);

      // 3 GCN conv layers with bias: H = ReLU(A · (H · W) + bias)
      let w1 = load_weight_matrix::<F>(&weights_path.join("conv1.lin.weight.bin"), hidden, num_features);
      let b1 = load_bias::<F>(&weights_path.join("conv1.bias.bin"), hidden);
      let w2 = load_weight_matrix::<F>(&weights_path.join("convs.0.lin.weight.bin"), hidden, hidden);
      let b2 = load_bias::<F>(&weights_path.join("convs.0.bias.bin"), hidden);
      let w3 = load_weight_matrix::<F>(&weights_path.join("convs.1.lin.weight.bin"), hidden, hidden);
      let b3 = load_bias::<F>(&weights_path.join("convs.1.bias.bin"), hidden);

      let mut h = x;
      for (w, b) in [(w1, b1), (w2, b2), (w3, b3)] {
        let w_id = g.param(w);
        let b_id = g.param(b);
        let y = g.einsum("bd,dh->bh".to_string(), vec![h, w_id], true)[0];
        let z = g.spmv(adj_id, y, true, false)[0];
        let z_bias = g.add(z, b_id)[0];
        h = g.relu(z_bias)[0];
      }

      // Global mean pool: (1, N_pad) · (N_pad, hidden) → (1, hidden)
      let pool_vec = build_mean_pool_vec(num_nodes);
      let pool_id = g.param(pool_vec);
      let pooled = g.einsum("bd,dh->bh".to_string(), vec![pool_id, h], true)[0];

      // MLP head: lin1 → ReLU → lin2
      let w_lin1 = load_weight_matrix::<F>(&weights_path.join("lin1.weight.bin"), hidden, hidden);
      let b_lin1 = load_bias::<F>(&weights_path.join("lin1.bias.bin"), hidden);
      let w_lin2 = load_weight_matrix::<F>(&weights_path.join("lin2.weight.bin"), num_classes, hidden);
      let b_lin2 = load_bias::<F>(&weights_path.join("lin2.bias.bin"), num_classes);

      let w_lin1_id = g.param(w_lin1);
      let b_lin1_id = g.param(b_lin1);
      let w_lin2_id = g.param(w_lin2);
      let b_lin2_id = g.param(b_lin2);

      let h1 = g.einsum("bd,dh->bh".to_string(), vec![pooled, w_lin1_id], true)[0];
      let h1_bias = g.add(h1, b_lin1_id)[0];
      let h1_relu = g.relu(h1_bias)[0];
      let h2 = g.einsum("bd,dh->bh".to_string(), vec![h1_relu, w_lin2_id], true)[0];
      let out = g.add(h2, b_lin2_id)[0];
      out
    },

    "graphsage" => {
      let sf_a = *SF_LOG;
      let mut adjacency = build_sage_adjacency(&edge_src, &edge_dst, num_nodes, sf_a);
      adjacency.additive_factorize(2).expect("SAGE adjacency factorization failed");
      let adj_id = g.param(adjacency);

      // 3 SAGEConv layers with bias: H = ReLU(H·W_self + A·(H·W_neighbor) + bias)
      // conv1
      let w1_l = load_weight_matrix::<F>(&weights_path.join("conv1.lin_l.weight.bin"), hidden, num_features);
      let b1 = load_bias::<F>(&weights_path.join("conv1.lin_l.bias.bin"), hidden);
      let w1_r = load_weight_matrix::<F>(&weights_path.join("conv1.lin_r.weight.bin"), hidden, num_features);
      // convs.0
      let w2_l = load_weight_matrix::<F>(&weights_path.join("convs.0.lin_l.weight.bin"), hidden, hidden);
      let b2 = load_bias::<F>(&weights_path.join("convs.0.lin_l.bias.bin"), hidden);
      let w2_r = load_weight_matrix::<F>(&weights_path.join("convs.0.lin_r.weight.bin"), hidden, hidden);
      // convs.1
      let w3_l = load_weight_matrix::<F>(&weights_path.join("convs.1.lin_l.weight.bin"), hidden, hidden);
      let b3 = load_bias::<F>(&weights_path.join("convs.1.lin_l.bias.bin"), hidden);
      let w3_r = load_weight_matrix::<F>(&weights_path.join("convs.1.lin_r.weight.bin"), hidden, hidden);

      let mut h = x;
      for (wl, wr, b) in [(w1_l, w1_r, b1), (w2_l, w2_r, b2), (w3_l, w3_r, b3)] {
        let wl_id = g.param(wl);
        let wr_id = g.param(wr);
        let b_id = g.param(b);
        // neighbor: A · (H · W_l)
        let y_neighbor = g.einsum("bd,dh->bh".to_string(), vec![h, wl_id], true)[0];
        let h_neighbor = g.spmv(adj_id, y_neighbor, true, false)[0];
        // self: H · W_r
        let h_self = g.einsum("bd,dh->bh".to_string(), vec![h, wr_id], true)[0];
        // combine
        let h_combined = g.add(h_self, h_neighbor)[0];
        let h_bias = g.add(h_combined, b_id)[0];
        h = g.relu(h_bias)[0];
      }

      // Global add pool: (1, N_pad) · (N_pad, hidden) → (1, hidden)
      let pool_vec = build_sum_pool_vec(num_nodes);
      let pool_id = g.param(pool_vec);
      let pooled = g.einsum("bd,dh->bh".to_string(), vec![pool_id, h], true)[0];

      // MLP head
      let w_lin1 = load_weight_matrix::<F>(&weights_path.join("lin1.weight.bin"), hidden, hidden);
      let b_lin1 = load_bias::<F>(&weights_path.join("lin1.bias.bin"), hidden);
      let w_lin2 = load_weight_matrix::<F>(&weights_path.join("lin2.weight.bin"), num_classes, hidden);
      let b_lin2 = load_bias::<F>(&weights_path.join("lin2.bias.bin"), num_classes);

      let w_lin1_id = g.param(w_lin1);
      let b_lin1_id = g.param(b_lin1);
      let w_lin2_id = g.param(w_lin2);
      let b_lin2_id = g.param(b_lin2);

      let h1 = g.einsum("bd,dh->bh".to_string(), vec![pooled, w_lin1_id], true)[0];
      let h1_bias = g.add(h1, b_lin1_id)[0];
      let h1_relu = g.relu(h1_bias)[0];
      let h2 = g.einsum("bd,dh->bh".to_string(), vec![h1_relu, w_lin2_id], true)[0];
      let out = g.add(h2, b_lin2_id)[0];
      out
    },

    "gin" => {
      let mut adjacency = build_gin_adjacency(&edge_src, &edge_dst, num_nodes);
      adjacency.additive_factorize(2).expect("GIN adjacency factorization failed");
      let adj_id = g.param(adjacency);

      // 3 GIN conv layers (BN folded): H' = (H + A·H), then MLP: Linear1→ReLU→Linear2→ReLU
      let mut h = x;
      for i in 0..3 {
        let in_dim = if i == 0 { num_features } else { hidden };
        let w1 = load_weight_matrix::<F>(
          &weights_path.join(format!("conv{}.w1.bin", i)), hidden, in_dim);
        let b1 = load_bias::<F>(
          &weights_path.join(format!("conv{}.b1.bin", i)), hidden);
        let w2 = load_weight_matrix::<F>(
          &weights_path.join(format!("conv{}.w2.bin", i)), hidden, hidden);
        let b2 = load_bias::<F>(
          &weights_path.join(format!("conv{}.b2.bin", i)), hidden);

        let w1_id = g.param(w1);
        let b1_id = g.param(b1);
        let w2_id = g.param(w2);
        let b2_id = g.param(b2);

        // Aggregate: Z = A · H (binary A, sf=0, no ScaleDown)
        let z = g.spmv(adj_id, h, false, false)[0];
        // H' = H + Z
        let h_prime = g.add(h, z)[0];
        // MLP: Linear1 → ReLU → Linear2 → ReLU
        let y1 = g.einsum("bd,dh->bh".to_string(), vec![h_prime, w1_id], true)[0];
        let y1_bias = g.add(y1, b1_id)[0];
        let y1_relu = g.relu(y1_bias)[0];
        let y2 = g.einsum("bd,dh->bh".to_string(), vec![y1_relu, w2_id], true)[0];
        let y2_bias = g.add(y2, b2_id)[0];
        h = g.relu(y2_bias)[0];
      }

      // Global mean pool
      let pool_vec = build_mean_pool_vec(num_nodes);
      let pool_id = g.param(pool_vec);
      let pooled = g.einsum("bd,dh->bh".to_string(), vec![pool_id, h], true)[0];

      // MLP head
      let w_lin1 = load_weight_matrix::<F>(&weights_path.join("lin1.weight.bin"), hidden, hidden);
      let b_lin1 = load_bias::<F>(&weights_path.join("lin1.bias.bin"), hidden);
      let w_lin2 = load_weight_matrix::<F>(&weights_path.join("lin2.weight.bin"), num_classes, hidden);
      let b_lin2 = load_bias::<F>(&weights_path.join("lin2.bias.bin"), num_classes);

      let w_lin1_id = g.param(w_lin1);
      let b_lin1_id = g.param(b_lin1);
      let w_lin2_id = g.param(w_lin2);
      let b_lin2_id = g.param(b_lin2);

      let h1 = g.einsum("bd,dh->bh".to_string(), vec![pooled, w_lin1_id], true)[0];
      let h1_bias = g.add(h1, b_lin1_id)[0];
      let h1_relu = g.relu(h1_bias)[0];
      let h2 = g.einsum("bd,dh->bh".to_string(), vec![h1_relu, w_lin2_id], true)[0];
      let out = g.add(h2, b_lin2_id)[0];
      out
    },

    "gat" => {
      use zk_torch_2::dag::gnn::gat_layer_with_offset;
      use zk_torch_2::dag::PolyType;

      let num_heads = 4usize;
      let head_dim = hidden / num_heads; // 16
      // r_offset_log for elem_div: must exceed log2(max softmax denominator).
      // The shifted remainder r + 2^r_offset must be non-negative and < 2^(r_offset+1).
      // For TU graphs, 24 is safe (covers max_degree * exp_range at internal sf).
      let r_offset_log = 24;

      // Build incidence matrices
      let num_edges = edge_src.len();
      let num_edges_log = log2_ceil(num_edges) as usize;
      let num_edges_pad = 1usize << num_edges_log;
      let dummy_node = num_nodes;

      let mut s_src_sel: Vec<(usize, usize)> = (0..num_edges)
        .map(|e| (edge_src[e] as usize, e)).collect();
      for e in num_edges..num_edges_pad { s_src_sel.push((dummy_node, e)); }
      let s_src_poly = SelectionPolynomial::new(num_nodes_log, num_edges_log, s_src_sel);
      let s_src = Witness::new_sparse(
        vec![num_nodes_pad, num_edges_pad], s_src_poly.to_sparse(),
        DataType::Float, 0, Role::Constant,
      );

      let mut s_dst_sel: Vec<(usize, usize)> = (0..num_edges)
        .map(|e| (edge_dst[e] as usize, e)).collect();
      for e in num_edges..num_edges_pad { s_dst_sel.push((dummy_node, e)); }
      let s_dst_poly = SelectionPolynomial::new(num_nodes_log, num_edges_log, s_dst_sel);
      let s_dst = Witness::new_sparse(
        vec![num_nodes_pad, num_edges_pad], s_dst_poly.to_sparse(),
        DataType::Float, 0, Role::Constant,
      );

      let s_src_id = g.param(s_src);
      let s_dst_id = g.param(s_dst);

      // 3 GAT conv layers with ReLU (no bias)
      // conv1: (num_features → hidden)
      let w1 = load_weight_matrix::<F>(&weights_path.join("conv1.lin.weight.bin"), hidden, num_features);
      let a1_src = load_attention_block(&weights_path.join("conv1.att_src.bin"), num_heads, head_dim);
      let a1_dst = load_attention_block(&weights_path.join("conv1.att_dst.bin"), num_heads, head_dim);

      let mut h = g.pipe(&[x], gat_layer_with_offset(s_src_id, s_dst_id, num_heads, w1, a1_src, a1_dst, r_offset_log))[0];
      h = g.relu(h)[0];

      // convs.0: (hidden → hidden)
      let w2 = load_weight_matrix::<F>(&weights_path.join("convs.0.lin.weight.bin"), hidden, hidden);
      let a2_src = load_attention_block(&weights_path.join("convs.0.att_src.bin"), num_heads, head_dim);
      let a2_dst = load_attention_block(&weights_path.join("convs.0.att_dst.bin"), num_heads, head_dim);

      h = g.pipe(&[h], gat_layer_with_offset(s_src_id, s_dst_id, num_heads, w2, a2_src, a2_dst, r_offset_log))[0];
      h = g.relu(h)[0];

      // convs.1: (hidden → hidden)
      let w3 = load_weight_matrix::<F>(&weights_path.join("convs.1.lin.weight.bin"), hidden, hidden);
      let a3_src = load_attention_block(&weights_path.join("convs.1.att_src.bin"), num_heads, head_dim);
      let a3_dst = load_attention_block(&weights_path.join("convs.1.att_dst.bin"), num_heads, head_dim);

      h = g.pipe(&[h], gat_layer_with_offset(s_src_id, s_dst_id, num_heads, w3, a3_src, a3_dst, r_offset_log))[0];
      h = g.relu(h)[0];

      // Global mean pool
      let pool_vec = build_mean_pool_vec(num_nodes);
      let pool_id = g.param(pool_vec);
      let pooled = g.einsum("bd,dh->bh".to_string(), vec![pool_id, h], true)[0];

      // MLP head: lin1 → ReLU → lin2
      let w_lin1 = load_weight_matrix::<F>(&weights_path.join("lin1.weight.bin"), hidden, hidden);
      let b_lin1 = load_bias::<F>(&weights_path.join("lin1.bias.bin"), hidden);
      let w_lin2 = load_weight_matrix::<F>(&weights_path.join("lin2.weight.bin"), num_classes, hidden);
      let b_lin2 = load_bias::<F>(&weights_path.join("lin2.bias.bin"), num_classes);

      let w_lin1_id = g.param(w_lin1);
      let b_lin1_id = g.param(b_lin1);
      let w_lin2_id = g.param(w_lin2);
      let b_lin2_id = g.param(b_lin2);

      let h1 = g.einsum("bd,dh->bh".to_string(), vec![pooled, w_lin1_id], true)[0];
      let h1_bias = g.add(h1, b_lin1_id)[0];
      let h1_relu = g.relu(h1_bias)[0];
      let h2 = g.einsum("bd,dh->bh".to_string(), vec![h1_relu, w_lin2_id], true)[0];
      let out = g.add(h2, b_lin2_id)[0];
      out
    },

    _ => panic!("Unknown model: {}", model_name),
  };

  println!("Compiling DAG...");
  let (dag, mut init) = g.compile();

  println!("Running forward pass...");
  let presplit_sparse = dag.run(&mut init, &vec![(x, input)]);
  println!("  Output shape: {:?}", init[output][0].shape);

  // Extract prediction (output shape is (1, num_classes))
  let output_witness = &init[output][0];
  let mut best_class = 0usize;
  let mut best_val = i128::MIN;
  for c in 0..num_classes {
    let field_val = output_witness.get(&[0, c]);
    let int_val = zk_torch_2::util::arith::f_to_int(field_val);
    if int_val > best_val {
      best_val = int_val;
      best_class = c;
    }
  }
  let correct = best_class == label as usize;
  println!("  Graph {}: Prediction={}, Label={}, Correct={}", graph_index, best_class, label, correct);

  if !do_prove {
    return (correct, 0.0, 0.0);
  }

  // --- Proving and Verification ---
  let mut transcript = Transcript::new(b"zkml");
  let mut dense_commitments: Vec<Option<KZH3Commitment<PairingType>>> = vec![None; dag.num_edges()];
  let mut sparse_commitments: Vec<Option<Vec<KZH3Commitment<PairingType>>>> = vec![None; dag.num_edges()];
  let mut factored_commitments: Vec<Option<Vec<KZH3Commitment<PairingType>>>> = vec![None; dag.num_edges()];

  let polynomial_sizes = dag.collect_polynomial_sizes(&init);

  let mut srs_map = HashMap::new();
  for &size in &polynomial_sizes {
    let size_srs = if std::fs::metadata(&format!("{}.srs", size)).is_ok() {
      load_kzh3_srs(size).expect("Failed to load SRS")
    } else {
      println!("  Generating SRS for size {}", size);
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
          load_kzh3_srs(mask_n).expect("Failed to load mask SRS")
        } else {
          println!("  Generating SRS for mask size {}", mask_n);
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

  let t0 = std::time::Instant::now();
  dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &kzh3, &sparse_kzh3, &init, &mut dense_commitments, &mut sparse_commitments, &mut factored_commitments, &mut timing,
  );
  let commit_ms = t0.elapsed().as_secs_f64() * 1000.0;

  let t0 = std::time::Instant::now();
  let (sc_proofs, op_proofs, range_proof, two_pow_proof, reducer_proofs, zk_proof) =
    dag.prove_zk::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &kzh3, &sparse_kzh3, &init, &presplit_sparse,
      &dense_commitments, &sparse_commitments, &factored_commitments, &mut transcript, &mut timing,
      zk_mode, mask_committer.clone(),
    );
  let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

  zk_torch_2::util::serialization::measure_total_proof_size(
    &sc_proofs, &op_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
  );

  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));

  let mut vt = Transcript::new(b"zkml");
  let t0 = std::time::Instant::now();
  let verified = dag.verify_zk::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &sc_proofs, &op_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
    &init, &dense_vk, &sparse_vk, &dense_commitments, &sparse_commitments, &factored_commitments, &mut vt,
    zk_mode, &zk_proof, mask_committer.clone(),
  );
  let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;
  println!("  Graph {}: commit={:.1}ms, prove={:.1}ms, verify={:.1}ms, verified={}{}",
    graph_index, commit_ms, prove_ms, verify_ms, verified, if zk_mode { " (ZK)" } else { "" });

  (correct, prove_ms, verify_ms)
}

fn main() {
  env_logger::init();

  let args: Vec<String> = std::env::args().collect();
  if args.len() < 5 {
    println!("Usage: kernel <config.yaml> <data_dir> <model> <dataset> [graph_indices] [--no-prove]");
    println!("  model: gcn | graphsage | gin | gat");
    println!("  dataset: mutag | proteins | imdb-binary | reddit-binary");
    println!("  graph_indices: single index, range (0-9), or 'all' (default: 0)");
    println!("  --no-prove: skip proving/verification, only run inference");
    println!("  --zk: enable zero-knowledge mode");
    return;
  }

  let data_dir = &args[2];
  let model_name = &args[3];
  let dataset_name = &args[4];
  let do_prove = !args.iter().any(|a| a == "--no-prove");
  let zk_mode = args.iter().any(|a| a == "--zk");

  // Parse graph indices
  let ds_meta_str = std::fs::read_to_string(
    Path::new(data_dir).join(dataset_name).join("meta.json")
  ).expect("Failed to read dataset meta.json");
  let ds_meta: serde_json::Value = serde_json::from_str(&ds_meta_str).unwrap();
  let num_classes = ds_meta["num_classes"].as_u64().unwrap() as usize;
  let num_features = ds_meta["num_features"].as_u64().unwrap() as usize;
  let num_graphs = ds_meta["num_graphs"].as_u64().unwrap() as usize;
  let hidden = 64;

  let graph_arg = if args.len() >= 6 && args[5] != "--no-prove" { &args[5] } else { "0" };
  let graph_indices: Vec<usize> = if graph_arg == "all" {
    (0..num_graphs).collect()
  } else if graph_arg.contains('-') {
    let parts: Vec<&str> = graph_arg.split('-').collect();
    let start: usize = parts[0].parse().unwrap();
    let end: usize = parts[1].parse().unwrap();
    (start..=end).collect()
  } else {
    vec![graph_arg.parse().unwrap()]
  };

  let weights_path = Path::new(data_dir).join(format!("{}_{}", model_name, dataset_name));

  println!("=== {} on {} ({} graphs, prove={}, zk={}) ===",
    model_name.to_uppercase(), dataset_name, graph_indices.len(), do_prove, zk_mode);

  let mut correct_count = 0usize;
  let mut total_prove_ms = 0.0f64;
  let mut total_verify_ms = 0.0f64;
  let mut proved_count = 0usize;

  for &gi in &graph_indices {
    let (correct, prove_ms, verify_ms) = run_graph(
      data_dir, model_name, dataset_name, gi,
      hidden, num_classes, &weights_path, do_prove, zk_mode,
    );
    if correct { correct_count += 1; }
    if do_prove {
      total_prove_ms += prove_ms;
      total_verify_ms += verify_ms;
      proved_count += 1;
    }
  }

  let accuracy = correct_count as f64 / graph_indices.len() as f64;
  println!("\n=== Results: {} on {} ===", model_name.to_uppercase(), dataset_name);
  println!("  Accuracy: {}/{} = {:.1}%", correct_count, graph_indices.len(), accuracy * 100.0);
  if do_prove && proved_count > 0 {
    println!("  Avg prove time:  {:.1}ms", total_prove_ms / proved_count as f64);
    println!("  Avg verify time: {:.1}ms", total_verify_ms / proved_count as f64);
  }
}

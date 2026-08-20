use crate::basicblock::spmv::SpMV;
use crate::basicblock::BasicBlockType;
use crate::util::arith::log2_ceil;
use crate::util::poly::CryptoField;
use crate::dag::{DagBuilder, EdgeId, Role, Witness};

impl<F: CryptoField + 'static> DagBuilder<F> {
    /// SpMV: sparse matrix × dense matrix multiply.
    /// Edge list and weights are read from the sparse witness at runtime.
    pub fn spmv(&mut self, a: EdgeId, y: EdgeId, scale_back: bool, transpose: bool) -> Vec<EdgeId> {
      let shape_a = self.init_values[a].as_ref().unwrap().shape.clone();
      let num_rows_log = log2_ceil(shape_a[0]) as usize;
      let num_cols_log = if shape_a.len() > 1 { log2_ceil(shape_a[1]) as usize } else { num_rows_log };
      let spmv_basicblock = BasicBlockType::SpMV(SpMV { num_rows_log, num_cols_log, transpose });

      let shape_y = self.init_values[y].as_ref().unwrap().shape.clone();
      let output_sf = self.init_values[y].as_ref().unwrap().sf;
      let data_type = self.init_values[y].as_ref().unwrap().data_type;
      let sf = self.init_values[a].as_ref().unwrap().sf + self.init_values[y].as_ref().unwrap().sf;
      // Output rows: shape_a[0] for normal, shape_a[1] for transpose
      let output_rows = if transpose { shape_a[1] } else { shape_a[0] };
      let output_shape = vec![output_rows, shape_y[1]];
      let out_value = Witness::new_wo_data(output_shape, data_type, sf, Role::Output);
      self.init_values.push(Some(out_value));

      let mut outs = self.add_gkr_node(vec![a, y], spmv_basicblock);
      if scale_back && sf != output_sf {
        outs = self.scale(outs[0], sf, output_sf);
      }
      outs
    }
}

/* =========================
GCN (Graph Convolutional Network)
========================= */

/// GCN layer decomposed into Einsum + SpMV:
///  H_{k+1} = σ(A · H_k · W_k)
/// Decomposed as:
///  1. Y = H_k · W_k (Einsum + ScaleDown)
///  2. Z = A · Y (SpMV + ScaleDown)
///  3. H_{k+1} = ReLU(Z)
pub fn gcn_layer<F: CryptoField + 'static>(
  adjacency: EdgeId,
  weight: Witness<F>,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "GCN layer expects 1 input");
    let h_k = x[0]; // (num_nodes, d_in)

    let weight_id = g.param(weight);

    // Step 1: Y = H_k · W_k + ScaleDown
    let y = g.einsum("bd,dh->bh".to_string(), vec![h_k, weight_id], true)[0];

    // Step 2: Z = A · Y + ScaleDown
    let z = g.spmv(adjacency, y, true, false)[0];

    // Step 3: H_{k+1} = ReLU(Z)
    let h_next = g.relu(z)[0];

    vec![h_next]
  }
}

pub fn gcn<F: CryptoField + 'static>(
  adjacency: Witness<F>,   // normalized adjacency (|V|, |V|) — weights in evaluations
  weights: Vec<Witness<F>>, // W_k per layer: (d_in, d_out)
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "GCN expects 1 input");
    let mut h = x[0];
    let adjacency_id = g.param(adjacency);

    for i in 0..weights.len() {
      h = g.pipe(&[h], gcn_layer(adjacency_id, weights[i].clone()))[0];
    }
    vec![h]
  }
}

/* =========================
GIN (Graph Isomorphism Network)
========================= */

/// GIN layer (ε=0 variant):
///  H_{k+1} = MLP( H_k + A · H_k )
/// where MLP is a 2-layer network: ReLU(H' · W1) · W2 + ReLU
///
/// Since A is binary (sf=0), SpMV does not increase the scale factor
/// and no ScaleDown is needed after aggregation.
///
/// Steps:
///  1. Z = A · H_k (SpMV, binary A, no ScaleDown)
///  2. H' = H_k + Z (Add)
///  3. Y = H' · W1 + ScaleDown (Einsum)
///  4. Y' = ReLU(Y)
///  5. H_{k+1} = ReLU(Y' · W2 + ScaleDown) (Einsum + ReLU)
pub fn gin_layer<F: CryptoField + 'static>(
  adjacency: EdgeId,
  w1: Witness<F>,  // MLP first layer: (d_in, d_hidden)
  w2: Witness<F>,  // MLP second layer: (d_hidden, d_out)
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "GIN layer expects 1 input");
    let h_k = x[0]; // (num_nodes, d_in)

    let w1_id = g.param(w1);
    let w2_id = g.param(w2);

    // Step 1: Z = A · H_k (binary A, sf_A=0, no ScaleDown)
    let z = g.spmv(adjacency, h_k, false, false)[0];

    // Step 2: H' = H_k + Z
    let h_prime = g.add(h_k, z)[0];

    // Step 3: Y = H' · W1 + ScaleDown
    let y = g.einsum("bd,dh->bh".to_string(), vec![h_prime, w1_id], true)[0];

    // Step 4: Y' = ReLU(Y)
    let y_relu = g.relu(y)[0];

    // Step 5: H_{k+1} = ReLU(Y' · W2 + ScaleDown)
    let out = g.einsum("bd,dh->bh".to_string(), vec![y_relu, w2_id], true)[0];
    let h_next = g.relu(out)[0];

    vec![h_next]
  }
}

pub fn gin<F: CryptoField + 'static>(
  adjacency: Witness<F>,    // binary adjacency (|V|, |V|) — sf=0
  weights_1: Vec<Witness<F>>, // MLP W1 per layer: (d_in, d_hidden)
  weights_2: Vec<Witness<F>>, // MLP W2 per layer: (d_hidden, d_out)
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "GIN expects 1 input");
    let mut h = x[0];
    let adjacency_id = g.param(adjacency);

    let num_layers = weights_1.len();
    for i in 0..num_layers {
      h = g.pipe(&[h], gin_layer(adjacency_id, weights_1[i].clone(), weights_2[i].clone()))[0];
    }
    vec![h]
  }
}

/* =========================
GraphSage
========================= */

/// Two-block decomposition GraphSage layer with bias:
/// 1. Y = Einsum(H_k, W_neighbor) with ScaleDown
/// 2. h_neighbor = SpMV(A, Y) with ScaleDown
/// 3. h_self = Einsum(H_k, W_self) with ScaleDown
/// 4. h = Add(h_self, h_neighbor)
/// 5. h = Add(h, bias)              — broadcast (1, d_out) to (n, d_out)
/// 6. H_{k+1} = ReLU(h)  (if apply_relu)
pub fn graph_sage_layer<F: CryptoField + 'static>(
  adjacency: EdgeId,
  weight_self: Witness<F>,
  weight_neighbor: Witness<F>,
  bias: Witness<F>,
  apply_relu: bool,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This custom GraphSage layer expects 1 input");
    let x = x[0]; // (num_nodes, input_dim)

    let weight_self_id = g.param(weight_self);
    let weight_neighbor_id = g.param(weight_neighbor);
    let bias_id = g.param(bias);

    // Step 1: Y = H_k · W_neighbor, then ScaleDown
    let y = g.einsum("bd,dh->bh".to_string(), vec![x, weight_neighbor_id], true)[0];

    // Step 2: h_neighbor = A · Y, then ScaleDown
    let h_neighbor = g.spmv(adjacency, y, true, false)[0];

    // Step 3: h_self = H_k · W_self, then ScaleDown
    let h_self = g.einsum("bd,dh->bh".to_string(), vec![x, weight_self_id], true)[0];

    // Step 4: h = h_self + h_neighbor
    let h = g.add(h_self, h_neighbor)[0];

    // Step 5: h = h + bias (broadcast (1, d_out) to (n, d_out))
    let mut h = g.add(h, bias_id)[0];

    // Step 6: ReLU (skipped on last layer to match PyTorch)
    if apply_relu {
      h = g.relu(h)[0];
    }
    vec![h]
  }
}

pub fn graph_sage<F: CryptoField + 'static>(
  adjacency: Witness<F>, // (num_nodes, num_nodes) — weights stored in evaluations
  weights_self: Vec<Witness<F>>,     // W_self per layer: (d_in, d_out)
  weights_neighbor: Vec<Witness<F>>, // W_neighbor per layer: (d_in, d_out)
  biases: Vec<Witness<F>>,           // bias per layer: (1, d_out)
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "This custom GraphSage expects 1 input");
    let mut x = x[0]; // (num_nodes, input_dim)
    let adjacency_id = g.param(adjacency);

    let num_layers = weights_self.len();
    for i in 0..num_layers {
      let weight_self = weights_self[i].clone();
      let weight_neighbor = weights_neighbor[i].clone();
      let bias = biases[i].clone();
      let apply_relu = i < num_layers - 1; // no ReLU on last layer
      x = g.pipe(&vec![x], graph_sage_layer(adjacency_id, weight_self, weight_neighbor, bias, apply_relu))[0];
    }
    vec![x]
  }
}

/* =========================
GAT (Graph Attention Network)
========================= */

/// Multi-head GAT layer.
///
/// Uses a combined projection W: (d_in, num_heads * head_dim) and block-diagonal
/// attention vectors a_src, a_dst: (num_heads * head_dim, num_heads).
/// Each head computes attention independently on its slice of features.
/// Output is concatenated: (num_nodes, num_heads * head_dim).
///
/// For single head (num_heads=1), this reduces to the standard GAT layer.
///
/// Steps:
///  1.  H' = H_k · W → (N, K*d_h)
///  2a. H'_src = S_src^T · H' → (E, K*d_h)
///  2b. H'_dst = S_dst^T · H' → (E, K*d_h)
///  3a. e_src = H'_src · a_src_blk → (E, K)     [block-diag extracts per-head scores]
///  3b. e_dst = H'_dst · a_dst_blk → (E, K)
///  3c. e = e_src + e_dst → (E, K)
///  3d. e' = LeakyReLU(e, negative_slope=0.2)
///  3e. Z' = exp(e') → (E, K)
///  4.  Z'' = S_dst · Z' → (N, K)
///  5.  Z''' = S_dst^T · Z'' → (E, K)
///  6.  α = Z' ⊘ Z''' → (E, K)
///  7.  change_shape H'_src to (E*K, d_h), α to (E*K, 1)
///      H'' = diag(α) · H'_src → (E*K, d_h)     [einsum "ea,ed->ed"]
///      change_shape H'' back to (E, K*d_h)
///  8.  O = S_dst · H'' → (N, K*d_h)
///  9.  H_{k+1} = ReLU(O)
pub fn gat_layer<F: CryptoField + 'static>(
  s_src: EdgeId,     // S_src sparse witness edge (|V| × |E|)
  s_dst: EdgeId,     // S_dst sparse witness edge (|V| × |E|)
  num_heads: usize,
  weight: Witness<F>,       // W: (d_in, num_heads * head_dim)
  attn_src: Witness<F>,     // block-diag a_src: (num_heads * head_dim, num_heads)
  attn_dst: Witness<F>,     // block-diag a_dst: (num_heads * head_dim, num_heads)
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  gat_layer_with_offset(s_src, s_dst, num_heads, weight, attn_src, attn_dst, 42)
}

/// GAT layer with a configurable range-table bound for element-wise division.
/// `r_offset_log` must satisfy 2^r_offset_log > max softmax denominator so
/// both nonnegative remainder witnesses fit.
/// For large graphs (N > 10K), use 42. For small graphs (N < 100), use 20.
pub fn gat_layer_with_offset<F: CryptoField + 'static>(
  s_src: EdgeId,
  s_dst: EdgeId,
  num_heads: usize,
  weight: Witness<F>,
  attn_src: Witness<F>,
  attn_dst: Witness<F>,
  r_offset_log: usize,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "GAT layer expects 1 input");
    let h_k = x[0]; // H_k: (num_nodes, d_in)

    let weight_id = g.param(weight);
    let attn_src_id = g.param(attn_src);
    let attn_dst_id = g.param(attn_dst);

    // Step 1: H' = H_k · W → (N, K*d_h)
    let h_prime = g.einsum("bd,dh->bh".to_string(), vec![h_k, weight_id], true)[0];

    // Infer shapes for change_shape later
    let h_prime_shape = g.init_values[h_prime].as_ref().unwrap().shape.clone();
    let num_edges = g.init_values[s_src].as_ref().unwrap().shape[1]; // |E| (padded)
    let total_dim = h_prime_shape[1]; // K * d_h
    let head_dim = total_dim / num_heads;

    // Step 2a: H'_src = S_src^T · H' → (E, K*d_h)
    let h_src = g.spmv(s_src, h_prime, false, true)[0];

    // Step 2b: H'_dst = S_dst^T · H' → (E, K*d_h)
    let h_dst = g.spmv(s_dst, h_prime, false, true)[0];

    // Step 3a: e_src = H'_src · a_src_blk → (E, K)
    let e_src = g.einsum("ed,dk->ek".to_string(), vec![h_src, attn_src_id], true)[0];

    // Step 3b: e_dst = H'_dst · a_dst_blk → (E, K)
    let e_dst = g.einsum("ed,dk->ek".to_string(), vec![h_dst, attn_dst_id], true)[0];

    // Step 3c: e = e_src + e_dst → (E, K)
    let e = g.add(e_src, e_dst)[0];

    // Step 3d: e' = LeakyReLU(e, negative_slope=0.2)
    let e_relu = g.leaky_relu(e, 0.2)[0];

    // Step 3e: Z' = exp(e') → (E, K)
    let z_prime = g.exp(e_relu)[0];

    // Step 4: Z'' = S_dst · Z' → (N, K)
    let z_double_prime = g.spmv(s_dst, z_prime, false, false)[0];

    // Step 5: Z''' = S_dst^T · Z'' → (E, K)
    let z_triple_prime = g.spmv(s_dst, z_double_prime, false, true)[0];

    // Step 6: α = Z' / Z''' → (E, K)
    // r_offset_log bounds every positive softmax denominator and remainder.
    let alpha = g.elem_div(z_prime, z_triple_prime, r_offset_log);

    // Step 7: Per-head weighted combination
    // Flatten head dimension into batch: (E, K*d_h) → (E*K, d_h), (E, K) → (E*K, 1)
    let h_src_flat = g.change_shape(h_src, vec![num_edges * num_heads, head_dim]);
    let alpha_flat = g.change_shape(alpha, vec![num_edges * num_heads, 1]);
    // einsum "ea,ed->ed" broadcasts α across features per (edge, head) pair
    let h_double_prime_flat = g.einsum("ea,ed->ed".to_string(), vec![alpha_flat, h_src_flat], true)[0];
    // Reshape back: (E*K, d_h) → (E, K*d_h)
    let h_double_prime = g.change_shape(h_double_prime_flat, vec![num_edges, total_dim]);

    // Step 8: O = S_dst · H'' → (N, K*d_h)
    let o = g.spmv(s_dst, h_double_prime, false, false)[0];

    // Step 9: H_{k+1} = ReLU(O)
    let h_next = g.relu(o)[0];

    vec![h_next]
  }
}

/* =========================
GAE (Graph Autoencoder)
========================= */

/// Graph Autoencoder:
///  Z = Encoder(X)          — any GNN encoder, output shape (|V|, d)
///  Â = sigmoid(Z · Z^T)    — reconstructed adjacency (|V|, |V|)
///
/// Decoder steps:
///  1. S = Z · Z^T (Einsum "bd,cd->bc" + ScaleDown)
///  2. Â = sigmoid(S)
pub fn gae<F: CryptoField + 'static>(
  encoder: impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId>,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    // Encoder: produces Z (|V|, d)
    let z = encoder(g, x)[0];

    // Decoder: Â = sigmoid(Z · Z^T)
    // Step 1: S = Z · Z^T + ScaleDown
    let s = g.einsum("bd,cd->bc".to_string(), vec![z, z], true)[0];

    // Step 2: Â = sigmoid(S)
    let a_hat = g.sigmoid(s)[0];

    vec![a_hat]
  }
}

/// Compute flat index for a Witness with given shape (column-major, padded to next power of 2).
/// Matches the indexing used by Witness::get().
fn flat_index(indices: &[usize], shape: &[usize]) -> usize {
  let mut index = 0;
  let mut stride = 1;
  for (i, &idx) in indices.iter().enumerate() {
    index += idx * stride;
    stride *= shape[i].next_power_of_two();
  }
  index
}

/// Build a block-diagonal attention matrix from per-head attention vectors.
/// Input: K vectors each of shape (head_dim, 1)
/// Output: (K * head_dim, K) where column k has nonzeros only in rows [k*d_h, (k+1)*d_h)
fn build_block_diag_attention<F: CryptoField + 'static>(
  per_head_vecs: &[Witness<F>],
) -> Witness<F> {
  let num_heads = per_head_vecs.len();
  let head_dim = per_head_vecs[0].shape[0];
  let total_dim = num_heads * head_dim;
  let sf = per_head_vecs[0].sf;
  let data_type = per_head_vecs[0].data_type;

  let out_shape = vec![total_dim, num_heads];
  let n = crate::util::arith::get_n(&out_shape);
  let mut data = vec![<F as CryptoField>::zero(); 1 << n];

  for k in 0..num_heads {
    for j in 0..head_dim {
      let val = per_head_vecs[k].get(&[j, 0]);
      let idx = flat_index(&[k * head_dim + j, k], &out_shape);
      data[idx] = val;
    }
  }

  Witness::new(
    out_shape,
    data,
    data_type,
    sf,
    Role::Constant,
  )
}

/// Build a combined weight matrix by horizontally concatenating per-head weights.
/// Input: K weights each of shape (d_in, head_dim)
/// Output: (d_in, K * head_dim) where columns [k*d_h, (k+1)*d_h) come from head k
fn build_combined_weight<F: CryptoField + 'static>(
  per_head_weights: &[Witness<F>],
) -> Witness<F> {
  let num_heads = per_head_weights.len();
  let d_in = per_head_weights[0].shape[0];
  let head_dim = per_head_weights[0].shape[1];
  let total_dim = num_heads * head_dim;
  let sf = per_head_weights[0].sf;
  let data_type = per_head_weights[0].data_type;

  let out_shape = vec![d_in, total_dim];
  let n = crate::util::arith::get_n(&out_shape);
  let mut data = vec![<F as CryptoField>::zero(); 1 << n];

  for k in 0..num_heads {
    for r in 0..d_in {
      for c in 0..head_dim {
        let val = per_head_weights[k].get(&[r, c]);
        let idx = flat_index(&[r, k * head_dim + c], &out_shape);
        data[idx] = val;
      }
    }
  }

  Witness::new(
    out_shape,
    data,
    data_type,
    sf,
    Role::Constant,
  )
}

/// Multi-head GAT model.
///
/// Per layer: `num_heads[i]` attention heads, each with `head_dim = weights[i][0].shape[1]`.
/// Intermediate layers concatenate heads; output dim = num_heads * head_dim.
/// Use num_heads=1 for the last layer to get (num_nodes, d_out) output.
///
/// Parameters per layer:
///  - weights[i]: Vec of K weight matrices, each (d_in, head_dim)
///  - attn_src_vecs[i]: Vec of K attention vectors, each (head_dim, 1)
///  - attn_dst_vecs[i]: Vec of K attention vectors, each (head_dim, 1)
pub fn gat<F: CryptoField + 'static>(
  s_src_witness: Witness<F>,              // S_src: (|V|, |E|)
  s_dst_witness: Witness<F>,              // S_dst: (|V|, |E|)
  weights: Vec<Vec<Witness<F>>>,          // per layer: Vec of per-head W_k: (d_in, head_dim)
  attn_src_vecs: Vec<Vec<Witness<F>>>,    // per layer: Vec of per-head a_src: (head_dim, 1)
  attn_dst_vecs: Vec<Vec<Witness<F>>>,    // per layer: Vec of per-head a_dst: (head_dim, 1)
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  gat_with_offset(s_src_witness, s_dst_witness, weights, attn_src_vecs, attn_dst_vecs, 42)
}

pub fn gat_with_offset<F: CryptoField + 'static>(
  s_src_witness: Witness<F>,
  s_dst_witness: Witness<F>,
  weights: Vec<Vec<Witness<F>>>,
  attn_src_vecs: Vec<Vec<Witness<F>>>,
  attn_dst_vecs: Vec<Vec<Witness<F>>>,
  r_offset_log: usize,
) -> impl FnOnce(&mut DagBuilder<F>, &[EdgeId]) -> Vec<EdgeId> {
  move |g, x| {
    assert!(x.len() == 1, "GAT expects 1 input");
    let mut h = x[0];
    let s_src_id = g.param(s_src_witness);
    let s_dst_id = g.param(s_dst_witness);

    let num_layers = weights.len();
    for i in 0..num_layers {
      let num_heads = weights[i].len();
      // Build combined weight and block-diagonal attention matrices
      let combined_weight = build_combined_weight(&weights[i]);
      let blk_attn_src = build_block_diag_attention(&attn_src_vecs[i]);
      let blk_attn_dst = build_block_diag_attention(&attn_dst_vecs[i]);

      h = g.pipe(
        &[h],
        gat_layer_with_offset(
          s_src_id, s_dst_id,
          num_heads,
          combined_weight, blk_attn_src, blk_attn_dst,
          r_offset_log,
        ),
      )[0];
    }
    vec![h]
  }
}

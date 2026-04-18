use crate::basicblock::BasicBlock;
use crate::crypto::{LinearSumcheckProver, ProveContext, SumcheckProof, SumcheckProver, SumcheckVerifier, ZkLinearSumcheckProver, replay_mask_opening_transcript, verify_mask_consistency, recompute_mask_eval};
use crate::dag::{Claim, Role, Witness};
use crate::util::arith::log2_ceil;
use crate::util::poly::CryptoField;
use crate::util::poly::{evaluate_lagrange_basis, fix_variables_from_right, DenseMLPoly, SparseMLPoly};
use crate::util::transcript::Transcript;

use std::sync::LazyLock;

/// When set (e.g., CLASSICAL_SPMV=1), SpMV uses dense fix_variables for
/// adjacency partial evaluation instead of sparse edge iteration.
/// This is for ablation study comparing O(N^2) classical vs O(|E|) proposed.
static CLASSICAL_SPMV: LazyLock<bool> = LazyLock::new(|| {
  std::env::var("CLASSICAL_SPMV").map(|v| v == "1" || v.to_lowercase() == "true").unwrap_or(false)
});

/// When set (e.g., NAIVE_SPMV=1), SpMV uses the sparse-naive O(M log N) approach:
/// for each edge, evaluate the Lagrange basis at that index in O(log N) time,
/// rather than precomputing all N basis values in O(N).
static NAIVE_SPMV: LazyLock<bool> = LazyLock::new(|| {
  std::env::var("NAIVE_SPMV").map(|v| v == "1" || v.to_lowercase() == "true").unwrap_or(false)
});

/// Evaluate a single Lagrange basis polynomial at index `idx` given challenge point `r`.
/// Cost: O(log N) multiplications where N = 2^{r.len()}.
/// L_idx(r) = Π_i (r_i if bit_i(idx)=1, else 1-r_i)
fn evaluate_lagrange_single<F: CryptoField>(r: &[F], idx: usize) -> F {
  let one = <F as CryptoField>::one();
  let mut result = one;
  for (i, &r_i) in r.iter().enumerate() {
    if (idx >> i) & 1 == 1 {
      result = result * r_i;
    } else {
      result = result * (one - r_i);
    }
  }
  result
}

/// SpMV: Sparse matrix × dense matrix multiply.
///
/// When `transpose = false` (S · H):
///   Input[0]: S (sparse, num_rows × num_cols)
///   Input[1]: H (dense, num_cols × num_features)
///   Output: Z = S · H (dense, num_rows × num_features)
///   Proving: Z(r_row, r_f) = Σ_col S(r_row, col) · H(col, r_f)
///
/// When `transpose = true` (S^T · H):
///   Input[0]: S (sparse, num_rows × num_cols) — same witness, interpreted as S^T
///   Input[1]: H (dense, num_rows × num_features)
///   Output: Z = S^T · H (dense, num_cols × num_features)
///   Proving: Z(r_col, r_f) = Σ_row S(row, r_col) · H(row, r_f)
///
/// The edge list is read from the sparse witness at runtime.
/// Transpose is handled at the polynomial level by swapping which dimension
/// is the output (fixed at challenge) vs the inner (sumcheck) dimension.
#[derive(Clone)]
pub struct SpMV {
  pub num_rows_log: usize,              // log2 of S's row dimension
  pub num_cols_log: usize,              // log2 of S's col dimension
  pub transpose: bool,                  // if true, compute S^T · H instead of S · H
}

impl std::fmt::Debug for SpMV {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "SpMV {{ rows_log: {}, cols_log: {}, transpose: {} }}",
      self.num_rows_log, self.num_cols_log, self.transpose)
  }
}

impl<F: CryptoField> BasicBlock<F> for SpMV {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    assert!(inputs.len() == 2, "SpMV expects 2 inputs (S, H)");
    let s = inputs[0];
    let h = inputs[1];

    // Read edge list and weights from the sparse witness
    let poly = s.data.as_ref().unwrap().as_any()
      .downcast_ref::<SparseMLPoly<F>>().expect("SpMV: input[0] must be a SparseMLPoly");
    let edges = &poly.selection.selection;

    let num_rows = 1usize << self.num_rows_log;
    let num_cols = 1usize << self.num_cols_log;
    let num_features = h.shape[1];
    let log_num_features = log2_ceil(num_features) as usize;
    let padded_features = 1usize << log_num_features;

    if self.transpose {
      // Z = S^T · H: Z[col, f] += weight * H[row, f]
      let total_size = num_cols * padded_features;
      let mut z_data = vec![<F as CryptoField>::zero(); total_size];

      for &(row, col) in edges {
        if row < num_rows && col < num_cols {
          let idx = row + col * (1 << poly.selection.input_num_vars);
          let w = poly.evaluations.get(&idx).copied().unwrap_or(<F as CryptoField>::one());
          for f in 0..padded_features {
            let h_idx = row + f * num_rows;
            let z_idx = col + f * num_cols;
            z_data[z_idx] = z_data[z_idx] + w * h.data.as_ref().unwrap().index(h_idx);
          }
        }
      }

      let output_shape = vec![s.shape[1], num_features];
      let sf = s.sf + h.sf;
      vec![Witness::new(output_shape, z_data, h.data_type, sf, Role::Output)]
    } else {
      // Z = S · H: Z[row, f] += weight * H[col, f]
      let total_size = num_rows * padded_features;
      let mut z_data = vec![<F as CryptoField>::zero(); total_size];

      for &(row, col) in edges {
        if row < num_rows && col < num_cols {
          let idx = row + col * (1 << poly.selection.input_num_vars);
          let w = poly.evaluations.get(&idx).copied().unwrap_or(<F as CryptoField>::one());
          for f in 0..padded_features {
            let h_idx = col + f * num_cols;
            let z_idx = row + f * num_rows;
            z_data[z_idx] = z_data[z_idx] + w * h.data.as_ref().unwrap().index(h_idx);
          }
        }
      }

      let output_shape = vec![s.shape[0], num_features];
      let sf = s.sf + h.sf;
      vec![Witness::new(output_shape, z_data, h.data_type, sf, Role::Output)]
    }
  }

  fn prove(
    &self,
    witnesses: &[&Witness<F>],
    edge_ids: &[usize],
    out_claims: &[&Claim<F>],
    transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    assert!(out_claims.len() == 1, "SpMV expects 1 output claim");

    // Read edge list and weights from the sparse witness (pre-split version)
    let s = witnesses[0];
    let poly = s.data.as_ref().unwrap().as_any()
      .downcast_ref::<SparseMLPoly<F>>().expect("SpMV prove: input[0] must be a SparseMLPoly");
    let edges = &poly.selection.selection;

    let h = witnesses[1];
    let num_features = h.shape[1];
    let point = out_claims[0].point.clone();

    if self.transpose {
      // S^T · H: Z(r_col, r_f) = Σ_row S(row, r_col) · H(row, r_f)
      // Output dim = cols, inner dim = rows
      let log_output = self.num_cols_log;  // output dimension (cols of S = rows of S^T)
      let log_inner = self.num_rows_log;   // inner/sumcheck dimension (rows of S)

      let mut sumcheck_prover = LinearSumcheckProver::new(log_inner, 2, transcript);

      let col_point = point[..log_output].to_vec();

      let start = std::time::Instant::now();
      let adjacency_eval = if *CLASSICAL_SPMV {
        // Classical: convert to dense, fix col vars (rightmost) at r_col — O(N^2)
        let dense_adj = poly.to_dense();
        let partial = fix_variables_from_right(&dense_adj, &col_point);
        partial.evaluations[..1 << log_inner].to_vec()
      } else if *NAIVE_SPMV {
        // Sparse-naive: evaluate Lagrange basis per edge — O(M log N)
        let mut adjacency_eval = vec![<F as CryptoField>::zero(); 1 << log_inner];
        for &(row, col) in edges {
          let idx = row + col * (1 << poly.selection.input_num_vars);
          let w = poly.evaluations.get(&idx).copied().unwrap_or(<F as CryptoField>::one());
          let basis_val = evaluate_lagrange_single(&col_point, col);
          adjacency_eval[row] = adjacency_eval[row] + w * basis_val;
        }
        adjacency_eval
      } else {
        // Proposed: sparse partial eval by iterating over edges — O(M)
        let challenge_vec = evaluate_lagrange_basis(&col_point);
        let mut adjacency_eval = vec![<F as CryptoField>::zero(); 1 << log_inner];
        for &(row, col) in edges {
          let idx = row + col * (1 << poly.selection.input_num_vars);
          let w = poly.evaluations.get(&idx).copied().unwrap_or(<F as CryptoField>::one());
          adjacency_eval[row] = adjacency_eval[row] + w * challenge_vec[col];
        }
        adjacency_eval
      };
      let elapsed = start.elapsed();
      let mode_str = if *CLASSICAL_SPMV { "classical" } else if *NAIVE_SPMV { "naive" } else { "proposed" };
      println!("SpMV (transpose) adjacency partial eval time: {:?} [{}]",
        elapsed, mode_str);

      // Partial eval of H: fix feature vars from right
      let h_point = point[log_output..].to_vec();
      let h_dense = h.data.as_ref().unwrap().as_any().downcast_ref::<DenseMLPoly<F>>().unwrap();
      let h_partial = crate::util::poly::fix_variables_from_right(h_dense, &h_point);

      // Sumcheck: Σ_row partial_S[row] · partial_H[row]
      let partial_adj = DenseMLPoly::new(log_inner, adjacency_eval);
      let sumcheck_proof = sumcheck_prover.prove(&vec![partial_adj, h_partial], transcript);
      let challenges = sumcheck_prover.challenges.clone();

      // Claim on H at (r_row, r_f) where r_row = sumcheck challenges
      let mut h_final_point = challenges.clone();
      h_final_point.extend(h_point.iter().cloned());
      let h_claim = Claim {
        edge_id: edge_ids[1],
        sparse_id: 0,
        point: h_final_point.clone(),
        eval: h.data.as_ref().unwrap().evaluate_at_point(&h_final_point),
      };

      // Adjacency claim: S(r_inner, r_col) where r_inner = row sumcheck challenges, r_col = fixed col point
      let mut adj_full_point = challenges.clone();
      adj_full_point.extend(col_point.iter().cloned());
      let adj_eval = s.data.as_ref().unwrap().evaluate_at_point(&adj_full_point);
      let adj_claim = Claim {
        edge_id: edge_ids[0],
        sparse_id: 0,
        point: adj_full_point,
        eval: adj_eval,
      };

      (vec![sumcheck_proof], vec![adj_claim, h_claim])
    } else {
      // S · H: Z(r_row, r_f) = Σ_col S(r_row, col) · H(col, r_f)
      // Output dim = rows, inner dim = cols
      let log_output = self.num_rows_log;
      let log_inner = self.num_cols_log;

      let mut sumcheck_prover = LinearSumcheckProver::new(log_inner, 2, transcript);

      let row_point = point[..log_output].to_vec();

      let start = std::time::Instant::now();
      let adjacency_eval = if *CLASSICAL_SPMV {
        // Classical: convert to dense, fix row vars (leftmost) at r_row — O(N^2)
        let dense_adj = poly.to_dense();
        let partial = dense_adj.fix_variables(&row_point);
        partial.evaluations[..1 << log_inner].to_vec()
      } else if *NAIVE_SPMV {
        // Sparse-naive: evaluate Lagrange basis per edge — O(M log N)
        let mut adjacency_eval = vec![<F as CryptoField>::zero(); 1 << log_inner];
        for &(row, col) in edges {
          let idx = row + col * (1 << poly.selection.input_num_vars);
          let w = poly.evaluations.get(&idx).copied().unwrap_or(<F as CryptoField>::one());
          let basis_val = evaluate_lagrange_single(&row_point, row);
          adjacency_eval[col] = adjacency_eval[col] + w * basis_val;
        }
        adjacency_eval
      } else {
        // Proposed: sparse partial eval by iterating over edges — O(M)
        let challenge_vec = evaluate_lagrange_basis(&row_point);
        let mut adjacency_eval = vec![<F as CryptoField>::zero(); 1 << log_inner];
        for &(row, col) in edges {
          let idx = row + col * (1 << poly.selection.input_num_vars);
          let w = poly.evaluations.get(&idx).copied().unwrap_or(<F as CryptoField>::one());
          adjacency_eval[col] = adjacency_eval[col] + w * challenge_vec[row];
        }
        adjacency_eval
      };
      let elapsed = start.elapsed();
      let mode_str = if *CLASSICAL_SPMV { "classical" } else if *NAIVE_SPMV { "naive" } else { "proposed" };
      println!("SpMV adjacency partial eval time: {:?} [{}]",
        elapsed, mode_str);

      // Partial eval of H: fix feature vars from right
      let h_point = point[log_output..].to_vec();
      let h_dense = h.data.as_ref().unwrap().as_any().downcast_ref::<DenseMLPoly<F>>().unwrap();
      let h_partial = crate::util::poly::fix_variables_from_right(h_dense, &h_point);

      // Sumcheck: Σ_col partial_S[col] · partial_H[col]
      let partial_adj = DenseMLPoly::new(log_inner, adjacency_eval);
      let sumcheck_proof = sumcheck_prover.prove(&vec![partial_adj, h_partial], transcript);
      let challenges = sumcheck_prover.challenges.clone();

      // Claim on H at (r_col, r_f) where r_col = sumcheck challenges
      let mut h_final_point = challenges.clone();
      h_final_point.extend(h_point.iter().cloned());
      let h_claim = Claim {
        edge_id: edge_ids[1],
        sparse_id: 0,
        point: h_final_point.clone(),
        eval: h.data.as_ref().unwrap().evaluate_at_point(&h_final_point),
      };

      // Adjacency claim: S(r_row, r_col) where r_row = fixed row point, r_col = sumcheck challenges
      let mut adj_full_point = row_point.clone();
      adj_full_point.extend(challenges.iter().cloned());
      let adj_eval = s.data.as_ref().unwrap().evaluate_at_point(&adj_full_point);
      let adj_claim = Claim {
        edge_id: edge_ids[0],
        sparse_id: 0,
        point: adj_full_point,
        eval: adj_eval,
      };

      (vec![sumcheck_proof], vec![adj_claim, h_claim])
    }
  }

  fn verify(
    &self,
    _witnesses: &[&Witness<F>],
    claims: &[&Claim<F>],
    sumcheck_proofs: &[&SumcheckProof<F>],
    transcript: &mut Transcript<F>,
  ) -> bool {
    let out_claim = claims[claims.len() - 1];
    let expected_sum = out_claim.eval;

    // Inner dimension: cols for normal, rows for transpose
    let log_inner = if self.transpose { self.num_rows_log } else { self.num_cols_log };

    let mut verifier = SumcheckVerifier::new(log_inner, 2, transcript);
    let (verification_result, _challenges) = verifier.verify(
      transcript,
      sumcheck_proofs[0].round_messages.clone(),
      expected_sum,
    );
    let running_sum = match verification_result {
      Some(v) => v,
      None => {
        println!("verified SpMV failed: sumcheck round check");
        return false;
      }
    };

    // Final eval check: running_sum == adj_eval * h_eval
    // claims = [adj_claim, h_claim, out_claim] — adj + H claims from prove(), last is output
    let adj_eval = claims[0].eval;
    let h_eval = claims[1].eval;
    let expected = adj_eval * h_eval;
    if running_sum != expected {
      println!("verified SpMV failed: final_eval check mismatch (running_sum != adj_eval * h_eval)");
      return false;
    }
    true
  }

  fn prove_zk(
    &self,
    witnesses: &[&Witness<F>],
    edge_ids: &[usize],
    out_claims: &[&Claim<F>],
    transcript: &mut Transcript<F>,
    ctx: &mut ProveContext<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    if !ctx.zk {
      return self.prove(witnesses, edge_ids, out_claims, transcript);
    }
    assert!(out_claims.len() == 1, "SpMV expects 1 output claim");

    let s = witnesses[0];
    let poly = s.data.as_ref().unwrap().as_any()
      .downcast_ref::<SparseMLPoly<F>>().expect("SpMV prove_zk: input[0] must be a SparseMLPoly");
    let edges = &poly.selection.selection;

    let h = witnesses[1];
    let point = out_claims[0].point.clone();

    let (log_output, log_inner, is_transpose) = if self.transpose {
      (self.num_cols_log, self.num_rows_log, true)
    } else {
      (self.num_rows_log, self.num_cols_log, false)
    };

    let mut zk_prover = ZkLinearSumcheckProver::new(log_inner, 2, transcript);

    let fixed_point = point[..log_output].to_vec();
    let adjacency_eval = if *CLASSICAL_SPMV {
      let dense_adj = poly.to_dense();
      if is_transpose {
        let partial = fix_variables_from_right(&dense_adj, &fixed_point);
        partial.evaluations[..1 << log_inner].to_vec()
      } else {
        let partial = dense_adj.fix_variables(&fixed_point);
        partial.evaluations[..1 << log_inner].to_vec()
      }
    } else if *NAIVE_SPMV {
      let mut adjacency_eval = vec![<F as CryptoField>::zero(); 1 << log_inner];
      for &(row, col) in edges {
        let idx = row + col * (1 << poly.selection.input_num_vars);
        let w = poly.evaluations.get(&idx).copied().unwrap_or(<F as CryptoField>::one());
        if is_transpose {
          let basis_val = evaluate_lagrange_single(&fixed_point, col);
          adjacency_eval[row] = adjacency_eval[row] + w * basis_val;
        } else {
          let basis_val = evaluate_lagrange_single(&fixed_point, row);
          adjacency_eval[col] = adjacency_eval[col] + w * basis_val;
        }
      }
      adjacency_eval
    } else {
      let challenge_vec = evaluate_lagrange_basis(&fixed_point);
      let mut adjacency_eval = vec![<F as CryptoField>::zero(); 1 << log_inner];
      for &(row, col) in edges {
        let idx = row + col * (1 << poly.selection.input_num_vars);
        let w = poly.evaluations.get(&idx).copied().unwrap_or(<F as CryptoField>::one());
        if is_transpose {
          adjacency_eval[row] = adjacency_eval[row] + w * challenge_vec[col];
        } else {
          adjacency_eval[col] = adjacency_eval[col] + w * challenge_vec[row];
        }
      }
      adjacency_eval
    };

    let h_point = point[log_output..].to_vec();
    let h_dense = h.data.as_ref().unwrap().as_any().downcast_ref::<DenseMLPoly<F>>().unwrap();
    let h_partial = crate::util::poly::fix_variables_from_right(h_dense, &h_point);

    let partial_adj = DenseMLPoly::new(log_inner, adjacency_eval);
    let sumcheck_proof = zk_prover.prove_zk(&vec![partial_adj, h_partial], transcript, ctx);
    let challenges = zk_prover.challenges.clone();

    let mut h_final_point = challenges.clone();
    h_final_point.extend(h_point.iter().cloned());
    let h_claim = Claim {
      edge_id: edge_ids[1],
      sparse_id: 0,
      point: h_final_point.clone(),
      eval: h.data.as_ref().unwrap().evaluate_at_point(&h_final_point),
    };

    // Adjacency claim
    let adj_full_point = if is_transpose {
      // S(r_inner, r_col): r_inner = row sumcheck challenges, r_col = fixed col point
      let mut p = challenges.clone();
      p.extend(fixed_point.iter().cloned());
      p
    } else {
      // S(r_row, r_col): r_row = fixed row point, r_col = sumcheck challenges
      let mut p = fixed_point.clone();
      p.extend(challenges.iter().cloned());
      p
    };
    let adj_eval = s.data.as_ref().unwrap().evaluate_at_point(&adj_full_point);
    let adj_claim = Claim {
      edge_id: edge_ids[0],
      sparse_id: 0,
      point: adj_full_point,
      eval: adj_eval,
    };

    (vec![sumcheck_proof], vec![adj_claim, h_claim])
  }

  fn verify_zk(
    &self,
    _witnesses: &[&Witness<F>],
    claims: &[&Claim<F>],
    sumcheck_proofs: &[&SumcheckProof<F>],
    transcript: &mut Transcript<F>,
    ctx: &ProveContext<F>,
  ) -> bool {
    if !ctx.zk {
      return self.verify(_witnesses, claims, sumcheck_proofs, transcript);
    }
    let out_claim = claims[claims.len() - 1];
    let expected_sum = out_claim.eval;
    let log_inner = if self.transpose { self.num_rows_log } else { self.num_cols_log };

    let mut verifier = SumcheckVerifier::new(log_inner, 2, transcript);
    let mask_sum = match sumcheck_proofs[0].zk_mask_sum {
      Some(v) => v,
      None => {
        println!("verified zk SpMV failed: missing mask_sum in ZK proof");
        return false;
      }
    };
    let (verification_result, challenges, rho) = verifier.verify_zk(
      transcript,
      sumcheck_proofs[0].round_messages.clone(),
      expected_sum,
      mask_sum,
      sumcheck_proofs[0].zk_mask_commitment.as_deref(),
    );
    let running_sum = match verification_result {
      Some(v) => v,
      None => {
        println!("verified zk SpMV failed: sumcheck round check");
        return false;
      }
    };

    // Final eval check: running_sum == adj_eval * h_eval + ρ * mask(x*)
    // In ZK mode, running_sum includes the mask term from the adjusted sum
    let adj_eval = claims[0].eval;
    let h_eval = claims[1].eval;
    let mask_eval = if let Some(ref mask_coeffs) = sumcheck_proofs[0].zk_mask_coeffs {
      recompute_mask_eval(mask_coeffs, &challenges)
    } else {
      <F as CryptoField>::zero()
    };
    let expected = adj_eval * h_eval + rho * mask_eval;
    if running_sum != expected {
      println!("verified zk SpMV failed: final_eval check mismatch (running_sum != adj_eval * h_eval + rho * mask_eval)");
      return false;
    }

    // Fix 4: Verify mask_sum consistency
    if !verify_mask_consistency(&sumcheck_proofs[0], ctx.mask_committer.as_deref()) {
      println!("verified zk SpMV failed: mask consistency check");
      return false;
    }

    // Replay mask opening challenges to keep transcript in sync with prover
    replay_mask_opening_transcript(&sumcheck_proofs[0], transcript);
    true
  }
}

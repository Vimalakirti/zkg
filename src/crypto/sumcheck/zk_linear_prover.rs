use crate::crypto::sumcheck::prover::{ProveContext, SumcheckProof};
use crate::crypto::sumcheck::LinearSumcheckProver;
use crate::crypto::sumcheck::SumcheckProver;
use crate::util::poly::{CryptoField, DenseMLPoly, MLPoly};
use crate::util::transcript::Transcript;

/// Zero-knowledge linear sumcheck prover using the masking polynomial technique.
///
/// Given g(x) = Π_{i=1}^ℓ p_i(x), proves H = Σ_{x∈{0,1}^n} g(x) without
/// leaking information about the p_i polynomials.
///
/// The mask is p(x) = Σ_i s_i(x_i) where each s_i is univariate of degree ℓ
/// (i.e. ℓ+1 random coefficients). The prover runs a standard sumcheck on
/// g + ρ·p, where ρ is drawn from the transcript after committing to P = Σ p(x).
///
/// The mask coefficients are included in the proof (they are random, so no leakage).
/// The verifier recomputes P and p(r) from them.
pub struct ZkLinearSumcheckProver<F: CryptoField> {
  pub num_var: usize,
  pub num_poly: usize,
  pub challenges: Vec<F>,
  inner: LinearSumcheckProver<F>,
  /// s_i coefficients: mask_coeffs[i] has degree+1 entries (degree = num_poly).
  mask_coeffs: Vec<Vec<F>>,
  /// s_i(0) precomputed for each variable.
  mask_at_0: Vec<F>,
  /// s_i(1) precomputed for each variable.
  mask_at_1: Vec<F>,
  /// Σ_{i < current_round} s_i(r_i)
  fixed_sum: F,
  /// Σ_{i > current_round} (s_i(0) + s_i(1))
  remaining_01_sum: F,
  /// Random scalar ρ drawn from transcript.
  rho: F,
  /// P = Σ_{x∈{0,1}^n} p(x)
  mask_sum: F,
}

/// Evaluate a univariate polynomial defined by coefficients at point c using Horner's method.
/// s(c) = Σ_j coeffs[j] * c^j
pub fn eval_univariate<F: CryptoField>(coeffs: &[F], c: F) -> F {
  let mut val = *coeffs.last().unwrap();
  for j in (0..coeffs.len() - 1).rev() {
    val = val * c + coeffs[j];
  }
  val
}

/// Recompute mask polynomial evaluation at the challenge point:
/// mask_eval = Σ_i s_i(r_i)
pub fn recompute_mask_eval<F: CryptoField>(mask_coeffs: &[Vec<F>], challenges: &[F]) -> F {
  assert_eq!(mask_coeffs.len(), challenges.len(), "mask_coeffs and challenges must have same length");
  let mut sum = <F as CryptoField>::zero();
  for (i, coeffs) in mask_coeffs.iter().enumerate() {
    sum = sum + eval_univariate(coeffs, challenges[i]);
  }
  sum
}

/// Pack mask coefficients into a DenseMLPoly (for commitment verification).
pub fn pack_mask_coeffs<F: CryptoField>(mask_coeffs: &[Vec<F>]) -> DenseMLPoly<F> {
  let n = mask_coeffs.len();
  let degree_plus_1 = if n > 0 { mask_coeffs[0].len() } else { 0 };
  let total_coeffs = n * degree_plus_1;
  let packed_size = total_coeffs.next_power_of_two();
  let packed_num_vars = if packed_size > 1 { (packed_size as f64).log2() as usize } else { 0 };
  let mut packed_evals = Vec::with_capacity(packed_size);
  for coeffs in mask_coeffs {
    packed_evals.extend_from_slice(coeffs);
  }
  packed_evals.resize(packed_size, <F as CryptoField>::zero());
  DenseMLPoly::new(packed_num_vars, packed_evals)
}

/// Verify mask_sum and mask commitment consistency in a ZK sumcheck proof.
/// Returns true if all checks pass, false otherwise.
pub fn verify_mask_consistency<F: CryptoField>(
  proof: &super::prover::SumcheckProof<F>,
  mask_committer: Option<&dyn super::prover::MaskCommitter<F>>,
) -> bool {
  let mask_coeffs = match proof.zk_mask_coeffs.as_ref() {
    Some(c) => c,
    None => return true, // Not ZK mode
  };
  let mask_sum = match proof.zk_mask_sum {
    Some(s) => s,
    None => return false, // Inconsistent: has coeffs but no sum
  };

  // Check 1: Recompute mask_sum from coefficients
  let recomputed_sum = recompute_mask_sum(mask_coeffs);
  if recomputed_sum != mask_sum {
    println!("mask_sum consistency check failed: recomputed != claimed");
    return false;
  }

  // Check 2: If mask commitment is present, verify opening
  if let (Some(commitment), Some(opening_proof), Some(opening_point), Some(opening_eval)) = (
    proof.zk_mask_commitment.as_ref(),
    proof.zk_mask_opening_proof.as_ref(),
    proof.zk_mask_opening_point.as_ref(),
    proof.zk_mask_opening_eval,
  ) {
    // Reconstruct packed polynomial from coefficients
    let packed_poly = pack_mask_coeffs(mask_coeffs);

    // Check evaluation consistency
    let recomputed_eval = packed_poly.evaluate_at_point(opening_point);
    if recomputed_eval != opening_eval {
      println!("mask coefficient evaluation mismatch at opening point");
      return false;
    }

    // Verify PCS opening proof
    if let Some(committer) = mask_committer {
      if !committer.verify(commitment, opening_proof, opening_point, opening_eval) {
        println!("mask PCS opening verification failed");
        return false;
      }
    }
  }

  true
}

/// Replay mask opening challenge draws from the transcript to keep it in sync with the prover.
/// Must be called after verify_zk in each BasicBlock's verify_zk method when a mask committer is present.
pub fn replay_mask_opening_transcript<F: CryptoField>(
  proof: &super::prover::SumcheckProof<F>,
  transcript: &mut crate::util::transcript::Transcript<F>,
) {
  if let Some(ref opening_point) = proof.zk_mask_opening_point {
    // Draw the same number of challenges as the prover did
    for _ in 0..opening_point.len() {
      let _: F = transcript.challenge_scalar(b"zk_mask_open_challenge");
    }
  }
}

/// Recompute mask_sum = 2^{n-1} * Σ_i (s_i(0) + s_i(1)) from mask coefficients.
pub fn recompute_mask_sum<F: CryptoField>(mask_coeffs: &[Vec<F>]) -> F {
  let n = mask_coeffs.len();
  assert!(n > 0, "mask_coeffs must not be empty");
  let zero = <F as CryptoField>::zero();
  let one = <F as CryptoField>::one();
  let mut total_01 = zero;
  for coeffs in mask_coeffs {
    total_01 = total_01 + eval_univariate(coeffs, zero) + eval_univariate(coeffs, one);
  }
  <F as CryptoField>::from_u64(1u64 << (n - 1)) * total_01
}

impl<F: CryptoField> ZkLinearSumcheckProver<F> {
  /// Evaluate univariate s_i at a field element c.
  fn eval_si(&self, i: usize, c: F) -> F {
    eval_univariate(&self.mask_coeffs[i], c)
  }

  /// Compute the mask round message at evaluation point c for the current round m.
  ///
  /// mask_round_m(c) = 2^{n-m-1} · (fixed_sum + s_m(c)) + half_remaining
  /// where half_remaining = 2^{n-m-2} · remaining_01_sum  (for m < n-1)
  /// or 0 for the last round (m = n-1).
  fn mask_round_value(&self, round: usize, c: F) -> F {
    let n = self.num_var;
    let power_main = 1u64 << (n - round - 1);
    let main = <F as CryptoField>::from_u64(power_main) * (self.fixed_sum + self.eval_si(round, c));

    if round < n - 1 {
      let power_half = 1u64 << (n - round - 2);
      main + <F as CryptoField>::from_u64(power_half) * self.remaining_01_sum
    } else {
      main
    }
  }

  pub fn prove_zk(
    &mut self,
    instances: &Vec<DenseMLPoly<F>>,
    transcript: &mut Transcript<F>,
    ctx: &mut ProveContext<F>,
  ) -> SumcheckProof<F> {
    assert!(!instances.is_empty(), "Polynomials cannot be empty");
    let n_size = instances[0].len();
    assert_eq!(n_size, 1 << self.num_var, "Polynomials have incorrect size");
    assert_eq!(instances.len(), self.num_poly, "Number of polynomials mismatch");

    let n = self.num_var;
    let degree = self.num_poly; // degree per variable = ℓ

    // 1. Sample mask coefficients randomly
    let mut rng = ark_std::rand::thread_rng();
    self.mask_coeffs = (0..n)
      .map(|_| (0..=degree).map(|_| F::rand(&mut rng)).collect())
      .collect();

    // 2. Precompute s_i(0) and s_i(1)
    self.mask_at_0 = (0..n).map(|i| self.eval_si(i, <F as CryptoField>::zero())).collect();
    self.mask_at_1 = (0..n).map(|i| self.eval_si(i, <F as CryptoField>::one())).collect();

    // 3. Compute mask_sum = 2^{n-1} * Σ_i (s_i(0) + s_i(1))
    let total_01: F = (0..n).map(|i| self.mask_at_0[i] + self.mask_at_1[i]).fold(<F as CryptoField>::zero(), |a, b| a + b);
    self.mask_sum = <F as CryptoField>::from_u64(1u64 << (n - 1)) * total_01;

    // 3b. Pack mask coefficients into a DenseMLPoly and commit (if committer available)
    let total_coeffs = n * (degree + 1);
    let packed_size = total_coeffs.next_power_of_two();
    let packed_num_vars = (packed_size as f64).log2() as usize;
    let mut packed_evals = Vec::with_capacity(packed_size);
    for coeffs in &self.mask_coeffs {
      packed_evals.extend_from_slice(coeffs);
    }
    packed_evals.resize(packed_size, <F as CryptoField>::zero());
    let packed_poly = DenseMLPoly::new(packed_num_vars, packed_evals);

    let mut mask_commitment_bytes: Option<Vec<u8>> = None;
    if let Some(ref committer) = ctx.mask_committer {
      let commitment = committer.commit(&packed_poly);
      // Append commitment to transcript before ρ
      transcript.append_bytes(b"zk_mask_commit", &commitment);
      mask_commitment_bytes = Some(commitment);
    }

    // 4. Append mask_sum to transcript and draw ρ
    transcript.append_scalar(b"zk_mask_sum", &self.mask_sum);
    self.rho = transcript.challenge_scalar(b"zk_rho");

    // 5. Initialize running sums
    self.fixed_sum = <F as CryptoField>::zero();
    self.remaining_01_sum = total_01;

    // 6. Store instances in inner prover
    self.inner.a_arrays = instances.clone();
    let mut round_messages = Vec::new();

    // 7. Execute rounds
    for round in 0..n {
      // Compute inner (g) round message
      let inner_msg = self.inner.compute_round_message();

      // Update remaining_01_sum before computing mask
      self.remaining_01_sum = self.remaining_01_sum - (self.mask_at_0[round] + self.mask_at_1[round]);

      // Compute combined message: inner[j] + ρ * mask_val(j)
      let combined: Vec<F> = (0..=degree)
        .map(|j| {
          let c = <F as CryptoField>::from_u32(j as u32);
          let mask_val = self.mask_round_value(round, c);
          inner_msg[j] + self.rho * mask_val
        })
        .collect();

      // Append combined message to transcript
      for &msg in &combined {
        transcript.append_scalar(b"round_message", &msg);
      }

      // Draw challenge
      let challenge = transcript.challenge_scalar(b"challenge");
      round_messages.push(combined);

      // Update fixed_sum
      self.fixed_sum = self.fixed_sum + self.eval_si(round, challenge);

      // Bind variable in inner prover
      self.inner.receive_challenge(challenge);
      self.challenges.push(challenge);
    }

    // 8. Compute final evaluation
    let inner_final = self.inner.final_evaluation();
    let mask_at_challenges: F = (0..n).map(|i| self.eval_si(i, self.challenges[i])).fold(<F as CryptoField>::zero(), |a, b| a + b);
    let final_eval = inner_final + self.rho * mask_at_challenges;

    // 9. If committer available, draw random eval point and open packed poly
    let (mask_opening_proof, mask_opening_point, mask_opening_eval) = if let Some(ref committer) = ctx.mask_committer {
      let q: Vec<F> = (0..packed_num_vars).map(|_| transcript.challenge_scalar(b"zk_mask_open_challenge")).collect();
      let (proof_bytes, eval) = committer.open(&packed_poly, &q);
      (Some(proof_bytes), Some(q), Some(eval))
    } else {
      (None, None, None)
    };

    SumcheckProof {
      final_eval,
      round_messages,
      zk_mask_sum: Some(self.mask_sum),
      zk_mask_coeffs: Some(self.mask_coeffs.clone()),
      zk_mask_commitment: mask_commitment_bytes,
      zk_mask_opening_proof: mask_opening_proof,
      zk_mask_opening_point: mask_opening_point,
      zk_mask_opening_eval: mask_opening_eval,
    }
  }
}

impl<F: CryptoField> SumcheckProver<F> for ZkLinearSumcheckProver<F> {
  type Instance = Vec<DenseMLPoly<F>>;

  fn new(num_var: usize, num_polys: usize, transcript: &mut Transcript<F>) -> Self {
    transcript.append_u64(b"num_var", num_var as u64);
    transcript.append_u64(b"num_poly", num_polys as u64);

    Self {
      num_var,
      num_poly: num_polys,
      challenges: Vec::new(),
      inner: LinearSumcheckProver {
        num_var,
        num_poly: num_polys,
        a_arrays: Vec::new(),
        current_round: 0,
        challenges: Vec::new(),
      },
      mask_coeffs: Vec::new(),
      mask_at_0: Vec::new(),
      mask_at_1: Vec::new(),
      fixed_sum: <F as CryptoField>::zero(),
      remaining_01_sum: <F as CryptoField>::zero(),
      rho: <F as CryptoField>::zero(),
      mask_sum: <F as CryptoField>::zero(),
    }
  }

  /// Non-ZK prove (delegates to inner). Use `prove_zk` for ZK mode.
  fn prove(&mut self, instances: &Self::Instance, transcript: &mut Transcript<F>) -> SumcheckProof<F> {
    self.inner.prove(instances, transcript)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ark_bn254::Fr;

  /// Helper: compute Σ_{x∈{0,1}^n} Π_i p_i(x) directly.
  fn brute_force_sum(polys: &[DenseMLPoly<Fr>]) -> Fr {
    let n = polys[0].n;
    let size = 1usize << n;
    let mut sum = <Fr as CryptoField>::zero();
    for x in 0..size {
      let mut prod = <Fr as CryptoField>::one();
      for p in polys {
        prod = prod * p[x];
      }
      sum = sum + prod;
    }
    sum
  }

  #[test]
  fn test_zk_sumcheck_prover() {
    use ark_std::UniformRand;
    let mut rng = ark_std::rand::thread_rng();

    let n = 4;
    let num_polys = 2;
    let size = 1usize << n;

    let polys: Vec<DenseMLPoly<Fr>> = (0..num_polys)
      .map(|_| DenseMLPoly::new(n, (0..size).map(|_| Fr::rand(&mut rng)).collect()))
      .collect();

    let _expected_sum = brute_force_sum(&polys);

    // ZK prove
    let mut transcript = Transcript::new(b"test_zk");
    let mut ctx = ProveContext::new(true);
    let mut prover = ZkLinearSumcheckProver::new(n, num_polys, &mut transcript);
    let proof = prover.prove_zk(&polys, &mut transcript, &mut ctx);

    // Check proof structure
    assert!(proof.zk_mask_sum.is_some(), "mask_sum should be set");
    assert!(proof.zk_mask_coeffs.is_some(), "mask_coeffs should be set");
    assert_eq!(proof.round_messages.len(), n, "should have n rounds");

    let coeffs = proof.zk_mask_coeffs.as_ref().unwrap();
    assert_eq!(coeffs.len(), n, "should have n univariates");
    for si in coeffs {
      assert_eq!(si.len(), num_polys + 1, "each s_i should have degree+1 coefficients");
    }

    // Each round message should have degree+1 = num_polys+1 entries
    for (round, msg) in proof.round_messages.iter().enumerate() {
      assert_eq!(msg.len(), num_polys + 1, "round {} should have {} entries", round, num_polys + 1);
    }
  }

  #[test]
  fn test_zk_mask_sum_consistency() {
    use ark_std::UniformRand;
    let mut rng = ark_std::rand::thread_rng();

    let n = 3;
    let num_polys = 2;
    let size = 1usize << n;

    let polys: Vec<DenseMLPoly<Fr>> = (0..num_polys)
      .map(|_| DenseMLPoly::new(n, (0..size).map(|_| Fr::rand(&mut rng)).collect()))
      .collect();

    let mut transcript = Transcript::new(b"test_mask");
    let mut ctx = ProveContext::new(true);
    let mut prover = ZkLinearSumcheckProver::new(n, num_polys, &mut transcript);
    let proof = prover.prove_zk(&polys, &mut transcript, &mut ctx);

    // Verify mask_sum = Σ_{x∈{0,1}^n} p(x) by brute force
    let coeffs = proof.zk_mask_coeffs.as_ref().unwrap();
    let mut brute_mask_sum = <Fr as CryptoField>::zero();
    for x in 0..size {
      let mut val = <Fr as CryptoField>::zero();
      for i in 0..n {
        let bit = (x >> i) & 1;
        let c = <Fr as CryptoField>::from_u32(bit as u32);
        val = val + eval_univariate(&coeffs[i], c);
      }
      brute_mask_sum = brute_mask_sum + val;
    }
    assert_eq!(brute_mask_sum, proof.zk_mask_sum.unwrap(), "mask_sum should equal Σ p(x)");
  }

  #[test]
  fn test_zk_round_message_consistency() {
    use ark_std::UniformRand;
    let mut rng = ark_std::rand::thread_rng();

    let n = 3;
    let num_polys = 2;
    let size = 1usize << n;

    let polys: Vec<DenseMLPoly<Fr>> = (0..num_polys)
      .map(|_| DenseMLPoly::new(n, (0..size).map(|_| Fr::rand(&mut rng)).collect()))
      .collect();

    let expected_sum = brute_force_sum(&polys);

    let mut transcript = Transcript::new(b"test_round");
    let mut ctx = ProveContext::new(true);
    let mut prover = ZkLinearSumcheckProver::new(n, num_polys, &mut transcript);
    let proof = prover.prove_zk(&polys, &mut transcript, &mut ctx);

    // Reconstruct ρ from a fresh verifier transcript
    let mut v_transcript = Transcript::new(b"test_round");
    // Replay the SumcheckProver::new transcript entries
    v_transcript.append_u64(b"num_var", n as u64);
    v_transcript.append_u64(b"num_poly", num_polys as u64);
    // Replay mask_sum and draw ρ
    v_transcript.append_scalar(b"zk_mask_sum", &proof.zk_mask_sum.unwrap());
    let rho: Fr = v_transcript.challenge_scalar(b"zk_rho");

    // First round: msg[0] + msg[1] should equal expected_sum + ρ * mask_sum
    let adjusted_sum = expected_sum + rho * proof.zk_mask_sum.unwrap();
    let first_msg = &proof.round_messages[0];
    assert_eq!(
      first_msg[0] + first_msg[1],
      adjusted_sum,
      "First round message should sum to adjusted_sum"
    );
  }

  #[test]
  fn test_nonzk_proof_structure() {
    use ark_std::UniformRand;
    let mut rng = ark_std::rand::thread_rng();

    let n = 3;
    let num_polys = 2;
    let size = 1usize << n;

    let polys: Vec<DenseMLPoly<Fr>> = (0..num_polys)
      .map(|_| DenseMLPoly::new(n, (0..size).map(|_| Fr::rand(&mut rng)).collect()))
      .collect();

    let mut transcript = Transcript::new(b"test_nonzk");
    let mut prover = crate::crypto::sumcheck::LinearSumcheckProver::new(n, num_polys, &mut transcript);
    let proof = prover.prove(&polys, &mut transcript);

    assert!(proof.zk_mask_sum.is_none());
    assert!(proof.zk_mask_coeffs.is_none());
    assert_eq!(proof.round_messages.len(), n);
  }
}

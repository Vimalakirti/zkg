use crate::basicblock::BasicBlock;
use crate::crypto::{LinearSumcheckProver, ProveContext, SumcheckProof, SumcheckProver, SumcheckVerifier, ZkLinearSumcheckProver, recompute_mask_eval, replay_mask_opening_transcript, verify_mask_consistency};
use crate::dag::{Claim, Witness};
use crate::util::arith::{calc_pow, get_n};
use crate::util::poly::{evaluate_lagrange_basis, DenseMLPoly};

use crate::util::poly::CryptoField;
use crate::util::transcript::Transcript;

#[derive(Debug, Clone)]
pub struct Reducer;
impl<F: CryptoField> BasicBlock<F> for Reducer {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    assert!(inputs.len() == 1, "Reducer expects 1 input");
    return vec![inputs[0].clone()];
  }

  fn prove(
    &self,
    witnesses: &[&Witness<F>],
    edge_ids: &[usize],
    out_claims: &[&Claim<F>],
    transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    assert!(witnesses.len() == 1, "Reducer expects 1 input");
    let alpha: F = transcript.challenge_scalar(b"reducer_alpha");
    let alphas = calc_pow(alpha, out_claims.len());
    let mut proofs = Vec::new();
    let mut claims = Vec::new();
    let x = witnesses[0];
    let mut sumcheck_prover = LinearSumcheckProver::new(x.data.as_ref().unwrap().n(), 2, transcript);

    let mut eq_partial_polys = vec![];
    let mut eq_points = Vec::with_capacity(out_claims.len());
    for claim in out_claims.iter() {
      eq_points.push(claim.point.clone());
    }
    for claim in out_claims {
      let eq_evaluations = evaluate_lagrange_basis(&claim.point);
      eq_partial_polys.push(DenseMLPoly::new(x.data.as_ref().unwrap().n(), eq_evaluations));
    }
    let eq_poly = (1..eq_partial_polys.len()).fold(eq_partial_polys[0].clone(), |acc, i| {
      acc.add(&eq_partial_polys[i].mul_by_scalar(alphas[i]))
    });
    let x_dense = x.data.as_ref().unwrap().as_any().downcast_ref::<DenseMLPoly<F>>().expect("Expected DenseMLPoly for x").clone();
    let sumcheck_proof = sumcheck_prover.prove(&vec![x_dense, eq_poly], transcript);
    let challenges = sumcheck_prover.challenges;
    let claim_x = Claim {
      edge_id: edge_ids[0],
      sparse_id: 0,
      point: challenges.clone(),
      eval: x.data.as_ref().unwrap().evaluate_at_point(&challenges),
    };

    proofs.push(sumcheck_proof);
    claims.push(claim_x);
    (proofs, claims)
  }

  fn verify(&self, witnesses: &[&Witness<F>], claims: &[&Claim<F>], sumcheck_proofs: &[&SumcheckProof<F>], transcript: &mut Transcript<F>) -> bool {
    assert!(witnesses.len() == 1, "Reducer expects 1 input");
    let alpha: F = transcript.challenge_scalar(b"reducer_alpha");
    let alphas = calc_pow(alpha, claims.len() - 1);
    let x = witnesses[0];

    let mut eval = <F as CryptoField>::zero();
    for (i, claim) in claims[..claims.len() - 1].iter().enumerate() {
      eval = eval + claim.eval * alphas[i];
    }
    let mut sumcheck_verifier = SumcheckVerifier::new(get_n(&x.shape), 2, transcript);
    let (verification_result, challenges) = sumcheck_verifier.verify(transcript, sumcheck_proofs[0].round_messages.clone(), eval);
    let running_sum = match verification_result {
      Some(v) => v,
      None => {
        println!("verified reducer failed: sumcheck round check");
        return false;
      }
    };

    // Compute eq_eval = Σ_i alpha_i * eq(challenges, claim_i.point)
    let one = <F as CryptoField>::one();
    let mut eq_eval = <F as CryptoField>::zero();
    for (i, claim) in claims[..claims.len() - 1].iter().enumerate() {
      let mut eq = one;
      for j in 0..challenges.len() {
        eq = eq * (challenges[j] * claim.point[j] + (one - challenges[j]) * (one - claim.point[j]));
      }
      eq_eval = eq_eval + alphas[i] * eq;
    }

    // Final eval check: running_sum == x_eval * eq_eval
    let x_eval = claims[claims.len() - 1].eval;
    let expected = x_eval * eq_eval;
    if running_sum != expected {
      println!("verified reducer failed: final_eval check mismatch");
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
    assert!(witnesses.len() == 1, "Reducer expects 1 input");
    let alpha: F = transcript.challenge_scalar(b"reducer_alpha");
    let alphas = calc_pow(alpha, out_claims.len());
    let x = witnesses[0];
    let mut zk_prover = ZkLinearSumcheckProver::new(x.data.as_ref().unwrap().n(), 2, transcript);

    let mut eq_partial_polys = vec![];
    for claim in out_claims {
      let eq_evaluations = evaluate_lagrange_basis(&claim.point);
      eq_partial_polys.push(DenseMLPoly::new(x.data.as_ref().unwrap().n(), eq_evaluations));
    }
    let eq_poly = (1..eq_partial_polys.len()).fold(eq_partial_polys[0].clone(), |acc, i| {
      acc.add(&eq_partial_polys[i].mul_by_scalar(alphas[i]))
    });
    let x_dense = x.data.as_ref().unwrap().as_any().downcast_ref::<DenseMLPoly<F>>().expect("Expected DenseMLPoly for x").clone();
    let sumcheck_proof = zk_prover.prove_zk(&vec![x_dense, eq_poly], transcript, ctx);
    let challenges = zk_prover.challenges;
    let claim_x = Claim {
      edge_id: edge_ids[0],
      sparse_id: 0,
      point: challenges.clone(),
      eval: x.data.as_ref().unwrap().evaluate_at_point(&challenges),
    };

    (vec![sumcheck_proof], vec![claim_x])
  }

  fn verify_zk(
    &self,
    witnesses: &[&Witness<F>],
    claims: &[&Claim<F>],
    sumcheck_proofs: &[&SumcheckProof<F>],
    transcript: &mut Transcript<F>,
    ctx: &ProveContext<F>,
  ) -> bool {
    if !ctx.zk {
      return self.verify(witnesses, claims, sumcheck_proofs, transcript);
    }
    assert!(witnesses.len() == 1, "Reducer expects 1 input");
    let alpha: F = transcript.challenge_scalar(b"reducer_alpha");
    let alphas = calc_pow(alpha, claims.len() - 1);
    let x = witnesses[0];

    let mut eval = <F as CryptoField>::zero();
    for (i, claim) in claims[..claims.len() - 1].iter().enumerate() {
      eval = eval + claim.eval * alphas[i];
    }
    let mut sumcheck_verifier = SumcheckVerifier::new(get_n(&x.shape), 2, transcript);
    let mask_sum = match sumcheck_proofs[0].zk_mask_sum {
      Some(v) => v,
      None => {
        println!("verified zk reducer failed: missing mask_sum in ZK proof");
        return false;
      }
    };
    let (verification_result, challenges, rho) = sumcheck_verifier.verify_zk(
      transcript, sumcheck_proofs[0].round_messages.clone(), eval, mask_sum,
      sumcheck_proofs[0].zk_mask_commitment.as_deref(),
    );
    let running_sum = match verification_result {
      Some(v) => v,
      None => {
        println!("verified zk reducer failed: sumcheck round check");
        return false;
      }
    };

    // Compute eq_eval = Σ_i alpha_i * eq(challenges, claim_i.point)
    let one = <F as CryptoField>::one();
    let mut eq_eval = <F as CryptoField>::zero();
    for (i, claim) in claims[..claims.len() - 1].iter().enumerate() {
      let mut eq = one;
      for j in 0..challenges.len() {
        eq = eq * (challenges[j] * claim.point[j] + (one - challenges[j]) * (one - claim.point[j]));
      }
      eq_eval = eq_eval + alphas[i] * eq;
    }

    // Final eval check (ZK): running_sum == x_eval * eq_eval + rho * mask_eval
    let x_eval = claims[claims.len() - 1].eval;
    let mask_eval = match sumcheck_proofs[0].zk_mask_coeffs.as_ref() {
      Some(coeffs) => recompute_mask_eval(coeffs, &challenges),
      None => <F as CryptoField>::zero(),
    };
    let expected = x_eval * eq_eval + rho * mask_eval;
    if running_sum != expected {
      println!("verified zk reducer failed: final_eval check mismatch");
      return false;
    }

    // Fix 4: Verify mask_sum consistency
    if !verify_mask_consistency(&sumcheck_proofs[0], ctx.mask_committer.as_deref()) {
      println!("verified zk reducer failed: mask consistency check");
      return false;
    }

    // Replay mask opening challenges to keep transcript in sync with prover
    replay_mask_opening_transcript(&sumcheck_proofs[0], transcript);
    true
  }
}

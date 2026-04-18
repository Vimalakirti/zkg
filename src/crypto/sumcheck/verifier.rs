use crate::util::poly::{evaluate_univariate_polynomial, CryptoField};
use crate::util::transcript::Transcript;

/// Verifier for sumcheck protocols with Fiat-Shamir
/// num_var: number of variables
/// num_poly: number of polynomials
pub struct SumcheckVerifier<F: CryptoField> {
  num_var: usize,
  num_poly: usize,
  _phantom: std::marker::PhantomData<F>,
}

impl<F: CryptoField + Send + Sync + 'static> SumcheckVerifier<F> {
  pub fn new(num_var: usize, num_poly: usize, transcript: &mut Transcript<F>) -> Self {
    // Commit to the same problem instance as prover
    transcript.append_u64(b"num_var", num_var as u64);
    transcript.append_u64(b"num_poly", num_poly as u64);

    Self {
      num_var,
      num_poly,
      _phantom: std::marker::PhantomData,
    }
  }

  /// Verify the sum-check proof
  pub fn verify(&mut self, transcript: &mut Transcript<F>, round_messages: Vec<Vec<F>>, claimed_sum: F) -> (Option<F>, Vec<F>) {
    if round_messages.len() != self.num_var {
      println!("Round messages length mismatch: {} != {}", round_messages.len(), self.num_var);
      return (None, vec![]);
    }

    let mut challenges = Vec::new();
    let mut running_sum = claimed_sum;

    // Verify each round
    for (round, round_message) in round_messages.iter().enumerate() {
      // Degree bound check: round polynomial must have exactly num_poly + 1 evaluations
      if round_message.len() != self.num_poly + 1 {
        println!(
          "Round {} degree mismatch: got {} evals, expected {}",
          round, round_message.len(), self.num_poly + 1
        );
        return (None, vec![]);
      }
      if round_message[0] + round_message[1] != running_sum {
        println!(
          "Round {} messages mismatch: {:?} != {:?}",
          round,
          round_message[0] + round_message[1],
          running_sum
        );
        return (None, vec![]);
      }

      // Append round message to transcript
      for (_i, &msg) in round_message.iter().enumerate() {
        transcript.append_scalar(b"round_message", &msg);
      }

      // Generate the same Fiat-Shamir challenge
      let challenge: F = transcript.challenge_scalar(b"challenge");
      // interpolate a univariate polynomial through the round messages (uses global cache)
      running_sum = evaluate_univariate_polynomial(round_message, challenge);

      challenges.push(challenge);
    }

    // In a complete implementation, would verify final evaluation
    // against the multilinear extension at the challenge point
    (Some(running_sum), challenges)
  }

  /// Verify a ZK sumcheck proof (masking polynomial technique).
  ///
  /// Returns (final_running_sum, challenges, rho) where rho is the masking scalar.
  /// The caller must additionally verify: final_eval == g(r) + ρ * mask_eval
  /// using a PCS opening proof on the mask commitment.
  /// Verify a ZK sumcheck proof (masking polynomial technique).
  ///
  /// Returns (final_running_sum, challenges, rho).
  /// The caller must additionally verify the final_eval check including the mask term.
  pub fn verify_zk(
    &mut self,
    transcript: &mut Transcript<F>,
    round_messages: Vec<Vec<F>>,
    claimed_sum: F,
    mask_sum: F,
    mask_commitment: Option<&[u8]>,
  ) -> (Option<F>, Vec<F>, F) {
    // Replay mask commitment bytes to transcript (must match prover's order)
    if let Some(commitment) = mask_commitment {
      transcript.append_bytes(b"zk_mask_commit", commitment);
    }
    // Reconstruct ρ from transcript
    transcript.append_scalar(b"zk_mask_sum", &mask_sum);
    let rho: F = transcript.challenge_scalar(b"zk_rho");

    // Adjusted sum: H + ρ·P
    let adjusted_sum = claimed_sum + rho * mask_sum;

    // Standard sumcheck verification on the adjusted sum
    let (result, challenges) = self.verify(transcript, round_messages, adjusted_sum);
    (result, challenges, rho)
  }
}

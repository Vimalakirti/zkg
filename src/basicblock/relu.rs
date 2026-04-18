use crate::basicblock::BasicBlock;
use crate::crypto::SumcheckProof;
use crate::dag::{Claim, Role, Witness};
use crate::util::arith::{f_to_int, next_pow};
use crate::util::poly::CryptoField;
use crate::util::transcript::Transcript;

/// SignBitHelper: advice op that computes s = sign_bit(x).
/// Input: x (1 input) → Output: s (1 output)
/// s[i] = 1 if x[i] >= 0, s[i] = 0 if x[i] < 0.
/// s is unscaled (sf = 0): it is literally 0 or 1 in the field.
///
/// Soundness comes from:
///   y = s * x (Einsum, algebraically enforced)
///   s * (1 - s) = 0 (binary check via NonNeg with table_size_log = 1)
///   NonNeg(y): y >= 0 — catches s=1 when x<0
///   NonNeg(y - x): y - x >= 0 — catches s=0 when x>=0
#[derive(Debug, Clone)]
pub struct SignBitHelper;

impl<F: CryptoField> BasicBlock<F> for SignBitHelper {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    assert_eq!(inputs.len(), 1, "SignBitHelper expects 1 input");
    let x = inputs[0];
    let x_shape = &x
      .shape
      .iter()
      .map(|&s| next_pow(s as u32) as usize)
      .collect::<Vec<usize>>();
    let n: usize = x_shape.iter().product();

    let mut sign_data = vec![<F as CryptoField>::zero(); n];
    for i in 0..n {
      let x_i = x.data.as_ref().unwrap().index(i);
      let x_i_num = f_to_int(x_i);
      if x_i_num >= 0 {
        sign_data[i] = <F as CryptoField>::one();
      }
      // else x_i is negative, sign_data[i] = 0 (already initialized)
    }

    vec![Witness::new(
      inputs[0].shape.clone(),
      sign_data,
      x.data_type,
      0, // sf = 0: s is unscaled (literal 0 or 1)
      Role::Output,
    )]
  }

  fn prove(
    &self,
    _witnesses: &[&Witness<F>],
    _edge_ids: &[usize],
    _out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    // Advice-only block: no sumcheck proof needed.
    // Soundness comes from the binary check s(1-s)=0, the Einsum y = s*x,
    // and the NonNeg range checks on y and y-x.
    (vec![], vec![])
  }

  fn verify(
    &self,
    _witnesses: &[&Witness<F>],
    _claims: &[&Claim<F>],
    _sumcheck_proofs: &[&SumcheckProof<F>],
    _transcript: &mut Transcript<F>,
  ) -> bool {
    true
  }
}

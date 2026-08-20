use crate::basicblock::BasicBlock;
use crate::crypto::SumcheckProof;
use crate::dag::{Claim, Role, Witness};
use crate::util::arith::{f_to_int, next_pow};
use crate::util::poly::CryptoField;
use crate::util::transcript::Transcript;
use crate::SF_LOG;

/// ElemDivHelper: advice op that computes z = floor(S * x / y) element-wise
/// for the positive numerator and divisor used by GAT attention normalization.
/// Input[0]: x (numerator), Input[1]: y (denominator)
/// Output: z (quotient, rounded toward zero)
///
/// Soundness comes from the DAG-level constraint verification:
/// 1. yz = Einsum(y, z) — verified product
/// 2. sx = ScaleUp(x) — S * x
/// 3. r = Sub(sx, yz) — derived remainder
/// 4. NonNegative(r) and NonNegative(y-r-1) — enforce 0 <= r < y
#[derive(Debug, Clone)]
pub struct ElemDivHelper;

impl<F: CryptoField> BasicBlock<F> for ElemDivHelper {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    assert_eq!(inputs.len(), 2, "ElemDivHelper expects 2 inputs (x, y)");
    let x = inputs[0];
    let y = inputs[1];
    let x_shape = &x
      .shape
      .iter()
      .map(|&s| next_pow(s as u32) as usize)
      .collect::<Vec<usize>>();
    let n: usize = x_shape.iter().product();

    let sf = *SF_LOG;
    let s = 1i128 << sf; // S = 2^sf

    let mut z_data = vec![<F as CryptoField>::zero(); n];
    for i in 0..n {
      let x_i = f_to_int(x.data.as_ref().unwrap().index(i));
      let y_i = f_to_int(y.data.as_ref().unwrap().index(i));

      if y_i != 0 {
        // x and y are positive in GAT, so Rust's integer division computes floor.
        let sx = s * x_i;
        let z_i = sx / y_i;
        z_data[i] = if z_i >= 0 {
          F::from(z_i as u32)
        } else {
          <F as CryptoField>::zero() - F::from((-z_i) as u32)
        };
      }
      // if y_i == 0, z_i = 0 (should not happen in valid GAT)
    }

    vec![Witness::new(
      inputs[0].shape.clone(),
      z_data,
      x.data_type,
      x.sf,
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
    // Soundness comes from the Einsum constraint (y*z verified) and
    // NonNegative range checks on r and y-r-1.
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

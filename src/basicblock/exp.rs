use crate::basicblock::BasicBlock;
use crate::crypto::SumcheckProof;
use crate::dag::{Claim, DataType, PolyType, Role, Witness};
use crate::{SF_FLOAT, SF_LOG};

use crate::util::arith::{f_to_int, get_n, next_pow};
use crate::util::poly::{SelectionPolynomial, SparseMLPoly};

use crate::util::poly::CryptoField;
use crate::util::transcript::Transcript;

/// Maximum k value for the exp decomposition. k_orig ranges [-K_MAX, K_MAX].
/// This controls the maximum input magnitude: exp handles |x| ≤ K_MAX * ln(2) * SF.
/// K_MAX=75 handles exp of real values up to ~52.
const K_MAX: usize = 75;

/// Number of bits needed for the selection polynomial (must hold [0, 2*K_MAX]).
const K_BITS: usize = 8; // 2^8 = 256 > 2*75 = 150

/// ExpHelper: decomposes input x into k_orig and remainder r such that
/// x = k_orig * (-ln2 * SF) + r.
///
/// To support both positive and negative inputs, k_orig ranges over [-K_MAX, K_MAX]
/// and is stored as k_shifted = k_orig + K_MAX in [0, 2*K_MAX].
///
/// Output[0]: r (dense polynomial, remainder for Taylor series)
/// Output[1]: auxiliary sparse polynomial (selection polynomial encoding k_shifted values)
#[derive(Debug, Clone)]
pub struct ExpHelper;
impl<F: CryptoField> BasicBlock<F> for ExpHelper {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    assert!(inputs.len() == 1, "ExpHelper expects 1 input");
    let x = inputs[0];
    let x_shape = &x.shape.iter().map(|&x| next_pow(x as u32) as usize).collect::<Vec<usize>>();
    let n: usize = x_shape.into_iter().product();
    let num_var = get_n(&x_shape);

    // Use f64 precision for ln2 computation
    let ln2 = (2.0_f64.ln() * (*SF_FLOAT as f64)).round() as i128;
    // neg_ln2_f = -ln2 in the field
    let neg_ln2_f = <F as CryptoField>::zero() - F::from(ln2 as u32);

    let mut r_data = vec![<F as CryptoField>::zero(); n];
    let mut selection = Vec::new();
    for i in 0..n {
      let x_i = x.data.as_ref().unwrap().index(i);
      let x_i_num = f_to_int(x_i) as f64;
      // k_orig = round(-x / (ln2*S)), then shift to unsigned: k_shifted = k_orig + K_MAX
      let k_orig = ((-x_i_num) / (ln2 as f64)).round().max(-(K_MAX as f64)).min(K_MAX as f64) as i32;
      let k_shifted = (k_orig + K_MAX as i32) as usize; // in [0, 2*K_MAX]

      // r = x - k_orig * neg_ln2_f = x + k_orig * ln2 (in field arithmetic)
      // k_orig = k_shifted - 15, so r = x - (k_shifted - 15) * neg_ln2_f
      let k_orig_f = if k_orig >= 0 {
        F::from(k_orig as u32)
      } else {
        <F as CryptoField>::zero() - F::from((-k_orig) as u32)
      };
      r_data[i] = x_i - (k_orig_f * neg_ln2_f);

      selection.push((i, k_shifted));
    }
    let selection_polynomial = SelectionPolynomial::new(num_var, K_BITS, selection);
    let r = Witness::new(inputs[0].shape.clone(), r_data, DataType::Float, *SF_LOG, Role::Output);
    let aux = Witness {
      shape: inputs[0].shape.clone(), // WORKAROUND: this shape is incorrect, but it is actually not used
      data: Some(Box::new(selection_polynomial.to_sparse()) as Box<dyn crate::util::poly::MLPoly<F>>),
      data_int: None,
      poly_type: PolyType::Sparse,
      data_type: x.data_type,
      sf: 0,
      role: Role::Auxiliary,
      factored: None,
      additive_factored: None,
    };
    vec![r, aux]
  }

  fn prove(
    &self,
    witnesses: &[&Witness<F>],
    edge_ids: &[usize],
    out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    let inp_claim = Claim {
      edge_id: edge_ids[0],
      sparse_id: 0,
      point: out_claims[0].point.clone(),
      eval: witnesses[0].data.as_ref().unwrap().evaluate_at_point(&out_claims[0].point),
    };
    (vec![], vec![inp_claim])
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

/// TwoPow: table lookup for 2^(K_MAX + 15 - k_shifted).
/// k_shifted = k_orig + K_MAX, so real value = 2^(K_MAX + 15 - k_shifted) / 2^15 = 2^(K_MAX - k_shifted) = 2^(-k_orig).
/// Input is a SparseMLPoly from ExpHelper containing the selection polynomial.
/// Output scale factor is always 15.
#[derive(Debug, Clone)]
pub struct TwoPow;
impl<F: CryptoField> BasicBlock<F> for TwoPow {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    assert!(inputs.len() == 1, "TwoPow expects 1 input");
    let x = inputs[0];
    let x_data = x.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap();
    let inp_num_vars = x_data.selection.input_num_vars;
    let selection = &x_data.selection.selection;
    let mut y_data = vec![<F as CryptoField>::zero(); 1 << inp_num_vars];
    let max_k_shifted = 2 * K_MAX;
    for (input_index, table_index) in selection {
      if *input_index < (1 << inp_num_vars) && *table_index <= max_k_shifted {
        // table[k_shifted] = 2^(K_MAX + 15 - k_shifted), representing 2^(-k_orig) at sf=15
        // When k_shifted > K_MAX + 15 (i.e., k_orig > 15), the exponent is negative,
        // meaning the real value is < 2^-1, which rounds to 0 at sf=15.
        if *table_index <= K_MAX + 15 {
          let exponent = K_MAX + 15 - table_index;
          // Build field element 2^exponent using repeated doubling to avoid u64 overflow
          let mut val = <F as CryptoField>::one();
          for _ in 0..exponent {
            val = val + val;
          }
          y_data[*input_index] = val;
        }
        // else: y_data[*input_index] stays 0 (exp of very negative input ≈ 0)
      }
    }
    let y = Witness::new(inputs[0].shape.clone(), y_data, DataType::Float, 15, Role::Output);
    vec![y]
  }

  fn prove(
    &self,
    _witnesses: &[&Witness<F>],
    _edge_ids: &[usize],
    _out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    // this will be batched proved later
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

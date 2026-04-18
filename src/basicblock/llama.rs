use crate::basicblock::BasicBlock;
use crate::crypto::SumcheckProof;
use crate::dag::{Claim, PolyType, Role, Witness};
use crate::util::arith::{f_to_int, get_n, next_pow};
use crate::util::poly::CryptoField;
use crate::util::poly::DenseMLPoly;
use crate::util::transcript::Transcript;

/// RMSReciprocal: computes 1/RMS(x) for RMSNorm.
/// Output shape: input shape with last dim = 1.
#[derive(Debug, Clone)]
pub struct RMSReciprocal;
impl<F: CryptoField> BasicBlock<F> for RMSReciprocal {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    let x = inputs[0];
    let mut y_shape = x.shape.clone();
    let ndim = y_shape.len();
    y_shape[ndim - 1] = 1;
    let n = get_n(&y_shape);
    let sf = (1u64 << x.sf) as f64;

    // Compute padded sizes and strides (little-endian: first dim has stride 1)
    let padded: Vec<usize> = x.shape.iter().map(|&s| next_pow(s as u32) as usize).collect();
    let d_pad = padded[ndim - 1]; // padded size of last dimension
    let stride_d: usize = padded[..ndim - 1].iter().product(); // stride for last dim
    let num_groups = stride_d;

    let mut result = Vec::with_capacity(num_groups);

    for g in 0..num_groups {
      let nn = d_pad as f64;
      let sum_sq: f64 = (0..d_pad)
        .map(|d| {
          let idx = g + d * stride_d;
          let xv = f_to_int(x.data.as_ref().unwrap().index(idx)) as f64 / sf;
          xv * xv
        })
        .sum();

      let rms = (sum_sq / nn).sqrt();
      let val = if rms == 0.0 {
        0i128
      } else {
        ((1.0 / rms) * sf).round() as i128
      };
      // 1/rms is always non-negative, but use proper signed encoding for safety
      let val_f = if val >= 0 {
        F::from(val as u32)
      } else {
        <F as CryptoField>::zero() - F::from((-val) as u32)
      };
      result.push(val_f);
    }

    // Pad to power-of-two size
    let total = 1usize << n;
    result.resize(total, <F as CryptoField>::zero());

    let y = Witness {
      shape: y_shape,
      data: Some(Box::new(DenseMLPoly::new(n, result)) as Box<dyn crate::util::poly::MLPoly<F>>),
      data_int: None,
      poly_type: PolyType::Dense,
      data_type: x.data_type,
      sf: x.sf,
      role: Role::Auxiliary,
      factored: None,
      additive_factored: None,
    };
    vec![y]
  }

  fn prove(
    &self,
    _witnesses: &[&Witness<F>],
    _edge_ids: &[usize],
    _out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
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

#[derive(Debug, Clone)]
pub struct DivConst {
  pub c: usize,
}
impl<F: CryptoField> BasicBlock<F> for DivConst {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    let x = inputs[0];
    let y_shape = x.shape.clone();
    let n = get_n(&y_shape);
    let c = self.c as f64;

    let x_shape = x.shape.iter().map(|&s| next_pow(s as u32) as usize).collect::<Vec<usize>>();
    let size: usize = x_shape.iter().product();
    let mut y_data = Vec::with_capacity(size);
    for i in 0..size {
      let x_i = x.data.as_ref().unwrap().index(i);
      let x_int = f_to_int(x_i) as f64;
      let y_int = (x_int / c).round() as i128;
      // Properly encode signed result back to field element
      let y_f = if y_int >= 0 {
        F::from(y_int as u32)
      } else {
        <F as CryptoField>::zero() - F::from((-y_int) as u32)
      };
      y_data.push(y_f);
    }

    let y = Witness {
      shape: y_shape,
      data: Some(Box::new(DenseMLPoly::new(n, y_data)) as Box<dyn crate::util::poly::MLPoly<F>>),
      data_int: None,
      poly_type: PolyType::Dense,
      data_type: x.data_type,
      sf: x.sf,
      role: Role::Auxiliary,
      factored: None,
      additive_factored: None,
    };
    vec![y]
  }

  fn prove(
    &self,
    _witnesses: &[&Witness<F>],
    _edge_ids: &[usize],
    _out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
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

/// SoftmaxConst: computes the per-row max subtraction constant for softmax numerical stability.
/// softmax_c[i] = -max(row) for all i in the same row (last dimension group).
/// Used as: scores = scores + softmax_c, then exp(scores).
#[derive(Debug, Clone)]
pub struct SoftmaxConst {
  pub dim: usize,
}
impl<F: CryptoField> BasicBlock<F> for SoftmaxConst {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    let x = inputs[0];
    let y_shape = x.shape.clone();
    let n = get_n(&y_shape);
    let ndim = x.shape.len();
    let padded: Vec<usize> = x.shape.iter().map(|&s| next_pow(s as u32) as usize).collect();
    let d_pad = padded[ndim - 1]; // padded size of last dimension
    let stride_d: usize = padded[..ndim - 1].iter().product(); // stride for last dim
    let num_groups = stride_d;
    let total_size: usize = padded.iter().product();

    let mut y_data = vec![<F as CryptoField>::zero(); total_size];

    for g in 0..num_groups {
      // Find max over real (non-padding) elements in last dim
      let mut max_val = i128::MIN;
      for d in 0..self.dim {
        let idx = g + d * stride_d;
        let val = f_to_int(x.data.as_ref().unwrap().index(idx));
        if val > max_val {
          max_val = val;
        }
      }

      // Set c = -max for all elements in the group
      let neg_max = if max_val >= 0 {
        <F as CryptoField>::zero() - F::from(max_val as u32)
      } else {
        F::from((-max_val) as u32)
      };
      for d in 0..d_pad {
        let idx = g + d * stride_d;
        y_data[idx] = neg_max;
      }
    }

    let y = Witness {
      shape: y_shape,
      data: Some(Box::new(DenseMLPoly::new(n, y_data)) as Box<dyn crate::util::poly::MLPoly<F>>),
      data_int: None,
      poly_type: PolyType::Dense,
      data_type: x.data_type,
      sf: x.sf,
      role: Role::Auxiliary,
      factored: None,
      additive_factored: None,
    };
    vec![y]
  }

  fn prove(
    &self,
    _witnesses: &[&Witness<F>],
    _edge_ids: &[usize],
    _out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
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

/// SigmoidConst: computes constant c such that exp((x+c)/sf) = sigmoid(x/sf).
/// sigmoid(t) = 1/(1+exp(-t)), so ln(sigmoid(t)) = -softplus(-t).
/// We need (x+c)/sf = ln(sigmoid(x/sf)), so c = -sf * softplus(-x/sf) - x.
#[derive(Debug, Clone)]
pub struct SigmoidConst;
impl<F: CryptoField> BasicBlock<F> for SigmoidConst {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    let x = inputs[0];
    let y_shape = x.shape.clone();
    let n = get_n(&y_shape);
    let sf = (1u64 << x.sf) as f64;

    let x_shape = x.shape.iter().map(|&s| next_pow(s as u32) as usize).collect::<Vec<usize>>();
    let size: usize = x_shape.iter().product();
    let mut y_data = Vec::with_capacity(size);

    for i in 0..size {
      let x_i = x.data.as_ref().unwrap().index(i);
      let x_int = f_to_int(x_i) as f64;
      let x_f = x_int / sf;
      // softplus(-x_f) = ln(1 + exp(-x_f)), numerically stable
      let t = -x_f;
      let sp = if t > 20.0 {
        t
      } else if t < -20.0 {
        0.0
      } else {
        (1.0_f64 + t.exp()).ln()
      };
      let c = -sf * sp - x_int;
      let c_int = c.round() as i128;
      let c_f = if c_int >= 0 {
        F::from(c_int as u32)
      } else {
        <F as CryptoField>::zero() - F::from((-c_int) as u32)
      };
      y_data.push(c_f);
    }

    let y = Witness {
      shape: y_shape,
      data: Some(Box::new(DenseMLPoly::new(n, y_data)) as Box<dyn crate::util::poly::MLPoly<F>>),
      data_int: None,
      poly_type: PolyType::Dense,
      data_type: x.data_type,
      sf: x.sf,
      role: Role::Auxiliary,
      factored: None,
      additive_factored: None,
    };
    vec![y]
  }

  fn prove(
    &self,
    _witnesses: &[&Witness<F>],
    _edge_ids: &[usize],
    _out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
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

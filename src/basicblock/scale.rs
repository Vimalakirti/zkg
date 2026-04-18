use crate::basicblock::BasicBlock;
use crate::crypto::SumcheckProof;
use crate::dag::{Claim, PolyType, Role, Witness};

use crate::util::arith::{f_to_int, get_n, next_pow};
use crate::util::poly::SelectionPolynomial;

use crate::util::poly::CryptoField;
use crate::util::transcript::Transcript;

#[derive(Debug, Clone)]
pub struct ScaleDown {
  pub input_sf: usize,
  pub output_sf: usize,
}
impl<F: CryptoField> BasicBlock<F> for ScaleDown {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    assert!(inputs.len() == 1, "ScaleDown expects 1 input");
    let x = inputs[0];
    let x_shape = &x.shape.iter().map(|&x| next_pow(x as u32) as usize).collect::<Vec<usize>>();
    let n: usize = x_shape.into_iter().product();
    let num_var = get_n(&x_shape);
    assert!(
      self.input_sf >= self.output_sf,
      "ScaleDown input SF must be greater than or equal to output SF"
    );
    let rescale_sf = self.input_sf - self.output_sf;
    let rescale_factor = 1 << rescale_sf;
    let rescale_factor_divided_by_2 = rescale_factor / 2;
    let rescale_factor_f = F::from(rescale_factor as u32);

    let mut y_data = vec![<F as CryptoField>::zero(); n];
    let mut selection = Vec::new();
    for i in 0..n {
      let x_i = x.data.as_ref().unwrap().index(i);
      let mut x_i_num = f_to_int(x_i) as f64;
      x_i_num /= rescale_factor as f64;

      // Properly encode signed result back to field element
      let mut rounded = x_i_num.round() as i64;
      y_data[i] = if rounded >= 0 {
        F::from(rounded as u32)
      } else {
        <F as CryptoField>::zero() - F::from((-rounded) as u32)
      };

      let aux_data = x_i - (y_data[i] * rescale_factor_f);
      let mut aux_num = f_to_int(aux_data) + rescale_factor_divided_by_2;
      // Fix boundary: if remainder == rescale/2, the aux_num equals rescale_factor
      // which overflows the table (max index = rescale_factor - 1). Round y up instead.
      if aux_num >= rescale_factor as i128 {
        rounded += 1;
        y_data[i] = if rounded >= 0 {
          F::from(rounded as u32)
        } else {
          <F as CryptoField>::zero() - F::from((-rounded) as u32)
        };
        let aux_data = x_i - (y_data[i] * rescale_factor_f);
        aux_num = f_to_int(aux_data) + rescale_factor_divided_by_2;
      }
      selection.push((i, aux_num as usize));
    }
    let selection_polynomial = SelectionPolynomial::new(num_var, rescale_sf, selection);
    let y = Witness::new(inputs[0].shape.clone(), y_data, x.data_type, self.output_sf, Role::Output);
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
    vec![y, aux]
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

#[derive(Debug, Clone)]
pub struct ScaleUp {
  pub input_sf: usize,
  pub output_sf: usize,
}
impl<F: CryptoField> BasicBlock<F> for ScaleUp {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    assert!(inputs.len() == 1, "ScaleUp expects 1 input");
    let x = inputs[0];
    let x_shape = &x.shape.iter().map(|&x| next_pow(x as u32) as usize).collect::<Vec<usize>>();
    let n: usize = x_shape.into_iter().product();
    let num_var = get_n(&x_shape);
    assert!(
      self.input_sf <= self.output_sf,
      "ScaleUp input SF must be less than or equal to output SF"
    );
    let rescale_sf = self.output_sf - self.input_sf;
    let rescale_factor = 1 << rescale_sf;
    let rescale_factor_divided_by_2 = rescale_factor / 2;
    let rescale_factor_f = F::from(rescale_factor as u32);

    let mut y_data = vec![<F as CryptoField>::zero(); n];
    let mut selection = Vec::new();
    for i in 0..n {
      let x_i = x.data.as_ref().unwrap().index(i);
      let x_i_num = f_to_int(x_i);
      let y_int = x_i_num * rescale_factor as i128;

      // Properly encode signed result back to field element
      y_data[i] = if y_int >= 0 {
        F::from(y_int as u32)
      } else {
        <F as CryptoField>::zero() - F::from((-y_int) as u32)
      };

      let aux_data = x_i * rescale_factor_f - y_data[i];
      let aux_num = f_to_int(aux_data) + rescale_factor_divided_by_2;
      selection.push((i, aux_num as usize));
    }
    let selection_polynomial = SelectionPolynomial::new(num_var, rescale_sf, selection);
    let y = Witness::new(inputs[0].shape.clone(), y_data, x.data_type, self.output_sf, Role::Output);
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
    vec![y, aux]
  }

  fn prove(
    &self,
    witnesses: &[&Witness<F>],
    edge_ids: &[usize],
    out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    // simply pass the claim
    let claim = vec![Claim {
      edge_id: edge_ids[0],
      sparse_id: 0,
      point: out_claims[0].point.clone(),
      eval: witnesses[0].data.as_ref().unwrap().evaluate_at_point(&out_claims[0].point),
    }];
    (vec![], claim)
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

use crate::basicblock::BasicBlock;
use crate::crypto::SumcheckProof;
use crate::dag::{Claim, Witness, PolyType};

use crate::util::arith::get_n;
use crate::util::poly::{CryptoField, SparseMLPoly};
use crate::util::transcript::Transcript;

#[derive(Debug, Clone)]
pub struct ChangeShape {
  pub new_shape: Vec<usize>,
}
impl<F: CryptoField + 'static> BasicBlock<F> for ChangeShape {
  fn run(&self, inputs: &[&Witness<F>]) -> Vec<Witness<F>> {
    assert!(inputs.len() == 1, "ChangeShape expects 1 input");
    let mut output = inputs[0].clone();
    let new_n = get_n(&self.new_shape);
    // Update sparse polynomial's variable count to match the new shape
    if output.poly_type == PolyType::Sparse {
      if let Some(ref data) = output.data {
        let sparse = data.as_any().downcast_ref::<SparseMLPoly<F>>().unwrap();
        if sparse.n != new_n {
          let mut new_sparse = sparse.clone();
          new_sparse.n = new_n;
          output.data = Some(Box::new(new_sparse));
        }
      }
    }
    output.shape = self.new_shape.clone();
    return vec![output];
  }

  fn prove(
    &self,
    _witnesses: &[&Witness<F>],
    edge_ids: &[usize],
    out_claims: &[&Claim<F>],
    _transcript: &mut Transcript<F>,
  ) -> (Vec<SumcheckProof<F>>, Vec<Claim<F>>) {
    // simply pass the claim
    let claim = vec![Claim {
      edge_id: edge_ids[0],
      sparse_id: 0,
      point: out_claims[0].point.clone(),
      eval: out_claims[0].eval,
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

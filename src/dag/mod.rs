use crate::util::serialization::{ark_de, ark_de_vec, ark_se, ark_se_vec};
use crate::util::transcript::Transcript;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ndarray::ArrayD;
use plonky2::{timed, util::timing::TimingTree};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;

use crate::basicblock::{reducer::Reducer, BasicBlock, BasicBlockType};
use crate::crypto::sumcheck::prover::SumcheckProver;
use crate::crypto::{polycommit::{Commitment, MLPolyCommit}, SumcheckProof};
use crate::crypto::{GeneralLinearSumcheckProver, LinearSumcheckProver, SumcheckVerifier};
use crate::util::arith::calc_pow;
use crate::util::arith::{f_to_int, get_n};
use crate::util::poly::CryptoField;
use crate::util::poly::{evaluate_lagrange_basis, get_cached_inverses, range_dense, two_pow_dense, AdditiveFactoredPoly, DenseMLPoly, FactoredDensePoly, MLPoly, SelectionPolynomial, SparseMLPoly};
use crate::TABLE_COMMIT_LOG;

#[cfg(feature = "arkworks")]
use rayon::prelude::*;

pub mod bert;
pub mod builder;
pub mod dense;
pub mod gnn;
pub mod gpt2;
pub mod gptj;
pub mod llama;

pub use builder::DagBuilder;
pub use dense::dense_add_relu;

pub type NodeId = usize;
pub type EdgeId = usize;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct AliasId(usize); // per-(edge, consumer node, input slot) alias

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
  Auxiliary,
  Constant,
  Input,
  Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
  Uint,
  Int,
  Bool,
  Float,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolyType {
  Dense,
  Sparse,
}

#[derive(Debug)]
pub struct Witness<F: CryptoField + 'static> {
  pub shape: Vec<usize>,
  pub data: Option<Box<dyn MLPoly<F>>>, // Could either be a dense or sparse multilinear polynomial
  pub data_int: Option<Vec<i128>>,      // integer representation of the data, only collected for dense and not auxiliary polys
  pub poly_type: PolyType,
  pub data_type: DataType,
  pub sf: usize,
  pub role: Role,
  /// Factored representation for large 2D dense polynomials.
  /// When set, commit/prove/verify use the factors instead of the full polynomial.
  pub factored: Option<FactoredDensePoly<F>>,
  /// Additive factored representation for sparse polynomials (adjacency matrices).
  /// When set, commit/prove/verify use the factor terms for PCS verification.
  pub additive_factored: Option<AdditiveFactoredPoly<F>>,
}

impl<F: CryptoField + 'static> Witness<F> {
  pub fn clear_data(&mut self) {
    self.data = None;
    self.data_int = None;
    // Keep additive_factored — needed by verifier for factor sizes/structure
  }

  pub fn get(&self, indices: &[usize]) -> F {
    let shape_next_pow = self.shape.iter().map(|&s| s.next_power_of_two()).collect::<Vec<usize>>();
    let index = indices.iter().enumerate().map(|(i, &index)| index * shape_next_pow[..i].iter().fold(1, |acc, &x| acc * x)).sum();
    self.data.as_ref().unwrap().index(index)
  }

  pub fn set(&mut self, indices: &[usize], value: F) {
    let shape_next_pow = self.shape.iter().map(|&s| s.next_power_of_two()).collect::<Vec<usize>>();
    let index = indices.iter().enumerate().map(|(i, &index)| index * shape_next_pow[..i].iter().fold(1, |acc, &x| acc * x)).sum();
    *self.data.as_mut().unwrap().index_mut(index) = value;
  }

  pub fn ndarray(&self) -> ArrayD<i128> {
    let shape = self.shape.clone().into_iter().rev().map(|s| s.next_power_of_two()).collect::<Vec<usize>>();
    let data = self
      .data
      .as_ref()
      .unwrap()
      .as_any()
      .downcast_ref::<DenseMLPoly<F>>()
      .unwrap()
      .evaluations
      .par_iter()
      .map(|f| f_to_int(*f))
      .collect::<Vec<i128>>();
    ArrayD::from_shape_vec(shape, data).unwrap().reversed_axes()
  }
}

impl<F: CryptoField + 'static> Witness<F> {
  pub fn new(shape: Vec<usize>, data: Vec<F>, data_type: DataType, sf: usize, role: Role) -> Self {
    let n = get_n(&shape);
    let data_int = data.par_iter().map(|f| f_to_int(*f)).collect::<Vec<i128>>();
    Self {
      shape,
      data: Some(Box::new(DenseMLPoly::new(n, data))),
      data_int: Some(data_int),
      poly_type: PolyType::Dense,
      data_type,
      sf,
      role,
      factored: None,
      additive_factored: None,
    }
  }

  pub fn new_sparse(shape: Vec<usize>, data: SparseMLPoly<F>, data_type: DataType, sf: usize, role: Role) -> Self {
    Self {
      shape,
      data: Some(Box::new(data)),
      data_int: None,
      poly_type: PolyType::Sparse,
      data_type,
      sf,
      role,
      factored: None,
      additive_factored: None,
    }
  }

  pub fn new_wo_data(shape: Vec<usize>, data_type: DataType, sf: usize, role: Role) -> Self {
    Self {
      shape,
      data: None,
      data_int: None,
      poly_type: PolyType::Dense,
      data_type,
      sf,
      role,
      factored: None,
      additive_factored: None,
    }
  }

  /// Factorize this witness's dense polynomial into smaller factors.
  /// The witness must be a 2D dense polynomial (shape has exactly 2 elements).
  /// chunk_sizes specifies how to split the column variables.
  pub fn factorize(&mut self, chunk_sizes: &[usize]) -> Result<(), String> {
    if self.poly_type != PolyType::Dense {
      return Err("Can only factorize dense polynomials".to_string());
    }
    if self.shape.len() != 2 {
      return Err(format!("Expected 2D shape, got {:?}", self.shape));
    }
    let row_vars = crate::util::arith::log2_ceil(self.shape[0]) as usize;
    let poly = self
      .data
      .as_ref()
      .ok_or("No data")?
      .as_any()
      .downcast_ref::<DenseMLPoly<F>>()
      .ok_or("Not a DenseMLPoly")?;
    let factored = FactoredDensePoly::factorize(poly, row_vars, chunk_sizes)?;
    self.factored = Some(factored);
    Ok(())
  }

  /// Decompose this witness's sparse polynomial into additive factored form.
  /// The witness must be a 2D sparse polynomial (adjacency matrix).
  /// num_shares controls how many groups the column variables are split into.
  pub fn additive_factorize(&mut self, num_shares: usize) -> Result<(), String> {
    if self.poly_type != PolyType::Sparse {
      return Err("Can only additive-factorize sparse polynomials".to_string());
    }
    if self.shape.len() != 2 {
      return Err(format!("Expected 2D shape, got {:?}", self.shape));
    }
    let row_vars = crate::util::arith::log2_ceil(self.shape[0]) as usize;
    let col_vars = crate::util::arith::log2_ceil(self.shape[1]) as usize;
    let poly = self
      .data
      .as_ref()
      .ok_or("No data")?
      .as_any()
      .downcast_ref::<SparseMLPoly<F>>()
      .ok_or("Not a SparseMLPoly")?;
    let af = AdditiveFactoredPoly::decompose(poly, row_vars, col_vars, num_shares)?;
    self.additive_factored = Some(af);
    Ok(())
  }

  /// Additively factorize a sparse matrix and pad its factorization to the
  /// authenticated public capacity used by data-independent zkGNN.
  pub fn additive_factorize_with_capacity(
    &mut self,
    num_shares: usize,
    capacity: usize,
  ) -> Result<(), String> {
    self.additive_factorize(num_shares)?;
    self
      .additive_factored
      .as_mut()
      .expect("factorization was just constructed")
      .pad_to_capacity(capacity)
  }

  /// Twist and Shout factorization for one-hot-per-column SelectionPolynomial
  /// witnesses (e.g., GAT incidence matrices). Transposes the polynomial so
  /// each row has exactly one nonzero, giving t=1 additive terms.
  pub fn selection_factorize(&mut self, num_shares: usize) -> Result<(), String> {
    if self.poly_type != PolyType::Sparse {
      return Err("Can only selection-factorize sparse polynomials".to_string());
    }
    let poly = self
      .data
      .as_ref()
      .ok_or("No data")?
      .as_any()
      .downcast_ref::<SparseMLPoly<F>>()
      .ok_or("Not a SparseMLPoly")?;
    if poly.selection.table_num_vars == 0 {
      return Err("Not a SelectionPolynomial (table_num_vars == 0)".to_string());
    }
    let af = AdditiveFactoredPoly::selection_decompose(poly, num_shares)?;
    self.additive_factored = Some(af);
    Ok(())
  }
}

impl<F: CryptoField + 'static> Clone for Witness<F> {
  fn clone(&self) -> Self {
    Self {
      shape: self.shape.clone(),
      data: self.data.as_ref().map(|d| d.clone_box()),
      data_int: self.data_int.clone(),
      poly_type: self.poly_type,
      data_type: self.data_type,
      sf: self.sf,
      role: self.role,
      factored: self.factored.clone(),
      additive_factored: self.additive_factored.clone(),
    }
  }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "F: CanonicalSerialize + CanonicalDeserialize")]
pub struct LookupProof<F: CryptoField + 'static> {
  pub table_proofs: Vec<SumcheckProof<F>>,
  #[serde(serialize_with = "se_nested_vec", deserialize_with = "de_nested_vec")]
  pub middle_claims: Vec<Vec<F>>,
  pub bool_proofs: Vec<SumcheckProof<F>>,
}

/// Wrapper struct for LookupProof without middle_claims (for proof size measurement)
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "F: CanonicalSerialize + CanonicalDeserialize")]
pub struct LookupProofOnly<F: CryptoField + 'static> {
  pub table_proofs: Vec<SumcheckProof<F>>,
  pub bool_proofs: Vec<SumcheckProof<F>>,
}

impl<F: CryptoField + 'static> LookupProofOnly<F> {
  pub fn from_lookup_proof(proof: &LookupProof<F>) -> Self
  where
    F: Clone,
  {
    Self {
      table_proofs: proof.table_proofs.clone(),
      bool_proofs: proof.bool_proofs.clone(),
    }
  }
}

/// Extract only the proofs from LookupProof (excluding middle_claims)
pub fn extract_lookup_proof_only<F: CryptoField + 'static>(proof: &LookupProof<F>) -> LookupProofOnly<F>
where
  F: Clone,
{
  LookupProofOnly::from_lookup_proof(proof)
}

/// Serialize nested Vec<Vec<F>> for arkworks types
fn se_nested_vec<S, A: CanonicalSerialize>(a: &Vec<Vec<A>>, s: S) -> Result<S::Ok, S::Error>
where
  S: serde::Serializer,
{
  use serde::ser::SerializeSeq;
  let mut seq = s.serialize_seq(Some(a.len()))?;
  for inner in a {
    let mut inner_bytes: Vec<Vec<u8>> = Vec::with_capacity(inner.len());
    for elem in inner {
      let mut bytes = vec![];
      elem.serialize_compressed(&mut bytes).map_err(serde::ser::Error::custom)?;
      inner_bytes.push(bytes);
    }
    seq.serialize_element(&inner_bytes)?;
  }
  seq.end()
}

/// Deserialize nested Vec<Vec<F>> for arkworks types
fn de_nested_vec<'de, D, A: CanonicalDeserialize>(data: D) -> Result<Vec<Vec<A>>, D::Error>
where
  D: serde::de::Deserializer<'de>,
{
  let v: Vec<Vec<Vec<u8>>> = serde::de::Deserialize::deserialize(data)?;
  v.into_iter()
    .map(|inner| inner.into_iter().map(|bytes| A::deserialize_compressed_unchecked(bytes.as_slice()).map_err(serde::de::Error::custom)).collect())
    .collect()
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
  serialize = "F: CanonicalSerialize + CanonicalDeserialize, DP::Proof: Serialize, SP::Proof: Serialize",
  deserialize = "F: CanonicalSerialize + CanonicalDeserialize, DP::Proof: for<'a> Deserialize<'a>, SP::Proof: for<'a> Deserialize<'a>"
))]
pub struct EdgeProof<F: CryptoField + 'static, DP: MLPolyCommit<F, DenseMLPoly<F>>, SP: MLPolyCommit<F, SparseMLPoly<F>>> {
  pub claims: Vec<Claim<F>>,
  pub dense_opening_proof: Vec<DP::Proof>,
  pub sparse_opening_proof: Vec<SP::Proof>,
  /// Sumcheck proofs for factored/additive-factored polynomial opening (one per claim).
  #[serde(skip)]
  pub factored_sumcheck_proofs: Vec<SumcheckProof<F>>,
}

impl<F: CryptoField + 'static, DP: MLPolyCommit<F, DenseMLPoly<F>>, SP: MLPolyCommit<F, SparseMLPoly<F>>> EdgeProof<F, DP, SP> {
  pub fn new() -> Self {
    Self {
      claims: Vec::new(),
      dense_opening_proof: Vec::new(),
      sparse_opening_proof: Vec::new(),
      factored_sumcheck_proofs: Vec::new(),
    }
  }
}

/// Wrapper struct for EdgeProof opening proofs only (excluding claims)
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
  serialize = "DP::Proof: Serialize, SP::Proof: Serialize",
  deserialize = "DP::Proof: for<'a> Deserialize<'a>, SP::Proof: for<'a> Deserialize<'a>"
))]
pub struct EdgeOpeningProof<F: CryptoField + 'static, DP: MLPolyCommit<F, DenseMLPoly<F>>, SP: MLPolyCommit<F, SparseMLPoly<F>>> {
  pub dense_opening_proof: Vec<DP::Proof>,
  pub sparse_opening_proof: Vec<SP::Proof>,
  #[serde(skip)]
  _phantom: std::marker::PhantomData<F>,
}

impl<F: CryptoField + 'static, DP: MLPolyCommit<F, DenseMLPoly<F>>, SP: MLPolyCommit<F, SparseMLPoly<F>>> EdgeOpeningProof<F, DP, SP> {
  pub fn from_edge_proof(edge_proof: &EdgeProof<F, DP, SP>) -> Self
  where
    DP::Proof: Clone,
    SP::Proof: Clone,
  {
    Self {
      dense_opening_proof: edge_proof.dense_opening_proof.clone(),
      sparse_opening_proof: edge_proof.sparse_opening_proof.clone(),
      _phantom: std::marker::PhantomData,
    }
  }
}

/// ZK proof data: placeholder for DAG-level ZK information.
/// Mask polynomial coefficients are embedded in each SumcheckProof's `zk_mask_coeffs` field.
pub struct ZkProof<F: CryptoField + 'static, DP: MLPolyCommit<F, DenseMLPoly<F>>> {
  _phantom_f: std::marker::PhantomData<F>,
  _phantom_dp: std::marker::PhantomData<DP>,
}

impl<F: CryptoField + 'static, DP: MLPolyCommit<F, DenseMLPoly<F>>> ZkProof<F, DP> {
  pub fn empty() -> Self {
    Self {
      _phantom_f: std::marker::PhantomData,
      _phantom_dp: std::marker::PhantomData,
    }
  }
}

/// Extract only sumcheck proofs from sumcheck_proofs (excluding claims)
pub fn extract_sumcheck_proofs_only<F: CryptoField + CanonicalSerialize + CanonicalDeserialize>(
  sumcheck_proofs: &[Option<(Vec<SumcheckProof<F>>, Vec<Claim<F>>)>],
) -> Vec<Option<Vec<SumcheckProof<F>>>>
where
  F: Clone,
{
  sumcheck_proofs.iter().map(|opt| opt.as_ref().map(|(proofs, _claims)| proofs.clone())).collect()
}

/// Extract only opening proofs from edge proofs (excluding claims)
pub fn extract_opening_proofs_only<F, DP, SP>(opening_proofs: &[EdgeProof<F, DP, SP>]) -> Vec<EdgeOpeningProof<F, DP, SP>>
where
  F: CryptoField + 'static,
  DP: MLPolyCommit<F, DenseMLPoly<F>>,
  SP: MLPolyCommit<F, SparseMLPoly<F>>,
  DP::Proof: Clone,
  SP::Proof: Clone,
{
  opening_proofs.iter().map(|ep| EdgeOpeningProof::from_edge_proof(ep)).collect()
}

impl<F: CryptoField + 'static, DP: MLPolyCommit<F, DenseMLPoly<F>>, SP: MLPolyCommit<F, SparseMLPoly<F>>> Clone for EdgeProof<F, DP, SP>
where
  DP::Proof: Clone,
  SP::Proof: Clone,
{
  fn clone(&self) -> Self {
    Self {
      claims: self.claims.clone(),
      dense_opening_proof: self.dense_opening_proof.clone(),
      sparse_opening_proof: self.sparse_opening_proof.clone(),
      factored_sumcheck_proofs: self.factored_sumcheck_proofs.clone(),
    }
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: CanonicalSerialize + CanonicalDeserialize")]
pub struct Claim<F: CryptoField> {
  pub edge_id: EdgeId,
  pub sparse_id: usize,
  #[serde(serialize_with = "ark_se_vec", deserialize_with = "ark_de_vec")]
  pub point: Vec<F>,
  #[serde(serialize_with = "ark_se", deserialize_with = "ark_de")]
  pub eval: F,
}

#[derive(Debug, Clone)]
pub struct Node {
  id: NodeId,
  kind: BasicBlockType,
  inputs: Vec<EdgeId>,  // physical edges
  outputs: Vec<EdgeId>, // physical edges (1..N)
}

#[derive(Debug)]
pub struct Dag {
  nodes: Vec<Node>,
  num_edges: usize,
  topo: Vec<NodeId>,
  range: Vec<NodeId>,   // currently only support non-negative range
  two_pow: Vec<NodeId>, // nodes that compute 2^(-k)
  // Physical connectivity
  consumers: Vec<Vec<NodeId>>,    // edge -> consumer node list
  producers: Vec<Option<NodeId>>, // edge -> producing node (None for graph inputs)
  input_ports: Vec<EdgeId>,
  output_ports: Vec<EdgeId>,

  // ----- Alias view (for readability in backward passes) -----
  // One alias per (edge, consumer node, consumer input slot).
  edge_aliases: Vec<Vec<AliasId>>, // physical edge -> alias list
  alias_to_edge: Vec<EdgeId>,
  alias_to_consumer: Vec<NodeId>,
  alias_input_slot: Vec<usize>, // which input index on the consumer
}

impl Dag {
  pub fn num_edges(&self) -> usize {
    self.num_edges
  }

  /// Collect all unique polynomial sizes (number of variables) from witnesses in the DAG.
  /// This is used to determine which SRS sizes need to be generated/loaded during setup.
  /// For factored witnesses, collects factor sizes instead of the full polynomial size.
  pub fn collect_polynomial_sizes<F: CryptoField + 'static>(&self, witnesses: &[Vec<Witness<F>>]) -> BTreeSet<usize> {
    let mut sizes = BTreeSet::new();

    for witness_opt in witnesses.iter() {
      for witness in witness_opt.iter() {
        if let Some(ref factored) = witness.factored {
          // Use factor sizes instead of full polynomial size
          for (i, _factor) in factored.factors.iter().enumerate() {
            sizes.insert(factored.row_vars + factored.chunk_sizes[i]);
          }
        } else if let Some(ref af) = witness.additive_factored {
          // Additive factored: only factor sizes are needed. The full sparse poly is
          // never committed/opened, so we don't need an SRS for its size.
          for i in 0..af.chunk_sizes.len() {
            sizes.insert(af.row_vars + af.chunk_sizes[i]);
          }
        } else if let Some(ref poly_data) = witness.data {
          let n = poly_data.n();
          sizes.insert(n);
        } else {
          let n = get_n(&witness.shape);
          sizes.insert(n);
        }
      }
    }

    sizes
  }

  /// Internal evaluator that produces values for **all** edges.
  /// Returns pre-split copies of sparse witnesses (indexed by edge id),
  /// needed by SpMV/GCN prove methods which read edge lists from the original polynomial.
  pub fn run<F: CryptoField + 'static>(&self, witnesses: &mut [Vec<Witness<F>>], feed: &[(EdgeId, Witness<F>)]) -> Vec<Option<Witness<F>>> {
    assert_eq!(witnesses.len(), self.num_edges, "init vec length must match num_edges");

    // Feed can override (graph inputs, params, anything)
    for (eid, t) in feed {
      witnesses[*eid] = vec![t.clone()];
    }

    // Evaluate in topological order
    for &nid in &self.topo {
      let node = &self.nodes[nid];

      let in_refs: Vec<&Witness<F>> = node.inputs.iter().map(|&e| &witnesses[e][0]).collect();

      // 1) compute
      println!("running node {} | kind {:?}", nid, node.kind);
      let outs = node.kind.run(&in_refs);

      // 2) publish
      assert_eq!(outs.len(), node.outputs.len(), "op output arity mismatch");
      for (&eid, out) in node.outputs.iter().zip(outs.into_iter()) {
        witnesses[eid] = vec![out];
      }
    }

    // Save pre-split sparse witnesses (needed for SpMV/GCN prove)
    let mut presplit_sparse: Vec<Option<Witness<F>>> = vec![None; witnesses.len()];
    for (e, witness) in witnesses.iter().enumerate() {
      if witness[0].poly_type == PolyType::Sparse {
        presplit_sparse[e] = Some(witness[0].clone());
      }
    }

    // Split sparse witnesses into blocks for manageable SRS sizes.
    // Sparse polys with additive_factored (adjacency, incidence) are committed/opened
    // via the factored path; the blocks here are only used for SpMV evaluation reference.
    for witness in witnesses.iter_mut() {
      let w = &witness[0];
      if w.poly_type == PolyType::Sparse {
        let af = w.additive_factored.clone();
        let poly = w.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap();
        let polys = poly.split_into_blocks(*TABLE_COMMIT_LOG);
        let mut new_witness: Vec<Witness<F>> = polys
          .iter()
          .map(|poly| Witness::new_sparse(w.shape.clone(), poly.clone(), w.data_type, w.sf, w.role))
          .collect();
        // Preserve additive_factored on the first (representative) witness
        if af.is_some() {
          new_witness[0].additive_factored = af;
        }
        *witness = new_witness;
      }
    }

    presplit_sparse
  }

  fn should_commit<F: CryptoField + 'static>(&self, witness: &Witness<F>, edge_id: EdgeId) -> bool {
    match witness.role {
      Role::Constant | Role::Auxiliary | Role::Input => true,
      Role::Output => self.consumers[edge_id].is_empty(),
    }
  }

  pub fn commit<F, DP, SP>(
    &self,
    dense_key: &DP::CommitmentKey,
    sparse_key: &SP::CommitmentKey,
    witnesses: &[Vec<Witness<F>>],
    dense_commitments: &mut [Option<DP::Commitment>],
    sparse_commitments: &mut [Option<Vec<SP::Commitment>>],
    factored_commitments: &mut [Option<Vec<DP::Commitment>>],
    timing: &mut TimingTree,
  ) where
    F: CryptoField + 'static,
    DP: MLPolyCommit<F, DenseMLPoly<F>>,
    SP: MLPolyCommit<F, SparseMLPoly<F>>,
    DP::CommitmentKey: Sync,
    DP::Commitment: Send,
    DenseMLPoly<F>: Sync,
  {
    for (e, witness) in witnesses.iter().enumerate() {
      if witness.len() > 0 {
        for w in witness.iter() {
          if self.should_commit(w, e) && w.data.is_some() {
            match w.poly_type {
              PolyType::Dense => {
                // If factored, commit to each factor instead of the full polynomial
                if let Some(ref factored) = w.factored {
                  if factored_commitments[e].is_none() {
                    #[cfg(feature = "arkworks")]
                    let factor_comms: Vec<DP::Commitment> = factored
                      .factors
                      .par_iter()
                      .map(|factor| DP::commit(factor, dense_key))
                      .collect();
                    #[cfg(not(feature = "arkworks"))]
                    let factor_comms: Vec<DP::Commitment> = factored
                      .factors
                      .iter()
                      .map(|factor| DP::commit(factor, dense_key))
                      .collect();
                    println!("  committed {} factors for factored edge {}", factor_comms.len(), e);
                    factored_commitments[e] = Some(factor_comms);
                  }
                } else if dense_commitments[e].is_none() {
                  let poly = w.data.as_ref().unwrap().as_any().downcast_ref::<DenseMLPoly<F>>().unwrap();
                  if w.role == Role::Constant {
                    timed!(timing, format!("commit to constant for edge {}", e).as_str(), {
                      dense_commitments[e] = Some(DP::commit(poly, dense_key));
                    });
                  } else {
                    dense_commitments[e] = Some(DP::commit(poly, dense_key));
                  }
                }
              }
              PolyType::Sparse => {
                // If additive factored, commit only the factor polynomials via dense PCS.
                // The full sparse poly is never opened/verified in that path, so the sparse
                // commit (and its huge SRS requirement) is unnecessary.
                if let Some(ref af) = w.additive_factored {
                  if factored_commitments[e].is_none() {
                    // Collect all factors into a flat vec, then commit in parallel
                    let all_factors: Vec<&DenseMLPoly<F>> = af.terms.iter()
                      .flat_map(|term_factors| term_factors.iter())
                      .collect();
                    #[cfg(feature = "arkworks")]
                    let factor_comms: Vec<DP::Commitment> = all_factors
                      .par_iter()
                      .map(|f| DP::commit(f, dense_key))
                      .collect();
                    #[cfg(not(feature = "arkworks"))]
                    let factor_comms: Vec<DP::Commitment> = all_factors
                      .iter()
                      .map(|f| DP::commit(f, dense_key))
                      .collect();
                    println!("  committed {} additive factors for sparse edge {} ({} terms × {} shares)",
                      factor_comms.len(), e, af.terms.len(), af.chunk_sizes.len());
                    factored_commitments[e] = Some(factor_comms);
                  }
                } else {
                  let poly = w.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap();
                  if sparse_commitments[e].is_none() {
                    sparse_commitments[e] = Some(vec![]);
                  }
                  sparse_commitments[e].as_mut().unwrap().push(SP::commit(poly, sparse_key));
                }
              }
            }
          }
        }
      }
    }
  }

  pub fn prove<F, DP, SP>(
    &self,
    dense_key: &DP::CommitmentKey,
    sparse_key: &SP::CommitmentKey,
    witnesses: &[Vec<Witness<F>>],
    presplit_sparse: &[Option<Witness<F>>],
    dense_commitments: &[Option<DP::Commitment>],
    sparse_commitments: &[Option<Vec<SP::Commitment>>],
    factored_commitments: &[Option<Vec<DP::Commitment>>],
    transcript: &mut Transcript<F>,
    timing: &mut TimingTree,
  ) -> (
    Vec<Option<(Vec<SumcheckProof<F>>, Vec<Claim<F>>)>>,
    Vec<EdgeProof<F, DP, SP>>,
    LookupProof<F>,
    LookupProof<F>,
    Vec<Option<Vec<SumcheckProof<F>>>>,
  )
  where
    F: CryptoField + 'static + Sync,
    DP: MLPolyCommit<F, DenseMLPoly<F>>,
    SP: MLPolyCommit<F, SparseMLPoly<F>>,
    DP::Proof: Clone + Send,
    SP::Proof: Clone + Send,
    DP::CommitmentKey: Sync,
    SP::CommitmentKey: Sync,
    DP::Commitment: Sync,
    SP::Commitment: Sync,
    DenseMLPoly<F>: Sync,
    SparseMLPoly<F>: Sync,
  {
    // 0. initialize proofs with pre-allocated capacity
    let reducer = BasicBlockType::Reducer(Reducer {});
    let mut node_proofs: Vec<Option<(Vec<SumcheckProof<F>>, Vec<Claim<F>>)>> = Vec::with_capacity(self.nodes.len());
    node_proofs.resize_with(self.nodes.len(), || None);
    let mut reducer_proofs: Vec<Option<Vec<SumcheckProof<F>>>> = Vec::with_capacity(self.nodes.len());
    reducer_proofs.resize_with(self.nodes.len(), || None);
    let mut edge_proofs: Vec<EdgeProof<F, DP, SP>> = Vec::with_capacity(self.num_edges);
    edge_proofs.resize_with(self.num_edges, || EdgeProof::new());

    // Absorb all polynomial commitments into Fiat-Shamir transcript before drawing challenges
    for comm in dense_commitments.iter() {
      if let Some(c) = comm {
        transcript.append_bytes(b"dense_commitment", &c.to_transcript_bytes());
      }
    }
    for comms in sparse_commitments.iter() {
      if let Some(cs) = comms {
        for c in cs {
          transcript.append_bytes(b"sparse_commitment", &c.to_transcript_bytes());
        }
      }
    }
    for comms in factored_commitments.iter() {
      if let Some(cs) = comms {
        for c in cs {
          transcript.append_bytes(b"factored_commitment", &c.to_transcript_bytes());
        }
      }
    }

    // 1. open the final output polynomials at some random points
    let mut claims: Vec<Vec<Claim<F>>> = Vec::with_capacity(self.num_edges);
    claims.resize_with(self.num_edges, || Vec::new());
    let mut nodes_to_prove = BTreeSet::new();
    self.output_ports.iter().for_each(|&e| {
      let w = &witnesses[e][0];
      if w.role == Role::Output {
        let point: Vec<F> = (0..w.data.as_ref().unwrap().n()).map(|_| transcript.challenge_scalar(b"challenge")).collect();
        let eval = w.data.as_ref().unwrap().evaluate_at_point(&point);
        claims[e].push(Claim {
          edge_id: e,
          sparse_id: 0,
          point,
          eval,
        });
        if self.consumers[e].len() > 0 && matches!(self.nodes[self.consumers[e][0]].kind, BasicBlockType::NonNegative(_)) {
          node_proofs[self.consumers[e][0]] = Some((vec![], vec![claims[e][0].clone()]));
        }
        nodes_to_prove.insert(self.producers[e].unwrap());
      }
    });

    // 2. prove from outputs to inputs by looping the following steps:
    // 2.1 pop a node from nodes_to_prove
    // 2.2 check if all the consumers of the node output(s) are not in nodes_to_prove
    // 2.2.1 if no, then add the node to preserve_nodes, and pop a node from nodes_to_prove and go to 2.2
    // 2.2.2 if yes, then go to 2.3
    // 2.3 proving and adding the claims to claims
    // 2.4 add the producer of the node input to preserve_nodes
    let mut preserve_nodes = BTreeSet::new();
    // Pre-compute consumer sets for faster lookups
    let consumer_sets: Vec<BTreeSet<NodeId>> = self.consumers.iter().map(|consumers| consumers.iter().copied().collect()).collect();

    while !nodes_to_prove.is_empty() {
      // 2.1
      let node_id = nodes_to_prove.pop_last().unwrap();
      // 2.2 - Optimized consumer check using pre-computed sets
      let mut can_prove = true;
      for &edge in &self.nodes[node_id].outputs {
        if consumer_sets[edge].iter().any(|&c| nodes_to_prove.contains(&c)) {
          can_prove = false;
          break;
        }
      }
      if !can_prove {
        nodes_to_prove.insert(node_id);
        continue;
      }
      // 2.3 - Optimized witness and claim collection
      let node = &self.nodes[node_id];

      // Avoid temporary Vec allocation by using slices directly when possible
      let total_edges = node.inputs.len() + node.outputs.len();
      let mut edge_ids = Vec::with_capacity(total_edges);
      edge_ids.extend_from_slice(&node.inputs);
      edge_ids.extend_from_slice(&node.outputs);

      // Pre-allocate with exact capacity
      // For sparse edges, use pre-split witness (needed by SpMV/GCN prove)
      let mut local_witnesses = Vec::with_capacity(edge_ids.len());
      for &e in &edge_ids {
        if let Some(ref unsplit) = presplit_sparse[e] {
          local_witnesses.push(unsplit);
        } else {
          local_witnesses.push(&witnesses[e][0]);
        }
      }

      // Optimized claim collection without flatten
      let mut local_claims = Vec::new();
      for &e in &node.outputs {
        local_claims.extend(claims[e].iter());
      }

      // if the node has multiple output claims, then we need to prove the reducer to reduce the claims
      let (reducer_proof, reducer_claims) = if local_claims.len() > 1 {
        println!("proving reducer for node {} | kind {:?}", node_id, self.nodes[node_id].kind);
        let reducer_witness = vec![&witnesses[node.outputs[0]][0]];
        let reducer_edge_ids = vec![edge_ids[edge_ids.len() - 1]];
        let (proofs, new_claims) = timed!(
          timing,
          format!("prove reducer for node {} | kind {:?}", node_id, self.nodes[node_id].kind).as_str(),
          { reducer.prove(&reducer_witness, &reducer_edge_ids, &local_claims, transcript) }
        );
        (proofs, new_claims)
      } else {
        (vec![], vec![])
      };
      if reducer_claims.len() > 0 {
        local_claims = reducer_claims.iter().collect();
        let output_edge_id = node.outputs[0];
        claims[output_edge_id].push(reducer_claims[0].clone());
        reducer_proofs[node_id] = Some(reducer_proof);
      }

      // prove the node
      println!("proving node {} | kind {:?}", node_id, self.nodes[node_id].kind);
      let node_start = std::time::Instant::now();
      let (proofs, new_claims) = timed!(
        timing,
        format!("prove node {} | kind {:?}", node_id, self.nodes[node_id].kind).as_str(),
        { self.nodes[node_id].kind.prove(&local_witnesses, &edge_ids, &local_claims, transcript) }
      );
      let node_elapsed = node_start.elapsed();
      let type_tag = match &self.nodes[node_id].kind {
        BasicBlockType::Einsum(_) => "MatMul",
        BasicBlockType::SpMV(_) => "SpMM",
        BasicBlockType::Add(_) | BasicBlockType::Sub(_) => "Add/Sub",
        BasicBlockType::SignBitHelper(_) => "SignBit",
        BasicBlockType::ScaleDown(_) | BasicBlockType::ScaleUp(_) => "Scale",
        BasicBlockType::NonNegative(_) => "RangeCheck",
        BasicBlockType::ExpHelper(_) | BasicBlockType::TwoPow(_) => "Exp",
        BasicBlockType::ElemDivHelper(_) => "ElemDiv",
        BasicBlockType::Reducer(_) => "Reducer",
        _ => "Other",
      };
      println!("BREAKDOWN|node|{}|{:.3}ms", type_tag, node_elapsed.as_secs_f64() * 1000.0);

      // Optimized node claims construction
      let mut node_claims = Vec::with_capacity(new_claims.len() + local_claims.len());
      node_claims.extend_from_slice(&new_claims);
      for claim in local_claims {
        node_claims.push(claim.clone());
      }
      node_proofs[node_id] = Some((proofs, node_claims));

      for c in new_claims {
        claims[c.edge_id].push(c);
      }
      // 2.4 - Optimized producer collection and set operations
      for &e in &node.inputs {
        if let Some(producer) = self.producers[e] {
          preserve_nodes.insert(producer);
        }
      }

      // Avoid expensive union + collect by extending in-place
      for &preserve_node in &preserve_nodes {
        nodes_to_prove.insert(preserve_node);
      }
      preserve_nodes.clear();
    }

    // Set node_proofs for NonNegative range nodes that weren't initialized during output port setup.
    // These nodes (e.g., from ReLU decomposition) consume internal edges, not output ports,
    // so they weren't covered by the output_ports loop above.
    for &n in &self.range {
      if node_proofs[n].is_none() && matches!(self.nodes[n].kind, BasicBlockType::NonNegative(_)) {
        let input_edge = self.nodes[n].inputs[0];
        if let Some(claim) = claims[input_edge].last() {
          node_proofs[n] = Some((vec![], vec![claim.clone()]));
        }
      }
    }

    claims.iter().enumerate().for_each(|(e, c)| edge_proofs[e].claims.extend(c.iter().cloned()));

    // 3. prove lookups
    let lookup_start = std::time::Instant::now();
    let (two_pow_table_proofs, two_pow_middle_claims, two_pow_bool_proofs) = self.prove_two_pow(witnesses, &mut claims, transcript, timing);
    let (range_table_proofs, range_middle_claims, range_bool_proofs) = self.prove_range(witnesses, &mut claims, &node_proofs, transcript, timing);
    println!("BREAKDOWN|lookup|Lookup|{:.3}ms", lookup_start.elapsed().as_secs_f64() * 1000.0);
    let two_pow_proof = LookupProof {
      table_proofs: two_pow_table_proofs,
      middle_claims: two_pow_middle_claims,
      bool_proofs: two_pow_bool_proofs,
    };
    let range_proof = LookupProof {
      table_proofs: range_table_proofs,
      middle_claims: range_middle_claims,
      bool_proofs: range_bool_proofs,
    };

    // 4. provide opening proofs for all the irreducible claims
    println!("proving opening proofs");
    let opening_start = std::time::Instant::now();
    timed!(timing, "provide opening proofs", {
      // Collect tasks by type: regular dense, factored dense, additive factored sparse, regular sparse
      let mut dense_tasks = Vec::new();
      let mut factored_tasks = Vec::new();
      let mut additive_factored_tasks = Vec::new();
      let mut sparse_tasks = Vec::new();

      for e in 0..self.num_edges {
        let w = &witnesses[e][0];
        if w.role == Role::Output {
          continue;
        }
        for c in &edge_proofs[e].claims {
          if w.poly_type == PolyType::Dense {
            if w.factored.is_some() {
              factored_tasks.push((e, c.clone()));
            } else {
              dense_tasks.push((e, c.clone()));
            }
          } else if w.additive_factored.is_some() {
            additive_factored_tasks.push((e, c.clone()));
          } else {
            sparse_tasks.push((e, c.clone()));
          }
        }
      }

      // 4a. Factored dense openings (sumcheck + factor openings)
      for (e, claim) in &factored_tasks {
        let w = &witnesses[*e][0];
        let factored = w.factored.as_ref().unwrap();
        let row_vars = factored.row_vars;
        let col_vars = factored.col_vars();

        // Split claim.point into (r_x, r_y)
        let r_x = &claim.point[..row_vars];
        let r_y = &claim.point[row_vars..row_vars + col_vars];

        // Prepare sumcheck polynomials: eq(r_x, ·) and f_i(·, r_{y_i})
        let (eq_poly, factor_polys) = factored.prepare_sumcheck_polys(r_x, r_y);

        // Build polynomial list: [eq, g_1, g_2, ..., g_m]
        let m = factored.factors.len();
        let mut polys = Vec::with_capacity(m + 1);
        polys.push(eq_poly);
        polys.extend(factor_polys);

        // Run sumcheck: Σ_{x'} eq(r_x, x') · Π_i f_i(x', r_{y_i}) = claim.eval
        let mut prover = LinearSumcheckProver::new(row_vars, m + 1, transcript);
        let sc_proof = prover.prove(&polys, transcript);

        // After sumcheck, get the challenge point x*
        let x_star = &prover.challenges;

        // Open each factor at (x*, r_{y_i}) — in parallel
        let factor_comms = factored_commitments[*e].as_ref().unwrap();
        let open_tasks: Vec<(usize, Vec<F>)> = {
          let mut tasks = Vec::with_capacity(factored.factors.len());
          let mut y_offset = 0;
          for (i, _factor) in factored.factors.iter().enumerate() {
            let k_i = factored.chunk_sizes[i];
            let r_yi = &r_y[y_offset..y_offset + k_i];
            y_offset += k_i;
            let mut opening_point = x_star.clone();
            opening_point.extend_from_slice(r_yi);
            tasks.push((i, opening_point));
          }
          tasks
        };

        #[cfg(feature = "arkworks")]
        let factor_proofs: Vec<_> = open_tasks.par_iter().map(|(i, opening_point)| {
          DP::open(&factor_comms[*i], &factored.factors[*i], dense_key, opening_point)
        }).collect();
        #[cfg(not(feature = "arkworks"))]
        let factor_proofs: Vec<_> = open_tasks.iter().map(|(i, opening_point)| {
          DP::open(&factor_comms[*i], &factored.factors[*i], dense_key, opening_point)
        }).collect();

        for proof in factor_proofs {
          edge_proofs[*e].dense_opening_proof.push(proof);
        }

        edge_proofs[*e].factored_sumcheck_proofs.push(sc_proof);
      }

      // 4a2. Additive factored sparse openings (sumcheck + factor openings)
      for (e, claim) in &additive_factored_tasks {
        let w = if let Some(ref unsplit) = presplit_sparse[*e] { unsplit } else { &witnesses[*e][0] };
        let af = w.additive_factored.as_ref().unwrap();
        let row_vars = af.row_vars;
        let col_vars = af.col_vars();
        let m = af.chunk_sizes.len();

        // For transposed decompositions (Twist and Shout), the claim point is
        // [input_vars, table_vars] but the decomposition expects [table_vars, input_vars].
        let (r_x, r_y) = if af.transposed {
          (&claim.point[col_vars..col_vars + row_vars], &claim.point[..col_vars])
        } else {
          (&claim.point[..row_vars], &claim.point[row_vars..row_vars + col_vars])
        };

        let (eq_poly, factor_polys_per_term) = af.prepare_sumcheck_inputs(r_x, r_y);

        let t = af.num_terms();
        let scalars: Vec<F> = vec![<F as CryptoField>::one(); t];
        let a_arrays: Vec<Vec<DenseMLPoly<F>>> = factor_polys_per_term;

        let mut prover = GeneralLinearSumcheckProver::<F>::new(row_vars, m + 1, transcript);
        let instance = (scalars, a_arrays, eq_poly);
        let sc_proof = prover.prove(&instance, transcript);
        let x_star = &prover.challenges;

        // Collect all factor opening tasks, then run in parallel
        let factor_comms = factored_commitments[*e].as_ref().unwrap();
        let all_factors: Vec<&DenseMLPoly<F>> = af.terms.iter()
          .flat_map(|term| term.iter())
          .collect();
        let open_tasks: Vec<(usize, Vec<F>)> = {
          let mut tasks = Vec::with_capacity(all_factors.len());
          let mut factor_idx = 0;
          for term in &af.terms {
            let mut y_offset = 0;
            for (i, _factor) in term.iter().enumerate() {
              let k_i = af.chunk_sizes[i];
              let r_yi = &r_y[y_offset..y_offset + k_i];
              y_offset += k_i;
              let mut opening_point = x_star.clone();
              opening_point.extend_from_slice(r_yi);
              tasks.push((factor_idx, opening_point));
              factor_idx += 1;
            }
          }
          tasks
        };

        #[cfg(feature = "arkworks")]
        let factor_proofs: Vec<_> = open_tasks.par_iter().map(|(idx, opening_point)| {
          DP::open(&factor_comms[*idx], &all_factors[*idx], dense_key, opening_point)
        }).collect();
        #[cfg(not(feature = "arkworks"))]
        let factor_proofs: Vec<_> = open_tasks.iter().map(|(idx, opening_point)| {
          DP::open(&factor_comms[*idx], &all_factors[*idx], dense_key, opening_point)
        }).collect();

        for proof in factor_proofs {
          edge_proofs[*e].dense_opening_proof.push(proof);
        }

        edge_proofs[*e].factored_sumcheck_proofs.push(sc_proof);
      }

      // 4b. Regular dense openings in parallel
      #[cfg(feature = "arkworks")]
      let dense_results: Vec<_> = dense_tasks
        .par_iter()
        .map(|(e, claim)| {
          let w = &witnesses[*e][0];
          let proof = DP::open(
            &dense_commitments[*e].as_ref().unwrap(),
            &w.data.as_ref().unwrap().as_any().downcast_ref::<DenseMLPoly<F>>().unwrap(),
            dense_key,
            &claim.point,
          );
          (*e, proof)
        })
        .collect();

      #[cfg(not(feature = "arkworks"))]
      let dense_results: Vec<_> = dense_tasks
        .iter()
        .map(|(e, claim)| {
          let w = witnesses[*e][0];
          let proof = DP::open(
            &dense_commitments[*e].as_ref().unwrap(),
            &w.data.as_ref().unwrap().as_any().downcast_ref::<DenseMLPoly<F>>().unwrap(),
            dense_key,
            &claim.point,
          );
          (*e, proof)
        })
        .collect();

      // 4c. Sparse openings in parallel
      #[cfg(feature = "arkworks")]
      let sparse_results: Vec<_> = sparse_tasks
        .par_iter()
        .map(|(e, claim)| {
          let w = &witnesses[*e][claim.sparse_id];
          let proof = SP::open(
            &sparse_commitments[*e].as_ref().unwrap()[claim.sparse_id],
            w.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap(),
            sparse_key,
            &claim.point,
          );
          (*e, proof)
        })
        .collect();

      #[cfg(not(feature = "arkworks"))]
      let sparse_results: Vec<_> = sparse_tasks
        .iter()
        .map(|(e, claim)| {
          let w = witnesses[*e][claim.sparse_id];
          let proof = SP::open(
            &sparse_commitments[*e].as_ref().unwrap()[claim.sparse_id],
            &w.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap(),
            sparse_key,
            &claim.point,
          );
          (*e, proof)
        })
        .collect();

      // Store dense results
      for (e, proof) in dense_results {
        edge_proofs[e].dense_opening_proof.push(proof);
      }

      // Store sparse results
      for (e, proof) in sparse_results {
        edge_proofs[e].sparse_opening_proof.push(proof);
      }
    });
    println!("BREAKDOWN|opening|Opening|{:.3}ms", opening_start.elapsed().as_secs_f64() * 1000.0);

    (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs)
  }

  pub fn verify<F, DP, SP>(
    &self,
    node_proofs: &[Option<(Vec<SumcheckProof<F>>, Vec<Claim<F>>)>],
    edge_proofs: &[EdgeProof<F, DP, SP>],
    range_proof: &LookupProof<F>,
    two_pow_proof: &LookupProof<F>,
    reducer_proofs: &[Option<Vec<SumcheckProof<F>>>],
    witnesses: &[Vec<Witness<F>>],
    dense_key: &DP::VerifierKey,
    sparse_key: &SP::VerifierKey,
    dense_commitments: &[Option<DP::Commitment>],
    sparse_commitments: &[Option<Vec<SP::Commitment>>],
    factored_commitments: &[Option<Vec<DP::Commitment>>],
    transcript: &mut Transcript<F>,
  ) -> bool
  where
    F: CryptoField + 'static,
    DP: MLPolyCommit<F, DenseMLPoly<F>>,
    SP: MLPolyCommit<F, SparseMLPoly<F>>,
    DP::Proof: Clone + Sync,
    SP::Proof: Clone + Sync,
    DP::VerifierKey: Sync,
    SP::VerifierKey: Sync,
    DP::Commitment: Sync,
    SP::Commitment: Sync,
  {
    let reducer = BasicBlockType::Reducer(Reducer {});
    let mut verified = true;
    let irreducable_edge_ids: Vec<EdgeId> =
      (0..self.num_edges).filter(|&e| self.producers[e].is_none() || witnesses[e][0].role == Role::Auxiliary).collect();

    // Absorb all polynomial commitments into Fiat-Shamir transcript before drawing challenges
    for comm in dense_commitments.iter() {
      if let Some(c) = comm {
        transcript.append_bytes(b"dense_commitment", &c.to_transcript_bytes());
      }
    }
    for comms in sparse_commitments.iter() {
      if let Some(cs) = comms {
        for c in cs {
          transcript.append_bytes(b"sparse_commitment", &c.to_transcript_bytes());
        }
      }
    }
    for comms in factored_commitments.iter() {
      if let Some(cs) = comms {
        for c in cs {
          transcript.append_bytes(b"factored_commitment", &c.to_transcript_bytes());
        }
      }
    }

    let mut claims: Vec<Vec<Claim<F>>> = Vec::with_capacity(self.num_edges);
    claims.resize_with(self.num_edges, || Vec::new());
    let mut nodes_to_verify = BTreeSet::new();
    self.output_ports.iter().for_each(|&e| {
      let w = &witnesses[e][0];
      if w.role == Role::Output {
        let point: Vec<F> = (0..get_n(&witnesses[e][0].shape)).map(|_| transcript.challenge_scalar(b"challenge")).collect();
        if edge_proofs[e].claims.is_empty() || edge_proofs[e].claims[0].point != point {
          verified = false;
        } else {
          claims[e].push(edge_proofs[e].claims[0].clone());
          match self.producers[e] {
            Some(producer) => { nodes_to_verify.insert(producer); }
            None => { verified = false; }
          }
        }
      }
    });

    let mut preserve_nodes = BTreeSet::new();
    // Pre-compute consumer sets for faster lookups
    let consumer_sets: Vec<BTreeSet<NodeId>> = self.consumers.iter().map(|consumers| consumers.iter().copied().collect()).collect();

    let start = std::time::Instant::now();
    while !nodes_to_verify.is_empty() {
      let u = nodes_to_verify.pop_last().unwrap();
      let mut can_verify = true;
      for &edge in &self.nodes[u].outputs {
        if consumer_sets[edge].iter().any(|&c| nodes_to_verify.contains(&c)) {
          can_verify = false;
          break;
        }
      }
      if !can_verify {
        nodes_to_verify.insert(u);
        continue;
      }
      let node = &self.nodes[u];

      let (local_sumcheck_proofs, local_claims) = match node_proofs[u].as_ref() {
        Some(p) => p,
        None => {
          println!("=== node proof missing for node {u} ===");
          verified = false;
          for &e in &node.inputs {
            if let Some(producer) = self.producers[e] {
              preserve_nodes.insert(producer);
            }
          }
          for &preserve_node in &preserve_nodes { nodes_to_verify.insert(preserve_node); }
          preserve_nodes.clear();
          continue;
        }
      };
      let local_claims: Vec<&Claim<F>> = local_claims.iter().collect();
      let local_witnesses: Vec<&Witness<F>> = local_claims.iter().map(|&c| &witnesses[c.edge_id][0]).collect();
      let local_sumcheck_proofs: Vec<&SumcheckProof<F>> = local_sumcheck_proofs.iter().collect();

      if claims[node.outputs[0]].len() > 1 {
        println!("verifying reducer for node {u} | kind {:?}", self.nodes[u].kind);
        let reducer_proof_ref = match reducer_proofs[u].as_ref() {
          Some(p) => p,
          None => {
            println!("=== reducer proof missing for node {u} ===");
            verified = false;
            continue;
          }
        };
        let reducer_witness = vec![&witnesses[node.outputs[0]][0]];
        let mut reducer_claims = claims[node.outputs[0]].clone();
        reducer_claims.push(local_claims[local_claims.len() - 1].clone());
        let reducer_claims: Vec<&Claim<F>> = reducer_claims.iter().collect();
        let reducer_sumcheck_proofs: Vec<&SumcheckProof<F>> = reducer_proof_ref.iter().collect();
        let reducer_verified = reducer.verify(&reducer_witness, &reducer_claims, &reducer_sumcheck_proofs, transcript);
        verified = verified && reducer_verified;
        if !reducer_verified {
          println!("=== verified reducer for node {u}: {reducer_verified} ===");
        }
      }

      println!("verifying node {u} | kind {:?}", self.nodes[u].kind);
      let node_verified = self.nodes[u].kind.verify(&local_witnesses, &local_claims, &local_sumcheck_proofs, transcript);
      if !node_verified {
        println!("=== verified node {u}: {node_verified} ===");
      }
      verified = verified && node_verified;

      for &e in &node.inputs {
        if let Some(producer) = self.producers[e] {
          preserve_nodes.insert(producer);
        }
      }

      // Avoid expensive union + collect by extending in-place
      for &preserve_node in &preserve_nodes {
        nodes_to_verify.insert(preserve_node);
      }
      preserve_nodes.clear();

      local_claims.iter().for_each(|&c| {
        if node.inputs.contains(&c.edge_id) {
          claims[c.edge_id].push(c.clone())
        }
      });
    }
    let end = start.elapsed();
    println!("time taken to verify nodes: {:?}", end);

    // verify the range proof and two_pow proof
    let start = std::time::Instant::now();
    let two_pow_verified = self.verify_two_pow(node_proofs, witnesses, two_pow_proof, transcript);
    let range_verified = self.verify_range(node_proofs, witnesses, range_proof, transcript);
    let end = start.elapsed();
    println!("time taken to verify two_pow and range: {:?}", end);
    if !two_pow_verified {
      println!("=== verified two_pow: {two_pow_verified} ===");
    }
    if !range_verified {
      println!("=== verified range: {range_verified} ===");
    }
    verified = verified && range_verified && two_pow_verified;

    // verify the irreducible edges (opening proofs)
    let start = std::time::Instant::now();

    // Separate factored, additive-factored, and regular edges
    let mut factored_edge_ids = Vec::new();
    let mut additive_factored_edge_ids = Vec::new();
    let mut regular_edge_ids = Vec::new();
    for &e in &irreducable_edge_ids {
      if witnesses[e][0].factored.is_some() {
        factored_edge_ids.push(e);
      } else if witnesses[e][0].additive_factored.is_some() {
        additive_factored_edge_ids.push(e);
      } else {
        regular_edge_ids.push(e);
      }
    }

    // Verify factored dense edge openings (sumcheck + factor openings)
    let mut factored_verified = true;
    for &e in &factored_edge_ids {
      let w = &witnesses[e][0];
      let factored = w.factored.as_ref().unwrap();
      if edge_proofs[e].factored_sumcheck_proofs.is_empty() {
        println!("Missing factored sumcheck proofs for edge {}", e);
        factored_verified = false;
        continue;
      }

      for (claim_idx, claim) in edge_proofs[e].claims.iter().enumerate() {
        let sc_proof = &edge_proofs[e].factored_sumcheck_proofs[claim_idx];
        let row_vars = factored.row_vars;
        let col_vars = factored.col_vars();
        let m = factored.factors.len();

        let mut sc_verifier = SumcheckVerifier::new(row_vars, m + 1, transcript);
        let (sc_result, sc_challenges) = sc_verifier.verify(
          transcript,
          sc_proof.round_messages.clone(),
          claim.eval,
        );
        let running_sum = match sc_result {
          Some(v) => v,
          None => {
            println!("Factored sumcheck verification failed for edge {}", e);
            factored_verified = false;
            continue;
          }
        };

        let r_x = &claim.point[..row_vars];
        let one = <F as CryptoField>::one();
        let eq_eval = r_x.iter().zip(sc_challenges.iter())
          .fold(one, |acc, (rx_j, xstar_j)| {
            acc * (*xstar_j * *rx_j + (one - *xstar_j) * (one - *rx_j))
          });

        assert_eq!(claim.point.len(), row_vars + col_vars,
          "Factored claim point should have row_vars + col_vars dimensions");

        let factor_comms = match factored_commitments[e].as_ref() {
          Some(c) => c,
          None => {
            println!("Missing factored commitments for edge {}", e);
            factored_verified = false;
            continue;
          }
        };
        // Verify factor PCS openings in parallel, extracting evaluations from proofs
        let proof_offset = claim_idx * factor_comms.len();
        // Build verification tasks
        let verify_tasks: Vec<(usize, Vec<F>)> = factor_comms.iter().enumerate().map(|(i, _)| {
          let mut opening_point = sc_challenges.clone();
          let k_i = factored.chunk_sizes[i];
          let y_start: usize = factored.chunk_sizes[..i].iter().sum();
          let r_yi = &claim.point[row_vars + y_start..row_vars + y_start + k_i];
          opening_point.extend_from_slice(r_yi);
          (i, opening_point)
        }).collect();

        let verify_results: Vec<(bool, F)> = verify_tasks.par_iter().map(|(i, opening_point)| {
          let global_proof_idx = proof_offset + *i;
          if global_proof_idx >= edge_proofs[e].dense_opening_proof.len() {
            return (false, <F as CryptoField>::one());
          }
          DP::verify_and_extract(
            &factor_comms[*i],
            &edge_proofs[e].dense_opening_proof[global_proof_idx],
            dense_key,
            opening_point,
          )
        }).collect();

        let mut factor_product = <F as CryptoField>::one();
        for (i, (ok, factor_eval)) in verify_results.into_iter().enumerate() {
          if !ok {
            println!("Factor opening verification failed for edge {} factor {}", e, i);
            factored_verified = false;
          }
          factor_product = factor_product * factor_eval;
        }

        // Final eval check: running_sum == eq_eval * Π_i f_i(x*, r_{y_i})
        let expected = eq_eval * factor_product;
        if running_sum != expected {
          println!("Factored final eval check failed for edge {}: running_sum != eq_eval * factor_product", e);
          factored_verified = false;
        }
      }
    }

    // Verify additive factored sparse edge openings
    for &e in &additive_factored_edge_ids {
      let w = &witnesses[e][0];
      let af = w.additive_factored.as_ref().unwrap();
      if edge_proofs[e].factored_sumcheck_proofs.is_empty() {
        println!("Missing additive factored sumcheck proofs for edge {}", e);
        factored_verified = false;
        continue;
      }

      let row_vars = af.row_vars;
      let col_vars = af.col_vars();
      let m = af.chunk_sizes.len();
      let t = af.num_terms();
      let factors_per_claim = m * t;

      let factor_comms = match factored_commitments[e].as_ref() {
        Some(c) => c,
        None => {
          println!("Missing additive factored commitments for edge {}", e);
          factored_verified = false;
          continue;
        }
      };

      for (claim_idx, claim) in edge_proofs[e].claims.iter().enumerate() {
        let sc_proof = &edge_proofs[e].factored_sumcheck_proofs[claim_idx];

        // Verify the degree-(m+1) sumcheck
        let mut sc_verifier = SumcheckVerifier::new(row_vars, m + 1, transcript);
        let (sc_result, sc_challenges) = sc_verifier.verify(
          transcript,
          sc_proof.round_messages.clone(),
          claim.eval,
        );
        let running_sum = match sc_result {
          Some(v) => v,
          None => {
            println!("Additive factored sumcheck verification failed for edge {}", e);
            factored_verified = false;
            continue;
          }
        };

        // Compute eq(r_x, x*)
        // For transposed decompositions: swap variable groups from claim point
        let (r_x, r_y) = if af.transposed {
          (&claim.point[col_vars..col_vars + row_vars], &claim.point[..col_vars])
        } else {
          (&claim.point[..row_vars], &claim.point[row_vars..row_vars + col_vars])
        };
        let one = <F as CryptoField>::one();
        let eq_eval = r_x.iter().zip(sc_challenges.iter())
          .fold(one, |acc, (rx_j, xstar_j)| {
            acc * (*xstar_j * *rx_j + (one - *xstar_j) * (one - *rx_j))
          });

        // Verify all m*t PCS openings in parallel, then compute factor evaluations
        // (verifier must NOT use polynomial data directly — adjacency is private in ZK mode)
        let proof_offset = claim_idx * factors_per_claim;

        // Build all verification tasks: (factor_idx, opening_point)
        let verify_tasks: Vec<(usize, Vec<F>)> = {
          let mut tasks = Vec::with_capacity(factors_per_claim);
          let mut factor_idx = 0;
          for _k in 0..t {
            let mut y_offset = 0;
            for i in 0..m {
              let k_i = af.chunk_sizes[i];
              let r_yi = &r_y[y_offset..y_offset + k_i];
              y_offset += k_i;
              let mut opening_point = sc_challenges.clone();
              opening_point.extend_from_slice(r_yi);
              tasks.push((factor_idx, opening_point));
              factor_idx += 1;
            }
          }
          tasks
        };

        // Run all verify_and_extract in parallel
        let verify_results: Vec<(bool, F)> = verify_tasks.par_iter().map(|(idx, opening_point)| {
          let global_proof_idx = proof_offset + *idx;
          if global_proof_idx >= edge_proofs[e].dense_opening_proof.len() || *idx >= factor_comms.len() {
            return (false, <F as CryptoField>::one());
          }
          DP::verify_and_extract(
            &factor_comms[*idx],
            &edge_proofs[e].dense_opening_proof[global_proof_idx],
            dense_key,
            opening_point,
          )
        }).collect();

        // Reconstruct sum_of_products from parallel results
        let mut sum_of_products = <F as CryptoField>::zero();
        let mut factor_idx = 0;
        for _k in 0..t {
          let mut product = <F as CryptoField>::one();
          for _i in 0..m {
            let (ok, factor_eval) = verify_results[factor_idx];
            if !ok {
              println!("Additive factor opening verification failed for edge {} factor {}", e, factor_idx);
              factored_verified = false;
            }
            product = product * factor_eval;
            factor_idx += 1;
          }
          sum_of_products = sum_of_products + product;
        }

        // Check: running_sum == eq_eval * Σ_k Π_i f_{k,i}(x*, r_{y_i})
        let expected = eq_eval * sum_of_products;
        if running_sum != expected {
          println!("Additive factored final eval check failed for edge {}: running_sum != eq_eval * sum_of_products", e);
          factored_verified = false;
        }
      }
    }

    // Verify regular (non-factored) edge openings in parallel
    let regular_opening_verified = regular_edge_ids.par_iter().all(|&e| {
      // Sparse edges may have no claims (adjacency matrices verified via SpMV sumcheck)
      if edge_proofs[e].claims.is_empty() && witnesses[e][0].poly_type == PolyType::Sparse {
        return true;
      }
      if witnesses[e][0].poly_type == PolyType::Dense {
        (0..edge_proofs[e].claims.len()).into_par_iter().all(|i| {
          DP::verify(
            dense_commitments[e].as_ref().unwrap(),
            &edge_proofs[e].dense_opening_proof[i],
            dense_key,
            &edge_proofs[e].claims[i].point,
          )
        })
      } else {
        (0..edge_proofs[e].claims.len()).into_par_iter().all(|i| {
          SP::verify(
            &sparse_commitments[e].as_ref().unwrap()[edge_proofs[e].claims[i].sparse_id],
            &edge_proofs[e].sparse_opening_proof[i],
            sparse_key,
            &edge_proofs[e].claims[i].point,
          )
        })
      }
    });
    let end = start.elapsed();
    println!("time taken to verify irreducible edges: {:?}", end);
    verified && factored_verified && regular_opening_verified
  }

  /// ZK-aware prove: runs standard prove but uses ZkLinearSumcheckProver for each node's sumcheck.
  /// After backward pass, commits each mask polynomial and produces opening proofs.
  /// Returns (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs, zk_proof).
  pub fn prove_zk<F, DP, SP>(
    &self,
    dense_key: &DP::CommitmentKey,
    sparse_key: &SP::CommitmentKey,
    witnesses: &[Vec<Witness<F>>],
    presplit_sparse: &[Option<Witness<F>>],
    dense_commitments: &[Option<DP::Commitment>],
    sparse_commitments: &[Option<Vec<SP::Commitment>>],
    factored_commitments: &[Option<Vec<DP::Commitment>>],
    transcript: &mut Transcript<F>,
    timing: &mut TimingTree,
    zk: bool,
    mask_committer: Option<Arc<dyn crate::crypto::MaskCommitter<F>>>,
  ) -> (
    Vec<Option<(Vec<SumcheckProof<F>>, Vec<Claim<F>>)>>,
    Vec<EdgeProof<F, DP, SP>>,
    LookupProof<F>,
    LookupProof<F>,
    Vec<Option<Vec<SumcheckProof<F>>>>,
    ZkProof<F, DP>,
  )
  where
    F: CryptoField + 'static + Sync,
    DP: MLPolyCommit<F, DenseMLPoly<F>>,
    SP: MLPolyCommit<F, SparseMLPoly<F>>,
    DP::Proof: Clone + Send,
    SP::Proof: Clone + Send,
    DP::CommitmentKey: Sync,
    SP::CommitmentKey: Sync,
    DP::Commitment: Sync + Clone,
    SP::Commitment: Sync,
    DenseMLPoly<F>: Sync,
    SparseMLPoly<F>: Sync,
  {
    if !zk {
      let (np, ep, rp, tp, rdp) = self.prove::<F, DP, SP>(
        dense_key, sparse_key, witnesses, presplit_sparse,
        dense_commitments, sparse_commitments, factored_commitments,
        transcript, timing,
      );
      return (np, ep, rp, tp, rdp, ZkProof::empty());
    }

    // ZK mode: create ProveContext and use prove_zk on each node
    let mut ctx = match mask_committer {
      Some(mc) => crate::crypto::ProveContext::<F>::with_mask_committer(true, mc),
      None => crate::crypto::ProveContext::<F>::new(true),
    };

    let reducer = BasicBlockType::Reducer(Reducer {});
    let mut node_proofs: Vec<Option<(Vec<SumcheckProof<F>>, Vec<Claim<F>>)>> = Vec::with_capacity(self.nodes.len());
    node_proofs.resize_with(self.nodes.len(), || None);
    let mut reducer_proofs: Vec<Option<Vec<SumcheckProof<F>>>> = Vec::with_capacity(self.nodes.len());
    reducer_proofs.resize_with(self.nodes.len(), || None);
    let mut edge_proofs: Vec<EdgeProof<F, DP, SP>> = Vec::with_capacity(self.num_edges);
    edge_proofs.resize_with(self.num_edges, || EdgeProof::new());

    // Absorb all polynomial commitments into Fiat-Shamir transcript before drawing challenges
    for comm in dense_commitments.iter() {
      if let Some(c) = comm {
        transcript.append_bytes(b"dense_commitment", &c.to_transcript_bytes());
      }
    }
    for comms in sparse_commitments.iter() {
      if let Some(cs) = comms {
        for c in cs {
          transcript.append_bytes(b"sparse_commitment", &c.to_transcript_bytes());
        }
      }
    }
    for comms in factored_commitments.iter() {
      if let Some(cs) = comms {
        for c in cs {
          transcript.append_bytes(b"factored_commitment", &c.to_transcript_bytes());
        }
      }
    }

    // 1. open the final output polynomials at some random points
    let mut claims: Vec<Vec<Claim<F>>> = Vec::with_capacity(self.num_edges);
    claims.resize_with(self.num_edges, || Vec::new());
    let mut nodes_to_prove = BTreeSet::new();
    self.output_ports.iter().for_each(|&e| {
      let w = &witnesses[e][0];
      if w.role == Role::Output {
        let point: Vec<F> = (0..w.data.as_ref().unwrap().n()).map(|_| transcript.challenge_scalar(b"challenge")).collect();
        let eval = w.data.as_ref().unwrap().evaluate_at_point(&point);
        claims[e].push(Claim { edge_id: e, sparse_id: 0, point, eval });
        if self.consumers[e].len() > 0 && matches!(self.nodes[self.consumers[e][0]].kind, BasicBlockType::NonNegative(_)) {
          node_proofs[self.consumers[e][0]] = Some((vec![], vec![claims[e][0].clone()]));
        }
        nodes_to_prove.insert(self.producers[e].unwrap());
      }
    });

    // 2. backward pass with ZK prove
    let mut preserve_nodes = BTreeSet::new();
    let consumer_sets: Vec<BTreeSet<NodeId>> = self.consumers.iter().map(|consumers| consumers.iter().copied().collect()).collect();

    while !nodes_to_prove.is_empty() {
      let node_id = nodes_to_prove.pop_last().unwrap();
      let mut can_prove = true;
      for &edge in &self.nodes[node_id].outputs {
        if consumer_sets[edge].iter().any(|&c| nodes_to_prove.contains(&c)) {
          can_prove = false;
          break;
        }
      }
      if !can_prove {
        nodes_to_prove.insert(node_id);
        continue;
      }
      let node = &self.nodes[node_id];

      let total_edges = node.inputs.len() + node.outputs.len();
      let mut edge_ids = Vec::with_capacity(total_edges);
      edge_ids.extend_from_slice(&node.inputs);
      edge_ids.extend_from_slice(&node.outputs);

      let mut local_witnesses = Vec::with_capacity(edge_ids.len());
      for &e in &edge_ids {
        if let Some(ref unsplit) = presplit_sparse[e] {
          local_witnesses.push(unsplit);
        } else {
          local_witnesses.push(&witnesses[e][0]);
        }
      }

      let mut local_claims = Vec::new();
      for &e in &node.outputs {
        local_claims.extend(claims[e].iter());
      }

      // Reducer (ZK-aware)
      let (reducer_proof, reducer_claims) = if local_claims.len() > 1 {
        println!("proving zk reducer for node {} | kind {:?}", node_id, self.nodes[node_id].kind);
        let reducer_witness = vec![&witnesses[node.outputs[0]][0]];
        let reducer_edge_ids = vec![edge_ids[edge_ids.len() - 1]];
        timed!(timing, format!("prove zk reducer for node {}", node_id).as_str(), {
          reducer.prove_zk(&reducer_witness, &reducer_edge_ids, &local_claims, transcript, &mut ctx)
        })
      } else {
        (vec![], vec![])
      };
      if reducer_claims.len() > 0 {
        local_claims = reducer_claims.iter().collect();
        let output_edge_id = node.outputs[0];
        claims[output_edge_id].push(reducer_claims[0].clone());
        reducer_proofs[node_id] = Some(reducer_proof);
      }

      // Prove node (ZK-aware)
      println!("proving zk node {} | kind {:?}", node_id, self.nodes[node_id].kind);
      let (proofs, new_claims) = timed!(
        timing,
        format!("prove zk node {} | kind {:?}", node_id, self.nodes[node_id].kind).as_str(),
        { self.nodes[node_id].kind.prove_zk(&local_witnesses, &edge_ids, &local_claims, transcript, &mut ctx) }
      );

      let mut node_claims = Vec::with_capacity(new_claims.len() + local_claims.len());
      node_claims.extend_from_slice(&new_claims);
      for claim in local_claims {
        node_claims.push(claim.clone());
      }
      node_proofs[node_id] = Some((proofs, node_claims));

      for c in new_claims {
        claims[c.edge_id].push(c);
      }
      for &e in &node.inputs {
        if let Some(producer) = self.producers[e] {
          preserve_nodes.insert(producer);
        }
      }
      for &preserve_node in &preserve_nodes {
        nodes_to_prove.insert(preserve_node);
      }
      preserve_nodes.clear();
    }

    // Set node_proofs for NonNegative range nodes
    for &n in &self.range {
      if node_proofs[n].is_none() && matches!(self.nodes[n].kind, BasicBlockType::NonNegative(_)) {
        let input_edge = self.nodes[n].inputs[0];
        if let Some(claim) = claims[input_edge].last() {
          node_proofs[n] = Some((vec![], vec![claim.clone()]));
        }
      }
    }

    claims.iter().enumerate().for_each(|(e, c)| edge_proofs[e].claims.extend(c.iter().cloned()));

    // 3. Lookups (same as non-ZK)
    let (two_pow_table_proofs, two_pow_middle_claims, two_pow_bool_proofs) = self.prove_two_pow(witnesses, &mut claims, transcript, timing);
    let (range_table_proofs, range_middle_claims, range_bool_proofs) = self.prove_range(witnesses, &mut claims, &node_proofs, transcript, timing);
    let two_pow_proof = LookupProof { table_proofs: two_pow_table_proofs, middle_claims: two_pow_middle_claims, bool_proofs: two_pow_bool_proofs };
    let range_proof = LookupProof { table_proofs: range_table_proofs, middle_claims: range_middle_claims, bool_proofs: range_bool_proofs };

    // 4. Opening proofs (same as non-ZK)
    println!("proving opening proofs (zk)");
    timed!(timing, "provide opening proofs (zk)", {
      let mut dense_tasks = Vec::new();
      let mut factored_tasks = Vec::new();
      let mut additive_factored_tasks = Vec::new();
      let mut sparse_tasks = Vec::new();

      for e in 0..self.num_edges {
        let w = &witnesses[e][0];
        if w.role == Role::Output { continue; }
        for c in &edge_proofs[e].claims {
          if w.poly_type == PolyType::Dense {
            if w.factored.is_some() {
              factored_tasks.push((e, c.clone()));
            } else {
              dense_tasks.push((e, c.clone()));
            }
          } else if w.additive_factored.is_some() {
            additive_factored_tasks.push((e, c.clone()));
          } else {
            sparse_tasks.push((e, c.clone()));
          }
        }
      }

      // Factored dense openings
      for (e, claim) in &factored_tasks {
        let w = &witnesses[*e][0];
        let factored = w.factored.as_ref().unwrap();
        let row_vars = factored.row_vars;
        let col_vars = factored.col_vars();
        let r_x = &claim.point[..row_vars];
        let r_y = &claim.point[row_vars..row_vars + col_vars];
        let (eq_poly, factor_polys) = factored.prepare_sumcheck_polys(r_x, r_y);
        let m = factored.factors.len();
        let mut polys = Vec::with_capacity(m + 1);
        polys.push(eq_poly);
        polys.extend(factor_polys);
        let mut prover = LinearSumcheckProver::new(row_vars, m + 1, transcript);
        let sc_proof = prover.prove(&polys, transcript);
        let x_star = &prover.challenges;

        // Open each factor in parallel
        let factor_comms = factored_commitments[*e].as_ref().unwrap();
        let open_tasks: Vec<(usize, Vec<F>)> = {
          let mut tasks = Vec::with_capacity(factored.factors.len());
          let mut y_offset = 0;
          for (i, _factor) in factored.factors.iter().enumerate() {
            let k_i = factored.chunk_sizes[i];
            let r_yi = &r_y[y_offset..y_offset + k_i];
            y_offset += k_i;
            let mut opening_point = x_star.clone();
            opening_point.extend_from_slice(r_yi);
            tasks.push((i, opening_point));
          }
          tasks
        };

        #[cfg(feature = "arkworks")]
        let factor_proofs: Vec<_> = open_tasks.par_iter().map(|(i, opening_point)| {
          DP::open(&factor_comms[*i], &factored.factors[*i], dense_key, opening_point)
        }).collect();
        #[cfg(not(feature = "arkworks"))]
        let factor_proofs: Vec<_> = open_tasks.iter().map(|(i, opening_point)| {
          DP::open(&factor_comms[*i], &factored.factors[*i], dense_key, opening_point)
        }).collect();

        for proof in factor_proofs {
          edge_proofs[*e].dense_opening_proof.push(proof);
        }
        edge_proofs[*e].factored_sumcheck_proofs.push(sc_proof);
      }

      // Additive factored sparse openings (same logic as non-ZK, adjacency is public)
      for (e, claim) in &additive_factored_tasks {
        let w = if let Some(ref unsplit) = presplit_sparse[*e] { unsplit } else { &witnesses[*e][0] };
        let af = w.additive_factored.as_ref().unwrap();
        let row_vars = af.row_vars;
        let col_vars = af.col_vars();
        let m = af.chunk_sizes.len();

        // For transposed decompositions (Twist and Shout), swap variable groups
        let (r_x, r_y) = if af.transposed {
          (&claim.point[col_vars..col_vars + row_vars], &claim.point[..col_vars])
        } else {
          (&claim.point[..row_vars], &claim.point[row_vars..row_vars + col_vars])
        };

        let (eq_poly, factor_polys_per_term) = af.prepare_sumcheck_inputs(r_x, r_y);
        let t = af.num_terms();
        let scalars: Vec<F> = vec![<F as CryptoField>::one(); t];
        let mut prover = GeneralLinearSumcheckProver::<F>::new(row_vars, m + 1, transcript);
        let instance = (scalars, factor_polys_per_term, eq_poly);
        let sc_proof = prover.prove(&instance, transcript);
        let x_star = &prover.challenges;

        // Collect all factor opening tasks, then run in parallel
        let factor_comms = factored_commitments[*e].as_ref().unwrap();
        let all_factors: Vec<&DenseMLPoly<F>> = af.terms.iter()
          .flat_map(|term| term.iter())
          .collect();
        let open_tasks: Vec<(usize, Vec<F>)> = {
          let mut tasks = Vec::with_capacity(all_factors.len());
          let mut factor_idx = 0;
          for term in &af.terms {
            let mut y_offset = 0;
            for (i, _factor) in term.iter().enumerate() {
              let k_i = af.chunk_sizes[i];
              let r_yi = &r_y[y_offset..y_offset + k_i];
              y_offset += k_i;
              let mut opening_point = x_star.clone();
              opening_point.extend_from_slice(r_yi);
              tasks.push((factor_idx, opening_point));
              factor_idx += 1;
            }
          }
          tasks
        };

        #[cfg(feature = "arkworks")]
        let factor_proofs: Vec<_> = open_tasks.par_iter().map(|(idx, opening_point)| {
          DP::open(&factor_comms[*idx], &all_factors[*idx], dense_key, opening_point)
        }).collect();
        #[cfg(not(feature = "arkworks"))]
        let factor_proofs: Vec<_> = open_tasks.iter().map(|(idx, opening_point)| {
          DP::open(&factor_comms[*idx], &all_factors[*idx], dense_key, opening_point)
        }).collect();

        for proof in factor_proofs {
          edge_proofs[*e].dense_opening_proof.push(proof);
        }
        edge_proofs[*e].factored_sumcheck_proofs.push(sc_proof);
      }

      // Regular dense openings in parallel
      #[cfg(feature = "arkworks")]
      let dense_results: Vec<_> = dense_tasks.par_iter().map(|(e, claim)| {
        let w = &witnesses[*e][0];
        let proof = DP::open(
          &dense_commitments[*e].as_ref().unwrap(),
          &w.data.as_ref().unwrap().as_any().downcast_ref::<DenseMLPoly<F>>().unwrap(),
          dense_key, &claim.point,
        );
        (*e, proof)
      }).collect();

      #[cfg(not(feature = "arkworks"))]
      let dense_results: Vec<_> = dense_tasks.iter().map(|(e, claim)| {
        let w = witnesses[*e][0];
        let proof = DP::open(
          &dense_commitments[*e].as_ref().unwrap(),
          &w.data.as_ref().unwrap().as_any().downcast_ref::<DenseMLPoly<F>>().unwrap(),
          dense_key, &claim.point,
        );
        (*e, proof)
      }).collect();

      // Sparse openings in parallel
      #[cfg(feature = "arkworks")]
      let sparse_results: Vec<_> = sparse_tasks.par_iter().map(|(e, claim)| {
        let w = &witnesses[*e][claim.sparse_id];
        let proof = SP::open(
          &sparse_commitments[*e].as_ref().unwrap()[claim.sparse_id],
          w.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap(),
          sparse_key, &claim.point,
        );
        (*e, proof)
      }).collect();

      #[cfg(not(feature = "arkworks"))]
      let sparse_results: Vec<_> = sparse_tasks.iter().map(|(e, claim)| {
        let w = witnesses[*e][claim.sparse_id];
        let proof = SP::open(
          &sparse_commitments[*e].as_ref().unwrap()[claim.sparse_id],
          &w.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap(),
          sparse_key, &claim.point,
        );
        (*e, proof)
      }).collect();

      for (e, proof) in dense_results { edge_proofs[e].dense_opening_proof.push(proof); }
      for (e, proof) in sparse_results { edge_proofs[e].sparse_opening_proof.push(proof); }
    });

    // 5. ZK mask coefficients are embedded in each SumcheckProof — no separate commit needed.
    let zk_proof = ZkProof::<F, DP>::empty();

    (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs, zk_proof)
  }

  /// ZK-aware verify: verifies the proof including mask polynomial commitments and openings.
  pub fn verify_zk<F, DP, SP>(
    &self,
    node_proofs: &[Option<(Vec<SumcheckProof<F>>, Vec<Claim<F>>)>],
    edge_proofs: &[EdgeProof<F, DP, SP>],
    range_proof: &LookupProof<F>,
    two_pow_proof: &LookupProof<F>,
    reducer_proofs: &[Option<Vec<SumcheckProof<F>>>],
    witnesses: &[Vec<Witness<F>>],
    dense_key: &DP::VerifierKey,
    sparse_key: &SP::VerifierKey,
    dense_commitments: &[Option<DP::Commitment>],
    sparse_commitments: &[Option<Vec<SP::Commitment>>],
    factored_commitments: &[Option<Vec<DP::Commitment>>],
    transcript: &mut Transcript<F>,
    zk: bool,
    zk_proof: &ZkProof<F, DP>,
    mask_committer: Option<Arc<dyn crate::crypto::MaskCommitter<F>>>,
  ) -> bool
  where
    F: CryptoField + 'static,
    DP: MLPolyCommit<F, DenseMLPoly<F>>,
    SP: MLPolyCommit<F, SparseMLPoly<F>>,
    DP::Proof: Clone + Sync,
    SP::Proof: Clone + Sync,
    DP::VerifierKey: Sync,
    SP::VerifierKey: Sync,
    DP::Commitment: Sync + Clone,
    SP::Commitment: Sync,
  {
    if !zk {
      return self.verify::<F, DP, SP>(
        node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs,
        witnesses, dense_key, sparse_key, dense_commitments, sparse_commitments,
        factored_commitments, transcript,
      );
    }

    let ctx = match mask_committer {
      Some(mc) => crate::crypto::ProveContext::<F>::with_mask_committer(true, mc),
      None => crate::crypto::ProveContext::<F>::new(true),
    };
    let reducer = BasicBlockType::Reducer(Reducer {});
    let mut verified = true;
    let irreducable_edge_ids: Vec<EdgeId> =
      (0..self.num_edges).filter(|&e| self.producers[e].is_none() || witnesses[e][0].role == Role::Auxiliary).collect();

    // Absorb all polynomial commitments into Fiat-Shamir transcript before drawing challenges
    for comm in dense_commitments.iter() {
      if let Some(c) = comm {
        transcript.append_bytes(b"dense_commitment", &c.to_transcript_bytes());
      }
    }
    for comms in sparse_commitments.iter() {
      if let Some(cs) = comms {
        for c in cs {
          transcript.append_bytes(b"sparse_commitment", &c.to_transcript_bytes());
        }
      }
    }
    for comms in factored_commitments.iter() {
      if let Some(cs) = comms {
        for c in cs {
          transcript.append_bytes(b"factored_commitment", &c.to_transcript_bytes());
        }
      }
    }

    let mut claims: Vec<Vec<Claim<F>>> = Vec::with_capacity(self.num_edges);
    claims.resize_with(self.num_edges, || Vec::new());
    let mut nodes_to_verify = BTreeSet::new();
    self.output_ports.iter().for_each(|&e| {
      let w = &witnesses[e][0];
      if w.role == Role::Output {
        let point: Vec<F> = (0..get_n(&witnesses[e][0].shape)).map(|_| transcript.challenge_scalar(b"challenge")).collect();
        if edge_proofs[e].claims.is_empty() || edge_proofs[e].claims[0].point != point {
          verified = false;
        } else {
          claims[e].push(edge_proofs[e].claims[0].clone());
          match self.producers[e] {
            Some(producer) => { nodes_to_verify.insert(producer); }
            None => { verified = false; }
          }
        }
      }
    });

    let mut preserve_nodes = BTreeSet::new();
    let consumer_sets: Vec<BTreeSet<NodeId>> = self.consumers.iter().map(|consumers| consumers.iter().copied().collect()).collect();

    while !nodes_to_verify.is_empty() {
      let u = nodes_to_verify.pop_last().unwrap();
      let mut can_verify = true;
      for &edge in &self.nodes[u].outputs {
        if consumer_sets[edge].iter().any(|&c| nodes_to_verify.contains(&c)) {
          can_verify = false;
          break;
        }
      }
      if !can_verify {
        nodes_to_verify.insert(u);
        continue;
      }
      let node = &self.nodes[u];

      let (local_sumcheck_proofs, local_claims) = match node_proofs[u].as_ref() {
        Some(p) => p,
        None => {
          println!("=== zk node proof missing for node {u} ===");
          verified = false;
          for &e in &node.inputs {
            if let Some(producer) = self.producers[e] {
              preserve_nodes.insert(producer);
            }
          }
          for &preserve_node in &preserve_nodes { nodes_to_verify.insert(preserve_node); }
          preserve_nodes.clear();
          continue;
        }
      };
      let local_claims: Vec<&Claim<F>> = local_claims.iter().collect();
      let local_witnesses: Vec<&Witness<F>> = local_claims.iter().map(|&c| &witnesses[c.edge_id][0]).collect();
      let local_sumcheck_proofs: Vec<&SumcheckProof<F>> = local_sumcheck_proofs.iter().collect();

      if claims[node.outputs[0]].len() > 1 {
        let reducer_proof_ref = match reducer_proofs[u].as_ref() {
          Some(p) => p,
          None => {
            println!("=== zk reducer proof missing for node {u} ===");
            verified = false;
            continue;
          }
        };
        let reducer_witness = vec![&witnesses[node.outputs[0]][0]];
        let mut reducer_claims = claims[node.outputs[0]].clone();
        reducer_claims.push(local_claims[local_claims.len() - 1].clone());
        let reducer_claims: Vec<&Claim<F>> = reducer_claims.iter().collect();
        let reducer_sumcheck_proofs: Vec<&SumcheckProof<F>> = reducer_proof_ref.iter().collect();
        let reducer_verified = reducer.verify_zk(&reducer_witness, &reducer_claims, &reducer_sumcheck_proofs, transcript, &ctx);
        verified = verified && reducer_verified;
        if !reducer_verified {
          println!("=== verified zk reducer for node {u}: {reducer_verified} ===");
        }
      }

      let node_verified = self.nodes[u].kind.verify_zk(&local_witnesses, &local_claims, &local_sumcheck_proofs, transcript, &ctx);
      if !node_verified {
        println!("=== verified zk node {u}: {node_verified} ===");
      }
      verified = verified && node_verified;

      for &e in &node.inputs {
        if let Some(producer) = self.producers[e] {
          preserve_nodes.insert(producer);
        }
      }
      for &preserve_node in &preserve_nodes {
        nodes_to_verify.insert(preserve_node);
      }
      preserve_nodes.clear();

      local_claims.iter().for_each(|&c| {
        if node.inputs.contains(&c.edge_id) {
          claims[c.edge_id].push(c.clone())
        }
      });
    }

    // Verify range and two_pow (same as non-ZK)
    let two_pow_verified = self.verify_two_pow(node_proofs, witnesses, two_pow_proof, transcript);
    let range_verified = self.verify_range(node_proofs, witnesses, range_proof, transcript);
    if !two_pow_verified { println!("=== verified two_pow: {two_pow_verified} ==="); }
    if !range_verified { println!("=== verified range: {range_verified} ==="); }
    verified = verified && range_verified && two_pow_verified;

    // Verify irreducible edges (same as non-ZK)
    let mut factored_edge_ids = Vec::new();
    let mut additive_factored_edge_ids = Vec::new();
    let mut regular_edge_ids = Vec::new();
    for &e in &irreducable_edge_ids {
      if witnesses[e][0].factored.is_some() {
        factored_edge_ids.push(e);
      } else if witnesses[e][0].additive_factored.is_some() {
        additive_factored_edge_ids.push(e);
      } else {
        regular_edge_ids.push(e);
      }
    }

    let mut factored_verified = true;
    for &e in &factored_edge_ids {
      let w = &witnesses[e][0];
      let factored = w.factored.as_ref().unwrap();
      if edge_proofs[e].factored_sumcheck_proofs.is_empty() {
        println!("Missing factored sumcheck proofs for edge {} (zk)", e);
        factored_verified = false;
        continue;
      }
      for (claim_idx, claim) in edge_proofs[e].claims.iter().enumerate() {
        let sc_proof = &edge_proofs[e].factored_sumcheck_proofs[claim_idx];
        let row_vars = factored.row_vars;
        let col_vars = factored.col_vars();
        let m = factored.factors.len();
        let mut sc_verifier = SumcheckVerifier::new(row_vars, m + 1, transcript);
        let (sc_result, sc_challenges) = sc_verifier.verify(transcript, sc_proof.round_messages.clone(), claim.eval);
        let running_sum = match sc_result {
          Some(v) => v,
          None => {
            println!("Factored sumcheck verification failed for edge {} (zk)", e);
            factored_verified = false;
            continue;
          }
        };

        assert_eq!(claim.point.len(), row_vars + col_vars,
          "Factored claim point should have row_vars + col_vars dimensions");

        let r_x = &claim.point[..row_vars];
        let one = <F as CryptoField>::one();
        let eq_eval = r_x.iter().zip(sc_challenges.iter())
          .fold(one, |acc, (rx_j, xstar_j)| {
            acc * (*xstar_j * *rx_j + (one - *xstar_j) * (one - *rx_j))
          });
        let factor_comms = match factored_commitments[e].as_ref() {
          Some(c) => c,
          None => {
            println!("Missing factored commitments for edge {} (zk)", e);
            factored_verified = false;
            continue;
          }
        };
        let proof_offset = claim_idx * factor_comms.len();
        // Verify factor PCS openings in parallel
        let verify_tasks: Vec<(usize, Vec<F>)> = factor_comms.iter().enumerate().map(|(i, _)| {
          let mut opening_point = sc_challenges.clone();
          let k_i = factored.chunk_sizes[i];
          let y_start: usize = factored.chunk_sizes[..i].iter().sum();
          let r_yi = &claim.point[row_vars + y_start..row_vars + y_start + k_i];
          opening_point.extend_from_slice(r_yi);
          (i, opening_point)
        }).collect();

        let verify_results: Vec<(bool, F)> = verify_tasks.par_iter().map(|(i, opening_point)| {
          let global_proof_idx = proof_offset + *i;
          if global_proof_idx >= edge_proofs[e].dense_opening_proof.len() {
            return (false, <F as CryptoField>::one());
          }
          DP::verify_and_extract(
            &factor_comms[*i],
            &edge_proofs[e].dense_opening_proof[global_proof_idx],
            dense_key,
            opening_point,
          )
        }).collect();

        let mut factor_product = <F as CryptoField>::one();
        for (i, (ok, factor_eval)) in verify_results.into_iter().enumerate() {
          if !ok {
            println!("Factor opening verification failed for edge {} factor {} (zk)", e, i);
            factored_verified = false;
          }
          factor_product = factor_product * factor_eval;
        }
        let expected = eq_eval * factor_product;
        if running_sum != expected {
          println!("Factored final eval check failed for edge {} (zk): running_sum != eq_eval * factor_product", e);
          factored_verified = false;
        }
      }
    }

    // Verify additive factored sparse edge openings (zk)
    for &e in &additive_factored_edge_ids {
      let w = &witnesses[e][0];
      let af = w.additive_factored.as_ref().unwrap();
      if edge_proofs[e].factored_sumcheck_proofs.is_empty() {
        println!("Missing additive factored sumcheck proofs for edge {} (zk)", e);
        factored_verified = false;
        continue;
      }
      let row_vars = af.row_vars;
      let col_vars = af.col_vars();
      let m = af.chunk_sizes.len();
      let t = af.num_terms();
      let factors_per_claim = m * t;
      let factor_comms = match factored_commitments[e].as_ref() {
        Some(c) => c,
        None => {
          println!("Missing additive factored commitments for edge {} (zk)", e);
          factored_verified = false;
          continue;
        }
      };
      for (claim_idx, claim) in edge_proofs[e].claims.iter().enumerate() {
        let sc_proof = &edge_proofs[e].factored_sumcheck_proofs[claim_idx];
        let mut sc_verifier = SumcheckVerifier::new(row_vars, m + 1, transcript);
        let (sc_result, sc_challenges) = sc_verifier.verify(transcript, sc_proof.round_messages.clone(), claim.eval);
        let running_sum = match sc_result {
          Some(v) => v,
          None => {
            println!("Additive factored sumcheck verification failed for edge {} (zk)", e);
            factored_verified = false;
            continue;
          }
        };
        // For transposed decompositions: swap variable groups from claim point
        let (r_x, r_y) = if af.transposed {
          (&claim.point[col_vars..col_vars + row_vars], &claim.point[..col_vars])
        } else {
          (&claim.point[..row_vars], &claim.point[row_vars..row_vars + col_vars])
        };
        let one = <F as CryptoField>::one();
        let eq_eval = r_x.iter().zip(sc_challenges.iter())
          .fold(one, |acc, (rx_j, xstar_j)| {
            acc * (*xstar_j * *rx_j + (one - *xstar_j) * (one - *rx_j))
          });
        let proof_offset = claim_idx * factors_per_claim;

        // Build all verification tasks in parallel
        let verify_tasks: Vec<(usize, Vec<F>)> = {
          let mut tasks = Vec::with_capacity(factors_per_claim);
          let mut factor_idx = 0;
          for _k in 0..t {
            let mut y_offset = 0;
            for i in 0..m {
              let k_i = af.chunk_sizes[i];
              let r_yi = &r_y[y_offset..y_offset + k_i];
              y_offset += k_i;
              let mut opening_point = sc_challenges.clone();
              opening_point.extend_from_slice(r_yi);
              tasks.push((factor_idx, opening_point));
              factor_idx += 1;
            }
          }
          tasks
        };

        let verify_results: Vec<(bool, F)> = verify_tasks.par_iter().map(|(idx, opening_point)| {
          let global_proof_idx = proof_offset + *idx;
          if global_proof_idx >= edge_proofs[e].dense_opening_proof.len() || *idx >= factor_comms.len() {
            return (false, <F as CryptoField>::one());
          }
          DP::verify_and_extract(
            &factor_comms[*idx],
            &edge_proofs[e].dense_opening_proof[global_proof_idx],
            dense_key,
            opening_point,
          )
        }).collect();

        // Reconstruct sum_of_products from parallel results
        let mut sum_of_products = <F as CryptoField>::zero();
        let mut factor_idx = 0;
        for _k in 0..t {
          let mut product = <F as CryptoField>::one();
          for _i in 0..m {
            let (ok, factor_eval) = verify_results[factor_idx];
            if !ok {
              println!("Additive factor opening verification failed for edge {} factor {} (zk)", e, factor_idx);
              factored_verified = false;
            }
            product = product * factor_eval;
            factor_idx += 1;
          }
          sum_of_products = sum_of_products + product;
        }
        let expected = eq_eval * sum_of_products;
        if running_sum != expected {
          println!("Additive factored final eval check failed for edge {} (zk)", e);
          factored_verified = false;
        }
      }
    }

    let regular_opening_verified = regular_edge_ids.par_iter().all(|&e| {
      // Sparse edges may have no claims (adjacency matrices verified via SpMV sumcheck)
      if edge_proofs[e].claims.is_empty() && witnesses[e][0].poly_type == PolyType::Sparse {
        return true;
      }
      if witnesses[e][0].poly_type == PolyType::Dense {
        (0..edge_proofs[e].claims.len()).into_par_iter().all(|i| {
          DP::verify(
            dense_commitments[e].as_ref().unwrap(),
            &edge_proofs[e].dense_opening_proof[i],
            dense_key,
            &edge_proofs[e].claims[i].point,
          )
        })
      } else {
        (0..edge_proofs[e].claims.len()).into_par_iter().all(|i| {
          SP::verify(
            &sparse_commitments[e].as_ref().unwrap()[edge_proofs[e].claims[i].sparse_id],
            &edge_proofs[e].sparse_opening_proof[i],
            sparse_key,
            &edge_proofs[e].claims[i].point,
          )
        })
      }
    });

    // ZK mask coefficients are verified implicitly: the verifier reconstructs ρ from the
    // transcript, computes the adjusted sum H + ρ·P, and verifies round message consistency.
    // Since the mask coefficients are random and public, no additional PCS verification is needed.
    verified && factored_verified && regular_opening_verified
  }

  pub fn verify_range<F: CryptoField + 'static>(
    &self,
    node_proofs: &[Option<(Vec<SumcheckProof<F>>, Vec<Claim<F>>)>],
    witnesses: &[Vec<Witness<F>>],
    range_proof: &LookupProof<F>,
    transcript: &mut Transcript<F>,
  ) -> bool {
    if self.range.len() == 0 {
      return true;
    }
    println!("verifying range");
    let mut verified = true;

    let range_table_proofs = &range_proof.table_proofs;
    let range_middle_claims = &range_proof.middle_claims;
    let range_bool_proofs = &range_proof.bool_proofs;

    let alpha: F = transcript.challenge_scalar(b"table_alpha");
    let alphas = calc_pow(alpha, 2);
    let beta: F = transcript.challenge_scalar(b"table_beta");
    let betas = calc_pow(beta, self.range.len());
    let gamma: F = transcript.challenge_scalar(b"table_gamma");
    let max_blocks = self.range.iter().map(|n| {
      let node = &self.nodes[*n];
      let aux_id = if matches!(node.kind, BasicBlockType::NonNegative(_)) { 0 } else { 1 };
      witnesses[node.outputs[aux_id]].len()
    }).max().unwrap_or(8);
    let gammas = calc_pow(gamma, max_blocks);

    let mut table_expected_sum = <F as CryptoField>::zero();
    for (i, n) in self.range.iter().enumerate() {
      println!("verifying range for node {n} | kind {:?}", self.nodes[*n].kind);
      let node = &self.nodes[*n];
      let aux_id = if matches!(node.kind, BasicBlockType::NonNegative(_)) { 0 } else { 1 };
      let auxs = &witnesses[node.outputs[aux_id]];
      let node_claim = &node_proofs[*n].as_ref().unwrap().1;
      let eval_to_check = if matches!(node.kind, BasicBlockType::ScaleDown(_)) {
        let input_sf = witnesses[node.inputs[0]][0].sf;
        let output_sf = witnesses[node.outputs[0]][0].sf;
        let rescale_factor = 1 << (input_sf - output_sf);
        let rescale_factor_divided_by_2 = rescale_factor / 2;
        let rescale_factor_f = F::from(rescale_factor as u32);
        let rescale_factor_divided_by_2_f = F::from(rescale_factor_divided_by_2 as u32);
        node_claim[0].eval - node_claim[1].eval * rescale_factor_f + rescale_factor_divided_by_2_f
      } else if matches!(node.kind, BasicBlockType::ScaleUp(_)) {
        let input_sf = witnesses[node.inputs[0]][0].sf;
        let output_sf = witnesses[node.outputs[0]][0].sf;
        let rescale_factor = 1 << (output_sf - input_sf);
        let rescale_factor_divided_by_2 = rescale_factor / 2;
        let rescale_factor_f = F::from(rescale_factor as u32);
        let rescale_factor_divided_by_2_f = F::from(rescale_factor_divided_by_2 as u32);
        node_claim[0].eval * rescale_factor_f - node_claim[1].eval + rescale_factor_divided_by_2_f
      } else if matches!(node.kind, BasicBlockType::ExpHelper(_)) {
        // ExpHelper stores k_shifted = k_orig + K_MAX where k_orig = (input - output) / (-ln2*sf)
        // The selection polynomial holds k_shifted, so eval_acc will be MLE(k_shifted)(r).
        // We need eval_to_check = MLE(k_shifted)(r) = MLE(k_orig)(r) + K_MAX
        let cache = get_cached_inverses::<F>();
        let k_max_f = F::from(75u32); // K_MAX = 75
        (node_claim[0].eval - node_claim[1].eval) * cache.neg_ln2_inv + k_max_f
      } else {
        // NonNegative
        node_claim[0].eval
      };
      let mut eval_acc = <F as CryptoField>::zero();

      for (sparse_id, _aux) in auxs.iter().enumerate() {
        let middle_sum = range_middle_claims[i][sparse_id];
        eval_acc += middle_sum * <F as CryptoField>::from_u64(1u64 << (sparse_id * *TABLE_COMMIT_LOG));
        let beta_gamma = betas[i] * gammas[sparse_id];
        table_expected_sum += (middle_sum + alphas[1]) * beta_gamma;
      }
      if eval_to_check != eval_acc {
        println!("range eval_to_check mismatch for node {n}");
        verified = false;
      }
    }
    // verify the table proof
    let mut sumcheck_verifier = SumcheckVerifier::new(range_table_proofs[0].round_messages.len(), 2, transcript);
    let (verification_result, _challenges) = sumcheck_verifier.verify(transcript, range_table_proofs[0].round_messages.clone(), table_expected_sum);
    if verification_result.is_none() {
      println!("range table proof verification failed");
      return false;
    }

    // verify the bool proofs (from GeneralLinearSumcheckProver)
    // Each bool proof corresponds to a group of auxiliary polynomials with the same num_var
    // We can determine aux_num_var directly from the proof's round_messages.len()
    for bool_proof in range_bool_proofs.iter() {
      // aux_num_var equals the number of sumcheck rounds
      let aux_num_var = bool_proof.round_messages.len();

      // Create verifier with same parameters as prover
      let mut bool_verifier = SumcheckVerifier::new(aux_num_var, 3, transcript);

      // Generate the same challenges for eq polynomial (before sumcheck)
      let _challenge: Vec<F> = (0..aux_num_var).map(|_| transcript.challenge_scalar(b"challenge")).collect();

      // The expected sum for boolean proof is 0: Σ_x eq(r,x) * Σ_j (beta_gamma_j * (aux_j(x)-1) * aux_j(x) * beta_gamma_j) = 0
      // because aux_j(x) * (aux_j(x) - 1) = 0 when aux_j is boolean
      let expected_sum = <F as CryptoField>::zero();

      // Verify the sumcheck proof
      let (verification_result, _challenges) = bool_verifier.verify(transcript, bool_proof.round_messages.clone(), expected_sum);
      if verification_result.is_none() {
        println!("bool proof verification failed for aux_num_var {}", aux_num_var);
        verified = false;
      }
    }

    verified
  }

  pub fn prove_range<F: CryptoField + 'static>(
    &self,
    witnesses: &[Vec<Witness<F>>],
    claims: &mut Vec<Vec<Claim<F>>>,
    node_proofs: &[Option<(Vec<SumcheckProof<F>>, Vec<Claim<F>>)>],
    transcript: &mut Transcript<F>,
    timing: &mut TimingTree,
  ) -> (Vec<SumcheckProof<F>>, Vec<Vec<F>>, Vec<SumcheckProof<F>>) {
    let mut table_proofs = Vec::new();
    let mut bool_proofs = Vec::new();
    let mut middle_claims: Vec<Vec<F>> = vec![vec![]; self.range.len()];
    // prove range
    if self.range.len() > 0 {
      println!("proving range");
      timed!(timing, "prove range", {
        let alpha: F = transcript.challenge_scalar(b"table_alpha");
        let alphas = calc_pow(alpha, 2);
        let beta: F = transcript.challenge_scalar(b"table_beta");
        let betas = calc_pow(beta, self.range.len());
        let gamma: F = transcript.challenge_scalar(b"table_gamma");
        let max_blocks = self.range.iter().map(|n| {
          let node = &self.nodes[*n];
          let aux_id = if matches!(node.kind, BasicBlockType::NonNegative(_)) { 0 } else { 1 };
          witnesses[node.outputs[aux_id]].len()
        }).max().unwrap_or(8);
        let gammas = calc_pow(gamma, max_blocks);
        let mut aux_polys = Vec::new();
        let mut bool_hashmap: BTreeMap<usize, (Vec<Vec<DenseMLPoly<F>>>, Vec<F>, Vec<(usize, usize, usize)>)> = BTreeMap::new();
        for (i, n) in self.range.iter().enumerate() {
          let node = &self.nodes[*n];
          let aux_id = if matches!(node.kind, BasicBlockType::NonNegative(_)) { 0 } else { 1 };
          // Use the node's own claim point (from its sumcheck), not the edge's last claim,
          // because the input edge may have additional claims from other consumers/reducers.
          let inp_claim = &node_proofs[*n].as_ref().unwrap().1[0];
          let lagrange_basis = evaluate_lagrange_basis(&inp_claim.point);
          let auxs = &witnesses[node.outputs[aux_id]];
          for (sparse_id, aux) in auxs.iter().enumerate() {
            let beta_gamma = betas[i] * gammas[sparse_id];
            let aux_poly = aux.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap();
            // prove boolean
            let dense_aux_poly = aux_poly.to_dense();
            let dense_aux_poly_beta_gamma = dense_aux_poly.mul_by_scalar(beta_gamma);
            let dense_aux_poly_minus_1 = dense_aux_poly.add_by_scalar(<F as CryptoField>::zero() - F::from(1));
            let aux_num_var = aux_poly.n();
            if !bool_hashmap.contains_key(&aux_num_var) {
              bool_hashmap.insert(aux_num_var, (Vec::new(), Vec::new(), Vec::new()));
            }
            let (bool_polys, bool_scalars, bool_indices) = bool_hashmap.get_mut(&aux_num_var).unwrap();
            bool_polys.push(vec![dense_aux_poly_minus_1, dense_aux_poly_beta_gamma]);
            bool_scalars.push(beta_gamma);
            bool_indices.push((*n, aux_id, sparse_id));
            // prove Shout
            let mut part_aux_poly_evals = vec![<F as CryptoField>::zero(); 1 << (aux_poly.n() - inp_claim.point.len())];
            let mut hs = HashSet::new();
            aux_poly.selection.selection.iter().for_each(|(input_idx, table_idx)| {
              hs.insert(*table_idx);
              part_aux_poly_evals[*table_idx] += lagrange_basis[*input_idx];
            });
            middle_claims[i].push(part_aux_poly_evals.par_iter().enumerate().map(|(idx, val)| *val * F::from(idx as u32)).sum::<F>());

            hs.iter().for_each(|idx| {
              part_aux_poly_evals[*idx] *= beta_gamma;
            });
            let part_aux_poly = DenseMLPoly::new(aux_poly.n() - inp_claim.point.len(), part_aux_poly_evals);
            aux_polys.push(part_aux_poly);
          }
        }

        let aux_poly = aux_polys[1..aux_polys.len()].iter().fold(aux_polys[0].clone(), |acc, x| acc.add(x));
        let range_poly = range_dense::<F>(aux_poly.n()).add_by_scalar(alphas[1]);
        let mut sumcheck_prover = LinearSumcheckProver::<F>::new(range_poly.n(), 2, transcript);
        let sumcheck_proof = sumcheck_prover.prove(&vec![aux_poly, range_poly], transcript);
        table_proofs.push(sumcheck_proof);

        bool_hashmap.iter().for_each(|(aux_num_var, (bool_polys, bool_scalars, bool_indices))| {
          let mut bool_prover = GeneralLinearSumcheckProver::<F>::new(*aux_num_var, 3, transcript);
          let challenge: Vec<F> = (0..*aux_num_var).map(|_| transcript.challenge_scalar(b"challenge")).collect();
          let eq = crate::util::poly::evaluate_lagrange_basis(&challenge);
          let eq = DenseMLPoly::new(*aux_num_var, eq);
          let bool_proof = bool_prover.prove(&(bool_scalars.clone(), bool_polys.clone(), eq), transcript);
          bool_proofs.push(bool_proof);
          for (n, aux_id, sparse_id) in bool_indices.iter() {
            let node = &self.nodes[*n];
            let bool_claim = Claim {
              edge_id: node.outputs[*aux_id],
              sparse_id: *sparse_id,
              point: bool_prover.challenges.clone(),
              eval: <F as CryptoField>::zero(),
            };
            claims[node.outputs[*aux_id]].push(bool_claim);
          }
        });

        // update claims for range selector polys
        for n in self.range.iter() {
          let node = &self.nodes[*n];
          let aux_id = if matches!(node.kind, BasicBlockType::NonNegative(_)) { 0 } else { 1 };
          let inp_claim_point = node_proofs[*n].as_ref().unwrap().1[0].point.clone();
          let auxs = &witnesses[node.outputs[aux_id]];
          for (sparse_id, aux) in auxs.iter().enumerate() {
            let aux_poly = aux.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap();
            let challenge_len = aux_poly.n() - inp_claim_point.len();
            let point = inp_claim_point.iter().chain(sumcheck_prover.challenges[..challenge_len].iter()).cloned().collect::<Vec<F>>();
            let claim = Claim {
              edge_id: node.outputs[aux_id],
              sparse_id: sparse_id,
              point: point,
              eval: <F as CryptoField>::zero(),
            };
            claims[node.outputs[aux_id]].push(claim);
          }
        }
      });
    }
    (table_proofs, middle_claims, bool_proofs)
  }

  pub fn verify_two_pow<F: CryptoField + 'static>(
    &self,
    node_proofs: &[Option<(Vec<SumcheckProof<F>>, Vec<Claim<F>>)>],
    _witnesses: &[Vec<Witness<F>>],
    two_pow_proof: &LookupProof<F>,
    transcript: &mut Transcript<F>,
  ) -> bool {
    if self.two_pow.len() == 0 {
      return true;
    }
    println!("verifying two_pow");

    let two_pow_table_proofs = &two_pow_proof.table_proofs;
    let two_pow_middle_claims = &two_pow_proof.middle_claims;

    let beta: F = transcript.challenge_scalar(b"table_beta");
    let betas = calc_pow(beta, self.two_pow.len());

    // Parallelize the loop: compute sum and check equalities in parallel
    let (table_expected_sum, all_equal) = self
      .two_pow
      .par_iter()
      .enumerate()
      .map(|(i, n)| {
        let node_claim = &node_proofs[*n].as_ref().unwrap().1;
        let eval_to_check = node_claim[0].eval;
        let eval_acc = two_pow_middle_claims[i][0];
        (eval_acc * betas[i], eval_to_check == eval_acc)
      })
      .reduce(
        || (<F as CryptoField>::zero(), true),
        |(sum1, eq1), (sum2, eq2)| (sum1 + sum2, eq1 && eq2),
      );

    if !all_equal {
      return false;
    }
    // verify the table proof
    let mut sumcheck_verifier = SumcheckVerifier::new(two_pow_table_proofs[0].round_messages.len(), 2, transcript);
    let (verification_result, _challenges) = sumcheck_verifier.verify(transcript, two_pow_table_proofs[0].round_messages.clone(), table_expected_sum);
    verification_result.is_some()
  }

  /// Reconstruct the unsplit selection polynomial from split blocks.
  /// Split blocks partition table bits into chunks of `block_size`; this merges them back.
  fn reconstruct_unsplit_poly<F: CryptoField + 'static>(
    auxs: &[Witness<F>],
    original_table_num_vars: usize,
  ) -> SparseMLPoly<F> {
    if auxs.len() == 1 {
      return auxs[0].data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap().clone();
    }
    let block_size = *TABLE_COMMIT_LOG;
    let first_poly = auxs[0].data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap();
    let input_num_vars = first_poly.selection.input_num_vars;

    // Merge partial table indices from each block back into full indices
    let mut combined: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (block_id, aux) in auxs.iter().enumerate() {
      let poly = aux.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap();
      for &(input_idx, partial_idx) in &poly.selection.selection {
        let entry = combined.entry(input_idx).or_insert(0);
        *entry |= partial_idx << (block_id * block_size);
      }
    }
    let selection: Vec<(usize, usize)> = combined.into_iter().collect();
    let full_sel = SelectionPolynomial::new(input_num_vars, original_table_num_vars, selection);
    full_sel.to_sparse()
  }

  pub fn prove_two_pow<F: CryptoField + 'static>(
    &self,
    witnesses: &[Vec<Witness<F>>],
    claims: &mut Vec<Vec<Claim<F>>>,
    transcript: &mut Transcript<F>,
    timing: &mut TimingTree,
  ) -> (Vec<SumcheckProof<F>>, Vec<Vec<F>>, Vec<SumcheckProof<F>>) {
    let mut table_proofs = Vec::new();
    let bool_proofs = Vec::new();
    let mut middle_claims: Vec<Vec<F>> = vec![vec![]; self.two_pow.len()];
    // prove two_pow
    if self.two_pow.len() > 0 {
      println!("proving two_pow");
      timed!(timing, "prove two_pow", {
        let beta: F = transcript.challenge_scalar(b"table_beta");
        let betas = calc_pow(beta, self.two_pow.len());
        let mut aux_polys = Vec::new();
        // The two_pow table has 8 variables (k_shifted in [0, 2*K_MAX=150])
        // Must match K_MAX=75 and K_BITS=8 in exp.rs
        let two_pow_table_num_vars: usize = 8;
        let k_max: usize = 75;
        // Precompute 2^(k_max+15-i) for i in [0, k_max+15] using repeated doubling
        let max_exp = k_max + 15; // = 90
        let mut two_pow_table: Vec<F> = vec![<F as CryptoField>::zero(); 2 * k_max + 1];
        for i in 0..=2 * k_max {
          if i <= max_exp {
            let exponent = max_exp - i;
            let mut val = <F as CryptoField>::one();
            for _ in 0..exponent {
              val = val + val;
            }
            two_pow_table[i] = val;
          }
          // else: stays zero (exp of very negative input ≈ 0)
        }
        for (i, n) in self.two_pow.iter().enumerate() {
          let node = &self.nodes[*n];
          let inp_claim = claims[node.outputs[0]].last().unwrap();
          let auxs = &witnesses[node.inputs[0]];
          // Reconstruct unsplit poly from split blocks for correct table index computation
          let aux_poly = Self::reconstruct_unsplit_poly::<F>(auxs, two_pow_table_num_vars);
          // we don't need to prove boolean for two_pow as we already proved it in the range proof
          // prove Shout
          let part_aux_poly = aux_poly.fix_variables(&inp_claim.point);
          middle_claims[i].push(part_aux_poly.evaluations.par_iter().map(|(idx, val)| {
            // table[k_shifted] = 2^(k_max + 15 - k_shifted)
            if *idx <= 2 * k_max { *val * two_pow_table[*idx] } else { <F as CryptoField>::zero() }
          }).sum::<F>());
          let part_aux_poly_beta = part_aux_poly.mul_by_scalar(betas[i]);
          aux_polys.push(part_aux_poly_beta.to_dense());
        }
        let aux_poly = aux_polys[1..aux_polys.len()].iter().fold(aux_polys[0].clone(), |acc, x| acc.add(x));
        let two_pow_poly = two_pow_dense::<F>(aux_poly.n());
        let mut sumcheck_prover = LinearSumcheckProver::<F>::new(two_pow_poly.n(), 2, transcript);
        let sumcheck_proof = sumcheck_prover.prove(&vec![aux_poly, two_pow_poly], transcript);
        table_proofs.push(sumcheck_proof);

        // update claims for two_pow selector polys (reuse two_pow_table_num_vars from above)
        for n in self.two_pow.iter() {
          let node = &self.nodes[*n];
          let inp_claim_point = claims[node.outputs[0]].last().unwrap().point.clone();
          let auxs = &witnesses[node.inputs[0]];
          let aux_poly = Self::reconstruct_unsplit_poly::<F>(auxs, 8);
          let challenge_len = aux_poly.n() - inp_claim_point.len();
          let point = inp_claim_point.iter().chain(sumcheck_prover.challenges[..challenge_len].iter()).cloned().collect::<Vec<F>>();
          let claim = Claim {
            edge_id: node.inputs[0],
            sparse_id: 0,
            point: point,
            eval: <F as CryptoField>::zero(),
          };
          claims[node.inputs[0]].push(claim);
        }
      });
    }
    (table_proofs, middle_claims, bool_proofs)
  }

  /* ---------- Alias helper ---------- */
  pub fn alias_info(&self, a: AliasId) -> (EdgeId, NodeId, usize) {
    (self.alias_to_edge[a.0], self.alias_to_consumer[a.0], self.alias_input_slot[a.0])
  }

  /// Pretty printer that shows NodeId, node kind, and both EdgeId and AliasId context with a post-order traversal.
  pub fn print_dag_structure(&self, targets: &[EdgeId]) {
    let (_need_edge, need_node) = self.backward_mark_needed(targets);
    let needed_out_aliases = self.collect_needed_out_aliases(&need_node);

    // Build initial successor alias counts
    let mut succ_alias_count: Vec<usize> = needed_out_aliases.iter().map(|v| v.len()).collect();

    // Initial queue: nodes with all needed outputs already reached (sinks).
    let mut q: Vec<NodeId> = (0..self.nodes.len()).filter(|&n| need_node[n] && succ_alias_count[n] == 0).collect();

    println!("=== reverse post-order (from targeted edges: #{:?}) ===", targets);

    while let Some(u) = q.pop() {
      // ---- Pretty print this node ----
      println!("node #{u}: {:?}", self.nodes[u].kind);

      // Outputs that contributed to the slice (original dependency set)
      if needed_out_aliases[u].is_empty() {
        println!("  outputs -> (targets or no downstream consumers)");
      } else {
        println!("  outputs:");
        for &(e, a) in &needed_out_aliases[u] {
          let (_pe, consumer, slot) = self.alias_info(a);
          println!("    edge #{e} via alias {:?} -> consumer node #{consumer} (input slot {slot})", a);
        }
      }

      // Inputs actually consumed to unlock predecessors
      if self.nodes[u].inputs.is_empty() {
        println!("  inputs: (none)");
      } else {
        println!("  inputs:");
        for (slot, &ie) in self.nodes[u].inputs.iter().enumerate() {
          if let Some(p) = self.producers[ie] {
            // find the exact alias from producer edge to this (u, slot)
            if let Some(&aid) = self.edge_aliases[ie].iter().find(|&&aid| {
              let (_e, c, s) = self.alias_info(aid);
              c == u && s == slot
            }) {
              println!("    edge #{ie}: alias {:?} from producer node #{p} (slot {slot})", aid);
            } else {
              println!("    edge #{ie}: from producer node #{p} (slot {slot})");
            }
          } else {
            if self.input_ports.contains(&ie) {
              println!("    edge #{ie}: GRAPH INPUT (slot {slot})");
            } else {
              println!("    edge #{ie}: CONSTANT (slot {slot})");
            }
          }
        }
      }

      // ---- Propagate to predecessors (decrement by matching aliases) ----
      for (slot, &ie) in self.nodes[u].inputs.iter().enumerate() {
        if let Some(p) = self.producers[ie] {
          if need_node[p] {
            let dec = self.edge_aliases[ie]
              .iter()
              .filter(|&&aid| {
                let (_e, c, s) = self.alias_info(aid);
                c == u && s == slot
              })
              .count();
            if dec > 0 {
              succ_alias_count[p] -= dec;
              if succ_alias_count[p] == 0 {
                q.push(p);
              }
            }
          }
        }
      }
    }

    println!("=== end reverse post-order ===");
  }

  // ------ small helpers reused by backward methods ------
  fn backward_mark_needed(&self, targets: &[EdgeId]) -> (Vec<bool>, Vec<bool>) {
    let mut need_edge = vec![false; self.num_edges];
    let mut need_node = vec![false; self.nodes.len()];
    let mut stack: Vec<EdgeId> = Vec::new();

    for &e in targets {
      if !need_edge[e] {
        need_edge[e] = true;
        stack.push(e);
      }
    }
    while let Some(e) = stack.pop() {
      if let Some(n) = self.producers[e] {
        if !need_node[n] {
          need_node[n] = true;
          for &ie in &self.nodes[n].inputs {
            if !need_edge[ie] {
              need_edge[ie] = true;
              stack.push(ie);
            }
          }
        }
      }
    }
    (need_edge, need_node)
  }

  fn collect_needed_out_aliases(&self, need_node: &[bool]) -> Vec<Vec<(EdgeId, AliasId)>> {
    let mut needed_out_aliases: Vec<Vec<(EdgeId, AliasId)>> = vec![Vec::new(); self.nodes.len()];
    for n in 0..self.nodes.len() {
      if !need_node[n] {
        continue;
      }
      for &e in &self.nodes[n].outputs {
        for &a in &self.edge_aliases[e] {
          let (_pe, consumer, _slot) = self.alias_info(a);
          if need_node[consumer] {
            needed_out_aliases[n].push((e, a));
          }
        }
      }
    }
    needed_out_aliases
  }
}

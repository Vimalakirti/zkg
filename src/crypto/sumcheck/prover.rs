use crate::util::poly::CryptoField;
use crate::util::poly::DenseMLPoly;
use crate::util::serialization::{ark_de, ark_se};
use crate::util::transcript::Transcript;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: CanonicalSerialize + CanonicalDeserialize")]
pub struct SumcheckProof<F: Clone> {
  #[serde(serialize_with = "ark_se", deserialize_with = "ark_de")]
  pub final_eval: F,
  #[serde(serialize_with = "se_nested_vec", deserialize_with = "de_nested_vec")]
  pub round_messages: Vec<Vec<F>>,
  /// ZK: mask polynomial sum P = Σ_{x∈{0,1}^n} p(x). None in non-ZK mode.
  #[serde(default, serialize_with = "se_opt_field", deserialize_with = "de_opt_field")]
  pub zk_mask_sum: Option<F>,
  /// ZK: univariate mask coefficients s_i. mask_coeffs[i] has degree+1 entries.
  #[serde(default, serialize_with = "se_opt_nested_vec", deserialize_with = "de_opt_nested_vec")]
  pub zk_mask_coeffs: Option<Vec<Vec<F>>>,
  /// ZK: PCS commitment to packed mask coefficients polynomial (opaque bytes).
  #[serde(default)]
  pub zk_mask_commitment: Option<Vec<u8>>,
  /// ZK: PCS opening proof for mask coefficients polynomial (opaque bytes).
  #[serde(default)]
  pub zk_mask_opening_proof: Option<Vec<u8>>,
  /// ZK: Point at which mask coefficients polynomial was opened.
  #[serde(default, serialize_with = "se_opt_vec", deserialize_with = "de_opt_vec")]
  pub zk_mask_opening_point: Option<Vec<F>>,
  /// ZK: Evaluation of mask coefficients polynomial at the opening point.
  #[serde(default, serialize_with = "se_opt_field", deserialize_with = "de_opt_field")]
  pub zk_mask_opening_eval: Option<F>,
}

impl<F: Clone> SumcheckProof<F> {
  /// Compute the number of extra bytes the ZK fields would add if serialized.
  /// Field element size = 32 bytes (BN254 scalar).
  pub fn zk_extra_bytes(&self) -> usize {
    let fe = 32usize; // field element size
    let mut total = 0;
    if let Some(ref _sum) = self.zk_mask_sum { total += fe; }
    if let Some(ref coeffs) = self.zk_mask_coeffs {
      total += coeffs.iter().map(|v| v.len() * fe).sum::<usize>();
    }
    if let Some(ref c) = self.zk_mask_commitment { total += c.len(); }
    if let Some(ref p) = self.zk_mask_opening_proof { total += p.len(); }
    if let Some(ref pt) = self.zk_mask_opening_point { total += pt.len() * fe; }
    if let Some(ref _e) = self.zk_mask_opening_eval { total += fe; }
    total
  }
}

/// Trait for committing to and opening mask coefficient polynomials.
/// Used to cryptographically bind mask coefficients before ρ is drawn.
pub trait MaskCommitter<F: CryptoField>: Send + Sync {
  /// Commit to a packed coefficients polynomial. Returns opaque commitment bytes.
  fn commit(&self, poly: &DenseMLPoly<F>) -> Vec<u8>;
  /// Open at a point. Returns (proof_bytes, evaluation).
  fn open(&self, poly: &DenseMLPoly<F>, point: &[F]) -> (Vec<u8>, F);
  /// Verify an opening.
  fn verify(&self, commitment: &[u8], proof: &[u8], point: &[F], eval: F) -> bool;
}

/// Context threaded through prove/verify for ZK mode.
pub struct ProveContext<F: CryptoField> {
  pub zk: bool,
  pub mask_committer: Option<Arc<dyn MaskCommitter<F>>>,
  _phantom: std::marker::PhantomData<F>,
}

impl<F: CryptoField> ProveContext<F> {
  pub fn new(zk: bool) -> Self {
    Self { zk, mask_committer: None, _phantom: std::marker::PhantomData }
  }

  pub fn with_mask_committer(zk: bool, committer: Arc<dyn MaskCommitter<F>>) -> Self {
    Self { zk, mask_committer: Some(committer), _phantom: std::marker::PhantomData }
  }
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

/// Serialize Option<F> for arkworks types
fn se_opt_field<S, A: CanonicalSerialize>(a: &Option<A>, s: S) -> Result<S::Ok, S::Error>
where
  S: serde::Serializer,
{
  match a {
    Some(v) => {
      let mut bytes = vec![];
      v.serialize_compressed(&mut bytes).map_err(serde::ser::Error::custom)?;
      s.serialize_some(&bytes)
    }
    None => s.serialize_none(),
  }
}

/// Deserialize Option<F> for arkworks types
fn de_opt_field<'de, D, A: CanonicalDeserialize>(data: D) -> Result<Option<A>, D::Error>
where
  D: serde::de::Deserializer<'de>,
{
  let v: Option<Vec<u8>> = serde::de::Deserialize::deserialize(data)?;
  match v {
    Some(bytes) => Ok(Some(A::deserialize_compressed_unchecked(bytes.as_slice()).map_err(serde::de::Error::custom)?)),
    None => Ok(None),
  }
}

/// Serialize Option<Vec<F>> for arkworks types
fn se_opt_vec<S, A: CanonicalSerialize>(a: &Option<Vec<A>>, s: S) -> Result<S::Ok, S::Error>
where
  S: serde::Serializer,
{
  match a {
    Some(v) => {
      let bytes: Vec<Vec<u8>> = v.iter().map(|elem| {
        let mut b = vec![];
        elem.serialize_compressed(&mut b).unwrap();
        b
      }).collect();
      s.serialize_some(&bytes)
    }
    None => s.serialize_none(),
  }
}

/// Deserialize Option<Vec<F>> for arkworks types
fn de_opt_vec<'de, D, A: CanonicalDeserialize>(data: D) -> Result<Option<Vec<A>>, D::Error>
where
  D: serde::de::Deserializer<'de>,
{
  let v: Option<Vec<Vec<u8>>> = serde::de::Deserialize::deserialize(data)?;
  match v {
    Some(inner) => Ok(Some(inner.into_iter().map(|bytes| A::deserialize_compressed_unchecked(bytes.as_slice()).map_err(serde::de::Error::custom)).collect::<Result<Vec<A>, _>>()?)),
    None => Ok(None),
  }
}

/// Serialize Option<Vec<Vec<F>>> for arkworks types
fn se_opt_nested_vec<S, A: CanonicalSerialize>(a: &Option<Vec<Vec<A>>>, s: S) -> Result<S::Ok, S::Error>
where
  S: serde::Serializer,
{
  match a {
    Some(v) => {
      let outer: Vec<Vec<Vec<u8>>> = v.iter().map(|inner| {
        inner.iter().map(|elem| {
          let mut b = vec![];
          elem.serialize_compressed(&mut b).unwrap();
          b
        }).collect()
      }).collect();
      s.serialize_some(&outer)
    }
    None => s.serialize_none(),
  }
}

/// Deserialize Option<Vec<Vec<F>>> for arkworks types
fn de_opt_nested_vec<'de, D, A: CanonicalDeserialize>(data: D) -> Result<Option<Vec<Vec<A>>>, D::Error>
where
  D: serde::de::Deserializer<'de>,
{
  let v: Option<Vec<Vec<Vec<u8>>>> = serde::de::Deserialize::deserialize(data)?;
  match v {
    Some(outer) => Ok(Some(outer.into_iter().map(|inner| {
      inner.into_iter().map(|bytes| A::deserialize_compressed_unchecked(bytes.as_slice()).map_err(serde::de::Error::custom)).collect::<Result<Vec<A>, _>>()
    }).collect::<Result<Vec<Vec<A>>, _>>()?)),
    None => Ok(None),
  }
}

pub trait SumcheckProver<F: CryptoField> {
  type Instance;

  fn new(n: usize, num_polys: usize, transcript: &mut Transcript<F>) -> Self;

  fn prove(&mut self, instances: &Self::Instance, transcript: &mut Transcript<F>) -> SumcheckProof<F>;
}

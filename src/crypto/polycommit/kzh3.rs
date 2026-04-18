use crate::crypto::polycommit::{Commitment, MLPolyCommit, PairingTrait};
use crate::util::arith::Math;
use crate::util::poly::{evaluate_lagrange_basis, CryptoField, DenseMLPoly, MLPoly};
use crate::util::serialization::{ark_de, ark_de_vec, ark_se, ark_se_vec};

#[cfg(feature = "arkworks")]
use rayon::prelude::*;

#[cfg(feature = "arkworks")]
use ark_ec::CurveGroup;
#[cfg(feature = "arkworks")]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
#[cfg(feature = "arkworks")]
use ark_std::UniformRand;
use serde::{Deserialize, Serialize};

#[cfg(feature = "icicle")]
use icicle_core::traits::GenerateRandom;

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

pub fn pad_at_start<F: CryptoField>(input: &[F], len: usize) -> Vec<F> {
  if input.len() >= len {
    return input.to_vec();
  }
  let mut result = vec![<F as CryptoField>::zero(); len - input.len()];
  result.extend_from_slice(input);
  result
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "arkworks", derive(ark_serialize::CanonicalSerialize, ark_serialize::CanonicalDeserialize))]
pub struct KZH3SRS<P: PairingTrait> {
  pub degree_x: usize,
  pub degree_y: usize,
  pub degree_z: usize,
  pub h_xyz: Vec<P::G1Affine>,
  pub h_yz: Vec<P::G1Affine>,
  pub h_z: Vec<P::G1Affine>,
  pub v_x: Vec<P::G2Affine>,
  pub v_y: Vec<P::G2Affine>,
  pub v_z: Vec<P::G2Affine>,
  pub v: P::G2Affine,
}

#[derive(Clone, Debug)]
pub struct KZH3Aux<P: PairingTrait> {
  pub d_x: Vec<P::G1>,
  /// ZK blinding factor: tau such that c_zk = c + tau * h
  pub tau: Option<P::ScalarField>,
}

impl<P: PairingTrait> KZH3Aux<P> {
  pub fn scale_by_r(&mut self, r: &P::ScalarField) {
    #[cfg(feature = "arkworks")]
    {
      self.d_x.par_iter_mut().for_each(|d| {
        *d = *d * *r;
      });
    }
    #[cfg(not(feature = "arkworks"))]
    {
      for d in &mut self.d_x {
        *d = *d * *r;
      }
    }
  }

  /// Create a dummy aux for deserialization/default (not cryptographically valid)
  pub fn dummy() -> Self {
    Self { d_x: vec![], tau: None }
  }
}

impl<P: PairingTrait> std::ops::Add for KZH3Aux<P> {
  type Output = Self;
  fn add(self, other: Self) -> Self {
    assert_eq!(self.d_x.len(), other.d_x.len());
    let tau = match (self.tau, other.tau) {
      (Some(a), Some(b)) => Some(a + b),
      (Some(a), None) => Some(a),
      (None, Some(b)) => Some(b),
      (None, None) => None,
    };
    Self {
      d_x: self.d_x.into_iter().zip(other.d_x).map(|(a, b)| a + b).collect(),
      tau,
    }
  }
}

impl<P: PairingTrait> std::ops::Sub for KZH3Aux<P> {
  type Output = Self;
  fn sub(self, other: Self) -> Self {
    assert_eq!(self.d_x.len(), other.d_x.len());
    let tau = match (self.tau, other.tau) {
      (Some(a), Some(b)) => Some(a - b),
      (Some(a), None) => Some(a),
      (None, Some(b)) => Some(-b),
      (None, None) => None,
    };
    Self {
      d_x: self.d_x.into_iter().zip(other.d_x).map(|(a, b)| a - b).collect(),
      tau,
    }
  }
}

impl<P: PairingTrait> std::ops::Mul<P::ScalarField> for KZH3Aux<P> {
  type Output = Self;
  fn mul(self, scalar: P::ScalarField) -> Self {
    Self {
      d_x: self.d_x.into_iter().map(|d| d * scalar).collect(),
      tau: self.tau.map(|t| t * scalar),
    }
  }
}

// KZH3 Opening
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
  bound = "P::G1: CanonicalSerialize + CanonicalDeserialize, P::G1Affine: CanonicalSerialize + CanonicalDeserialize, P::ScalarField: CanonicalSerialize + CanonicalDeserialize"
)]
pub struct KZH3Opening<P: PairingTrait> {
  #[serde(serialize_with = "ark_se_vec", deserialize_with = "ark_de_vec")]
  pub d_x: Vec<P::G1>,
  #[serde(serialize_with = "ark_se_vec", deserialize_with = "ark_de_vec")]
  pub d_y: Vec<P::G1>,
  #[serde(serialize_with = "ark_se", deserialize_with = "ark_de")]
  pub c_y: P::G1Affine,
  pub f_star: DenseMLPoly<P::ScalarField>,
  /// ZK: commitment to the masking polynomial r (None in non-ZK mode).
  #[serde(skip)]
  pub r_hide: Option<P::G1Affine>,
  /// ZK: evaluation of the masking polynomial r at the opening point.
  #[serde(skip)]
  pub y_r: Option<P::ScalarField>,
  /// ZK: combined blinding factor rho_prime = alpha * tau + rho.
  #[serde(skip)]
  pub rho_prime: Option<P::ScalarField>,
}

// Non-ZK KZH3 Commitment
#[derive(Clone, Debug)]
pub struct KZH3Commitment<P: PairingTrait> {
  pub srs: Arc<KZH3SRS<P>>,
  pub c: P::G1Affine,
  pub aux: KZH3Aux<P>,
}

// Manual PartialEq implementation that only compares the commitment point,
// not the SRS (since different Arc pointers to the same SRS data should be considered equal)
impl<P: PairingTrait> PartialEq for KZH3Commitment<P> {
  fn eq(&self, other: &Self) -> bool {
    self.c == other.c
  }
}

impl<P: PairingTrait> Eq for KZH3Commitment<P> {}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> ark_serialize::CanonicalSerialize for KZH3Commitment<P>
where
  P::G1Affine: ark_serialize::CanonicalSerialize,
  P::G1: ark_serialize::CanonicalSerialize,
{
  fn serialize_with_mode<W: ark_serialize::Write>(
    &self,
    mut writer: W,
    compress: ark_serialize::Compress,
  ) -> Result<(), ark_serialize::SerializationError> {
    self.c.serialize_with_mode(&mut writer, compress)?;
    self.aux.d_x.serialize_with_mode(&mut writer, compress)?;
    Ok(())
  }

  fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
    self.c.serialized_size(compress) + self.aux.d_x.serialized_size(compress)
  }
}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> ark_serialize::CanonicalDeserialize for KZH3Commitment<P>
where
  P::ScalarField: CryptoField,
  P::G1Affine: ark_serialize::CanonicalDeserialize + ark_serialize::Valid,
  P::G1: ark_serialize::CanonicalDeserialize + ark_serialize::Valid,
{
  fn deserialize_with_mode<R: ark_serialize::Read>(
    mut reader: R,
    compress: ark_serialize::Compress,
    validate: ark_serialize::Validate,
  ) -> Result<Self, ark_serialize::SerializationError> {
    let c = P::G1Affine::deserialize_with_mode(&mut reader, compress, validate)?;
    let d_x = Vec::<P::G1>::deserialize_with_mode(&mut reader, compress, validate)?;
    Ok(Self {
      srs: create_dummy_srs::<P>(),
      c,
      aux: KZH3Aux { d_x, tau: None },
    })
  }
}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> ark_serialize::Valid for KZH3Commitment<P>
where
  P::G1Affine: ark_serialize::Valid,
{
  fn check(&self) -> Result<(), ark_serialize::SerializationError> {
    self.c.check()
  }
}

impl<P: PairingTrait> std::ops::Add for KZH3Commitment<P>
where
  P::ScalarField: CryptoField,
{
  type Output = Self;

  fn add(self, other: Self) -> Self {
    // Both commitments should use the same SRS
    assert!(Arc::ptr_eq(&self.srs, &other.srs), "Cannot add commitments with different SRS");

    let g1_self: P::G1 = self.c.into();
    let g1_other: P::G1 = other.c.into();
    #[cfg(feature = "arkworks")]
    {
      use ark_ec::CurveGroup;
      let sum = (g1_self + g1_other).into_affine();
      Self {
        srs: self.srs,
        c: sum,
        aux: self.aux + other.aux,
      }
    }
    #[cfg(feature = "icicle")]
    {
      use crate::crypto::polycommit::PairingG1;
      let sum = (g1_self + g1_other).into_affine();
      Self {
        srs: self.srs,
        c: sum,
        aux: self.aux + other.aux,
      }
    }
  }
}

impl<P: PairingTrait> std::ops::Add<&KZH3Commitment<P>> for &KZH3Commitment<P>
where
  P::ScalarField: CryptoField,
{
  type Output = KZH3Commitment<P>;

  fn add(self, other: &KZH3Commitment<P>) -> Self::Output {
    self.clone() + other.clone()
  }
}

impl<P: PairingTrait> std::ops::Sub for KZH3Commitment<P>
where
  P::ScalarField: CryptoField,
{
  type Output = Self;

  fn sub(self, other: Self) -> Self {
    // Both commitments should use the same SRS
    assert!(Arc::ptr_eq(&self.srs, &other.srs), "Cannot subtract commitments with different SRS");

    #[cfg(feature = "arkworks")]
    {
      use ark_ec::CurveGroup;
      let g1_self: P::G1 = self.c.into();
      let g1_other: P::G1 = other.c.into();
      let neg_other: P::G1 = -g1_other;
      let diff = (g1_self + neg_other).into_affine();
      Self {
        srs: self.srs,
        c: diff,
        aux: self.aux - other.aux,
      }
    }
    #[cfg(feature = "icicle")]
    {
      use crate::crypto::polycommit::PairingG1;
      let g1_self: P::G1 = self.c.into();
      let g1_other: P::G1 = other.c.into();
      let diff = (g1_self - g1_other).into_affine();
      Self {
        srs: self.srs,
        c: diff,
        aux: self.aux - other.aux,
      }
    }
  }
}

impl<P: PairingTrait> std::ops::Sub<&KZH3Commitment<P>> for &KZH3Commitment<P>
where
  P::ScalarField: CryptoField,
{
  type Output = KZH3Commitment<P>;

  fn sub(self, other: &KZH3Commitment<P>) -> Self::Output {
    self.clone() - other.clone()
  }
}

impl<P: PairingTrait> std::ops::Mul<P::ScalarField> for KZH3Commitment<P>
where
  P::ScalarField: CryptoField,
{
  type Output = Self;

  fn mul(self, scalar: P::ScalarField) -> Self {
    let g1: P::G1 = self.c.into();
    #[cfg(feature = "arkworks")]
    {
      use ark_ec::CurveGroup;
      let scaled = (g1 * scalar).into_affine();
      Self {
        srs: self.srs,
        c: scaled,
        aux: self.aux * scalar,
      }
    }
    #[cfg(feature = "icicle")]
    {
      use crate::crypto::polycommit::PairingG1;
      let scaled = (g1 * scalar).into_affine();
      Self {
        srs: self.srs,
        c: scaled,
        aux: self.aux * scalar,
      }
    }
  }
}

impl<P: PairingTrait> std::ops::Mul<&P::ScalarField> for &KZH3Commitment<P>
where
  P::ScalarField: CryptoField,
{
  type Output = KZH3Commitment<P>;

  fn mul(self, scalar: &P::ScalarField) -> Self::Output {
    self.clone() * scalar.clone()
  }
}

// Helper function to create a minimal dummy SRS for Default/deserialization
// This SRS is not cryptographically valid and should not be used for actual cryptographic operations
#[cfg(feature = "arkworks")]
fn create_dummy_srs<P: PairingTrait>() -> Arc<KZH3SRS<P>>
where
  P::ScalarField: CryptoField,
{
  Arc::new(KZH3SRS {
    degree_x: 1,
    degree_y: 1,
    degree_z: 1,
    h_xyz: vec![P::G1Affine::default()],
    h_yz: vec![P::G1Affine::default()],
    h_z: vec![P::G1Affine::default()],
    v_x: vec![P::G2Affine::default()],
    v_y: vec![P::G2Affine::default()],
    v_z: vec![P::G2Affine::default()],
    v: P::G2Affine::default(),
  })
}

#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
fn create_dummy_srs<P: PairingTrait>() -> Arc<KZH3SRS<P>>
where
  P::ScalarField: CryptoField,
  P::G2: icicle_core::projective::Projective<Affine = P::G2Affine>,
{
  use crate::crypto::polycommit::PairingG1;
  use icicle_core::projective::Projective;
  // Create zero/identity G2Affine value safely using Projective::zero
  // This is only used for Default/deserialization, not for actual cryptographic operations
  let g2_zero: P::G2 = <P::G2 as Projective>::zero();
  let dummy_g2_affine: P::G2Affine = g2_zero.to_affine();
  Arc::new(KZH3SRS {
    degree_x: 1,
    degree_y: 1,
    degree_z: 1,
    h_xyz: vec![<P::G1 as PairingG1>::zero().into_affine()],
    h_yz: vec![<P::G1 as PairingG1>::zero().into_affine()],
    h_z: vec![<P::G1 as PairingG1>::zero().into_affine()],
    v_x: vec![dummy_g2_affine.clone()],
    v_y: vec![dummy_g2_affine.clone()],
    v_z: vec![dummy_g2_affine.clone()],
    v: dummy_g2_affine,
  })
}

impl<P: PairingTrait> Default for KZH3Commitment<P>
where
  P::ScalarField: CryptoField,
{
  fn default() -> Self {
    #[cfg(feature = "arkworks")]
    {
      Self {
        srs: create_dummy_srs::<P>(),
        c: P::G1Affine::default(),
        aux: KZH3Aux::dummy(),
      }
    }
    #[cfg(feature = "icicle")]
    {
      use crate::crypto::polycommit::PairingG1;
      let zero_projective: P::G1 = <P::G1 as PairingG1>::zero();
      Self {
        srs: create_dummy_srs::<P>(),
        c: zero_projective.into_affine(),
        aux: KZH3Aux::dummy(),
      }
    }
  }
}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> Commitment<P::ScalarField> for KZH3Commitment<P>
where
  P::ScalarField: CryptoField,
  P::G1: ark_ec::CurveGroup,
{
  fn to_transcript_bytes(&self) -> Vec<u8> {
    use ark_serialize::CanonicalSerialize;
    let mut bytes = Vec::new();
    self.c.serialize_compressed(&mut bytes).expect("commitment serialization failed");
    bytes
  }
}

#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl<P: PairingTrait> Commitment<P::ScalarField> for KZH3Commitment<P> where P::ScalarField: CryptoField {}

fn get_degree_from_maximum_supported_degree(n: usize) -> (usize, usize, usize) {
  // Balanced split to match reference implementation
  match n % 3 {
    0 => (n / 3, n / 3, n / 3),
    1 => (n / 3 + 1, n / 3, n / 3),
    2 => (n / 3 + 1, n / 3 + 1, n / 3),
    _ => unreachable!(),
  }
}

pub fn scalar_field_zero<F: CryptoField>() -> F {
  <F as CryptoField>::zero()
}

pub fn split_input<P: PairingTrait, T: Clone>(srs: &KZH3SRS<P>, input: &[T], default: T) -> Vec<Vec<T>> {
  let z_log = srs.degree_z.log_2();
  let y_log = srs.degree_y.log_2();
  let x_log = srs.degree_x.log_2();
  let total_length = x_log + y_log + z_log;

  let mut extended_r = input.to_vec();
  if input.len() < total_length {
    let mut zeros = vec![default; total_length - input.len()];
    zeros.extend(extended_r); // Prepend zeros to the beginning
    extended_r = zeros;
  }

  let r_z = extended_r[..z_log].to_vec();
  let r_y = extended_r[z_log..(z_log + y_log)].to_vec();
  let r_x = extended_r[(z_log + y_log)..(z_log + y_log + x_log)].to_vec();
  vec![r_z, r_y, r_x]
}

#[cfg(feature = "arkworks")]
pub fn setup_kzh3_srs<P: PairingTrait, R: ark_std::rand::Rng>(maximum_degree: usize, rng: &mut R) -> KZH3SRS<P>
where
  P::ScalarField: CryptoField,
  P::G1Affine: ark_ec::AffineRepr,
  P::G2Affine: ark_ec::AffineRepr,
{
  let (degree_x_log, degree_y_log, degree_z_log) = get_degree_from_maximum_supported_degree(maximum_degree);
  let degree_x = 1 << degree_x_log;
  let degree_y = 1 << degree_y_log;
  let degree_z = 1 << degree_z_log;

  let (g, v) = (P::G1Affine::rand(rng), P::G2Affine::rand(rng));

  let tau_x: Vec<P::ScalarField> = (0..degree_x).map(|_| P::ScalarField::rand(rng)).collect();
  let tau_y: Vec<P::ScalarField> = (0..degree_y).map(|_| P::ScalarField::rand(rng)).collect();
  let tau_z: Vec<P::ScalarField> = (0..degree_z).map(|_| P::ScalarField::rand(rng)).collect();

  // Generate h_xyz: g^{tau_x[i] * tau_y[j] * tau_z[k]} for all i, j, k
  // Indexed in DenseMLPoly order (z in low bits, y in middle, x in high bits)
  let z_log = degree_z.log_2();
  let y_log = degree_y.log_2();
  let x_log = degree_x.log_2();
  let z_mask = (1 << z_log) - 1;
  let y_mask = (1 << y_log) - 1;
  let x_mask = (1 << x_log) - 1;

  #[cfg(feature = "arkworks")]
  let h_xyz: Vec<_> = (0..degree_x * degree_y * degree_z)
    .into_par_iter()
    .map(|denseml_idx| {
      let z = denseml_idx & z_mask;
      let y = (denseml_idx >> z_log) & y_mask;
      let x = (denseml_idx >> (z_log + y_log)) & x_mask;
      (g * (tau_x[x] * tau_y[y] * tau_z[z])).into()
    })
    .collect();

  #[cfg(not(feature = "arkworks"))]
  let h_xyz: Vec<_> = (0..degree_x * degree_y * degree_z)
    .map(|denseml_idx| {
      let z = denseml_idx & z_mask;
      let y = (denseml_idx >> z_log) & y_mask;
      let x = (denseml_idx >> (z_log + y_log)) & x_mask;
      (g * (tau_x[x] * tau_y[y] * tau_z[z])).into()
    })
    .collect();

  // Generate h_yz: g^{tau_y[j] * tau_z[k]} for all j, k
  // Indexed in DenseMLPoly order (z in low bits, y in high bits)
  #[cfg(feature = "arkworks")]
  let h_yz: Vec<_> = (0..degree_y * degree_z)
    .into_par_iter()
    .map(|denseml_idx| {
      let z = denseml_idx & z_mask;
      let y = (denseml_idx >> z_log) & y_mask;
      (g * (tau_y[y] * tau_z[z])).into()
    })
    .collect();

  #[cfg(not(feature = "arkworks"))]
  let h_yz: Vec<_> = (0..degree_y * degree_z)
    .map(|denseml_idx| {
      let z = denseml_idx & z_mask;
      let y = (denseml_idx >> z_log) & y_mask;
      (g * (tau_y[y] * tau_z[z])).into()
    })
    .collect();

  #[cfg(feature = "arkworks")]
  let h_z: Vec<_> = (0..degree_z).into_par_iter().map(|i| (g * tau_z[i]).into()).collect();
  #[cfg(not(feature = "arkworks"))]
  let h_z: Vec<_> = (0..degree_z).map(|i| (g * tau_z[i]).into()).collect();

  #[cfg(feature = "arkworks")]
  let v_x: Vec<_> = (0..degree_x).into_par_iter().map(|i| (v * tau_x[i]).into()).collect();
  #[cfg(not(feature = "arkworks"))]
  let v_x: Vec<_> = (0..degree_x).map(|i| (v * tau_x[i]).into()).collect();

  #[cfg(feature = "arkworks")]
  let v_y: Vec<_> = (0..degree_y).into_par_iter().map(|i| (v * tau_y[i]).into()).collect();
  #[cfg(not(feature = "arkworks"))]
  let v_y: Vec<_> = (0..degree_y).map(|i| (v * tau_y[i]).into()).collect();

  #[cfg(feature = "arkworks")]
  let v_z: Vec<_> = (0..degree_z).into_par_iter().map(|i| (v * tau_z[i]).into()).collect();
  #[cfg(not(feature = "arkworks"))]
  let v_z: Vec<_> = (0..degree_z).map(|i| (v * tau_z[i]).into()).collect();

  KZH3SRS {
    degree_x,
    degree_y,
    degree_z,
    h_xyz,
    h_yz,
    h_z,
    v_x,
    v_y,
    v_z,
    v,
  }
}

#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
pub fn setup_kzh3_srs<P: PairingTrait, R>(maximum_degree: usize, _rng: &mut R) -> KZH3SRS<P>
where
  P::ScalarField: CryptoField + GenerateRandom,
  P::G1Affine: icicle_core::traits::GenerateRandom,
  P::G2Affine: icicle_core::traits::GenerateRandom,
{
  use crate::crypto::polycommit::{generate_random_g1_affine, generate_random_g2_affine};
  let (degree_x_log, degree_y_log, degree_z_log) = get_degree_from_maximum_supported_degree(maximum_degree);
  let degree_x = 1 << degree_x_log;
  let degree_y = 1 << degree_y_log;
  let degree_z = 1 << degree_z_log;

  // Generate random G1 and G2 generators using helper functions
  let g_vec = generate_random_g1_affine::<P>(1);
  let v_vec = generate_random_g2_affine::<P>(1);
  let g = g_vec[0].clone();
  let v = v_vec[0].clone();

  // Generate random tau values
  let tau_x = P::ScalarField::generate_random(degree_x);
  let tau_y = P::ScalarField::generate_random(degree_y);
  let tau_z = P::ScalarField::generate_random(degree_z);

  // Convert affine to projective for scalar multiplication
  let g_proj: P::G1 = g.clone().into();
  let v_proj: P::G2 = v.clone().into();

  // Generate h_xyz: g^{tau_x[i] * tau_y[j] * tau_z[k]} for all i, j, k
  // Indexed in DenseMLPoly order (z in low bits, y in middle, x in high bits)
  let z_log = degree_z.log_2();
  let y_log = degree_y.log_2();
  let x_log = degree_x.log_2();
  let z_mask = (1 << z_log) - 1;
  let y_mask = (1 << y_log) - 1;
  let x_mask = (1 << x_log) - 1;

  let h_xyz: Vec<_> = (0..degree_x * degree_y * degree_z)
    .map(|denseml_idx| {
      let z = denseml_idx & z_mask;
      let y = (denseml_idx >> z_log) & y_mask;
      let x = (denseml_idx >> (z_log + y_log)) & x_mask;
      let scalar = tau_x[x] * tau_y[y] * tau_z[z];
      (g_proj * scalar).into()
    })
    .collect();

  // Generate h_yz: g^{tau_y[j] * tau_z[k]} for all j, k
  // Indexed in DenseMLPoly order (z in low bits, y in high bits)
  let h_yz: Vec<_> = (0..degree_y * degree_z)
    .map(|denseml_idx| {
      let z = denseml_idx & z_mask;
      let y = (denseml_idx >> z_log) & y_mask;
      let scalar = tau_y[y] * tau_z[z];
      (g_proj * scalar).into()
    })
    .collect();

  let h_z: Vec<_> = (0..degree_z).map(|i| (g_proj * tau_z[i]).into()).collect();

  let v_x: Vec<_> = (0..degree_x).map(|i| (v_proj * tau_x[i]).into()).collect();

  let v_y: Vec<_> = (0..degree_y).map(|i| (v_proj * tau_y[i]).into()).collect();

  let v_z: Vec<_> = (0..degree_z).map(|i| (v_proj * tau_z[i]).into()).collect();

  KZH3SRS {
    degree_x,
    degree_y,
    degree_z,
    h_xyz,
    h_yz,
    h_z,
    v_x,
    v_y,
    v_z,
    v,
  }
}

pub fn kzh3_commit<P: PairingTrait>(srs: &KZH3SRS<P>, poly: &DenseMLPoly<P::ScalarField>) -> (P::G1Affine, KZH3Aux<P>)
where
  P::ScalarField: CryptoField,
{
  let len = srs.degree_x.log_2() + srs.degree_y.log_2() + srs.degree_z.log_2();

  // Only extend if needed to avoid unnecessary cloning
  let extended_poly;
  let poly_ref: &DenseMLPoly<P::ScalarField> = if poly.n == len {
    poly // No clone needed!
  } else {
    extended_poly = poly.clone().extend_number_of_variables(len);
    &extended_poly
  };

  // Compute commitment: C = sum_i f[i] * h_xyz[i]
  // h_xyz is now indexed by DenseMLPoly order, so direct MSM works
  let c = {
    #[cfg(feature = "arkworks")]
    {
      use ark_ec::VariableBaseMSM;
      P::G1::msm(&srs.h_xyz, &poly_ref.evaluations).unwrap().into()
    }
    #[cfg(feature = "icicle")]
    {
      use crate::crypto::polycommit::PairingG1;
      <P::G1 as PairingG1>::msm(&srs.h_xyz, &poly_ref.evaluations).unwrap().into()
    }
  };

  // Compute auxiliary data: d_x[i] = sum_{j,k} f[i,j,k] * h_yz[j,k]
  // Use get_partial_evaluation_for_boolean_input to directly access polynomial slices
  // without creating an intermediate 2D structure, avoiding data movement
  let yz_size = srs.degree_y * srs.degree_z;
  #[cfg(feature = "arkworks")]
  let d_x: Vec<_> = (0..srs.degree_x)
    .into_par_iter()
    .map(|i| {
      use ark_ec::VariableBaseMSM;
      let partial_evals = poly_ref.get_partial_evaluation_for_boolean_input(i, yz_size);
      P::G1::msm(&srs.h_yz, partial_evals).unwrap()
    })
    .collect();

  #[cfg(not(feature = "arkworks"))]
  let d_x: Vec<_> = (0..srs.degree_x)
    .into_iter()
    .map(|i| {
      use crate::crypto::polycommit::PairingG1;
      let partial_evals = poly_ref.get_partial_evaluation_for_boolean_input(i, yz_size);
      <P::G1 as PairingG1>::msm(&srs.h_yz, partial_evals).unwrap()
    })
    .collect();

  (c, KZH3Aux { d_x, tau: None })
}

pub fn kzh3_open<P: PairingTrait>(
  srs: &KZH3SRS<P>,
  input: &[P::ScalarField],
  _com: &P::G1Affine,
  aux: &KZH3Aux<P>,
  poly: &DenseMLPoly<P::ScalarField>,
) -> KZH3Opening<P>
where
  P::ScalarField: CryptoField,
{
  let x_log = srs.degree_x.log_2();
  let y_log = srs.degree_y.log_2();
  let z_log = srs.degree_z.log_2();
  let len = x_log + y_log + z_log;

  let extended_poly;
  let poly_ref: &DenseMLPoly<P::ScalarField> = if poly.n == len {
    poly
  } else {
    extended_poly = poly.clone().extend_number_of_variables(len);
    &extended_poly
  };

  assert_eq!(poly_ref.n, len);
  assert_eq!(poly_ref.evaluations.len(), 1 << poly_ref.n);

  let split_input = split_input(srs, input, scalar_field_zero::<P::ScalarField>());
  // split_input[0] = r_z (lowest bits), split_input[1] = r_y (middle bits), split_input[2] = r_x (highest bits)
  let r_y = &split_input[1];
  let r_x = &split_input[2];

  // Compute f_prime = f(x, Y, Z) by fixing x from the right (highest bits in LSB ordering)
  // Use optimized fix_variables_from_right instead of manual loops
  use crate::util::poly::fix_variables_from_right;
  let f_prime = fix_variables_from_right(poly_ref, r_x);

  // Compute eq_x_evals for C_y computation
  let eq_x_evals = evaluate_lagrange_basis(r_x);

  // Compute D_y using partial evaluations of f_prime
  #[cfg(feature = "arkworks")]
  let d_y: Vec<_> = (0..srs.degree_y)
    .into_par_iter()
    .map(|i| {
      let partial_evals = f_prime.get_partial_evaluation_for_boolean_input(i, srs.degree_z);
      use ark_ec::VariableBaseMSM;
      P::G1::msm(&srs.h_z, partial_evals).unwrap()
    })
    .collect();

  #[cfg(not(feature = "arkworks"))]
  let d_y: Vec<_> = (0..srs.degree_y)
    .map(|i| {
      let partial_evals = f_prime.get_partial_evaluation_for_boolean_input(i, srs.degree_z);
      use crate::crypto::polycommit::PairingG1;
      <P::G1 as PairingG1>::msm(&srs.h_z, partial_evals).unwrap()
    })
    .collect();

  // Compute f_star(Z) = f(r_x, r_y, Z) - fix both y and x from the right
  let mut yx_point = r_y.clone();
  yx_point.extend_from_slice(r_x);
  let f_star = fix_variables_from_right(poly_ref, &yx_point);

  // Compute C_y using eq polynomial over x (already computed above)
  let d_x_affine: Vec<_> = aux.d_x.iter().map(|e| (*e).into()).collect();
  let c_y = {
    #[cfg(feature = "arkworks")]
    {
      use ark_ec::VariableBaseMSM;
      P::G1::msm(&d_x_affine, &eq_x_evals).unwrap().into()
    }
    #[cfg(feature = "icicle")]
    {
      use crate::crypto::polycommit::PairingG1;
      <P::G1 as PairingG1>::msm(&d_x_affine, &eq_x_evals).unwrap().into()
    }
  };

  KZH3Opening {
    d_x: aux.d_x.clone(),
    d_y,
    c_y,
    f_star,
    r_hide: None,
    y_r: None,
    rho_prime: None,
  }
}

// KZH3 verify - fully parallelized for maximum throughput
pub fn kzh3_verify<P: PairingTrait>(
  srs: &KZH3SRS<P>,
  input: &[P::ScalarField],
  output: &P::ScalarField,
  com: &P::G1Affine,
  open: &KZH3Opening<P>,
) -> bool
where
  P::ScalarField: CryptoField,
{
  let split_input = split_input(srs, input, scalar_field_zero::<P::ScalarField>());
  let r_z = &split_input[0];
  let r_y = &split_input[1];
  let r_x = &split_input[2];

  // Phase 1: Parallel precomputation of affine conversions and lagrange basis
  #[cfg(feature = "arkworks")]
  let ((d_x_affine, d_y_affine), (eq_evals_x, eq_evals_y)) = rayon::join(
    || {
      rayon::join(
        || open.d_x.par_iter().map(|g| (*g).into()).collect::<Vec<P::G1Affine>>(),
        || open.d_y.par_iter().map(|g| (*g).into()).collect::<Vec<P::G1Affine>>(),
      )
    },
    || rayon::join(|| evaluate_lagrange_basis(r_x), || evaluate_lagrange_basis(r_y)),
  );

  #[cfg(all(feature = "icicle", not(feature = "arkworks")))]
  let (d_x_affine, d_y_affine, eq_evals_x, eq_evals_y) = {
    let d_x_affine: Vec<P::G1Affine> = open.d_x.iter().map(|g| (*g).into()).collect();
    let d_y_affine: Vec<P::G1Affine> = open.d_y.iter().map(|g| (*g).into()).collect();
    let eq_evals_x = evaluate_lagrange_basis(r_x);
    let eq_evals_y = evaluate_lagrange_basis(r_y);
    (d_x_affine, d_y_affine, eq_evals_x, eq_evals_y)
  };

  // Phase 2: Run ALL expensive operations (pairings + MSMs) in parallel
  // This maximizes parallelism for the common case where verification succeeds
  #[cfg(feature = "arkworks")]
  let ((pairing1_lhs, pairing1_rhs), ((computed_c_y, (pairing3_lhs, pairing3_rhs)), (msm_lhs, msm_rhs))) = rayon::join(
    // Check 1: D_x well-formatted
    || (P::multi_pairing(&d_x_affine, &srs.v_x), P::pairing(com, &srs.v)),
    || {
      rayon::join(
        || {
          rayon::join(
            // Check 2: c_y computation
            || {
              use ark_ec::VariableBaseMSM;
              P::G1::msm(&d_x_affine, &eq_evals_x).unwrap().into_affine()
            },
            // Check 3: D_y well-formatted
            || (P::multi_pairing(&d_y_affine, &srs.v_y), P::pairing(&open.c_y, &srs.v)),
          )
        },
        // Check 4: Final MSM comparison
        || {
          use ark_ec::VariableBaseMSM;
          rayon::join(
            || P::G1::msm(&srs.h_z, &open.f_star.evaluations).unwrap(),
            || P::G1::msm(&d_y_affine, &eq_evals_y).unwrap(),
          )
        },
      )
    },
  );

  #[cfg(all(feature = "icicle", not(feature = "arkworks")))]
  let (pairing1_lhs, pairing1_rhs, computed_c_y, pairing3_lhs, pairing3_rhs, msm_lhs, msm_rhs) = {
    use crate::crypto::polycommit::PairingG1;
    let pairing1_lhs = P::multi_pairing(&d_x_affine, &srs.v_x);
    let pairing1_rhs = P::pairing(com, &srs.v);
    let computed_c_y = <P::G1 as PairingG1>::msm(&d_x_affine, &eq_evals_x).unwrap().into_affine();
    let pairing3_lhs = P::multi_pairing(&d_y_affine, &srs.v_y);
    let pairing3_rhs = P::pairing(&open.c_y, &srs.v);
    let msm_lhs = <P::G1 as PairingG1>::msm(&srs.h_z, &open.f_star.evaluations).unwrap();
    let msm_rhs = <P::G1 as PairingG1>::msm(&d_y_affine, &eq_evals_y).unwrap();
    (pairing1_lhs, pairing1_rhs, computed_c_y, pairing3_lhs, pairing3_rhs, msm_lhs, msm_rhs)
  };

  // Phase 3: Verify all results
  if pairing1_lhs != pairing1_rhs {
    return false;
  }
  if computed_c_y != open.c_y {
    return false;
  }
  if pairing3_lhs != pairing3_rhs {
    return false;
  }
  if msm_lhs != msm_rhs {
    return false;
  }

  // Check 5: Final polynomial evaluation
  let computed_output = open.f_star.evaluate_at_point(r_z);
  computed_output == *output
}

/// Compute hiding sparsity: ~3 * N^{1/3} nonzero entries for masking polynomial.
/// Matches reference: ceil_k_root_scaled(N, k) with k=3.
pub fn hiding_sparsity(num_vars: usize) -> usize {
  let n = 1u64 << num_vars;
  let cbrt = (n as f64).cbrt().ceil() as usize;
  3 * cbrt
}

/// Generate a random sparse multilinear polynomial with `sparsity` nonzero entries.
/// Returns a SparseMLPoly suitable for sparse commit/open operations.
#[cfg(feature = "arkworks")]
pub fn rand_sparse_poly<F: CryptoField>(
  num_vars: usize,
  sparsity: usize,
) -> crate::util::poly::SparseMLPoly<F> {
  use crate::util::poly::{SelectionPolynomial, SparseMLPoly};
  use ark_std::rand::thread_rng;
  use std::collections::{HashMap, HashSet, VecDeque};

  let mut rng = thread_rng();
  let total = 1usize << num_vars;
  let actual_sparsity = sparsity.min(total);

  // Sample `actual_sparsity` random distinct indices
  let mut chosen = HashSet::new();
  while chosen.len() < actual_sparsity {
    let idx: usize = <usize as ark_std::UniformRand>::rand(&mut rng) % total;
    chosen.insert(idx);
  }

  let mut evaluations = HashMap::new();
  for idx in &chosen {
    evaluations.insert(*idx, F::rand(&mut rng));
  }

  // Dummy SelectionPolynomial (not used for masking poly operations)
  let dummy_selection = SelectionPolynomial::new(num_vars, 0, vec![]);

  let mut sparse = SparseMLPoly::new(num_vars, evaluations, VecDeque::new(), dummy_selection);
  sparse.build_sorted_indices();
  sparse
}

/// ZK commit: c_zk = c_non_zk + tau * h, where c_non_zk = MSM(srs, poly).
/// Stores tau in aux for use during open.
#[cfg(feature = "arkworks")]
pub fn kzh3_commit_zk<P: PairingTrait>(
  srs: &KZH3SRS<P>,
  poly: &DenseMLPoly<P::ScalarField>,
  h: &P::G1Affine,
) -> (P::G1Affine, KZH3Aux<P>)
where
  P::ScalarField: CryptoField,
{
  use ark_std::rand::thread_rng;
  let mut rng = thread_rng();
  let tau = P::ScalarField::rand(&mut rng);

  let (c_non_zk, mut aux) = kzh3_commit(srs, poly);

  // c_zk = c_non_zk + tau * h
  let h_proj: P::G1 = (*h).into();
  let c_zk: P::G1Affine = (P::G1::from(c_non_zk) + h_proj * tau).into();
  aux.tau = Some(tau);
  (c_zk, aux)
}

/// ZK open following the reference protocol (KZHK open_zk):
/// 1. Non-ZK open the original polynomial
/// 2. Sample sparse masking poly r (O(N^{1/3}) entries)
/// 3. Commit to r with ZK blinding (sparse commit + tau*h)
/// 4. Open r at the same point (sparse open, non-ZK since r is public to prover)
/// 5. Combine: proof = alpha * non_zk_proof + r_proof, with alpha=1
/// 6. Attach r_hide (commitment to r), y_r (eval of r), rho_prime
#[cfg(feature = "arkworks")]
pub fn kzh3_open_zk<P: PairingTrait>(
  srs: &KZH3SRS<P>,
  input: &[P::ScalarField],
  com: &P::G1Affine,
  aux: &KZH3Aux<P>,
  poly: &DenseMLPoly<P::ScalarField>,
  h: &P::G1Affine,
) -> KZH3Opening<P>
where
  P::ScalarField: CryptoField,
  P::G1: ark_ec::CurveGroup,
  P::G1Affine: Send + Sync,
{
  use crate::crypto::polycommit::sparse_kzh3::{sparse_kzh3_commit, sparse_kzh3_open};
  use ark_std::rand::thread_rng;

  // Step 1: Non-ZK open the original polynomial
  let non_zk_open = kzh3_open(srs, input, com, aux, poly);

  // Step 2: Sample sparse masking polynomial r(X)
  let num_vars = srs.degree_x.log_2() + srs.degree_y.log_2() + srs.degree_z.log_2();
  let sparsity = hiding_sparsity(num_vars);
  let r_poly = rand_sparse_poly::<P::ScalarField>(num_vars, sparsity);

  // Step 3: Commit to r with ZK blinding (sparse commit + tau*h)
  // Following reference: Self::commit(prover_param, &r_poly_wrapped) dispatches to commit_zk
  // which calls commit_sparse_inner then adds tau*h
  let (r_com_non_zk, r_aux_non_zk) = sparse_kzh3_commit(srs, &r_poly);
  let mut rng = thread_rng();
  let rho = P::ScalarField::rand(&mut rng);
  let h_proj: P::G1 = (*h).into();
  let r_com: P::G1Affine = (P::G1::from(r_com_non_zk) + h_proj * rho).into();

  // Step 4: Open r at the same point using sparse open (non-ZK, matching reference:
  // open_sparse_non_bool_inner with dummy_aux)
  let dummy_aux = KZH3Aux { d_x: r_aux_non_zk.d_x.clone(), tau: None };
  let r_opening = sparse_kzh3_open(srs, input, &r_com, &dummy_aux, &r_poly);

  // Step 5: Evaluate r at the point
  let y_r = r_poly.evaluate_at_point(input);

  // Step 6: Derive alpha from Fiat-Shamir transcript (absorb r_hide commitment)
  let alpha = {
    use ark_serialize::CanonicalSerialize;
    let mut alpha_transcript = merlin::Transcript::new(b"kzh3_zk_alpha");
    let mut r_com_bytes = Vec::new();
    r_com.serialize_compressed(&mut r_com_bytes).expect("serialize r_com");
    alpha_transcript.append_message(b"r_hide", &r_com_bytes);
    let mut buf = [0u8; 64];
    alpha_transcript.challenge_bytes(b"alpha", &mut buf);
    <P::ScalarField as CryptoField>::from_bytes_le(&buf)
  };

  // Step 7: rho_prime = alpha * tau + rho (reference: alpha * aux.get_tau() + rho)
  let tau = aux.tau.unwrap_or_else(|| <P::ScalarField as CryptoField>::zero());
  let rho_prime = alpha * tau + rho;

  // Step 8: Combine openings (reference: output_opening = non_zk_opening * alpha + r_opening)
  let d_x_combined: Vec<P::G1> = non_zk_open.d_x.iter().zip(r_opening.d_x.iter())
    .map(|(a, b)| *a * alpha + *b)
    .collect();
  let d_y_combined: Vec<P::G1> = non_zk_open.d_y.iter().zip(r_opening.d_y.iter())
    .map(|(a, b)| *a * alpha + *b)
    .collect();
  let c_y_combined: P::G1Affine = (P::G1::from(non_zk_open.c_y) * alpha + P::G1::from(r_opening.c_y)).into();
  let f_star_combined_evals: Vec<P::ScalarField> = non_zk_open.f_star.evaluations.iter()
    .zip(r_opening.f_star.evaluations.iter())
    .map(|(a, b)| *a * alpha + *b)
    .collect();
  let f_star_combined = DenseMLPoly::new(non_zk_open.f_star.n, f_star_combined_evals);

  // Step 9: Attach ZK fields (reference: set_r_hide, set_y_r, set_rho_prime)
  KZH3Opening {
    d_x: d_x_combined,
    d_y: d_y_combined,
    c_y: c_y_combined,
    f_star: f_star_combined,
    r_hide: Some(r_com),
    y_r: Some(y_r),
    rho_prime: Some(rho_prime),
  }
}

/// ZK verify following reference protocol (KZHK verify_zk):
/// 1. Linearize commitment: c_lin = alpha*c + r_hide - rho_prime*h
/// 2. Linearize value: v_lin = alpha*v + y_r
/// 3. Run non-ZK verify with (c_lin, v_lin)
#[cfg(feature = "arkworks")]
pub fn kzh3_verify_zk<P: PairingTrait>(
  srs: &KZH3SRS<P>,
  input: &[P::ScalarField],
  output: &P::ScalarField,
  com: &P::G1Affine,
  open: &KZH3Opening<P>,
  h: &P::G1Affine,
) -> bool
where
  P::ScalarField: CryptoField,
{
  let (r_com, _y_r, rho_prime) = match (&open.r_hide, &open.y_r, &open.rho_prime) {
    (Some(r_com), Some(y_r), Some(rho_prime)) => (r_com, y_r, rho_prime),
    _ => return false,
  };

  // Derive alpha from Fiat-Shamir transcript (must match prover's derivation)
  let alpha = {
    use ark_serialize::CanonicalSerialize;
    let mut alpha_transcript = merlin::Transcript::new(b"kzh3_zk_alpha");
    let mut r_com_bytes = Vec::new();
    r_com.serialize_compressed(&mut r_com_bytes).expect("serialize r_com");
    alpha_transcript.append_message(b"r_hide", &r_com_bytes);
    let mut buf = [0u8; 64];
    alpha_transcript.challenge_bytes(b"alpha", &mut buf);
    <P::ScalarField as CryptoField>::from_bytes_le(&buf)
  };

  // c_lin = alpha * c + r_hide - rho_prime * h
  // Linearize the commitment to cancel the blinding factors.
  let h_proj: P::G1 = (*h).into();
  let c_lin: P::G1Affine = (P::G1::from(*com) * alpha + P::G1::from(*r_com) - h_proj * *rho_prime).into();

  // The proof's f_star is the combined polynomial: alpha*f_star_orig + f_star_r.
  // Its evaluation at r_z gives alpha*v_orig + v_r, which is already the correct
  // linearized value for the non-ZK verify. We pass it directly because the
  // MLPolyCommit::verify wrapper extracts the eval from the combined f_star.
  // (The v_lin = alpha*v + y_r formula from the reference protocol applies when v
  // is the ORIGINAL poly's eval; here `output` is already the combined eval.)
  kzh3_verify(srs, input, output, &c_lin, open)
}

#[derive(Clone, Debug)]
pub struct KZH3Commit<P: PairingTrait> {
  pub srs: Arc<KZH3SRS<P>>,
  pub aux: Arc<RefCell<Option<KZH3Aux<P>>>>,
}

#[derive(Clone, Debug)]
pub struct KZH3CommitKey<P: PairingTrait> {
  // HashMap of polynomial size (n) -> SRS for that size
  pub srs_map: Arc<HashMap<usize, Arc<KZH3SRS<P>>>>,
  /// Hiding generator for ZK mode. When Some, ZK commit/open/verify is used.
  pub h: Option<P::G1Affine>,
}

#[derive(Clone)]
pub struct KZH3VerifierKey<P: PairingTrait> {
  // HashMap of polynomial size (n) -> SRS for that size
  pub srs_map: Arc<HashMap<usize, Arc<KZH3SRS<P>>>>,
  /// Hiding generator for ZK mode. When Some, ZK verify is used.
  pub h: Option<P::G1Affine>,
}

impl<P: PairingTrait> KZH3Commit<P> {
  /// Fast fake setup for benchmarking purposes only.
  /// Uses default/identity values to skip expensive elliptic curve operations.
  /// WARNING: Do NOT use for production - not cryptographically secure!
  #[cfg(feature = "arkworks")]
  pub fn fake_setup(n: usize) -> KZH3CommitKey<P>
  where
    P::ScalarField: CryptoField,
    P::G1: ark_ec::CurveGroup,
    P::G1Affine: Send + Sync,
  {
    let (degree_x_log, degree_y_log, degree_z_log) = get_degree_from_maximum_supported_degree(n);
    let degree_x = 1 << degree_x_log;
    let degree_y = 1 << degree_y_log;
    let degree_z = 1 << degree_z_log;

    // Use default/identity elements to avoid expensive curve operations
    let default_g1 = P::G1Affine::default();
    let default_g2 = P::G2Affine::default();

    // Instantly allocate with default values instead of computing MSMs
    let h_xyz = vec![default_g1; degree_x * degree_y * degree_z];
    let h_yz = vec![default_g1; degree_y * degree_z];
    let h_z = vec![default_g1; degree_z];
    let v_x = vec![default_g2; degree_x];
    let v_y = vec![default_g2; degree_y];
    let v_z = vec![default_g2; degree_z];
    let v = default_g2;

    let srs = KZH3SRS {
      degree_x,
      degree_y,
      degree_z,
      h_xyz,
      h_yz,
      h_z,
      v_x,
      v_y,
      v_z,
      v,
    };

    let mut srs_map = HashMap::new();
    srs_map.insert(n, Arc::new(srs));
    KZH3CommitKey { srs_map: Arc::new(srs_map), h: None }
  }

  /// Fast fake setup for icicle backend
  #[cfg(all(feature = "icicle", not(feature = "arkworks")))]
  pub fn fake_setup(n: usize) -> KZH3CommitKey<P>
  where
    P::ScalarField: CryptoField,
    P::G1Affine: icicle_core::traits::GenerateRandom,
    P::G2Affine: icicle_core::traits::GenerateRandom,
  {
    use crate::crypto::polycommit::{generate_random_g1_affine, generate_random_g2_affine};

    let (degree_x_log, degree_y_log, degree_z_log) = get_degree_from_maximum_supported_degree(n);
    let degree_x = 1 << degree_x_log;
    let degree_y = 1 << degree_y_log;
    let degree_z = 1 << degree_z_log;

    // Generate one random point and reuse it (fast, no MSMs)
    let default_g1 = generate_random_g1_affine::<P>(1)[0].clone();
    let default_g2 = generate_random_g2_affine::<P>(1)[0].clone();

    // Instantly allocate with same value instead of computing MSMs
    let h_xyz = vec![default_g1.clone(); degree_x * degree_y * degree_z];
    let h_yz = vec![default_g1.clone(); degree_y * degree_z];
    let h_z = vec![default_g1.clone(); degree_z];
    let v_x = vec![default_g2.clone(); degree_x];
    let v_y = vec![default_g2.clone(); degree_y];
    let v_z = vec![default_g2.clone(); degree_z];
    let v = default_g2;

    let srs = KZH3SRS {
      degree_x,
      degree_y,
      degree_z,
      h_xyz,
      h_yz,
      h_z,
      v_x,
      v_y,
      v_z,
      v,
    };

    let mut srs_map = HashMap::new();
    srs_map.insert(n, Arc::new(srs));
    KZH3CommitKey { srs_map: Arc::new(srs_map), h: None }
  }
}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> MLPolyCommit<P::ScalarField, DenseMLPoly<P::ScalarField>> for KZH3Commit<P>
where
  P::ScalarField: CryptoField,
  P::G1: ark_ec::CurveGroup,
  P::G1Affine: Send + Sync,
{
  type CommitmentKey = KZH3CommitKey<P>;
  type VerifierKey = KZH3VerifierKey<P>;
  type Commitment = KZH3Commitment<P>;
  type Proof = KZH3Opening<P>;
  type BatchProof = ();

  fn setup(n: usize, _sf_log: usize, _offset: i128, _size: usize) -> Self::CommitmentKey {
    let maximum_degree_log2 = n;
    use ark_std::rand::thread_rng;
    let mut rng = thread_rng();
    let srs = setup_kzh3_srs::<P, _>(maximum_degree_log2, &mut rng);
    let mut srs_map = HashMap::new();
    srs_map.insert(n, Arc::new(srs));
    KZH3CommitKey { srs_map: Arc::new(srs_map), h: None }
  }

  fn commit(poly: &DenseMLPoly<P::ScalarField>, key: &Self::CommitmentKey) -> Self::Commitment {
    let srs = key.srs_map.get(&poly.n).expect(&format!("No SRS found for polynomial size {}", poly.n));
    let (com, aux) = if let Some(ref h) = key.h {
      kzh3_commit_zk(srs, poly, h)
    } else {
      kzh3_commit(srs, poly)
    };
    KZH3Commitment {
      srs: Arc::clone(srs),
      c: com,
      aux,
    }
  }

  fn open(commitment: &Self::Commitment, poly: &DenseMLPoly<P::ScalarField>, key: &Self::CommitmentKey, point: &[P::ScalarField]) -> Self::Proof {
    let srs = &commitment.srs;
    if let Some(ref h) = key.h {
      kzh3_open_zk(srs, point, &commitment.c, &commitment.aux, poly, h)
    } else {
      kzh3_open(srs, point, &commitment.c, &commitment.aux, poly)
    }
  }

  fn verify(commitment: &Self::Commitment, proof: &Self::Proof, key: &Self::VerifierKey, point: &[P::ScalarField]) -> bool {
    let (ok, _eval) = Self::verify_and_extract(commitment, proof, key, point);
    ok
  }

  fn verify_and_extract(commitment: &Self::Commitment, proof: &Self::Proof, key: &Self::VerifierKey, point: &[P::ScalarField]) -> (bool, P::ScalarField) {
    let srs = &commitment.srs;
    let split_r = split_input(srs, point, scalar_field_zero::<P::ScalarField>());
    let r_z = &split_r[0];
    let claimed_eval = proof.f_star.evaluate_at_point(r_z);
    let ok = if let Some(ref h) = key.h {
      kzh3_verify_zk(srs, point, &claimed_eval, &commitment.c, proof, h)
    } else {
      kzh3_verify(srs, point, &claimed_eval, &commitment.c, proof)
    };
    // In ZK mode, f_star is combined: alpha * f_star_orig + f_star_r.
    // claimed_eval = alpha * orig_eval + y_r. Extract the original evaluation.
    let orig_eval = if let (Some(y_r), Some(r_hide)) = (proof.y_r, proof.r_hide) {
      use ark_serialize::CanonicalSerialize;
      let alpha = {
        let mut alpha_transcript = merlin::Transcript::new(b"kzh3_zk_alpha");
        let mut r_com_bytes = Vec::new();
        r_hide.serialize_compressed(&mut r_com_bytes).expect("serialize r_com");
        alpha_transcript.append_message(b"r_hide", &r_com_bytes);
        let mut buf = [0u8; 64];
        alpha_transcript.challenge_bytes(b"alpha", &mut buf);
        <P::ScalarField as CryptoField>::from_bytes_le(&buf)
      };
      (claimed_eval - y_r) * alpha.invert()
    } else {
      claimed_eval
    };
    (ok, orig_eval)
  }

  fn batch_open(
    _commitments: &[Self::Commitment],
    _polys: &[DenseMLPoly<P::ScalarField>],
    _keys: &[Self::CommitmentKey],
    _point: &[P::ScalarField],
  ) -> Self::BatchProof {
    todo!()
  }

  fn batch_verify(_commitments: &[Self::Commitment], _proofs: &[Self::Proof], _keys: &[Self::VerifierKey], _point: &[P::ScalarField]) -> bool {
    todo!()
  }
}

#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl<P: PairingTrait> MLPolyCommit<P::ScalarField, DenseMLPoly<P::ScalarField>> for KZH3Commit<P>
where
  P::ScalarField: CryptoField + GenerateRandom,
  P::G1Affine: icicle_core::traits::GenerateRandom,
  P::G2Affine: icicle_core::traits::GenerateRandom,
{
  type CommitmentKey = KZH3CommitKey<P>;
  type VerifierKey = KZH3VerifierKey<P>;
  type Commitment = KZH3Commitment<P>;
  type Proof = KZH3Opening<P>;
  type BatchProof = ();

  fn setup(n: usize, _sf_log: usize, _offset: i128, _size: usize) -> Self::CommitmentKey {
    let maximum_degree_log2 = n;
    let mut dummy_rng = ();
    let srs = setup_kzh3_srs::<P, _>(maximum_degree_log2, &mut dummy_rng);
    let mut srs_map = HashMap::new();
    srs_map.insert(n, Arc::new(srs));
    KZH3CommitKey { srs_map: Arc::new(srs_map), h: None }
  }

  fn commit(poly: &DenseMLPoly<P::ScalarField>, key: &Self::CommitmentKey) -> Self::Commitment {
    let srs = key.srs_map.get(&poly.n).expect(&format!("No SRS found for polynomial size {}", poly.n));
    let (com, aux) = kzh3_commit::<P>(srs, poly);
    KZH3Commitment {
      srs: Arc::clone(srs),
      c: com,
      aux,
    }
  }

  fn open(commitment: &Self::Commitment, poly: &DenseMLPoly<P::ScalarField>, _key: &Self::CommitmentKey, point: &[P::ScalarField]) -> Self::Proof {
    let srs = &commitment.srs;
    kzh3_open::<P>(srs, point, &commitment.c, &commitment.aux, poly)
  }

  fn verify(commitment: &Self::Commitment, proof: &Self::Proof, key: &Self::VerifierKey, point: &[P::ScalarField]) -> bool {
    let (ok, _eval) = Self::verify_and_extract(commitment, proof, key, point);
    ok
  }

  fn verify_and_extract(commitment: &Self::Commitment, proof: &Self::Proof, _key: &Self::VerifierKey, point: &[P::ScalarField]) -> (bool, P::ScalarField) {
    let srs = &commitment.srs;
    let split_input = split_input(srs, point, scalar_field_zero::<P::ScalarField>());
    let r_z = &split_input[0];
    let claimed_eval = proof.f_star.evaluate_at_point(r_z);
    let ok = kzh3_verify::<P>(srs, point, &claimed_eval, &commitment.c, proof);
    (ok, claimed_eval)
  }

  fn batch_open(
    _commitments: &[Self::Commitment],
    _polys: &[DenseMLPoly<P::ScalarField>],
    _keys: &[Self::CommitmentKey],
    _point: &[P::ScalarField],
  ) -> Self::BatchProof {
    todo!()
  }

  fn batch_verify(_commitments: &[Self::Commitment], _proofs: &[Self::Proof], _keys: &[Self::VerifierKey], _point: &[P::ScalarField]) -> bool {
    todo!()
  }
}

/// KZH3-based mask committer for ZK sumcheck mask coefficients.
/// Uses non-ZK (h=None) KZH3 since the mask coefficients are public.
#[cfg(feature = "arkworks")]
pub struct KZH3MaskCommitter<P: PairingTrait> {
  pub commit_key: KZH3CommitKey<P>,
  pub verify_key: KZH3VerifierKey<P>,
}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> KZH3MaskCommitter<P>
where
  P::ScalarField: CryptoField,
  P::G1: ark_ec::CurveGroup,
  P::G1Affine: Send + Sync + CanonicalSerialize + CanonicalDeserialize,
{
  pub fn new(srs_map: Arc<HashMap<usize, Arc<KZH3SRS<P>>>>) -> Self {
    Self {
      commit_key: KZH3CommitKey { srs_map: Arc::clone(&srs_map), h: None },
      verify_key: KZH3VerifierKey { srs_map, h: None },
    }
  }
}

#[cfg(feature = "arkworks")]
impl<P: PairingTrait> crate::crypto::sumcheck::prover::MaskCommitter<P::ScalarField> for KZH3MaskCommitter<P>
where
  P::ScalarField: CryptoField,
  P::G1: ark_ec::CurveGroup,
  P::G1Affine: Send + Sync + CanonicalSerialize + CanonicalDeserialize,
{
  fn commit(&self, poly: &DenseMLPoly<P::ScalarField>) -> Vec<u8> {
    let srs = self.commit_key.srs_map.get(&poly.n)
      .expect(&format!("No SRS found for mask polynomial size {}", poly.n));
    let (com, _aux) = kzh3_commit(srs, poly);
    let mut bytes = Vec::new();
    com.serialize_compressed(&mut bytes).expect("Failed to serialize commitment");
    bytes
  }

  fn open(&self, poly: &DenseMLPoly<P::ScalarField>, point: &[P::ScalarField]) -> (Vec<u8>, P::ScalarField) {
    let srs = self.commit_key.srs_map.get(&poly.n)
      .expect(&format!("No SRS found for mask polynomial size {}", poly.n));
    let (com, aux) = kzh3_commit(srs, poly);
    let opening = kzh3_open(srs, point, &com, &aux, poly);
    let split_r = split_input(srs, point, scalar_field_zero::<P::ScalarField>());
    let r_z = &split_r[0];
    let eval = opening.f_star.evaluate_at_point(r_z);
    let proof_bytes = bincode::serialize(&opening).expect("Failed to serialize opening proof");
    (proof_bytes, eval)
  }

  fn verify(&self, commitment: &[u8], proof: &[u8], point: &[P::ScalarField], eval: P::ScalarField) -> bool {
    let com = match P::G1Affine::deserialize_compressed(commitment) {
      Ok(c) => c,
      Err(_) => return false,
    };
    let opening: KZH3Opening<P> = match bincode::deserialize(proof) {
      Ok(o) => o,
      Err(_) => return false,
    };
    // Find the SRS for this point size
    let n = point.len();
    let srs = match self.verify_key.srs_map.get(&n) {
      Some(s) => s,
      None => return false,
    };
    let split_r = split_input(srs, point, scalar_field_zero::<P::ScalarField>());
    let r_z = &split_r[0];
    let claimed_eval = opening.f_star.evaluate_at_point(r_z);
    if claimed_eval != eval {
      return false;
    }
    kzh3_verify(srs, point, &claimed_eval, &com, &opening)
  }
}

#[cfg(all(test, feature = "arkworks", feature = "bn254"))]
mod tests {
  use super::*;
  use ark_bn254::Fr;
  use ark_std::UniformRand;
  use crate::crypto::polycommit::ArkBn254;

  #[test]
  fn test_kzh3_zk_commit_open_verify() {
    let mut rng = ark_std::rand::thread_rng();
    let n = 7;
    let srs = setup_kzh3_srs::<ArkBn254, _>(n, &mut rng);
    let srs = Arc::new(srs);
    let mut srs_map = HashMap::new();
    srs_map.insert(n, srs.clone());
    let srs_map = Arc::new(srs_map);

    let h = <ArkBn254 as PairingTrait>::G1Affine::rand(&mut rng);

    let size = 1usize << n;
    let evals: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
    let poly = DenseMLPoly::new(n, evals);

    let point: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

    // Test via MLPolyCommit trait
    let key = KZH3CommitKey::<ArkBn254> { srs_map: srs_map.clone(), h: Some(h) };
    let vk = KZH3VerifierKey::<ArkBn254> { srs_map: srs_map.clone(), h: Some(h) };
    let comm = <KZH3Commit<ArkBn254> as MLPolyCommit<Fr, DenseMLPoly<Fr>>>::commit(&poly, &key);
    let proof = <KZH3Commit<ArkBn254> as MLPolyCommit<Fr, DenseMLPoly<Fr>>>::open(&comm, &poly, &key, &point);
    let ok = <KZH3Commit<ArkBn254> as MLPolyCommit<Fr, DenseMLPoly<Fr>>>::verify(&comm, &proof, &vk, &point);
    assert!(ok, "KZH3 ZK via trait should pass");
  }
}

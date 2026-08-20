use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;

use crate::util::serialization::{ark_de_vec, ark_se_vec};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(feature = "arkworks")]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

// Arkworks imports (only imported when arkworks feature is enabled)
#[cfg(feature = "arkworks")]
use ark_crypto_primitives::sponge::Absorb;
#[cfg(feature = "arkworks")]
use ark_ff::{PrimeField, UniformRand};

// Icicle imports (only imported when icicle feature is enabled and arkworks is not)
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
use icicle_core::field::Field;

use crate::SF_FLOAT;
/// Field with cryptographic traits
///
/// Provides field arithmetic operations and cryptographic traits required for proof systems:
/// - Basic arithmetic: zero, one, from_u32, from_bytes_le, to_bytes_le, invert
/// - Fiat-Shamir transcripts (Absorb/challenges)
/// - Random sampling (UniformRand/GenerateRandom)
/// - Serialization and other proof system requirements
///
/// This trait is defined differently based on which feature is enabled:
/// - feature="arkworks": Requires PrimeField + Absorb + UniformRand
/// - feature="icicle": Requires Field + GenerateRandom
///
/// Use this for all proof-related code (sumcheck, commitments, etc.)
#[cfg(feature = "arkworks")]
pub trait CryptoField:
  PrimeField
  + Absorb
  + UniformRand
  + Clone
  + Copy
  + std::fmt::Debug
  + PartialEq
  + 'static
  + std::ops::Add<Output = Self>
  + std::ops::Sub<Output = Self>
  + std::ops::Mul<Output = Self>
{
  fn zero() -> Self;
  fn one() -> Self;
  fn from_u32(n: u32) -> Self;
  fn from_u64(n: u64) -> Self;
  fn from_bytes_le(bytes: &[u8]) -> Self;
  fn to_bytes_le(&self) -> Vec<u8>;
  fn invert(&self) -> Self;
}

// Arkworks implementation of CryptoField
#[cfg(feature = "arkworks")]
impl<F> CryptoField for F
where
  F: PrimeField + Absorb + UniformRand + std::ops::Add<Output = F> + std::ops::Sub<Output = F> + std::ops::Mul<Output = F>,
{
  fn zero() -> Self {
    F::ZERO
  }
  fn one() -> Self {
    F::ONE
  }
  fn from_u32(n: u32) -> Self {
    F::from(n as u64)
  }
  fn from_u64(n: u64) -> Self {
    F::from(n)
  }
  fn from_bytes_le(bytes: &[u8]) -> Self {
    // Pad to field size if needed
    let mut padded = vec![0u8; (F::MODULUS_BIT_SIZE as usize + 7) / 8];
    let copy_len = std::cmp::min(bytes.len(), padded.len());
    padded[..copy_len].copy_from_slice(&bytes[..copy_len]);
    F::from_le_bytes_mod_order(&padded)
  }
  fn to_bytes_le(&self) -> Vec<u8> {
    let mut bytes = vec![0u8; (F::MODULUS_BIT_SIZE as usize + 7) / 8];
    self.serialize_uncompressed(&mut bytes[..]).unwrap();
    bytes
  }
  fn invert(&self) -> Self {
    self.inverse().unwrap_or(F::ZERO)
  }
}

// Icicle path: CryptoField extends Field
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
pub trait CryptoField: Field + Clone + Copy + std::fmt::Debug + PartialEq + 'static {
  fn zero() -> Self;
  fn one() -> Self;
  fn from_u32(n: u32) -> Self;
  fn from_u64(n: u64) -> Self;
  fn from_bytes_le(bytes: &[u8]) -> Self;
  fn to_bytes_le(&self) -> Vec<u8>;
  fn invert(&self) -> Self;
}

// Macro to implement CryptoField for icicle field types
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
macro_rules! impl_crypto_field_for_icicle {
  ($field_type:ty, $byte_size:expr) => {
    impl CryptoField for $field_type {
      fn zero() -> Self {
        // Use Default::default() which creates a zero value for icicle fields
        std::default::Default::default()
      }
      fn one() -> Self {
        use icicle_core::bignum::BigNum;
        <$field_type as BigNum>::from_bytes_le(&[1u8])
      }
      fn from_u32(n: u32) -> Self {
        <Self as CryptoField>::from_u64(n as u64)
      }
      fn from_u64(n: u64) -> Self {
        if n == 0 {
          return <Self as CryptoField>::zero();
        }
        if n == 1 {
          return <Self as CryptoField>::one();
        }
        use icicle_core::bignum::BigNum;
        let bytes = n.to_le_bytes();
        <$field_type as BigNum>::from_bytes_le(&bytes)
      }
      fn from_bytes_le(bytes: &[u8]) -> Self {
        // Pad to field size for consistency with arkworks behavior
        use icicle_core::bignum::BigNum;
        const FIELD_BYTE_SIZE: usize = $byte_size;
        let mut padded = vec![0u8; FIELD_BYTE_SIZE];
        let copy_len = std::cmp::min(bytes.len(), FIELD_BYTE_SIZE);
        padded[..copy_len].copy_from_slice(&bytes[..copy_len]);
        <$field_type as BigNum>::from_bytes_le(&padded)
      }
      fn to_bytes_le(&self) -> Vec<u8> {
        use icicle_core::bignum::BigNum;
        let bytes = <$field_type as BigNum>::to_bytes_le(self);
        // Pad to fixed size to match arkworks behavior
        // This ensures transcript consistency between prover and verifier
        const FIELD_BYTE_SIZE: usize = $byte_size;
        let mut padded = vec![0u8; FIELD_BYTE_SIZE];
        let copy_len = std::cmp::min(bytes.len(), FIELD_BYTE_SIZE);
        padded[..copy_len].copy_from_slice(&bytes[..copy_len]);
        padded
      }
      fn invert(&self) -> Self {
        use icicle_core::traits::Invertible;
        if *self == <Self as CryptoField>::zero() {
          <Self as CryptoField>::zero()
        } else {
          self.inv()
        }
      }
    }
  };
}

// Implement CryptoField for icicle field types
// Map each field type to its byte size for consistent serialization
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl_crypto_field_for_icicle!(icicle_bls12_381::curve::ScalarField, 32);
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl_crypto_field_for_icicle!(icicle_bls12_381::curve::BaseField, 48);
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl_crypto_field_for_icicle!(icicle_bls12_381::curve::G2BaseField, 96);

// BN254 field types for icicle
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl_crypto_field_for_icicle!(icicle_bn254::curve::ScalarField, 32);
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl_crypto_field_for_icicle!(icicle_bn254::curve::BaseField, 32);
#[cfg(all(feature = "icicle", not(feature = "arkworks")))]
impl_crypto_field_for_icicle!(icicle_bn254::curve::G2BaseField, 64);

/// Pads a vector at the start with zeros to reach the target length
pub fn pad_at_start<F: CryptoField>(input: &[F], len: usize) -> Vec<F> {
  if input.len() >= len {
    return input.to_vec();
  }
  let mut result = vec![<F as CryptoField>::zero(); len - input.len()];
  result.extend_from_slice(input);
  result
}

/// Convert usize to little-endian Vec<u8> representation
pub fn usize_to_le_vec(value: usize) -> Vec<u8> {
  let bytes = value.to_le_bytes();
  // Remove trailing zeros to keep it minimal
  let mut end = bytes.len();
  while end > 1 && bytes[end - 1] == 0 {
    end -= 1;
  }
  bytes[..end].to_vec()
}

pub fn le_vec_to_usize(value: &[u8]) -> usize {
  let mut result = 0;
  for (i, &byte) in value.iter().enumerate() {
    result |= (byte as usize) << (i * 8);
  }
  result
}

/// Extract low 'bits' bits from a little-endian Vec<u8> number
/// Equivalent to: number & ((1 << bits) - 1)
pub fn extract_low_bits(number: &[u8], bits: usize) -> Vec<u8> {
  if bits == 0 {
    return vec![0];
  }

  let full_bytes = bits / 8;
  let remaining_bits = bits % 8;
  let mut result = vec![];

  // Extract full bytes
  for i in 0..full_bytes.min(number.len()).min(std::mem::size_of::<usize>()) {
    result.push(number[i]);
  }

  // Extract remaining bits from the next byte
  if remaining_bits > 0 && full_bytes < number.len() && full_bytes < std::mem::size_of::<usize>() {
    let mask = (1u8 << remaining_bits) - 1;
    let masked_byte = number[full_bytes] & mask;
    result.push(masked_byte);
  }

  result
}

/// Right shift a little-endian Vec<u8> number by 'bits' bits
/// Equivalent to: number >> bits
pub fn right_shift_le_vec(number: &[u8], bits: usize) -> Vec<u8> {
  if bits == 0 {
    return number.to_vec();
  }

  let byte_shift = bits / 8;
  let bit_shift = bits % 8;

  if byte_shift >= number.len() {
    return vec![0];
  }

  let mut result = Vec::new();

  if bit_shift == 0 {
    // Simple byte shift
    result.extend_from_slice(&number[byte_shift..]);
  } else {
    // Need to shift bits within bytes
    let mut carry = 0u8;
    for i in (byte_shift..number.len()).rev() {
      let current = number[i];
      let new_byte = (current >> bit_shift) | (carry << (8 - bit_shift));
      carry = current & ((1 << bit_shift) - 1);
      result.push(new_byte);
    }
    result.reverse();
  }

  // Remove leading zeros
  while result.len() > 1 && result.last() == Some(&0) {
    result.pop();
  }

  if result.is_empty() {
    vec![0]
  } else {
    result
  }
}

/// Fix multiple variables at arbitrary positions in a dense multilinear polynomial.
/// This allows fixing any subset of variables, not just from left or right.
pub fn fix_many<F: CryptoField>(
  dense_poly: &DenseMLPoly<F>,
  fixed: &[(usize, F)], // (var_index, value), where var_index in 0..n
) -> DenseMLPoly<F> {
  let n = dense_poly.n;
  assert!(fixed.len() <= n, "too many fixed variables");

  // Map each variable index -> Option<value>
  let mut value_by_var: Vec<Option<F>> = vec![None; n];
  for &(var, val) in fixed {
    assert!(var < n, "variable index {} out of range 0..{}", var, n - 1);
    assert!(value_by_var[var].is_none(), "duplicate assignment for variable {}", var);
    value_by_var[var] = Some(val);
  }

  // Split into unfixed and fixed variables (keeps original order).
  let mut unfixed_vars = Vec::new();
  let mut fixed_vars = Vec::new();
  for v in 0..n {
    if value_by_var[v].is_some() {
      fixed_vars.push(v);
    } else {
      unfixed_vars.push(v);
    }
  }

  let m = fixed_vars.len();
  if m == 0 {
    // nothing to do; you can also return a clone or just copy the slice
    return dense_poly.clone();
  }

  let total = 1usize << n;
  let src = &dense_poly.evaluations;

  // 1) Build permutation: old var index -> new var index.
  // New order: [unfixed_vars..., fixed_vars...].
  let mut pos_new = vec![0usize; n];
  for (i, &v) in unfixed_vars.iter().enumerate() {
    pos_new[v] = i;
  }
  for (j, &v) in fixed_vars.iter().enumerate() {
    pos_new[v] = unfixed_vars.len() + j;
  }

  // 2) Permute evaluation table according to this variable reordering.
  // Index bits: old_index bit v -> new_index bit pos_new[v].
  let mut evals = vec![src[0]; total];
  for idx_old in 0..total {
    let mut idx_new = 0usize;
    let mut mask = 1usize;
    for v in 0..n {
      if idx_old & mask != 0 {
        idx_new |= 1usize << pos_new[v];
      }
      mask <<= 1;
    }
    evals[idx_new] = src[idx_old];
  }

  // 3) Prepare suffix values for the now-rightmost m variables.
  let mut partial_suffix = Vec::with_capacity(m);
  for &v in &fixed_vars {
    partial_suffix.push(value_by_var[v].unwrap());
  }

  // 4) Collapse the rightmost m variables (same logic as old `fix_from_right`).
  let mut poly = evals;
  for i in 0..m {
    let r = partial_suffix[m - 1 - i]; // collapse last var first
    let half = 1usize << (n - 1 - i); // left half length
    for j in 0..half {
      let left = poly[j]; // bit = 0
      let right = poly[j + half]; // bit = 1
      poly[j] = left + r * (right - left);
    }
    // After this, valid region is poly[0 .. half)
  }

  DenseMLPoly::from_evaluations_slice(n - m, &poly[..(1 << (n - m))])
}

pub struct FixedBlock<'a, F> {
  pub start: usize,    // 0-based index of first variable in this block (0 -> x1)
  pub points: &'a [F], // values for x_{start+1}, ..., x_{start+values.len()}
}

pub fn fix_blocks<F: CryptoField>(dense_poly: &DenseMLPoly<F>, blocks: &[FixedBlock<'_, F>]) -> DenseMLPoly<F> {
  // Flatten blocks into (var_index, value) assignments.
  let mut assignments = Vec::new();
  for block in blocks {
    let start = block.start;
    for (offset, &val) in block.points.iter().enumerate() {
      assignments.push((start + offset, val));
    }
  }
  fix_many(dense_poly, &assignments)
}

/// Trait for multilinear polynomials that can be used in polynomial commitments
/// Now works with any field type that implements basic arithmetic operations
pub trait MLPoly<F>: std::fmt::Debug + std::any::Any + Send + Sync
where
  F: Clone + Copy + std::fmt::Debug + 'static,
{
  fn fix_variables(&self, partial_point: &[F]) -> Box<dyn MLPoly<F>>;
  fn n(&self) -> usize; // number of variables
  fn len(&self) -> usize; // number of evaluations (2^n)
  fn evaluate_at_point(&self, point: &[F]) -> F;
  fn evaluations(&self) -> Vec<F>;
  fn index(&self, index: usize) -> F;
  fn index_mut(&mut self, index: usize) -> &mut F;
  fn clone_box(&self) -> Box<dyn MLPoly<F>>;
  fn as_any(&self) -> &dyn std::any::Any;
  fn mul_by_scalar(&self, scalar: F) -> Box<dyn MLPoly<F>>;
  fn add(&self, other: &Box<dyn MLPoly<F>>) -> Box<dyn MLPoly<F>>;
}

use std::any::{Any, TypeId};
use std::sync::{OnceLock, RwLock};

/// Cached inverse constants for Lagrange interpolation
#[derive(Clone, Copy)]
pub struct CachedInverses<F: Copy> {
  pub two_inv: F,
  pub six_inv: F,
  pub neg_ln2_inv: F,
  pub neg_two_inv: F,
  pub neg_six_inv: F,
}

/// Global cache for inverse constants, keyed by field type
pub static LAGRANGE_INVERSE_CACHE: OnceLock<RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>> = OnceLock::new();

/// Get cached inverse constants for a field type, computing them if not already cached
pub fn get_cached_inverses<F: CryptoField + Send + Sync + 'static>() -> CachedInverses<F> {
  let cache = LAGRANGE_INVERSE_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
  let type_id = TypeId::of::<F>();

  // Try read first (fast path)
  {
    let read_guard = cache.read().unwrap();
    if let Some(cached) = read_guard.get(&type_id) {
      return *cached.downcast_ref::<CachedInverses<F>>().unwrap();
    }
  }

  // Compute inverses (only happens once per field type)
  let two_inv = <F as CryptoField>::from_u32(2).invert();
  let six_inv = <F as CryptoField>::from_u32(6).invert();
  let ln2 = -((2.0_f32.ln() * *SF_FLOAT).round() as i128);
  let neg_ln2_inv = F::from(ln2).invert();
  let inverses = CachedInverses {
    two_inv,
    six_inv,
    neg_ln2_inv,
    neg_two_inv: <F as CryptoField>::zero() - two_inv,
    neg_six_inv: <F as CryptoField>::zero() - six_inv,
  };

  // Insert into cache
  {
    let mut write_guard = cache.write().unwrap();
    write_guard.entry(type_id).or_insert_with(|| Box::new(inverses));
  }

  inverses
}

/// Evaluates a univariate polynomial in evaluation form at a given challenge point
/// Implementation is based on the Lagrange interpolation formula
/// Uses a global cache for inverse constants to avoid repeated field inversions
pub fn evaluate_univariate_polynomial<F: CryptoField + Send + Sync + 'static>(points: &Vec<F>, challenge: F) -> F {
  let n = points.len();

  // Special cases for small n (common in sumcheck: degree 2, 3, or 4)
  match n {
    0 => return <F as CryptoField>::zero(),
    1 => return points[0],
    2 => {
      // L_0(x) = (1 - x), L_1(x) = x
      // f(x) = points[0] + x * (points[1] - points[0])
      return points[0] + challenge * (points[1] - points[0]);
    }
    3 => {
      // L_0(x) = (x-1)(x-2)/2, L_1(x) = -x(x-2), L_2(x) = x(x-1)/2
      let cached = get_cached_inverses::<F>();
      let x = challenge;
      let x_minus_1 = x - <F as CryptoField>::one();
      let x_minus_2 = x - <F as CryptoField>::from_u32(2);

      let l0 = x_minus_1 * x_minus_2 * cached.two_inv;
      let l1 = <F as CryptoField>::zero() - x * x_minus_2;
      let l2 = x * x_minus_1 * cached.two_inv;

      return points[0] * l0 + points[1] * l1 + points[2] * l2;
    }
    4 => {
      // L_0(x) = (x-1)(x-2)(x-3)/(-6), L_1(x) = x(x-2)(x-3)/2
      // L_2(x) = x(x-1)(x-3)/(-2), L_3(x) = x(x-1)(x-2)/6
      let cached = get_cached_inverses::<F>();
      let x = challenge;
      let x_minus_1 = x - <F as CryptoField>::one();
      let x_minus_2 = x - <F as CryptoField>::from_u32(2);
      let x_minus_3 = x - <F as CryptoField>::from_u32(3);

      let l0 = x_minus_1 * x_minus_2 * x_minus_3 * cached.neg_six_inv;
      let l1 = x * x_minus_2 * x_minus_3 * cached.two_inv;
      let l2 = x * x_minus_1 * x_minus_3 * cached.neg_two_inv;
      let l3 = x * x_minus_1 * x_minus_2 * cached.six_inv;

      return points[0] * l0 + points[1] * l1 + points[2] * l2 + points[3] * l3;
    }
    _ => {}
  }

  // General case: use batch inversion (Montgomery's trick) to minimize field inversions
  // Total: only 2 field inversions instead of O(n²)

  // Step 1: Compute (challenge - i) for all i
  let diffs: Vec<F> = (0..n).map(|i| challenge - <F as CryptoField>::from_u32(i as u32)).collect();

  // Step 2: Batch inversion for diffs using Montgomery's trick
  let mut prefix_products = vec![<F as CryptoField>::one(); n];
  for i in 1..n {
    prefix_products[i] = prefix_products[i - 1] * diffs[i - 1];
  }

  let mut suffix_products = vec![<F as CryptoField>::one(); n];
  for i in (0..n - 1).rev() {
    suffix_products[i] = suffix_products[i + 1] * diffs[i + 1];
  }

  // Product of all (challenge - j)
  let all_product = prefix_products[n - 1] * diffs[n - 1];
  let all_product_inv = all_product.invert();

  // Compute inverses: inv[i] = prefix[i] * suffix[i] * all_product_inv
  let inv_diffs: Vec<F> = (0..n).map(|i| prefix_products[i] * suffix_products[i] * all_product_inv).collect();

  // Step 3: Precompute barycentric weights w_i = 1 / Π_{j≠i} (i - j)
  // For evaluation points 0, 1, ..., n-1:
  // w_i = (-1)^(n-1-i) / (i! * (n-1-i)!)
  let mut factorials = vec![<F as CryptoField>::one(); n];
  for i in 1..n {
    factorials[i] = factorials[i - 1] * <F as CryptoField>::from_u32(i as u32);
  }

  // Compute denominators for batch inversion
  let denoms: Vec<F> = (0..n).map(|i| factorials[i] * factorials[n - 1 - i]).collect();

  // Batch inversion for denominators
  let mut denom_prefix = vec![<F as CryptoField>::one(); n];
  for i in 1..n {
    denom_prefix[i] = denom_prefix[i - 1] * denoms[i - 1];
  }

  let mut denom_suffix = vec![<F as CryptoField>::one(); n];
  for i in (0..n - 1).rev() {
    denom_suffix[i] = denom_suffix[i + 1] * denoms[i + 1];
  }

  let all_denoms = denom_prefix[n - 1] * denoms[n - 1];
  let all_denoms_inv = all_denoms.invert();

  // Compute barycentric weights with proper signs
  let bary_weights: Vec<F> = (0..n)
    .map(|i| {
      let denom_inv = denom_prefix[i] * denom_suffix[i] * all_denoms_inv;
      let sign = if (n - 1 - i) % 2 == 0 {
        <F as CryptoField>::one()
      } else {
        <F as CryptoField>::zero() - <F as CryptoField>::one()
      };
      sign * denom_inv
    })
    .collect();

  // Step 4: Compute result using barycentric formula
  // f(x) = [Π_{j=0}^{n-1} (x - j)] * Σ_{i=0}^{n-1} [points[i] * w_i / (x - i)]
  let sum: F = (0..n).map(|i| points[i] * bary_weights[i] * inv_diffs[i]).fold(<F as CryptoField>::zero(), |acc, x| acc + x);

  all_product * sum
}

/// Efficient evaluation of multilinear Lagrange basis polynomials
///
/// Given a point r ∈ F^(log m), evaluates all Lagrange basis polynomials at r using only m field multiplications.
/// [VSBW13] Victor Vu, Srinath Setty, Andrew J. Blumberg, and Michael Walfish. A hybrid architecture for
/// verifiable computation. In Proceedings of the IEEE Symposium on Security and Privacy (S&P), 2013.
///
/// Also known as eq_polynomial_evals in some contexts - evaluates eq(i, r) for all i in the boolean hypercube.
/// Optimized to pre-allocate the final array size and perform in-place updates with parallelization.
pub fn evaluate_lagrange_basis<F: CryptoField>(r: &[F]) -> Vec<F> {
  let log_m = r.len();
  if log_m == 0 {
    return vec![<F as CryptoField>::one()];
  }

  let final_size = 1usize << log_m;
  // Pre-allocate the final array size upfront
  let mut a = vec![<F as CryptoField>::zero(); final_size];
  a[0] = <F as CryptoField>::one();

  for i in 0..log_m {
    let r_i = r[i];
    let current_size = 1usize << i;
    let offset = current_size;

    if current_size >= 1024 {
      // Parallel in-place update in two phases to avoid data races:
      // Phase 1: Write "high" values (read from [0..current_size], write to [current_size..2*current_size])
      // Phase 2: Update "low" values (read from both halves, write to [0..current_size])

      // Split the array into two non-overlapping halves
      let (left, right) = a[..current_size * 2].split_at_mut(current_size);

      // Phase 1: Compute high values in parallel (right[x] = r_i * left[x])
      left.par_iter().zip(right.par_iter_mut()).for_each(|(&left_val, right_val)| {
        *right_val = r_i * left_val;
      });

      // Phase 2: Update low values in parallel (left[x] = left[x] - right[x])
      left.par_iter_mut().zip(right.par_iter()).for_each(|(left_val, &right_val)| {
        *left_val = *left_val - right_val;
      });
    } else {
      // Sequential in-place update for smaller arrays (process in reverse order)
      for x in (0..current_size).rev() {
        let a_x = a[x];
        let mul_result = r_i * a_x;
        a[x + offset] = mul_result;
        a[x] = a_x - mul_result;
      }
    }
  }

  a
}

/// Computes the multilinear extension ũ(r) for vector u at point r
/// Uses exactly 2m field multiplications as per [VSBW13]
pub fn evaluate_multilinear_extension<F: CryptoField>(u: &[F], r: &[F]) -> F {
  let m = u.len();
  let log_m = r.len();

  // Verify that m = 2^(log_m)
  assert_eq!(m, 1 << log_m, "Vector length must be a power of 2 matching r dimension");

  // First m multiplications: evaluate Lagrange basis polynomials
  let lagrange_basis = evaluate_lagrange_basis(r); // uses m-1 multiplications

  // Additional m multiplications: compute ũ(r) = Σ u[i] * ẽq(i, r)
  // Parallelize the final sum when vector is large enough
  let result = if m >= 1024 {
    // Use parallel reduction for large vectors (1024+ elements)
    u.par_iter()
      .zip(lagrange_basis.par_iter())
      .map(|(&u_i, &basis_i)| u_i * basis_i)
      .reduce(|| <F as CryptoField>::zero(), |a, b| a + b)
  } else {
    // Use sequential iteration for smaller vectors
    let mut result = <F as CryptoField>::zero();
    for i in 0..m {
      result = result + u[i] * lagrange_basis[i]; // 1 multiplication per term, m total
    }
    result
  };
  // Total: (m-1) + m = 2m-1 multiplications
  // We need to account for 1 more multiplication somewhere to match Lemma 1's "2m" claim

  result
}

/// Precomputes the eq polynomial evaluations for a given partial point
/// Used in sparse polynomial fix_variables operation
/// Optimized to avoid intermediate allocations by using in-place updates with parallelization.
pub fn precompute_eq<F: CryptoField>(g: &[F]) -> Vec<F> {
  let dim = g.len();
  if dim == 0 {
    return vec![<F as CryptoField>::one()];
  }

  let mut dp = vec![<F as CryptoField>::zero(); 1 << dim];
  dp[0] = <F as CryptoField>::one() - g[0];
  dp[1] = g[0];

  for i in 1..dim {
    let num_elements = 1 << i;
    let g_i = g[i];

    if num_elements >= 1024 {
      // Parallel in-place update in two phases:
      // Phase 1: Write "high" values (read from [0..num_elements], write to [num_elements..2*num_elements])
      // Phase 2: Update "low" values

      // Split the array into two non-overlapping halves
      let (left, right) = dp[..num_elements * 2].split_at_mut(num_elements);

      // Phase 1: Compute high values in parallel (right[b] = g_i * left[b])
      left.par_iter().zip(right.par_iter_mut()).for_each(|(&left_val, right_val)| {
        *right_val = left_val * g_i;
      });

      // Phase 2: Update low values in parallel (left[b] = left[b] - right[b])
      left.par_iter_mut().zip(right.par_iter()).for_each(|(left_val, &right_val)| {
        *left_val = *left_val - right_val;
      });
    } else {
      // Sequential in-place update for smaller arrays (process in reverse order)
      for b in (0..num_elements).rev() {
        let prev = dp[b];
        let new_high = prev * g_i;
        dp[b + num_elements] = new_high;
        dp[b] = prev - new_high;
      }
    }
  }
  dp
}

/// Fix variables from the right (highest bits) in a multilinear polynomial.
/// This is useful when you need to fix the last N variables in LSB-first ordering.
/// For a polynomial with variables [v0, v1, ..., v_{n-1}] in LSB-first order,
/// this fixes [v_{n-k}, ..., v_{n-1}] where k = fixed_vars.len().
pub fn fix_variables_from_right<F: CryptoField>(poly: &DenseMLPoly<F>, fixed_vars: &[F]) -> DenseMLPoly<F> {
  assert!(fixed_vars.len() <= poly.n, "invalid size of partial point");
  if fixed_vars.is_empty() {
    return poly.clone();
  }

  let dim = fixed_vars.len();
  let remaining_vars = poly.n - dim;

  // Use eq polynomial evaluation to compute the partial evaluation
  let eq_evals = precompute_eq(fixed_vars);

  let result_evals = if poly.evaluations.len() >= (1 << 15) {
    // Use parallel iteration for large polynomials (2^15 evaluations or more)
    (0..poly.evaluations.len())
      .into_par_iter()
      .fold(
        || vec![<F as CryptoField>::zero(); 1 << remaining_vars],
        |mut local_result, idx| {
          let high_bits = idx >> remaining_vars;
          let low_bits = idx & ((1 << remaining_vars) - 1);
          if high_bits < eq_evals.len() {
            let eq_val = eq_evals[high_bits];
            local_result[low_bits] = local_result[low_bits] + eq_val * poly.evaluations[idx];
          }
          local_result
        },
      )
      .reduce(
        || vec![<F as CryptoField>::zero(); 1 << remaining_vars],
        |mut a, b| {
          for i in 0..a.len() {
            a[i] = a[i] + b[i];
          }
          a
        },
      )
  } else {
    // Use sequential iteration for smaller polynomials
    let mut result_evals = vec![<F as CryptoField>::zero(); 1 << remaining_vars];
    for idx in 0..poly.evaluations.len() {
      let high_bits = idx >> remaining_vars;
      let low_bits = idx & ((1 << remaining_vars) - 1);

      if high_bits < eq_evals.len() {
        let eq_val = eq_evals[high_bits];
        result_evals[low_bits] = result_evals[low_bits] + eq_val * poly.evaluations[idx];
      }
    }
    result_evals
  };

  DenseMLPoly::new(remaining_vars, result_evals)
}

/// Dense multilinear polynomial represented as a vector of evaluations at all points in the domain
/// Now supports both icicle's Field and arkworks' PrimeField
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "F: CanonicalSerialize + CanonicalDeserialize")]
pub struct DenseMLPoly<F> {
  pub n: usize, // number of variables
  #[serde(serialize_with = "ark_se_vec", deserialize_with = "ark_de_vec")]
  pub evaluations: Vec<F>,
}

impl<F: CryptoField + 'static> MLPoly<F> for DenseMLPoly<F> {
  fn len(&self) -> usize {
    self.evaluations.len()
  }

  fn index_mut(&mut self, index: usize) -> &mut F {
    &mut self.evaluations[index]
  }

  fn fix_variables(&self, partial_point: &[F]) -> Box<dyn MLPoly<F>> {
    Box::new(self.fix_variables(partial_point))
  }

  fn n(&self) -> usize {
    self.n
  }

  fn evaluate_at_point(&self, point: &[F]) -> F {
    assert!(point.len() == self.n, "invalid size of point");
    evaluate_multilinear_extension(&self.evaluations, &point)
  }

  fn evaluations(&self) -> Vec<F> {
    self.evaluations.clone()
  }

  fn index(&self, index: usize) -> F {
    self.evaluations[index]
  }

  fn clone_box(&self) -> Box<dyn MLPoly<F>> {
    Box::new(self.clone())
  }

  fn as_any(&self) -> &dyn std::any::Any {
    self
  }

  fn mul_by_scalar(&self, scalar: F) -> Box<dyn MLPoly<F>> {
    Box::new(self.mul_by_scalar(scalar))
  }

  fn add(&self, other: &Box<dyn MLPoly<F>>) -> Box<dyn MLPoly<F>> {
    Box::new(self.add(other.as_any().downcast_ref::<DenseMLPoly<F>>().unwrap()))
  }
}

impl<F: CryptoField> DenseMLPoly<F> {
  pub fn new(n: usize, evaluations: Vec<F>) -> Self {
    Self { n, evaluations }
  }

  pub fn len(&self) -> usize {
    self.evaluations.len()
  }

  pub fn is_empty(&self) -> bool {
    self.evaluations.is_empty()
  }

  pub fn from_evaluations(evaluations: Vec<F>) -> Self {
    let n = (evaluations.len() as f64).log2() as usize;
    assert_eq!(1 << n, evaluations.len(), "Vector length must be a power of 2");
    Self::new(n, evaluations)
  }

  pub fn from_evaluations_slice(n: usize, evaluations: &[F]) -> Self {
    Self::new(n, evaluations.to_vec())
  }

  /// Fix variables from the left side of the polynomial.
  /// Optimized to avoid intermediate allocations by using in-place updates with parallelization.
  pub fn fix_variables(&self, partial_point: &[F]) -> Self {
    assert!(partial_point.len() <= self.n, "invalid size of partial point");

    let mut poly = self.evaluations.to_vec();
    let nv = self.n;
    let dim = partial_point.len();

    // Evaluate single variable of partial point from left to right.
    // Each iteration halves the polynomial size.
    // In-place update: read from [2b, 2b+1], write to [b].
    for i in 1..dim + 1 {
      let r = partial_point[i - 1];
      let num_pairs = 1 << (nv - i);

      if num_pairs >= 1024 {
        // Parallel in-place update:
        // We read from pairs (poly[2b], poly[2b+1]) and write to poly[b].
        // Since output index b < input indices 2b and 2b+1, there's no conflict
        // between different values of b when processed in parallel.
        // Use chunks to process pairs and write results.
        let slice = &mut poly[..num_pairs * 2];
        // Process pairs: each pair at indices [2b, 2b+1] produces result at index b
        // We need to collect results first, then write back to avoid conflicts within rayon
        let results: Vec<F> = slice
          .par_chunks(2)
          .map(|pair| {
            let left = pair[0];
            let right = pair[1];
            left + r * (right - left)
          })
          .collect();

        // Write results back (safe because results.len() == num_pairs <= slice.len())
        poly[..num_pairs].copy_from_slice(&results);
      } else {
        // Sequential in-place update for smaller arrays
        for b in 0..num_pairs {
          let left = poly[b << 1];
          let right = poly[(b << 1) + 1];
          poly[b] = left + r * (right - left);
        }
      }
    }

    Self::from_evaluations_slice(nv - dim, &poly[..(1 << (nv - dim))])
  }

  pub fn mul_by_scalar(&self, scalar: F) -> Self
  where
    F: std::ops::Mul<Output = F>,
  {
    // Parallelize scalar multiplication when vector is large enough
    let new_evaluations = if self.evaluations.len() >= 1024 {
      self.evaluations.par_iter().map(|&x| x * scalar).collect()
    } else {
      self.evaluations.iter().map(|&x| x * scalar).collect()
    };
    Self::new(self.n, new_evaluations)
  }

  pub fn add(&self, other: &Self) -> Self {
    assert!(self.n == other.n, "mismatched number of variables");

    // Parallelize addition when vectors are large enough
    let new_evaluations = if self.evaluations.len() >= 1024 {
      self.evaluations.par_iter().zip(other.evaluations.par_iter()).map(|(&x, &y)| x + y).collect()
    } else {
      self.evaluations.iter().zip(other.evaluations.iter()).map(|(&x, &y)| x + y).collect()
    };

    Self::new(self.n, new_evaluations)
  }

  pub fn add_by_scalar(&self, scalar: F) -> Self
  where
    F: std::ops::Add<Output = F>,
  {
    // Parallelize scalar addition when vector is large enough
    let new_evaluations = if self.evaluations.len() >= 1024 {
      self.evaluations.par_iter().map(|&x| x + scalar).collect()
    } else {
      self.evaluations.iter().map(|&x| x + scalar).collect()
    };
    Self::new(self.n, new_evaluations)
  }

  /// Extend the polynomial to have more variables by padding with zeros.
  /// This increases the number of variables from `self.n` to `num_variables`.
  /// Takes self by value to avoid cloning when no extension is needed.
  pub fn extend_number_of_variables(mut self, num_variables: usize) -> Self {
    assert!(self.n <= num_variables, "Cannot extend to fewer variables");

    // Fast path: if already at the right size, return self without any cloning
    if self.n == num_variables {
      return self;
    }

    // Extend the evaluations vector in place
    let new_len = 1 << num_variables;
    self.evaluations.extend(vec![<F as CryptoField>::zero(); new_len - self.evaluations.len()]);
    self.n = num_variables;
    self
  }

  /// Get a slice of evaluations for a specific boolean input index.
  /// This extracts `n` consecutive evaluations starting at `n * index`.
  /// Returns a slice to avoid copying data.
  pub fn get_partial_evaluation_for_boolean_input(&self, index: usize, n: usize) -> &[F] {
    let start = n * index;
    let end = start + n;
    &self.evaluations[start..end]
  }

  /// Perform partial evaluation by fixing variables from the left.
  /// This is equivalent to `fix_variables` but returns a new polynomial.
  pub fn partial_evaluation(&self, fixed_vars: &[F]) -> Self {
    self.fix_variables(fixed_vars)
  }
}

pub fn add_broadcast<F: CryptoField>(a: &DenseMLPoly<F>, b: &DenseMLPoly<F>) -> DenseMLPoly<F> {
  let n_out = a.n.max(b.n);
  let out_len = 1usize << n_out;

  // We'll map each output index i (n_out bits) to indices in a and b
  // by dropping the highest (n_out - n_small) bits, i.e. keeping the lowest n_small bits.
  //
  // This assumes variable order is:
  //   x1 is the "lowest bit" (fastest-changing),
  //   x_n is the "highest bit" (slowest-changing).
  // TODO: Check if this is correct.
  let mask_a = if a.n == 0 { 1 } else { (1usize << a.n) - 1 };
  let mask_b = if b.n == 0 { 1 } else { (1usize << b.n) - 1 };

  // Parallelize broadcasting when output is large enough
  let out = if out_len >= 1024 {
    (0..out_len)
      .into_par_iter()
      .map(|i| {
        let ia = i & mask_a;
        let ib = i & mask_b;
        a.evaluations[ia] + b.evaluations[ib]
      })
      .collect()
  } else {
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
      let ia = i & mask_a;
      let ib = i & mask_b;
      out.push(a.evaluations[ia] + b.evaluations[ib]);
    }
    out
  };

  DenseMLPoly { n: n_out, evaluations: out }
}

impl<F: CryptoField> std::ops::Index<usize> for DenseMLPoly<F> {
  type Output = F;

  fn index(&self, index: usize) -> &Self::Output {
    &self.evaluations[index]
  }
}

impl<F: CryptoField> std::ops::IndexMut<usize> for DenseMLPoly<F> {
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    &mut self.evaluations[index]
  }
}

/// Factored representation of a dense 2D polynomial.
///
/// Represents f(x, y) = Σ_{x' ∈ {0,1}^row_vars} eq(x, x') · Π_{i=1}^{m} f_i(x', y_i)
/// where y = (y_1, ..., y_m) with |y_i| = chunk_sizes[i].
///
/// Each factor f_i has (row_vars + chunk_sizes[i]) variables.
/// Commitment uses smaller SRS: max(row_vars + chunk_sizes[i]) instead of row_vars + col_vars.
#[derive(Clone, Debug)]
pub struct FactoredDensePoly<F: CryptoField> {
  /// Number of row (x) variables
  pub row_vars: usize,
  /// Sizes of column variable chunks: |y_1|, |y_2|, ..., |y_m|
  pub chunk_sizes: Vec<usize>,
  /// Factor polynomials: f_1, f_2, ..., f_m.
  /// Each f_i has (row_vars + chunk_sizes[i]) variables.
  /// Layout: f_i.evaluations[x + y_i * 2^row_vars] = f_i(x, y_i)
  pub factors: Vec<DenseMLPoly<F>>,
}

impl<F: CryptoField> FactoredDensePoly<F> {
  /// Attempt to factorize a dense polynomial representing a 2D matrix.
  ///
  /// The polynomial must have row_vars + sum(chunk_sizes) variables.
  /// For each row x, the row values reshaped as a tensor [2^k_1, ..., 2^k_m]
  /// must be rank-1 (an outer product). Returns Err if any row is not rank-1.
  ///
  /// Variable layout: first row_vars variables are x (row), remaining are y (column).
  /// Evaluations: poly[x + y * 2^row_vars] = matrix[x, y].
  pub fn factorize(
    poly: &DenseMLPoly<F>,
    row_vars: usize,
    chunk_sizes: &[usize],
  ) -> Result<Self, String> {
    let col_vars: usize = chunk_sizes.iter().sum();
    if row_vars + col_vars != poly.n {
      return Err(format!(
        "Variable count mismatch: row_vars({}) + col_vars({}) != poly.n({})",
        row_vars, col_vars, poly.n
      ));
    }
    let m = chunk_sizes.len();
    if m == 0 {
      return Err("Need at least one chunk".to_string());
    }

    let num_rows = 1usize << row_vars;
    let num_cols = 1usize << col_vars;

    // Initialize factor evaluation arrays
    // factor_evals[i] has 2^{row_vars + chunk_sizes[i]} entries
    let mut factor_evals: Vec<Vec<F>> = chunk_sizes
      .iter()
      .map(|&k| vec![<F as CryptoField>::zero(); num_rows * (1 << k)])
      .collect();

    // For each row, decompose into rank-1 tensor
    for x in 0..num_rows {
      // Extract row: poly[x + y * num_rows] for y = 0..num_cols
      let row: Vec<F> = (0..num_cols)
        .map(|y| poly.evaluations[x + y * num_rows])
        .collect();

      // Check for zero row
      if row.iter().all(|&v| v == <F as CryptoField>::zero()) {
        // All factors already zero-initialized
        continue;
      }

      // Find a non-zero entry as pivot
      let pivot_y = row.iter().position(|&v| v != <F as CryptoField>::zero()).unwrap();

      // Decompose pivot_y into chunk indices
      let mut pivot_indices = Vec::with_capacity(m);
      let mut remaining_bits = pivot_y;
      for &k in chunk_sizes.iter() {
        pivot_indices.push(remaining_bits & ((1 << k) - 1));
        remaining_bits >>= k;
      }

      let pivot_val = row[pivot_y];
      let pivot_val_inv = pivot_val.invert();

      // Extract factor vectors using rank-1 decomposition:
      // v_1[i_1] = row[i_1, pivot_2, ..., pivot_m]       (absorbs full scale)
      // v_j[i_j] = row[pivot_1, ..., i_j, ..., pivot_m] / pivot_val   (j >= 2)
      for factor_idx in 0..m {
        let k_i = chunk_sizes[factor_idx];
        let chunk_size = 1usize << k_i;

        for yi in 0..chunk_size {
          // Build full y index: this chunk = yi, others = pivot
          let mut y_full = 0usize;
          let mut bit_offset = 0;
          for j in 0..m {
            let idx = if j == factor_idx { yi } else { pivot_indices[j] };
            y_full += idx << bit_offset;
            bit_offset += chunk_sizes[j];
          }

          let val = row[y_full];
          if factor_idx == 0 {
            factor_evals[factor_idx][x + yi * num_rows] = val;
          } else {
            factor_evals[factor_idx][x + yi * num_rows] = val * pivot_val_inv;
          }
        }
      }

      // Verify the decomposition for this row
      for y in 0..num_cols {
        let expected = row[y];
        let mut product = <F as CryptoField>::one();
        let mut bit_offset = 0;
        for i in 0..m {
          let yi = (y >> bit_offset) & ((1 << chunk_sizes[i]) - 1);
          product = product * factor_evals[i][x + yi * num_rows];
          bit_offset += chunk_sizes[i];
        }
        if product != expected {
          return Err(format!(
            "Row {} is not rank-1 at y={}: expected != product of factors",
            x, y
          ));
        }
      }
    }

    // Build factor polynomials
    let factors: Vec<DenseMLPoly<F>> = factor_evals
      .into_iter()
      .enumerate()
      .map(|(i, evals)| DenseMLPoly::new(row_vars + chunk_sizes[i], evals))
      .collect();

    Ok(Self {
      row_vars,
      chunk_sizes: chunk_sizes.to_vec(),
      factors,
    })
  }

  /// Total column variables (sum of chunk sizes).
  pub fn col_vars(&self) -> usize {
    self.chunk_sizes.iter().sum()
  }

  /// Evaluate at (r_x, r_y) using the factored form:
  /// f(r_x, r_y) = Σ_{x'} eq(r_x, x') · Π_i f_i(x', r_{y_i})
  pub fn evaluate(&self, r_x: &[F], r_y: &[F]) -> F {
    assert_eq!(r_x.len(), self.row_vars);
    assert_eq!(r_y.len(), self.col_vars());

    let num_rows = 1usize << self.row_vars;

    // eq(r_x, x') for all x' ∈ {0,1}^row_vars
    let eq_evals = evaluate_lagrange_basis(r_x);

    // For each factor, fix the y_i variables to get a polynomial of x only
    let mut y_offset = 0;
    let fixed_factors: Vec<DenseMLPoly<F>> = self
      .factors
      .iter()
      .enumerate()
      .map(|(i, f_i)| {
        let k_i = self.chunk_sizes[i];
        let r_yi = &r_y[y_offset..y_offset + k_i];
        y_offset += k_i;
        // Fix variables at indices row_vars..row_vars+k_i
        let fixed: Vec<(usize, F)> = r_yi
          .iter()
          .enumerate()
          .map(|(j, &val)| (self.row_vars + j, val))
          .collect();
        fix_many(f_i, &fixed)
      })
      .collect();

    // Σ_{x'} eq(r_x, x') · Π_i f_i(x', r_{y_i})
    let mut result = <F as CryptoField>::zero();
    for x in 0..num_rows {
      let mut term = eq_evals[x];
      for fixed_f in &fixed_factors {
        term = term * fixed_f.evaluations[x];
      }
      result = result + term;
    }
    result
  }

  /// Fix the y variables in each factor to r_y, returning polynomials of x only.
  /// Returns (eq_poly, [g_1, g_2, ..., g_m]) where:
  ///   eq_poly = eq(r_x, ·) as a DenseMLPoly of row_vars variables
  ///   g_i(x') = f_i(x', r_{y_i}) as a DenseMLPoly of row_vars variables
  pub fn prepare_sumcheck_polys(&self, r_x: &[F], r_y: &[F]) -> (DenseMLPoly<F>, Vec<DenseMLPoly<F>>) {
    assert_eq!(r_x.len(), self.row_vars);
    assert_eq!(r_y.len(), self.col_vars());

    // eq(r_x, x') for all x' ∈ {0,1}^row_vars
    let eq_evals = evaluate_lagrange_basis(r_x);
    let eq_poly = DenseMLPoly::new(self.row_vars, eq_evals);

    // Fix y_i in each factor
    let mut y_offset = 0;
    let factor_polys: Vec<DenseMLPoly<F>> = self
      .factors
      .iter()
      .enumerate()
      .map(|(i, f_i)| {
        let k_i = self.chunk_sizes[i];
        let r_yi = &r_y[y_offset..y_offset + k_i];
        y_offset += k_i;
        let fixed: Vec<(usize, F)> = r_yi
          .iter()
          .enumerate()
          .map(|(j, &val)| (self.row_vars + j, val))
          .collect();
        fix_many(f_i, &fixed)
      })
      .collect();

    (eq_poly, factor_polys)
  }
}

/// Additive factored representation: A = Σ_k Π_i A^{(k)}_i
/// where each term k is a rank-1 tensor over m column variable groups.
///
/// The number of groups m = chunk_sizes.len() is configurable (num_shares parameter).
/// Each factor A^{(k)}_i has (row_vars + chunk_sizes[i]) variables.
/// Layout: factor.evaluations[x + z_i * 2^row_vars] = factor(x, z_i)
#[derive(Clone, Debug)]
pub struct AdditiveFactoredPoly<F: CryptoField> {
  pub row_vars: usize,
  pub chunk_sizes: Vec<usize>,
  /// t terms: each term has m factors (one per chunk group)
  /// terms[k][i] = A^{(k)}_i with (row_vars + chunk_sizes[i]) variables
  pub terms: Vec<Vec<DenseMLPoly<F>>>,
  /// If true, the decomposition was built on the transposed polynomial
  /// (e.g., one-hot-per-column matrix viewed as one-hot-per-row by swapping
  /// input/table roles). The claim point from SpMM has the original variable
  /// ordering [input, table], so the prove/verify paths must swap to
  /// [table, input] = [row_vars, col_vars] before running the sumcheck.
  pub transposed: bool,
}

impl<F: CryptoField> AdditiveFactoredPoly<F> {
  /// Decompose a sparse polynomial into additive factored form.
  ///
  /// Input: sparse polynomial with row_vars + col_vars variables, num_shares groups.
  /// For each row, performs outer product removal (rank decomposition) to express
  /// the row as a sum of rank-1 terms over the column variable groups.
  pub fn decompose(
    poly: &SparseMLPoly<F>,
    row_vars: usize,
    col_vars: usize,
    num_shares: usize,
  ) -> Result<Self, String> {
    if row_vars + col_vars != poly.n {
      return Err(format!(
        "Variable count mismatch: row_vars({}) + col_vars({}) != poly.n({})",
        row_vars, col_vars, poly.n
      ));
    }
    if num_shares == 0 || num_shares > col_vars {
      return Err(format!(
        "num_shares must be in [1, col_vars]: got {} with col_vars={}",
        num_shares, col_vars
      ));
    }

    // Phase 1: Determine chunk sizes
    let m = num_shares;
    let base = col_vars / m;
    let remainder = col_vars % m;
    let chunk_sizes: Vec<usize> = (0..m)
      .map(|i| if i < remainder { base + 1 } else { base })
      .collect();

    let num_rows = 1usize << row_vars;
    let num_cols = 1usize << col_vars;

    // Phase 2: Per-row tensor rank factorization
    // For m=2, use outer product removal directly.
    // For m>2, first decompose into 2 groups (first chunk vs rest), then recursively factorize the rest.

    // Collect rows from sparse poly
    let mut row_entries: Vec<Vec<(usize, F)>> = vec![Vec::new(); num_rows];
    for (&idx, &val) in poly.evaluations.iter() {
      if idx < (1 << poly.n) {
        let x = idx & ((1 << row_vars) - 1);
        let y = idx >> row_vars;
        if y < num_cols {
          row_entries[x].push((y, val));
        }
      }
    }

    // Per-row factorization
    let mut all_row_terms: Vec<Vec<Vec<Vec<F>>>> = Vec::with_capacity(num_rows);
    // all_row_terms[x][k][i] = vector of 2^chunk_sizes[i] entries for term k, factor i, row x

    for x in 0..num_rows {
      let entries = &row_entries[x];
      if entries.is_empty() {
        all_row_terms.push(Vec::new());
        continue;
      }

      // Build the row as a dense vector for factorization
      let row_terms = Self::factorize_row(entries, &chunk_sizes)?;
      all_row_terms.push(row_terms);
    }

    // Phase 3: Assemble factor polynomials
    let t = all_row_terms.iter().map(|terms| terms.len()).max().unwrap_or(0);

    let mut terms: Vec<Vec<DenseMLPoly<F>>> = Vec::with_capacity(t);
    for k in 0..t {
      let mut factors: Vec<DenseMLPoly<F>> = Vec::with_capacity(m);
      for i in 0..m {
        let chunk_size = 1usize << chunk_sizes[i];
        let mut evals = vec![<F as CryptoField>::zero(); num_rows * chunk_size];
        for x in 0..num_rows {
          if k < all_row_terms[x].len() {
            let factor_vec = &all_row_terms[x][k][i];
            for zi in 0..chunk_size {
              evals[x + zi * num_rows] = factor_vec[zi];
            }
          }
          // else: zero-padded (already initialized)
        }
        factors.push(DenseMLPoly::new(row_vars + chunk_sizes[i], evals));
      }
      terms.push(factors);
    }

    Ok(Self {
      row_vars,
      chunk_sizes,
      terms,
      transposed: false,
    })
  }

  /// Pad an additive decomposition to a public, witness-independent number of
  /// terms. Each appended term contains zero factor polynomials, so it does
  /// not change the represented polynomial.
  pub fn pad_to_capacity(&mut self, capacity: usize) -> Result<(), String> {
    if self.terms.len() > capacity {
      return Err(format!(
        "decomposition needs {} terms, exceeding public capacity {}",
        self.terms.len(), capacity
      ));
    }

    while self.terms.len() < capacity {
      let factors = self
        .chunk_sizes
        .iter()
        .map(|&chunk_vars| {
          let num_vars = self.row_vars + chunk_vars;
          DenseMLPoly::new(num_vars, vec![<F as CryptoField>::zero(); 1usize << num_vars])
        })
        .collect();
      self.terms.push(factors);
    }
    Ok(())
  }

  /// Decompose a one-hot-per-column SelectionPolynomial into factored form using
  /// the Twist and Shout technique. Transposes the matrix so that each row has
  /// exactly one nonzero (one-hot per row), giving t=1 (single additive term).
  ///
  /// The resulting AdditiveFactoredPoly has:
  ///   row_vars = table_num_vars (edges)  — sumcheck dimension
  ///   col_vars = input_num_vars (nodes)  — factored dimension
  ///   transposed = true
  ///   terms.len() = 1 (t=1, Twist and Shout)
  ///   chunk_sizes splits input_num_vars into num_shares groups
  ///
  /// Optimized: since each transposed row has exactly one nonzero (=1), the
  /// per-row factorization is trivially a product of unit vectors. We build the
  /// factor polynomials directly without calling the general decompose/factorize_row.
  pub fn selection_decompose(
    poly: &SparseMLPoly<F>,
    num_shares: usize,
  ) -> Result<Self, String> {
    let sel = &poly.selection;
    let input_vars = sel.input_num_vars;   // log N (nodes) — becomes col_vars
    let table_vars = sel.table_num_vars;   // log E (edges) — becomes row_vars

    if input_vars == 0 || table_vars == 0 {
      return Err("SelectionPolynomial must have both input and table variables".to_string());
    }
    if num_shares == 0 || num_shares > input_vars {
      return Err(format!(
        "num_shares must be in [1, input_vars={}]: got {}",
        input_vars, num_shares
      ));
    }

    // Determine chunk sizes for splitting input_vars (node variables)
    let m = num_shares;
    let base = input_vars / m;
    let remainder = input_vars % m;
    let chunk_sizes: Vec<usize> = (0..m)
      .map(|i| if i < remainder { base + 1 } else { base })
      .collect();

    let num_rows = 1usize << table_vars;  // 2^log_E
    let one = <F as CryptoField>::one();
    let zero = <F as CryptoField>::zero();

    // Build factor polynomials directly.
    // The transposed matrix T has T[edge, node] = 1 for each (node, edge) in selection.
    // For each selection pair, row = edge, col = node.
    // The col is split into m chunk groups. Factor i at row x has a 1 at the
    // chunk_i bits of the column (node) for that row's selected column.
    //
    // Factor layout: factor.evaluations[x + z_i * 2^row_vars] = 1 if chunk_i(col(x)) == z_i
    let mut factors: Vec<Vec<F>> = Vec::with_capacity(m);
    for i in 0..m {
      let chunk_size = 1usize << chunk_sizes[i];
      factors.push(vec![zero; num_rows * chunk_size]);
    }

    for &(node, edge) in &sel.selection {
      // Transposed: row = edge, col = node
      let row = edge;
      let mut col_remaining = node;
      for i in 0..m {
        let k_i = chunk_sizes[i];
        let z_i = col_remaining & ((1 << k_i) - 1);
        col_remaining >>= k_i;
        // factor[i][row + z_i * num_rows] = 1
        factors[i][row + z_i * num_rows] = one;
      }
    }

    let terms = vec![
      factors.into_iter().enumerate().map(|(i, evals)| {
        DenseMLPoly::new(table_vars + chunk_sizes[i], evals)
      }).collect()
    ];

    Ok(Self {
      row_vars: table_vars,
      chunk_sizes,
      terms,
      transposed: true,
    })
  }

  /// Factorize a single row's nonzero entries into rank-1 terms over chunk groups.
  /// Returns Vec of terms, where each term is Vec of m factor vectors.
  fn factorize_row(
    entries: &[(usize, F)],
    chunk_sizes: &[usize],
  ) -> Result<Vec<Vec<Vec<F>>>, String> {
    let m = chunk_sizes.len();
    let col_vars: usize = chunk_sizes.iter().sum();
    let num_cols = 1usize << col_vars;

    if m == 1 {
      // Single group: each nonzero row is one term with the row as its single factor
      let chunk_size = 1usize << chunk_sizes[0];
      let mut v = vec![<F as CryptoField>::zero(); chunk_size];
      for &(y, val) in entries {
        if y < chunk_size {
          v[y] = val;
        }
      }
      return Ok(vec![vec![v]]);
    }

    // Build dense row vector
    let mut row = vec![<F as CryptoField>::zero(); num_cols];
    for &(y, val) in entries {
      if y < num_cols {
        row[y] = val;
      }
    }

    // Split into first chunk and rest
    let k1 = chunk_sizes[0];
    let k_rest: usize = chunk_sizes[1..].iter().sum();
    let size1 = 1usize << k1;
    let size_rest = 1usize << k_rest;

    // Reshape row as a size1 × size_rest matrix M where M[i1][i_rest] = row[i1 + i_rest * size1]
    // (LSB-first: first chunk occupies lowest bits)

    // Outer product removal on M
    let mut residual: Vec<Vec<F>> = vec![vec![<F as CryptoField>::zero(); size_rest]; size1];
    for y in 0..num_cols {
      let i1 = y & (size1 - 1);
      let i_rest = y >> k1;
      residual[i1][i_rest] = row[y];
    }

    let mut terms: Vec<Vec<Vec<F>>> = Vec::new();
    let zero = <F as CryptoField>::zero();

    loop {
      // Find a nonzero pivot
      let mut pivot = None;
      'outer: for i1 in 0..size1 {
        for i_rest in 0..size_rest {
          if residual[i1][i_rest] != zero {
            pivot = Some((i1, i_rest));
            break 'outer;
          }
        }
      }

      let (pi, pj) = match pivot {
        Some(p) => p,
        None => break, // residual is zero
      };

      let p_val = residual[pi][pj];
      let p_val_inv = p_val.invert();

      // u_k = column pj of residual (size1 entries)
      let u: Vec<F> = (0..size1).map(|i1| residual[i1][pj]).collect();
      // v_k = row pi of residual, normalized by pivot (size_rest entries)
      let v: Vec<F> = (0..size_rest).map(|i_rest| residual[pi][i_rest] * p_val_inv).collect();

      // Subtract rank-1 outer product: R = R - u * v^T
      for i1 in 0..size1 {
        for i_rest in 0..size_rest {
          residual[i1][i_rest] = residual[i1][i_rest] - u[i1] * v[i_rest];
        }
      }

      // Now recursively factorize v (which lives in the remaining chunk groups)
      if m == 2 {
        // v is already the second factor
        terms.push(vec![u, v]);
      } else {
        // Recursively factorize v into (m-1) factors over chunk_sizes[1..]
        let v_entries: Vec<(usize, F)> = v.iter().enumerate()
          .filter(|(_, &val)| val != zero)
          .map(|(idx, &val)| (idx, val))
          .collect();

        if v_entries.is_empty() {
          // This shouldn't happen since u*v^T was nonzero
          continue;
        }

        let sub_terms = Self::factorize_row(&v_entries, &chunk_sizes[1..])?;
        // Each sub_term gives (m-1) factors; prepend u to get m factors
        for sub_term in sub_terms {
          let mut full_term = vec![u.clone()];
          full_term.extend(sub_term);
          terms.push(full_term);
        }
      }
    }

    Ok(terms)
  }

  /// Total column variables.
  pub fn col_vars(&self) -> usize {
    self.chunk_sizes.iter().sum()
  }

  /// Number of terms in the additive decomposition.
  pub fn num_terms(&self) -> usize {
    self.terms.len()
  }

  /// Evaluate at (r_x, r_y):
  /// A(r_x, r_y) = Σ_{x'} eq(r_x, x') · Σ_k Π_i f^{(k)}_i(x', r_{y_i})
  pub fn evaluate(&self, r_x: &[F], r_y: &[F]) -> F {
    assert_eq!(r_x.len(), self.row_vars);
    assert_eq!(r_y.len(), self.col_vars());

    let num_rows = 1usize << self.row_vars;
    let eq_evals = evaluate_lagrange_basis(r_x);

    let mut result = <F as CryptoField>::zero();
    for x in 0..num_rows {
      let mut sum_over_terms = <F as CryptoField>::zero();
      for term in &self.terms {
        let mut product = <F as CryptoField>::one();
        let mut y_offset = 0;
        for (i, factor) in term.iter().enumerate() {
          let k_i = self.chunk_sizes[i];
          let r_yi = &r_y[y_offset..y_offset + k_i];
          y_offset += k_i;
          // Evaluate factor at (x, r_yi) by fixing y variables
          let fixed: Vec<(usize, F)> = r_yi.iter().enumerate()
            .map(|(j, &val)| (self.row_vars + j, val))
            .collect();
          let fixed_poly = fix_many(factor, &fixed);
          product = product * fixed_poly.evaluations[x];
        }
        sum_over_terms = sum_over_terms + product;
      }
      result = result + eq_evals[x] * sum_over_terms;
    }
    result
  }

  /// Prepare sumcheck inputs for verifying adj_eval.
  ///
  /// The sumcheck proves:
  ///   adj_eval = Σ_{x'} eq(r_x, x') · Σ_k Π_i f^{(k)}_i(x', r_{y_i})
  ///
  /// Returns (eq_poly, factor_polys_per_term) where:
  ///   eq_poly has row_vars variables
  ///   factor_polys_per_term[k][i] has row_vars variables (y vars fixed at r_{y_i})
  pub fn prepare_sumcheck_inputs(
    &self, r_x: &[F], r_y: &[F],
  ) -> (DenseMLPoly<F>, Vec<Vec<DenseMLPoly<F>>>) {
    assert_eq!(r_x.len(), self.row_vars);
    assert_eq!(r_y.len(), self.col_vars());

    let eq_evals = evaluate_lagrange_basis(r_x);
    let eq_poly = DenseMLPoly::new(self.row_vars, eq_evals);

    let mut factor_polys_per_term: Vec<Vec<DenseMLPoly<F>>> = Vec::with_capacity(self.terms.len());
    for term in &self.terms {
      let mut y_offset = 0;
      let mut fixed_factors: Vec<DenseMLPoly<F>> = Vec::with_capacity(term.len());
      for (i, factor) in term.iter().enumerate() {
        let k_i = self.chunk_sizes[i];
        let r_yi = &r_y[y_offset..y_offset + k_i];
        y_offset += k_i;
        let fixed: Vec<(usize, F)> = r_yi.iter().enumerate()
          .map(|(j, &val)| (self.row_vars + j, val))
          .collect();
        fixed_factors.push(fix_many(factor, &fixed));
      }
      factor_polys_per_term.push(fixed_factors);
    }

    (eq_poly, factor_polys_per_term)
  }
}

/// Selection polynomial representation
/// f(k1, k2, ..., kn, t1, t2, ..., tm) = f_1(k1, k2, ..., k_(n/2), t1, t2, ..., tm) * f_2(k_(n/2+1), ..., kn, t1, t2, ..., tm)
/// n = logK, m = logT
#[derive(Clone, Debug)]
pub struct SelectionPolynomial<F: CryptoField> {
  pub input_num_vars: usize,          // Number of variables for input (logT variables)
  pub table_num_vars: usize,          // Number of variables for table (logK variables)
  pub selection: Vec<(usize, usize)>, // (input_index, table_index) pairs
  _phantom: PhantomData<F>,
}

impl<F: CryptoField> SelectionPolynomial<F> {
  pub fn new(input_num_vars: usize, table_num_vars: usize, selection: Vec<(usize, usize)>) -> Self {
    Self {
      input_num_vars,
      table_num_vars,
      selection,
      _phantom: PhantomData,
    }
  }

  pub fn to_sparse(self) -> SparseMLPoly<F> {
    let n = self.input_num_vars + self.table_num_vars;
    let mut evaluations = HashMap::new();
    for (input_index, table_index) in self.selection.iter() {
      let index = input_index + table_index * (1 << self.input_num_vars);
      evaluations.insert(index, <F as CryptoField>::one());
    }
    let mut sparse = SparseMLPoly::new(n, evaluations, VecDeque::new(), self);
    sparse.build_sorted_indices();
    sparse
  }
}

/// Sparse multilinear polynomial represented as a map from point index to evaluation
/// and a queue of point indices for efficient iteration
#[derive(Clone, Debug)]
pub struct SparseMLPoly<F: CryptoField> {
  pub n: usize, // number of variables
  pub evaluations: HashMap<usize, F>,
  pub indices: VecDeque<usize>,
  pub selection: SelectionPolynomial<F>,
}

impl<F: CryptoField + 'static> MLPoly<F> for SparseMLPoly<F> {
  fn fix_variables(&self, partial_point: &[F]) -> Box<dyn MLPoly<F>> {
    Box::new(self.fix_variables(partial_point))
  }

  fn n(&self) -> usize {
    self.n
  }

  fn evaluate_at_point(&self, point: &[F]) -> F {
    assert!(point.len() == self.n, "SparseMLPoly: point.len()={} != poly.n={}", point.len(), self.n);

    // Sparse evaluation: directly compute sum of eq(i, point) * poly[i] for non-zero entries
    // This is O(k) instead of O(k log k) from fix_variables (which sorts indices)
    if self.evaluations.is_empty() {
      return <F as CryptoField>::zero();
    }

    let eq_evals = precompute_eq(point);
    let mut result = <F as CryptoField>::zero();

    for (idx, &val) in self.evaluations.iter() {
      if *idx < eq_evals.len() {
        result = result + val * eq_evals[*idx];
      }
    }

    result
  }

  fn evaluations(&self) -> Vec<F> {
    let mut result = vec![<F as CryptoField>::zero(); 1 << self.n];
    for (idx, val) in self.evaluations.iter() {
      result[*idx] = *val;
    }
    result
  }

  fn len(&self) -> usize {
    1 << self.n
  }

  // index only works for indices that are less than 2^64 (usize)
  fn index(&self, index: usize) -> F {
    *self.evaluations.get(&index).unwrap_or(&<F as CryptoField>::zero())
  }

  fn index_mut(&mut self, index: usize) -> &mut F {
    self.evaluations.entry(index).or_insert(<F as CryptoField>::zero())
  }

  fn clone_box(&self) -> Box<dyn MLPoly<F>> {
    Box::new(self.clone())
  }

  fn as_any(&self) -> &dyn std::any::Any {
    self
  }

  fn mul_by_scalar(&self, scalar: F) -> Box<dyn MLPoly<F>> {
    Box::new(self.mul_by_scalar(scalar))
  }

  fn add(&self, other: &Box<dyn MLPoly<F>>) -> Box<dyn MLPoly<F>> {
    Box::new(self.add(other.as_any().downcast_ref::<SparseMLPoly<F>>().unwrap()))
  }
}

impl<F: CryptoField> SparseMLPoly<F> {
  pub fn new(n: usize, evaluations: HashMap<usize, F>, indices: VecDeque<usize>, selection: SelectionPolynomial<F>) -> Self {
    Self {
      n,
      evaluations,
      indices,
      selection,
    }
  }

  /// Build sorted indices from evaluations - call this before operations that need sorted indices (e.g., SD sumcheck)
  pub fn build_sorted_indices(&mut self) {
    let mut indices: Vec<usize> = self.evaluations.keys().cloned().collect();
    indices.sort_unstable_by(|a, b| a.cmp(b));
    self.indices = indices.into();
  }

  pub fn len(&self) -> usize {
    self.evaluations.len()
  }

  pub fn is_empty(&self) -> bool {
    self.evaluations.is_empty()
  }

  pub fn fix_variables(&self, partial_point: &[F]) -> Self {
    let dim = partial_point.len();
    assert!(dim <= self.n, "invalid partial point dimension");

    let mut window = (self.evaluations.len() as f64).log2() as usize;
    if window == 0 {
      window = 1;
    }

    let mut point = partial_point;
    let mut last = self.evaluations.clone();

    // batch evaluation
    while !point.is_empty() {
      let focus_length = if point.len() > window { window } else { point.len() };
      let focus = &point[..focus_length];
      point = &point[focus_length..];
      let pre = precompute_eq(focus);
      let dim = focus.len();
      let mut result: HashMap<usize, F> = HashMap::new();

      // Parallelize the sparse entry processing when there are enough entries
      if last.len() >= 128 {
        // Use parallel processing for larger sparse polynomials
        let partial_results: Vec<HashMap<usize, F>> = last
          .par_iter()
          .map(|src_entry| {
            let old_idx_bytes = src_entry.0;
            let low_bits = *old_idx_bytes & ((1 << dim) - 1);
            let gz = pre[low_bits];
            let new_idx_bytes = old_idx_bytes >> dim;
            let mut local_result = HashMap::new();
            local_result.insert(new_idx_bytes, gz * *src_entry.1);
            local_result
          })
          .collect();

        // Combine partial results
        for partial in partial_results {
          for (key, value) in partial {
            result.entry(key).and_modify(|e| *e = *e + value).or_insert(value);
          }
        }
      } else {
        // Use sequential processing for smaller sparse polynomials
        for src_entry in last.iter() {
          let old_idx_bytes = src_entry.0;
          let low_bits = *old_idx_bytes & ((1 << dim) - 1);
          let gz = pre[low_bits];
          let new_idx_bytes = old_idx_bytes >> dim;
          let dst_entry = result.entry(new_idx_bytes).or_insert(<F as CryptoField>::zero());
          *dst_entry = *dst_entry + gz * *src_entry.1;
        }
      }
      last = result;
    }

    // Return result without sorted indices - caller can call build_sorted_indices() if needed
    Self::new(self.n - dim, last, VecDeque::new(), self.selection.clone())
  }

  pub fn mul_by_scalar(&self, scalar: F) -> Self {
    // If scalar is zero, return empty sparse polynomial
    if scalar == <F as CryptoField>::zero() {
      return Self::new(self.n, HashMap::new(), VecDeque::new(), self.selection.clone());
    }

    // Parallelize scalar multiplication for large sparse polynomials
    let new_evaluations = if self.evaluations.len() >= 128 {
      // Use parallel processing for larger sparse polynomials
      let partial_results: Vec<(usize, F)> =
        self.evaluations.par_iter().map(|(k, v)| (*k, *v * scalar)).filter(|(_, result)| *result != <F as CryptoField>::zero()).collect();

      partial_results.into_iter().collect()
    } else {
      // Use sequential processing for smaller sparse polynomials
      let mut new_evaluations: HashMap<usize, F> = HashMap::new();
      for (k, v) in self.evaluations.iter() {
        let result = *v * scalar;
        if result != <F as CryptoField>::zero() {
          new_evaluations.insert(*k, result);
        }
      }
      new_evaluations
    };

    Self::new(self.n, new_evaluations, VecDeque::new(), self.selection.clone())
  }

  pub fn add(&self, other: &Self) -> Self {
    assert!(self.n == other.n, "mismatched number of variables");

    // Parallelize addition for large sparse polynomials
    let new_evaluations = if self.evaluations.len() + other.evaluations.len() >= 256 {
      // Use parallel processing for larger sparse polynomials
      let mut combined_keys: std::collections::HashSet<usize> = std::collections::HashSet::new();
      combined_keys.extend(self.evaluations.keys());
      combined_keys.extend(other.evaluations.keys());

      let partial_results: Vec<(usize, F)> = combined_keys
        .into_par_iter()
        .map(|k| {
          let self_val = self.evaluations.get(&k).copied().unwrap_or(<F as CryptoField>::zero());
          let other_val = other.evaluations.get(&k).copied().unwrap_or(<F as CryptoField>::zero());
          let sum = self_val + other_val;
          (k, sum)
        })
        .filter(|(_, v)| *v != <F as CryptoField>::zero())
        .collect();

      partial_results.into_iter().collect()
    } else {
      // Use sequential processing for smaller sparse polynomials
      let mut new_evaluations = self.evaluations.clone();

      // Add entries from other polynomial during iteration
      for (k, v) in other.evaluations.iter() {
        new_evaluations.entry(*k).and_modify(|e| *e = *e + *v).or_insert(*v);
      }

      // Filter out zeros that may result from cancellation (e.g., a + (-a) = 0)
      new_evaluations.retain(|_, v| *v != <F as CryptoField>::zero());

      new_evaluations
    };

    Self::new(self.n, new_evaluations, VecDeque::new(), self.selection.clone())
  }

  /// Materialize the sparse multilinear polynomial into a dense table of size 2^n.
  ///
  /// Indices are stored in little-endian Vec<u8> form; we convert them with
  /// `le_vec_to_usize`. This assumes `n` is small enough that 2^n fits in `usize`.
  pub fn to_dense(&self) -> DenseMLPoly<F> {
    // Sanity check: avoid shifting past usize width
    assert!(
      self.n < usize::BITS as usize,
      "too many variables ({}) to materialize into a dense table on this platform",
      self.n
    );

    let size = 1usize << self.n;
    let mut evaluations = vec![<F as CryptoField>::zero(); size];

    // Fill in non-zero entries from the sparse representation.
    // Missing entries stay as zero.
    for (idx, &val) in &self.evaluations {
      debug_assert!(*idx < size, "sparse index {} exceeds dense table size 2^{} = {}", idx, self.n, size);
      evaluations[*idx] = val;
    }

    DenseMLPoly::new(self.n, evaluations)
  }

  pub fn split_into_blocks(&self, block_size: usize) -> Vec<Self> {
    assert!(block_size > 0, "block_size must be > 0");

    let selection = &self.selection;

    // If there are no table vars, there is nothing to split.
    if selection.table_num_vars == 0 {
      return vec![self.clone()];
    }

    // ceil(table_num_vars / block_size)
    let num_blocks = (selection.table_num_vars + block_size - 1) / block_size;
    debug_assert!(num_blocks > 0);

    let mut blocks = Vec::with_capacity(num_blocks);

    // Constant mask range for a full block of size `block_size`.
    let full_block_mod = 1usize << block_size;

    for i in 0..num_blocks {
      let offset = i * block_size; // bit offset into table_index
      let start = 1usize << offset;

      // Number of table vars in this block (last block may be smaller)
      let remaining = selection.table_num_vars - offset;
      let this_block_vars = remaining.min(block_size);

      // Modulus matching the number of bits we actually keep in this block.
      let this_block_mod = 1usize << this_block_vars;

      let block_selection = selection
        .selection
        .iter()
        .map(|(input_index, table_index)| {
          // Extract bits [offset .. offset+this_block_vars)
          // (table_index >> offset) & ((1<<this_block_vars)-1)
          let v = (table_index / start) % full_block_mod;
          (*input_index, v % this_block_mod)
        })
        .collect::<Vec<(usize, usize)>>();

      let block_selection_polynomial = SelectionPolynomial::new(selection.input_num_vars, block_size, block_selection);
      let block_selection_polynomial_sparse = block_selection_polynomial.to_sparse();

      blocks.push(block_selection_polynomial_sparse);
    }

    blocks
  }
}

pub fn range_dense<F: CryptoField>(num_vars: usize) -> DenseMLPoly<F> {
  let vec_len = 1 << num_vars;
  let mut evaluations = Vec::with_capacity(vec_len);
  for i in 0..vec_len {
    evaluations.push(F::from(i as u32));
  }
  DenseMLPoly::new(num_vars, evaluations)
}

pub fn two_pow_dense<F: CryptoField>(num_vars: usize) -> DenseMLPoly<F> {
  assert!(num_vars == 8, "two_pow_dense only supports 8 variables (k_shifted in [0,150])");
  let vec_len = 1 << num_vars; // 256
  let k_max = 75usize; // must match K_MAX in exp.rs
  let max_k_shifted = 2 * k_max; // 150
  let mut evaluations = Vec::with_capacity(vec_len);
  for i in 0..vec_len {
    if i <= max_k_shifted && i <= k_max + 15 {
      // table[k_shifted] = 2^(k_max + 15 - k_shifted)
      let exponent = k_max + 15 - i;
      let mut val = <F as CryptoField>::one();
      for _ in 0..exponent {
        val = val + val;
      }
      evaluations.push(val);
    } else {
      evaluations.push(<F as CryptoField>::zero()); // exp ≈ 0 for very negative inputs
    }
  }
  DenseMLPoly::new(num_vars, evaluations)
}

pub fn fix_variables_zkgpt<F: CryptoField>(
  num_vars: usize,
  t_idx: &[i128], // n x m, entries in [0, 2^Q]
  r: &[F],        // len = log2(n)
  q_bits: usize,
) -> DenseMLPoly<F> {
  let n = 1usize << r.len();
  let logm = num_vars - r.len();
  let m = 1usize << logm;

  let logn = r.len();
  let left_bits = logn / 2;
  let right_bits = logn - left_bits;

  let y = &r[..right_bits];
  let x = &r[right_bits..];

  let eqy = precompute_eq::<F>(y);
  let eqx = precompute_eq::<F>(x);

  let max_t: usize = 1usize << q_bits;
  let fm_width = max_t + 1;
  let right_size = 1usize << right_bits;

  // 1) Flattened FM: fm[br * fm_width + t]
  let fm: Vec<F> = (0..right_size)
    .into_par_iter()
    .flat_map_iter(|br| {
      let mut row = vec![F::ZERO; fm_width];
      let step = eqy[br];
      for t in 1..=max_t {
        row[t] = row[t - 1] + step; // additions only
      }
      row
    })
    .collect();

  let right_mask = right_size - 1;

  // 2) Process columns in chunks (fewer Rayon tasks, better locality)
  let chunk = 64usize; // tune: 32/64/128 often good
  let mut out = vec![F::ZERO; m];

  out.par_chunks_mut(chunk).enumerate().for_each(|(chunk_id, out_chunk)| {
    let c0 = chunk_id * chunk;
    let c1 = (c0 + out_chunk.len()).min(m);

    // thread-local AC buffer reused across columns in this chunk
    let mut ac = vec![F::ZERO; 1usize << left_bits];

    for (j, c) in (c0..c1).enumerate() {
      // reset ac
      for v in ac.iter_mut() {
        *v = F::ZERO;
      }

      // IMPORTANT: this assumes your layout is b + c*n
      let col_base = c * n;

      for b in 0..n {
        let br = b & right_mask;
        let bl = b >> right_bits;

        let t = t_idx[col_base + b];
        if t > 0 {
          // ac[bl] += fm[br][t]
          ac[bl] += F::from(fm[br * fm_width + t as usize]);
        } else {
          ac[bl] -= F::from(fm[br * fm_width + (-t) as usize]);
        }
      }

      // outer dot with eqx
      let mut acc = F::ZERO;
      for (l, &ac_l) in ac.iter().enumerate() {
        acc += eqx[l] * ac_l; // only 2^{left_bits} multiplications per column
      }
      out_chunk[j] = acc;
    }
  });

  DenseMLPoly::new(logm, out)
}

#[cfg(test)]
mod tests {
  use super::*;
  use ark_bn254::Fr as F;

  #[test]
  fn test_factored_dense_poly_rank1_matrix() {
    // Build a 4×4 matrix where each row is rank-1 when split into 2×2 (m=2, each chunk = 1 bit)
    // Row i: v_i ⊗ w_i where v_i, w_i are 2-element vectors
    // Matrix[i, j] = v_i[j_1] * w_i[j_2] where j = j_1 + 2*j_2
    let row_vars = 2; // log(4) = 2
    let col_vars = 2; // log(4) = 2

    // Define factors per row (4 rows):
    // Row 0: v=[1, 2], w=[3, 5]  → [1*3, 2*3, 1*5, 2*5] = [3, 6, 5, 10]
    // Row 1: v=[1, 0], w=[1, 1]  → [1, 0, 1, 0]
    // Row 2: v=[2, 3], w=[1, 4]  → [2, 3, 8, 12]
    // Row 3: v=[0, 0], w=[0, 0]  → [0, 0, 0, 0]
    let n_rows = 4usize;
    let n_cols = 4usize;

    // Build evaluation table: evals[x + y * n_rows]
    let mut evals = vec![<F as CryptoField>::zero(); n_rows * n_cols];

    // Row 0: [3, 6, 5, 10]
    evals[0 + 0 * 4] = <F as CryptoField>::from_u32(3);
    evals[0 + 1 * 4] = <F as CryptoField>::from_u32(6);
    evals[0 + 2 * 4] = <F as CryptoField>::from_u32(5);
    evals[0 + 3 * 4] = <F as CryptoField>::from_u32(10);

    // Row 1: [1, 0, 1, 0]
    evals[1 + 0 * 4] = <F as CryptoField>::from_u32(1);
    evals[1 + 1 * 4] = <F as CryptoField>::zero();
    evals[1 + 2 * 4] = <F as CryptoField>::from_u32(1);
    evals[1 + 3 * 4] = <F as CryptoField>::zero();

    // Row 2: [2, 3, 8, 12]
    evals[2 + 0 * 4] = <F as CryptoField>::from_u32(2);
    evals[2 + 1 * 4] = <F as CryptoField>::from_u32(3);
    evals[2 + 2 * 4] = <F as CryptoField>::from_u32(8);
    evals[2 + 3 * 4] = <F as CryptoField>::from_u32(12);

    // Row 3: all zeros (already initialized)

    let poly = DenseMLPoly::new(row_vars + col_vars, evals);
    let chunk_sizes = vec![1, 1]; // split col into two 1-bit chunks

    let factored = FactoredDensePoly::factorize(&poly, row_vars, &chunk_sizes)
      .expect("Factorization should succeed for rank-1 rows");

    assert_eq!(factored.factors.len(), 2);
    assert_eq!(factored.factors[0].n, row_vars + 1); // 2 + 1 = 3
    assert_eq!(factored.factors[1].n, row_vars + 1); // 2 + 1 = 3

    // Verify evaluation at a random-ish point
    let r_x = vec![<F as CryptoField>::from_u32(7), <F as CryptoField>::from_u32(13)];
    let r_y = vec![<F as CryptoField>::from_u32(3), <F as CryptoField>::from_u32(5)];

    let factored_eval = factored.evaluate(&r_x, &r_y);
    let direct_eval = poly.evaluate_at_point(&[r_x[0], r_x[1], r_y[0], r_y[1]]);

    assert_eq!(factored_eval, direct_eval, "Factored evaluation must match direct evaluation");
  }

  #[test]
  fn test_factored_dense_poly_non_rank1_fails() {
    // Build a 4×4 matrix where row 0 is NOT rank-1 when split into 2×2
    // Row 0: [1, 0, 0, 1] → reshaped as [[1, 0], [0, 1]] = identity, rank 2
    let row_vars = 2;
    let n_rows = 4usize;
    let n_cols = 4usize;
    let mut evals = vec![<F as CryptoField>::zero(); n_rows * n_cols];

    evals[0 + 0 * 4] = <F as CryptoField>::from_u32(1); // M[0,0] = 1
    evals[0 + 3 * 4] = <F as CryptoField>::from_u32(1); // M[0,3] = 1

    let poly = DenseMLPoly::new(row_vars + 2, evals);
    let result = FactoredDensePoly::factorize(&poly, row_vars, &[1, 1]);
    assert!(result.is_err(), "Should fail for non-rank-1 rows");
  }

  #[test]
  fn test_additive_factored_poly_rank1_row() {
    // A 4×4 sparse matrix with one rank-1 row (row 0: [3, 6, 5, 10] = [1,2] ⊗ [3,5])
    let row_vars = 2;
    let col_vars = 2;
    let n = row_vars + col_vars;

    let mut evaluations = std::collections::HashMap::new();
    // Row 0: [3, 6, 5, 10] at indices (0,0), (0,1), (0,2), (0,3)
    let num_rows = 1usize << row_vars;
    evaluations.insert(0 + 0 * num_rows, <F as CryptoField>::from_u32(3));
    evaluations.insert(0 + 1 * num_rows, <F as CryptoField>::from_u32(6));
    evaluations.insert(0 + 2 * num_rows, <F as CryptoField>::from_u32(5));
    evaluations.insert(0 + 3 * num_rows, <F as CryptoField>::from_u32(10));

    let indices = evaluations.keys().copied().collect();
    let selection = SelectionPolynomial::<F>::new(row_vars, col_vars, vec![]);
    let sparse = SparseMLPoly::new(n, evaluations, indices, selection);

    let af = AdditiveFactoredPoly::decompose(&sparse, row_vars, col_vars, 2)
      .expect("Decomposition should succeed");

    // rank-1 row → 1 term
    assert_eq!(af.num_terms(), 1, "Rank-1 row should decompose into 1 term");
    assert_eq!(af.chunk_sizes, vec![1, 1]);

    // Verify evaluation at a test point
    let r_x = vec![<F as CryptoField>::from_u32(7), <F as CryptoField>::from_u32(13)];
    let r_y = vec![<F as CryptoField>::from_u32(3), <F as CryptoField>::from_u32(5)];
    let af_eval = af.evaluate(&r_x, &r_y);
    let mut full_point = r_x.clone();
    full_point.extend_from_slice(&r_y);
    let direct_eval = sparse.evaluate_at_point(&full_point);
    assert_eq!(af_eval, direct_eval, "Additive factored eval must match sparse eval");
  }

  #[test]
  fn test_additive_factored_poly_rank2_row() {
    // Anti-diagonal: row 0 has M[0,0]=1, M[0,3]=1 → reshaped [[1,0],[0,1]] = rank 2
    let row_vars = 2;
    let col_vars = 2;
    let n = row_vars + col_vars;
    let num_rows = 1usize << row_vars;

    let mut evaluations = std::collections::HashMap::new();
    evaluations.insert(0 + 0 * num_rows, <F as CryptoField>::from_u32(1));
    evaluations.insert(0 + 3 * num_rows, <F as CryptoField>::from_u32(1));

    let indices = evaluations.keys().copied().collect();
    let selection = SelectionPolynomial::<F>::new(row_vars, col_vars, vec![]);
    let sparse = SparseMLPoly::new(n, evaluations, indices, selection);

    let af = AdditiveFactoredPoly::decompose(&sparse, row_vars, col_vars, 2)
      .expect("Decomposition should succeed");

    // rank-2 row → 2 terms
    assert_eq!(af.num_terms(), 2, "Rank-2 row should decompose into 2 terms");

    // Verify evaluation
    let r_x = vec![<F as CryptoField>::from_u32(7), <F as CryptoField>::from_u32(13)];
    let r_y = vec![<F as CryptoField>::from_u32(3), <F as CryptoField>::from_u32(5)];
    let af_eval = af.evaluate(&r_x, &r_y);
    let mut full_point = r_x.clone();
    full_point.extend_from_slice(&r_y);
    let direct_eval = sparse.evaluate_at_point(&full_point);
    assert_eq!(af_eval, direct_eval, "Additive factored eval must match for rank-2 row");
  }

  #[test]
  fn test_additive_factored_poly_public_capacity_padding() {
    let row_vars = 2;
    let col_vars = 2;
    let num_rows = 1usize << row_vars;
    let mut evaluations = std::collections::HashMap::new();
    evaluations.insert(0, <F as CryptoField>::from_u32(1));
    evaluations.insert(3 * num_rows, <F as CryptoField>::from_u32(1));

    let indices = evaluations.keys().copied().collect();
    let selection = SelectionPolynomial::<F>::new(row_vars, col_vars, vec![]);
    let sparse = SparseMLPoly::new(row_vars + col_vars, evaluations, indices, selection);
    let mut af = AdditiveFactoredPoly::decompose(&sparse, row_vars, col_vars, 2)
      .expect("Decomposition should succeed");
    assert_eq!(af.num_terms(), 2);

    let r_x = vec![<F as CryptoField>::from_u32(7), <F as CryptoField>::from_u32(13)];
    let r_y = vec![<F as CryptoField>::from_u32(3), <F as CryptoField>::from_u32(5)];
    let before = af.evaluate(&r_x, &r_y);
    af.pad_to_capacity(4).expect("Public capacity should fit");
    assert_eq!(af.num_terms(), 4);
    assert_eq!(af.evaluate(&r_x, &r_y), before, "Zero terms must preserve the polynomial");
    assert!(af.pad_to_capacity(1).is_err(), "Capacity below the true rank must fail");
  }

  #[test]
  fn test_additive_factored_poly_num_shares_3() {
    // 8×8 sparse matrix (row_vars=3, col_vars=3), num_shares=3 (each chunk = 1 bit)
    let row_vars = 3;
    let col_vars = 3;
    let n = row_vars + col_vars;
    let num_rows = 1usize << row_vars;

    let mut evaluations = std::collections::HashMap::new();
    // Row 0: sparse entries at columns 0, 3, 5, 7
    evaluations.insert(0 + 0 * num_rows, <F as CryptoField>::from_u32(2));
    evaluations.insert(0 + 3 * num_rows, <F as CryptoField>::from_u32(7));
    evaluations.insert(0 + 5 * num_rows, <F as CryptoField>::from_u32(11));
    evaluations.insert(0 + 7 * num_rows, <F as CryptoField>::from_u32(3));
    // Row 2: one entry
    evaluations.insert(2 + 1 * num_rows, <F as CryptoField>::from_u32(5));

    let indices = evaluations.keys().copied().collect();
    let selection = SelectionPolynomial::<F>::new(row_vars, col_vars, vec![]);
    let sparse = SparseMLPoly::new(n, evaluations, indices, selection);

    let af = AdditiveFactoredPoly::decompose(&sparse, row_vars, col_vars, 3)
      .expect("Decomposition with 3 shares should succeed");

    assert_eq!(af.chunk_sizes, vec![1, 1, 1]);

    // Verify evaluation at a test point
    let r_x = vec![<F as CryptoField>::from_u32(2), <F as CryptoField>::from_u32(5), <F as CryptoField>::from_u32(9)];
    let r_y = vec![<F as CryptoField>::from_u32(3), <F as CryptoField>::from_u32(7), <F as CryptoField>::from_u32(11)];
    let af_eval = af.evaluate(&r_x, &r_y);
    let mut full_point = r_x.clone();
    full_point.extend_from_slice(&r_y);
    let direct_eval = sparse.evaluate_at_point(&full_point);
    assert_eq!(af_eval, direct_eval, "Additive factored eval must match for num_shares=3");
  }
}

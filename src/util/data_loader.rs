//! Utility functions for loading raw binary data files exported from Python.
//!
//! File conventions:
//!   - `.bin` with f32: raw little-endian float32 values, row-major
//!   - `.bin` with i32: raw little-endian int32 values
//!   - `.bin` with u8:  raw bytes (0/1 for masks)
//!   - `meta.json`: JSON object with shape metadata

use std::fs;
use std::path::Path;

use crate::util::poly::CryptoField;
use crate::{SF_FLOAT, SF_LOG};
use crate::dag::{DataType, Role, Witness};

/// Read a binary file as a Vec<f32>.
pub fn read_f32_bin(path: &Path) -> Vec<f32> {
  let bytes = fs::read(path).unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
  assert!(bytes.len() % 4 == 0, "File size not a multiple of 4: {}", path.display());
  bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Read a binary file as a Vec<i32>.
pub fn read_i32_bin(path: &Path) -> Vec<i32> {
  let bytes = fs::read(path).unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
  assert!(bytes.len() % 4 == 0, "File size not a multiple of 4: {}", path.display());
  bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Read a binary file as a Vec<u8>.
pub fn read_u8_bin(path: &Path) -> Vec<u8> {
  fs::read(path).unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e))
}

/// Convert f32 values to fixed-point field elements using the global SF_LOG.
/// Negative values are represented as field negation.
pub fn f32_to_field_vec<F: CryptoField>(data: &[f32]) -> Vec<F> {
  let sf = *SF_FLOAT;
  data.iter().map(|&x| {
    let y = (x * sf).round();
    let y = y.clamp(-(1i64 << 30) as f32, (1i64 << 30) as f32);
    if y < 0.0 {
      <F as CryptoField>::zero() - F::from((-y) as u64)
    } else {
      F::from(y as u64)
    }
  }).collect()
}

/// Load node features from a raw f32 binary file.
/// Returns a Witness of shape (num_nodes, num_features) with Role::Input.
pub fn load_node_features<F: CryptoField + 'static>(
  path: &Path,
  num_nodes: usize,
  num_features: usize,
) -> Witness<F> {
  let data_f32 = read_f32_bin(path);
  assert_eq!(data_f32.len(), num_nodes * num_features,
    "Expected {} values, got {}", num_nodes * num_features, data_f32.len());
  // Data is row-major (node, feature). Need to convert to Witness column-major layout.
  let mut field_data = vec![<F as CryptoField>::zero(); num_nodes.next_power_of_two() * num_features.next_power_of_two()];
  let stride0 = 1; // column-major: stride for dim 0
  let stride1 = num_nodes.next_power_of_two(); // stride for dim 1
  let sf = *SF_FLOAT;
  for n in 0..num_nodes {
    for f in 0..num_features {
      let val = data_f32[n * num_features + f]; // row-major input
      let y = (val * sf).round();
      let y = y.clamp(-(1i64 << 30) as f32, (1i64 << 30) as f32);
      let field_val = if y < 0.0 {
        <F as CryptoField>::zero() - F::from((-y) as u64)
      } else {
        F::from(y as u64)
      };
      field_data[n * stride0 + f * stride1] = field_val;
    }
  }
  Witness::new(
    vec![num_nodes, num_features],
    field_data,
    DataType::Float,
    *SF_LOG as usize,
    Role::Input,
  )
}

/// Load a weight matrix from a raw f32 binary file.
/// PyTorch stores Linear weights as (out_features, in_features).
/// zk-torch-3 expects (in_features, out_features) — so we transpose.
pub fn load_weight_matrix<F: CryptoField + 'static>(
  path: &Path,
  out_features: usize,
  in_features: usize,
) -> Witness<F> {
  let data_f32 = read_f32_bin(path);
  assert_eq!(data_f32.len(), out_features * in_features,
    "Expected {} values for weight ({}×{}), got {}", out_features * in_features, out_features, in_features, data_f32.len());
  // PyTorch: (out, in) row-major → zk-torch-3: (in, out) column-major
  // Witness column-major: index = row + col * row_pad
  let row_pad = in_features.next_power_of_two();
  let col_pad = out_features.next_power_of_two();
  let mut field_data = vec![<F as CryptoField>::zero(); row_pad * col_pad];
  let sf = *SF_FLOAT;
  for o in 0..out_features {
    for i in 0..in_features {
      let val = data_f32[o * in_features + i]; // PyTorch row-major (out, in)
      let y = (val * sf).round();
      let y = y.clamp(-(1i64 << 30) as f32, (1i64 << 30) as f32);
      let field_val = if y < 0.0 {
        <F as CryptoField>::zero() - F::from((-y) as u64)
      } else {
        F::from(y as u64)
      };
      // zk-torch-3 shape: (in_features, out_features), column-major
      field_data[i + o * row_pad] = field_val;
    }
  }
  Witness::new(
    vec![in_features, out_features],
    field_data,
    DataType::Float,
    *SF_LOG as usize,
    Role::Constant,
  )
}

/// Load a 1D bias vector from a raw f32 binary file.
/// Returns a Witness of shape (1, dim) for broadcast addition.
pub fn load_bias<F: CryptoField + 'static>(
  path: &Path,
  dim: usize,
) -> Witness<F> {
  let data_f32 = read_f32_bin(path);
  assert_eq!(data_f32.len(), dim, "Expected {} bias values, got {}", dim, data_f32.len());
  // Shape (1, dim), column-major: index = row + col * 1_pad = 0 + col * 1 = col
  let row_pad = 1usize.next_power_of_two(); // = 1
  let col_pad = dim.next_power_of_two();
  let mut field_data = vec![<F as CryptoField>::zero(); row_pad * col_pad];
  let sf = *SF_FLOAT;
  for d in 0..dim {
    let val = data_f32[d];
    let y = (val * sf).round();
    let y = y.clamp(-(1i64 << 30) as f32, (1i64 << 30) as f32);
    let field_val = if y < 0.0 {
      <F as CryptoField>::zero() - F::from((-y) as u64)
    } else {
      F::from(y as u64)
    };
    field_data[d * row_pad] = field_val;
  }
  Witness::new(
    vec![1, dim],
    field_data,
    DataType::Float,
    *SF_LOG as usize,
    Role::Constant,
  )
}

/// Load an attention vector from a raw f32 binary file.
/// PyTorch GAT stores att as (1, heads, head_dim). For single-head extraction,
/// pass the flat (head_dim,) slice. Returns Witness of shape (head_dim, 1).
pub fn load_attention_vector<F: CryptoField + 'static>(
  data_f32: &[f32],
) -> Witness<F> {
  let head_dim = data_f32.len();
  // Shape (head_dim, 1), column-major: index = row + col * head_dim_pad = row
  let row_pad = head_dim.next_power_of_two();
  let col_pad = 1usize.next_power_of_two();
  let mut field_data = vec![<F as CryptoField>::zero(); row_pad * col_pad];
  let sf = *SF_FLOAT;
  for d in 0..head_dim {
    let val = data_f32[d];
    let y = (val * sf).round();
    let y = y.clamp(-(1i64 << 30) as f32, (1i64 << 30) as f32);
    let field_val = if y < 0.0 {
      <F as CryptoField>::zero() - F::from((-y) as u64)
    } else {
      F::from(y as u64)
    };
    field_data[d] = field_val;
  }
  Witness::new(
    vec![head_dim, 1],
    field_data,
    DataType::Float,
    *SF_LOG as usize,
    Role::Constant,
  )
}

/// Extract the output of a DAG run as integer values and compute accuracy.
/// The output witness has shape (num_nodes_padded, num_classes) in fixed-point.
/// `num_nodes` is the actual (unpadded) number of nodes.
/// Returns (predictions, train_acc, val_acc, test_acc).
pub fn compute_accuracy<F: CryptoField + 'static>(
  output_witness: &Witness<F>,
  labels: &[i32],
  num_nodes: usize,
  masks: Option<(&[u8], &[u8], &[u8])>, // (train_mask, val_mask, test_mask)
) -> (Vec<usize>, f64, f64, f64) {
  let num_classes = output_witness.shape[1];

  let mut predictions = Vec::with_capacity(num_nodes);

  for n in 0..num_nodes {
    let mut best_class = 0usize;
    let mut best_val = i128::MIN;
    for c in 0..num_classes {
      let field_val = output_witness.get(&[n, c]);
      let int_val = crate::util::arith::f_to_int(field_val);
      if int_val > best_val {
        best_val = int_val;
        best_class = c;
      }
    }
    predictions.push(best_class);
  }

  // Compute accuracy for each split
  let (train_acc, val_acc, test_acc) = if let Some((train_mask, val_mask, test_mask)) = masks {
    let compute_split_acc = |mask: &[u8]| -> f64 {
      let mut correct = 0usize;
      let mut total = 0usize;
      for i in 0..num_nodes {
        if mask[i] != 0 {
          total += 1;
          if predictions[i] == labels[i] as usize {
            correct += 1;
          }
        }
      }
      if total > 0 { correct as f64 / total as f64 } else { 0.0 }
    };
    (compute_split_acc(train_mask), compute_split_acc(val_mask), compute_split_acc(test_mask))
  } else {
    let mut correct = 0usize;
    for i in 0..num_nodes {
      if predictions[i] == labels[i] as usize {
        correct += 1;
      }
    }
    let acc = correct as f64 / num_nodes as f64;
    (acc, acc, acc)
  };

  (predictions, train_acc, val_acc, test_acc)
}

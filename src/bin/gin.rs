// Field type selection based on backend and curve
#[cfg(all(feature = "arkworks", feature = "bls12_381"))]
use ark_bls12_381::Fr as F;
#[cfg(all(feature = "arkworks", feature = "bn254"))]
use ark_bn254::Fr as F;

#[cfg(all(feature = "icicle", feature = "bls12_381"))]
use icicle_bls12_381::curve::ScalarField as F;
#[cfg(all(feature = "icicle", feature = "bn254"))]
use icicle_bn254::curve::ScalarField as F;
#[cfg(all(feature = "icicle", feature = "goldilocks"))]
use icicle_goldilocks::field::ScalarField as F;

use plonky2::{timed, util::timing::TimingTree};
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::sync::Arc;
use zk_torch_2::util::transcript::Transcript;
use zk_torch_2::{
  crypto::polycommit::kzh3::{setup_kzh3_srs, KZH3Commit, KZH3CommitKey, KZH3Commitment, KZH3MaskCommitter, KZH3VerifierKey},
  crypto::polycommit::sparse_kzh3::{SparseKZH3Commit, SparseKZH3CommitKey, SparseKZH3VerifierKey},
  crypto::srs_storage::{load_kzh3_srs, store_kzh3_srs},
  crypto::MaskCommitter,
  dag::{gnn::gin, DagBuilder, DataType, Role, Witness},
  util::poly::{CryptoField, SelectionPolynomial},
  SF_LOG,
};

#[cfg(all(feature = "arkworks", feature = "bn254"))]
use zk_torch_2::crypto::polycommit::ArkBn254 as PairingType;
#[cfg(all(feature = "icicle", feature = "bn254"))]
use zk_torch_2::crypto::polycommit::IcicleBn254 as PairingType;

// GIN hyperparameters
const NUM_NODES_LOG: usize = 5;
const NUM_NODES: usize = 1 << NUM_NODES_LOG; // 32 nodes
const NUM_EDGES: usize = 128;
const INPUT_DIM: usize = 8;
const HIDDEN_DIM: usize = 8;
const NUM_LAYERS: usize = 1;

fn generate_random_field_vec(size: usize) -> Vec<F> {
  let mut rng = rand::thread_rng();
  (0..size).map(|_| <F as CryptoField>::from_u32(rng.gen::<u32>() % 500)).collect()
}

/// Generate random directed edges (no self-loops).
fn generate_random_edges(num_nodes: usize, num_edges: usize) -> Vec<(usize, usize)> {
  let mut rng = rand::thread_rng();
  let mut edge_set = HashSet::new();

  while edge_set.len() < num_edges {
    let src = rng.gen::<usize>() % num_nodes;
    let dst = rng.gen::<usize>() % num_nodes;
    if src != dst {
      edge_set.insert((src, dst));
    }
  }

  edge_set.into_iter().collect()
}

/// Create a binary sparse adjacency matrix (all weights = 1, sf = 0).
fn create_binary_adjacency(num_nodes_log: usize, edges: &[(usize, usize)]) -> Witness<F> {
  let selection_poly = SelectionPolynomial::<F>::new(
    num_nodes_log, // input_num_vars (source node index)
    num_nodes_log, // table_num_vars (dest node index)
    edges.to_vec(),
  );

  // to_sparse() sets all evaluations to 1 (binary)
  let sparse_poly = selection_poly.to_sparse();

  let num_nodes = 1 << num_nodes_log;
  Witness::new_sparse(
    vec![num_nodes, num_nodes],
    sparse_poly,
    DataType::Float,
    0, // sf = 0 for binary adjacency
    Role::Constant,
  )
}

fn main() {
  let mut timing = TimingTree::default();
  env_logger::init();

  let args: Vec<String> = std::env::args().collect();
  let zk_mode = args.iter().any(|a| a == "--zk");

  println!("usize bits {}", usize::BITS);

  #[cfg(feature = "arkworks")]
  println!("using arkworks");
  #[cfg(feature = "icicle")]
  println!("using icicle");

  let thread_num = rayon::current_num_threads();
  println!("using {} threads", thread_num);

  println!("Generating GIN with binary adjacency matrix...");
  println!("  NUM_NODES: {} (2^{})", NUM_NODES, NUM_NODES_LOG);
  println!("  NUM_EDGES: {}", NUM_EDGES);
  println!("  INPUT_DIM: {}", INPUT_DIM);
  println!("  HIDDEN_DIM: {}", HIDDEN_DIM);
  println!("  NUM_LAYERS: {}", NUM_LAYERS);

  // Generate random edges (binary, no normalization)
  let edges = generate_random_edges(NUM_NODES, NUM_EDGES);
  println!("Generated {} random directed edges", edges.len());

  // Create binary adjacency matrix (sf = 0)
  let mut adjacency = create_binary_adjacency(NUM_NODES_LOG, &edges);
  let num_shares = 2;
  adjacency.additive_factorize(num_shares).expect("Adjacency factorization failed");
  let af = adjacency.additive_factored.as_ref().unwrap();
  println!("Adjacency decomposed: {} terms, {} shares, chunk_sizes={:?}",
    af.terms.len(), af.chunk_sizes.len(), af.chunk_sizes);

  // Generate MLP weight matrices for each GIN layer
  // Each layer has a 2-layer MLP: W1 (d_in → d_hidden) and W2 (d_hidden → d_out)
  let mut weights_1 = Vec::new();
  let mut weights_2 = Vec::new();
  for i in 0..NUM_LAYERS {
    let in_dim = if i == 0 { INPUT_DIM } else { HIDDEN_DIM };
    let out_dim = HIDDEN_DIM;

    // W1: (in_dim, hidden_dim)
    weights_1.push(Witness::new(
      vec![in_dim, HIDDEN_DIM],
      generate_random_field_vec(in_dim * HIDDEN_DIM),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));

    // W2: (hidden_dim, out_dim)
    weights_2.push(Witness::new(
      vec![HIDDEN_DIM, out_dim],
      generate_random_field_vec(HIDDEN_DIM * out_dim),
      DataType::Float,
      *SF_LOG as usize,
      Role::Constant,
    ));
  }

  // --- Circuit compilation ---
  let mut g = DagBuilder::new();

  // Input: node features (num_nodes, input_dim)
  let x = g.input(vec![NUM_NODES, INPUT_DIM], DataType::Float);

  // Run GIN model
  let output = g.pipe(
    &[x],
    gin(adjacency, weights_1, weights_2),
  )[0];

  // Compile -> (Dag, initial edge values)
  println!("Compiling DAG...");
  let (dag, mut init) = g.compile();

  // --- Prover ---
  let mut transcript = Transcript::new(b"zkml");

  let mut dense_commitments: Vec<Option<KZH3Commitment<PairingType>>> = vec![None; dag.num_edges()];
  let mut sparse_commitments: Vec<Option<Vec<KZH3Commitment<PairingType>>>> = vec![None; dag.num_edges()];
  let mut factored_commitments: Vec<Option<Vec<KZH3Commitment<PairingType>>>> = vec![None; dag.num_edges()];

  // Witness generation from input
  println!("Generating witness from random input...");
  let input = Witness::new(
    vec![NUM_NODES, INPUT_DIM],
    generate_random_field_vec(NUM_NODES * INPUT_DIM),
    DataType::Float,
    *SF_LOG as usize,
    Role::Input,
  );

  let presplit_sparse = dag.run(&mut init, &vec![(x, input)]);
  println!("GIN output shape: {:?}", init[output][0].shape);

  // Collect polynomial sizes from the DAG after witness generation
  let polynomial_sizes = dag.collect_polynomial_sizes(&init);
  println!("\n=== Polynomial sizes in GIN DAG ===");
  println!("Sizes needed: {:?}", polynomial_sizes);
  println!("Number of different sizes: {}", polynomial_sizes.len());
  println!("====================================\n");

  // Load or generate SRS for all required polynomial sizes
  let mut srs_map = HashMap::new();
  for &size in &polynomial_sizes {
    let size_srs = if fs::metadata(&format!("{}.srs", size)).is_ok() {
      println!("Loading existing SRS for polynomial size {}", size);
      load_kzh3_srs(size).expect(&format!("Failed to load SRS for size {}", size))
    } else {
      println!("Generating SRS for polynomial size {}", size);
      #[cfg(feature = "arkworks")]
      let size_srs = {
        use rand::thread_rng;
        setup_kzh3_srs::<PairingType, _>(size, &mut thread_rng())
      };
      #[cfg(feature = "icicle")]
      let size_srs = setup_kzh3_srs::<PairingType, _>(size, &mut ());
      store_kzh3_srs(&size_srs, size).expect(&format!("Failed to store SRS for size {}", size));
      size_srs
    };
    srs_map.insert(size, Arc::new(size_srs));
  }
  // In ZK mode, add SRS entries for small mask polynomial sizes (1-8 vars)
  if zk_mode {
    for mask_n in 1..=8usize {
      if !srs_map.contains_key(&mask_n) {
        let mask_srs = if fs::metadata(&format!("{}.srs", mask_n)).is_ok() {
          println!("Loading SRS for mask size {}", mask_n);
          load_kzh3_srs(mask_n).expect("Failed to load mask SRS")
        } else {
          println!("Generating SRS for mask size {}", mask_n);
          #[cfg(feature = "arkworks")]
          let mask_srs = { use rand::thread_rng; setup_kzh3_srs::<PairingType, _>(mask_n, &mut thread_rng()) };
          #[cfg(feature = "icicle")]
          let mask_srs = setup_kzh3_srs::<PairingType, _>(mask_n, &mut ());
          store_kzh3_srs(&mask_srs, mask_n).expect("Failed to store mask SRS");
          mask_srs
        };
        srs_map.insert(mask_n, Arc::new(mask_srs));
      }
    }
  }
  let srs_map = Arc::new(srs_map);

  let mask_committer: Option<Arc<dyn MaskCommitter<F>>> = if zk_mode {
    Some(Arc::new(KZH3MaskCommitter::<PairingType>::new(srs_map.clone())))
  } else {
    None
  };

  let h_point = if zk_mode {
    println!("ZK mode enabled: generating hiding generator h");
    use zk_torch_2::crypto::polycommit::PairingTrait;
    use ark_std::UniformRand;
    let mut rng = ark_std::rand::thread_rng();
    Some(<PairingType as PairingTrait>::G1Affine::rand(&mut rng))
  } else {
    None
  };

  // Create commitment keys with the collected SRS sizes
  let kzh3 = KZH3CommitKey::<PairingType> { srs_map: srs_map.clone(), h: h_point };
  let sparse_kzh3 = SparseKZH3CommitKey::<PairingType> { srs_map: srs_map.clone(), h: h_point };

  // Create verifier keys with the same srs_map
  let dense_verifier_key = KZH3VerifierKey { srs_map: srs_map.clone(), h: h_point };
  let sparse_verifier_key = SparseKZH3VerifierKey { srs_map: srs_map.clone(), h: h_point };

  // Commit to all witnesses
  println!("Committing to witnesses...");
  timed!(timing, "commit", {
    dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &kzh3,
      &sparse_kzh3,
      &init,
      &mut dense_commitments,
      &mut sparse_commitments,
      &mut factored_commitments,
      &mut timing,
    )
  });

  println!("Generating proof{}...", if zk_mode { " (ZK)" } else { "" });
  let (sumcheck_proofs, opening_proofs, range_proof, two_pow_proof, reducer_proofs, zk_proof) = timed!(
    timing,
    "prove",
    dag.prove_zk::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &kzh3,
      &sparse_kzh3,
      &init,
      &presplit_sparse,
      &dense_commitments,
      &sparse_commitments,
      &factored_commitments,
      &mut transcript,
      &mut timing,
      zk_mode,
      mask_committer.clone(),
    )
  );

  // Clear the data from the witnesses before verification
  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));

  // --- Verifier ---
  let mut verifier_transcript = Transcript::new(b"zkml");
  let verified = timed!(
    timing,
    "verify",
    dag.verify_zk::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &sumcheck_proofs,
      &opening_proofs,
      &range_proof,
      &two_pow_proof,
      &reducer_proofs,
      &init,
      &dense_verifier_key,
      &sparse_verifier_key,
      &dense_commitments,
      &sparse_commitments,
      &factored_commitments,
      &mut verifier_transcript,
      zk_mode,
      &zk_proof,
      mask_committer.clone(),
    )
  );
  timing.print();
  println!("verified: {:?}", verified);
}

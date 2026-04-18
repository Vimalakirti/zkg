use std::time::Instant;
use zk_torch_2::crypto::polycommit::kzh3::{
  kzh3_commit, kzh3_commit_zk, kzh3_open, kzh3_open_zk, kzh3_verify, kzh3_verify_zk,
  scalar_field_zero, setup_kzh3_srs, split_input,
};
use zk_torch_2::crypto::polycommit::kzh3::rand_sparse_poly;
use zk_torch_2::crypto::polycommit::sparse_kzh3::{
  sparse_kzh3_commit, sparse_kzh3_commit_zk, sparse_kzh3_open, sparse_kzh3_open_zk,
  sparse_kzh3_verify,
};
use zk_torch_2::util::poly::{DenseMLPoly, MLPoly};

#[cfg(feature = "arkworks")]
use ark_std::UniformRand;

#[cfg(feature = "arkworks")]
type PairingType = ark_bn254::Bn254;

#[cfg(feature = "arkworks")]
type Fr = ark_bn254::Fr;

struct BenchResult {
  n: usize,
  commit_ns: u128,
  open_ns: u128,
  verify_ns: u128,
}

#[cfg(feature = "arkworks")]
fn bench_non_zk(n: usize) -> BenchResult {
  use ark_std::rand::thread_rng;
  let mut rng = thread_rng();

  // Setup
  let srs = setup_kzh3_srs::<PairingType, _>(n, &mut rng);

  // Random polynomial
  let evals: Vec<Fr> = (0..(1 << n)).map(|_| Fr::rand(&mut rng)).collect();
  let poly = DenseMLPoly::new(n, evals);

  // Random evaluation point
  let point: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

  // Commit
  let start = Instant::now();
  let (com, aux) = kzh3_commit::<PairingType>(&srs, &poly);
  let commit_ns = start.elapsed().as_nanos();

  // Open
  let start = Instant::now();
  let proof = kzh3_open::<PairingType>(&srs, &point, &com, &aux, &poly);
  let open_ns = start.elapsed().as_nanos();

  // Verify
  let split_r = split_input(&srs, &point, scalar_field_zero::<Fr>());
  let r_z = &split_r[0];
  let claimed_eval = proof.f_star.evaluate_at_point(r_z);
  let start = Instant::now();
  let ok = kzh3_verify::<PairingType>(&srs, &point, &claimed_eval, &com, &proof);
  let verify_ns = start.elapsed().as_nanos();
  assert!(ok, "Non-ZK verification failed for n={}", n);

  BenchResult { n, commit_ns, open_ns, verify_ns }
}

#[cfg(feature = "arkworks")]
fn bench_zk(n: usize) -> BenchResult {
  use ark_std::rand::thread_rng;
  let mut rng = thread_rng();

  // Setup
  let srs = setup_kzh3_srs::<PairingType, _>(n, &mut rng);

  // Generate hiding generator h
  let h = <ark_bn254::G1Affine as ark_std::UniformRand>::rand(&mut rng);

  // Random polynomial
  let evals: Vec<Fr> = (0..(1 << n)).map(|_| Fr::rand(&mut rng)).collect();
  let poly = DenseMLPoly::new(n, evals);

  // Random evaluation point
  let point: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

  // Commit (ZK)
  let start = Instant::now();
  let (com, aux) = kzh3_commit_zk::<PairingType>(&srs, &poly, &h);
  let commit_ns = start.elapsed().as_nanos();

  // Open (ZK)
  let start = Instant::now();
  let proof = kzh3_open_zk::<PairingType>(&srs, &point, &com, &aux, &poly, &h);
  let open_ns = start.elapsed().as_nanos();

  // Verify (ZK)
  let eval = poly.evaluate_at_point(&point);
  let start = Instant::now();
  let ok = kzh3_verify_zk::<PairingType>(&srs, &point, &eval, &com, &proof, &h);
  let verify_ns = start.elapsed().as_nanos();
  assert!(ok, "ZK verification failed for n={}", n);

  BenchResult { n, commit_ns, open_ns, verify_ns }
}

#[cfg(feature = "arkworks")]
fn bench_sparse_non_zk(n: usize, sparsity_pct: f64) -> BenchResult {
  use ark_std::rand::thread_rng;
  let mut rng = thread_rng();

  let srs = setup_kzh3_srs::<PairingType, _>(n, &mut rng);

  // Random sparse polynomial using the proper constructor
  let total = 1usize << n;
  let nnz = ((total as f64) * sparsity_pct).max(1.0) as usize;
  let poly = rand_sparse_poly::<Fr>(n, nnz);

  let point: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

  let start = Instant::now();
  let (com, aux) = sparse_kzh3_commit::<PairingType>(&srs, &poly);
  let commit_ns = start.elapsed().as_nanos();

  let start = Instant::now();
  let proof = sparse_kzh3_open::<PairingType>(&srs, &point, &com, &aux, &poly);
  let open_ns = start.elapsed().as_nanos();

  let split_r = split_input(&srs, &point, scalar_field_zero::<Fr>());
  let r_z = &split_r[0];
  let claimed_eval = proof.f_star.evaluate_at_point(r_z);
  let start = Instant::now();
  let ok = sparse_kzh3_verify::<PairingType>(&srs, &point, &claimed_eval, &com, &proof);
  let verify_ns = start.elapsed().as_nanos();
  assert!(ok, "Sparse non-ZK verification failed for n={}", n);

  BenchResult { n, commit_ns, open_ns, verify_ns }
}

#[cfg(feature = "arkworks")]
fn bench_sparse_zk(n: usize, sparsity_pct: f64) -> BenchResult {
  use ark_std::rand::thread_rng;
  let mut rng = thread_rng();

  let srs = setup_kzh3_srs::<PairingType, _>(n, &mut rng);
  let h = <ark_bn254::G1Affine as ark_std::UniformRand>::rand(&mut rng);

  // Random sparse polynomial using the proper constructor
  let total = 1usize << n;
  let nnz = ((total as f64) * sparsity_pct).max(1.0) as usize;
  let poly = rand_sparse_poly::<Fr>(n, nnz);

  let point: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

  let start = Instant::now();
  let (com, aux) = sparse_kzh3_commit_zk::<PairingType>(&srs, &poly, &h);
  let commit_ns = start.elapsed().as_nanos();

  let start = Instant::now();
  let proof = sparse_kzh3_open_zk::<PairingType>(&srs, &point, &com, &aux, &poly, &h);
  let open_ns = start.elapsed().as_nanos();

  let eval = poly.evaluate_at_point(&point);
  let start = Instant::now();
  let ok = kzh3_verify_zk::<PairingType>(&srs, &point, &eval, &com, &proof, &h);
  let verify_ns = start.elapsed().as_nanos();
  assert!(ok, "Sparse ZK verification failed for n={}", n);

  BenchResult { n, commit_ns, open_ns, verify_ns }
}

fn ns_to_ms(ns: u128) -> f64 {
  ns as f64 / 1_000_000.0
}

fn overhead_pct(base: u128, zk: u128) -> f64 {
  if base == 0 { return 0.0; }
  ((zk as f64 - base as f64) / base as f64) * 100.0
}

#[cfg(feature = "arkworks")]
fn main() {
  let sizes = vec![10, 12, 14, 16, 18];

  println!("ZK PCS Overhead Benchmark (KZH3)");
  println!("=================================\n");
  println!(
    "{:>5} | {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10} | {:>8} {:>8} {:>8}",
    "n", "Commit", "Open", "Verify",
    "ZK Commit", "ZK Open", "ZK Verify",
    "C %", "O %", "V %"
  );
  println!("{}", "-".repeat(120));

  let mut results: Vec<(BenchResult, BenchResult)> = Vec::new();

  for &n in &sizes {
    eprintln!("Benchmarking n={}...", n);

    let non_zk = bench_non_zk(n);
    let zk = bench_zk(n);

    println!(
      "{:>5} | {:>9.2}ms {:>9.2}ms {:>9.2}ms | {:>9.2}ms {:>9.2}ms {:>9.2}ms | {:>7.1}% {:>7.1}% {:>7.1}%",
      n,
      ns_to_ms(non_zk.commit_ns), ns_to_ms(non_zk.open_ns), ns_to_ms(non_zk.verify_ns),
      ns_to_ms(zk.commit_ns), ns_to_ms(zk.open_ns), ns_to_ms(zk.verify_ns),
      overhead_pct(non_zk.commit_ns, zk.commit_ns),
      overhead_pct(non_zk.open_ns, zk.open_ns),
      overhead_pct(non_zk.verify_ns, zk.verify_ns),
    );

    results.push((non_zk, zk));
  }

  // --- Sparse KZH3 benchmarks ---
  let sparsity_pct = 0.1; // 10% nonzero entries
  println!("\n\nSparse ZK PCS Overhead Benchmark (KZH3, {:.0}% nonzero)", sparsity_pct * 100.0);
  println!("=====================================================\n");
  println!(
    "{:>5} | {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10} | {:>8} {:>8} {:>8}",
    "n", "Commit", "Open", "Verify",
    "ZK Commit", "ZK Open", "ZK Verify",
    "C %", "O %", "V %"
  );
  println!("{}", "-".repeat(120));

  let mut sparse_results: Vec<(BenchResult, BenchResult)> = Vec::new();

  for &n in &sizes {
    eprintln!("Benchmarking sparse n={}...", n);

    let non_zk = bench_sparse_non_zk(n, sparsity_pct);
    let zk = bench_sparse_zk(n, sparsity_pct);

    println!(
      "{:>5} | {:>9.2}ms {:>9.2}ms {:>9.2}ms | {:>9.2}ms {:>9.2}ms {:>9.2}ms | {:>7.1}% {:>7.1}% {:>7.1}%",
      n,
      ns_to_ms(non_zk.commit_ns), ns_to_ms(non_zk.open_ns), ns_to_ms(non_zk.verify_ns),
      ns_to_ms(zk.commit_ns), ns_to_ms(zk.open_ns), ns_to_ms(zk.verify_ns),
      overhead_pct(non_zk.commit_ns, zk.commit_ns),
      overhead_pct(non_zk.open_ns, zk.open_ns),
      overhead_pct(non_zk.verify_ns, zk.verify_ns),
    );

    sparse_results.push((non_zk, zk));
  }

  // Write markdown report
  let mut md = String::new();
  md.push_str("# ZK PCS Overhead Benchmark (KZH3)\n\n");

  // Dense section
  md.push_str("## Dense Polynomial\n\n");
  md.push_str("Comparison of non-ZK vs ZK modes for KZH3 with dense polynomials.\n\n");
  md.push_str("| Poly Size (n) | Entries | Commit (ms) | ZK Commit (ms) | Overhead | Open (ms) | ZK Open (ms) | Overhead | Verify (ms) | ZK Verify (ms) | Overhead |\n");
  md.push_str("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n");

  for (non_zk, zk) in &results {
    md.push_str(&format!(
      "| {} | {} | {:.2} | {:.2} | {:.1}% | {:.2} | {:.2} | {:.1}% | {:.2} | {:.2} | {:.1}% |\n",
      non_zk.n,
      1u64 << non_zk.n,
      ns_to_ms(non_zk.commit_ns),
      ns_to_ms(zk.commit_ns),
      overhead_pct(non_zk.commit_ns, zk.commit_ns),
      ns_to_ms(non_zk.open_ns),
      ns_to_ms(zk.open_ns),
      overhead_pct(non_zk.open_ns, zk.open_ns),
      ns_to_ms(non_zk.verify_ns),
      ns_to_ms(zk.verify_ns),
      overhead_pct(non_zk.verify_ns, zk.verify_ns),
    ));
  }

  // Sparse section
  md.push_str(&format!("\n## Sparse Polynomial ({:.0}% nonzero)\n\n", sparsity_pct * 100.0));
  md.push_str("Comparison of non-ZK vs ZK modes for KZH3 with sparse polynomials.\n\n");
  md.push_str("| Poly Size (n) | Entries | Commit (ms) | ZK Commit (ms) | Overhead | Open (ms) | ZK Open (ms) | Overhead | Verify (ms) | ZK Verify (ms) | Overhead |\n");
  md.push_str("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n");

  for (non_zk, zk) in &sparse_results {
    md.push_str(&format!(
      "| {} | {} | {:.2} | {:.2} | {:.1}% | {:.2} | {:.2} | {:.1}% | {:.2} | {:.2} | {:.1}% |\n",
      non_zk.n,
      1u64 << non_zk.n,
      ns_to_ms(non_zk.commit_ns),
      ns_to_ms(zk.commit_ns),
      overhead_pct(non_zk.commit_ns, zk.commit_ns),
      ns_to_ms(non_zk.open_ns),
      ns_to_ms(zk.open_ns),
      overhead_pct(non_zk.open_ns, zk.open_ns),
      ns_to_ms(non_zk.verify_ns),
      ns_to_ms(zk.verify_ns),
      overhead_pct(non_zk.verify_ns, zk.verify_ns),
    ));
  }

  std::fs::create_dir_all("benchmarks").ok();
  std::fs::write("benchmarks/zk_pcs_overhead.md", &md).expect("Failed to write benchmark results");
  eprintln!("\nResults written to benchmarks/zk_pcs_overhead.md");
}

#[cfg(not(feature = "arkworks"))]
fn main() {
  eprintln!("ZK PCS benchmark requires the 'arkworks' feature. Run with: cargo run --bin bench_zk_pcs --features arkworks,bn254");
}

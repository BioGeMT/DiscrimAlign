"""
Benchmark discrimalign's backends on this machine.

Run from the repository root:

    python benchmarks/bench_backends.py                 # all workloads
    python benchmarks/bench_backends.py --quick         # smaller and faster
    python benchmarks/bench_backends.py --fingerprint   # hashes for cross-machine checks

For each workload it times one subgradient iteration, split into the alignment
(and scores), the intercept fit, the gradient and the rest, for the nwgrad
backend at several thread counts and for the Biopython backend on one thread
(its threads contend for the GIL). It also times the initial estimate.
Results are printed as JSON lines (or written with --json) and summarized in a
table at the end.

All inputs are generated with Python's own random module, which produces the
same numbers on every platform, so --fingerprint hashes from different
machines can be compared directly: nwgrad's results are meant to be
bit-identical across CPUs and instruction sets.
"""
import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices
from scipy.optimize import minimize

import nwgrad
from src.discrimalign import _BiopythonEngine, _align_pairs, discrimalign
from src.logit_link import logit_logL, logit_partial_scores
from src.nwgrad_backend import NwgradEngine, baseline_parameters
from src.optimization import (create_powerstep, get_initial_estimate,
                              get_initial_estimate_from_counts)

DNA = "ACGT"
PROTEIN = "ACDEFGHIKLMNPQRSTVWY"

# name: (alphabet, mode, gap model, substitution mode, pair maker, default pair count)
WORKLOADS = {
    "mirna-local": (DNA, "local", "affine", "general", "mirna", 10000),
    "mirna-global": (DNA, "global", "linear", "simple", "mirna", 10000),
    "protein": (PROTEIN, "local", "affine", "general", "protein", 2000),
}


# --- portable inputs ----------------------------------------------------------

def random_seq(rng, length, alphabet):
    return "".join(rng.choice(alphabet) for _ in range(length))


def mutate(rng, seq, alphabet, sub_rate, indel_rate):
    out = []
    for char in seq:
        u = rng.random()
        if u < indel_rate / 2:
            continue
        if u < indel_rate:
            out.append(rng.choice(alphabet))
        if rng.random() < sub_rate:
            char = rng.choice([c for c in alphabet if c != char])
        out.append(char)
    return "".join(out) or seq[0]


def mirna_pairs(rng, n):
    """22-nt queries against 50-nt targets; every other target hides a mutated copy."""
    A, B, y = [], [], []
    for k in range(n):
        query, target = random_seq(rng, 22, DNA), random_seq(rng, 50, DNA)
        if k % 2 == 0:
            site = mutate(rng, query, DNA, 0.15, 0.05)
            pos = rng.randrange(0, 50 - len(site))
            target = target[:pos] + site + target[pos + len(site):]
        A.append(query)
        B.append(target[:50])
        y.append(1 - k % 2)
    return A, B, np.array(y)


def protein_pairs(rng, n, length=300):
    """Mutated homologs and unrelated pairs of 300 residues."""
    A, B, y = [], [], []
    for k in range(n):
        seq = random_seq(rng, length, PROTEIN)
        A.append(seq)
        B.append(mutate(rng, seq, PROTEIN, 0.3, 0.05) if k % 2 == 0 else random_seq(rng, length, PROTEIN))
        y.append(1 - k % 2)
    return A, B, np.array(y)


def random_params(rng, gap_mode, substitution_mode, alphabet):
    """Non-integer parameters, so alignments with different counts do not tie."""
    params = {"alpha": rng.uniform(-1.5, -0.5)}
    if gap_mode == "affine":
        params["open_gap_score"] = rng.uniform(-4.0, -2.5)
        params["extend_gap_score"] = rng.uniform(-1.5, -0.3)
    else:
        params["gap_score"] = rng.uniform(-3.0, -1.0)
    if substitution_mode == "simple":
        params["match_score"] = rng.uniform(1.5, 3.0)
        params["mismatch_score"] = rng.uniform(-2.0, -0.5)
    else:
        n = len(alphabet)
        data = np.array([[rng.uniform(1.5, 3.0) if i == j else rng.uniform(-2.0, -0.5)
                          for j in range(n)] for i in range(n)])
        if substitution_mode == "symmetric":
            data = (data + data.T) / 2
        params["substitution_matrix"] = substitution_matrices.Array(alphabet=alphabet, data=data)
    return params


def make_workload(name, n_pairs, seed=0):
    alphabet, mode, gap_mode, substitution_mode, maker, _ = WORKLOADS[name]
    rng = random.Random(seed)
    A, B, y = (mirna_pairs if maker == "mirna" else protein_pairs)(rng, n_pairs)
    params = random_params(random.Random(seed + 1), gap_mode, substitution_mode, alphabet)
    return A, B, y, params, alphabet, mode, gap_mode, substitution_mode


def default_baseline(mode, gap_mode, substitution_mode, alphabet):
    """discrimalign()'s default baseline aligner."""
    aligner = PairwiseAligner()
    aligner.mode = mode
    if gap_mode == "affine":
        aligner.open_gap_score, aligner.extend_gap_score = -8, -0.5
    else:
        aligner.gap_score = -6
    if substitution_mode == "simple":
        aligner.match_score, aligner.mismatch_score = 5, -4
    else:
        aligner.substitution_matrix = substitution_matrices.Array(
            data=9 * np.eye(len(alphabet)) - 4, alphabet=alphabet)
    return aligner


def make_aligner(mode, params):
    aligner = PairwiseAligner()
    aligner.mode = mode
    if "gap_score" in params:
        aligner.gap_score = params["gap_score"]
    else:
        aligner.open_gap_score = params["open_gap_score"]
        aligner.extend_gap_score = params["extend_gap_score"]
    if "substitution_matrix" in params:
        aligner.substitution_matrix = params["substitution_matrix"]
    else:
        aligner.match_score = params["match_score"]
        aligner.mismatch_score = params["mismatch_score"]
    return aligner


# --- timing -------------------------------------------------------------------

def fit_alpha(scores, labels, alpha0):
    target = lambda a: -logit_logL(logit_partial_scores(scores, a), labels)
    fprime = lambda a: -np.sum(labels - logit_partial_scores(scores, a))
    return minimize(target, alpha0, jac=fprime)["x"][0]


def time_iterations(engine, params, labels, reps):
    """Median seconds per part of one iteration, after one warm-up iteration."""
    parts = {k: [] for k in ("align", "alpha", "grad", "other")}
    for rep in range(reps + 1):
        t0 = time.perf_counter()
        engine.set_params(params)
        scores = engine.scores()
        t1 = time.perf_counter()
        logit_logL(logit_partial_scores(scores, params["alpha"]), labels)
        t2 = time.perf_counter()
        alpha = fit_alpha(scores, labels, params["alpha"])
        t3 = time.perf_counter()
        engine.raw_subgradient(logit_partial_scores(scores, alpha), labels, alpha)
        t4 = time.perf_counter()
        if rep:
            parts["align"].append(t1 - t0)
            parts["other"].append(t2 - t1)
            parts["alpha"].append(t3 - t2)
            parts["grad"].append(t4 - t3)
    out = {k: statistics.median(v) for k, v in parts.items()}
    out["total"] = sum(out.values())
    return out


def bench_workload(name, n_pairs, threads, reps, biopython, emit):
    A, B, y, params, alphabet, mode, gap_mode, substitution_mode = make_workload(name, n_pairs)
    for t in threads:
        engine = NwgradEngine(A, B, mode, gap_mode, substitution_mode, alphabet, t)
        emit({"kind": "iteration", "workload": name, "pairs": n_pairs, "backend": "nwgrad",
              "threads": t, **time_iterations(engine, params, y, reps)})
    if biopython:
        engine = _BiopythonEngine(A, B, make_aligner(mode, params), gap_mode,
                                  substitution_mode, alphabet, 1)
        emit({"kind": "iteration", "workload": name, "pairs": n_pairs, "backend": "biopython",
              "threads": 1, **time_iterations(engine, params, y, max(1, reps // 2))})
    baseline = default_baseline(mode, gap_mode, substitution_mode, alphabet)
    t0 = time.perf_counter()
    if biopython:
        get_initial_estimate(_align_pairs(A, B, baseline, 1), y, substitution_mode, gap_mode, alphabet)
    t1 = time.perf_counter()
    engine = NwgradEngine(A, B, mode, gap_mode, substitution_mode, alphabet, 1)
    engine.set_params(baseline_parameters(baseline, gap_mode, alphabet), substitution_mode="general")
    engine.scores()
    get_initial_estimate_from_counts(engine.raw_counts(), y, substitution_mode, gap_mode, alphabet)
    t2 = time.perf_counter()
    emit({"kind": "initial_estimate", "workload": name, "pairs": n_pairs, "threads": 1,
          "biopython": (t1 - t0) if biopython else None, "nwgrad": t2 - t1})


# --- fingerprints -------------------------------------------------------------

def sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()[:16]


def fingerprints(emit):
    """
    Hashes of nwgrad's scores and weighted gradient, and of a 10-iteration fit
    (fitted parameters and intercept), from fixed portable inputs. The fit's
    reported log-likelihood is left out: numpy may sum it in a different order
    on different CPUs.
    """
    for name in ("mirna-local", "mirna-global"):
        A, B, y, params, alphabet, mode, gap_mode, substitution_mode = make_workload(name, 2000, seed=7)
        record = {"kind": "fingerprint", "workload": name}
        weights = np.array([2 * k / (len(A) - 1) - 1 for k in range(len(A))])
        for t in (1, 4):
            engine = NwgradEngine(A, B, mode, gap_mode, substitution_mode, alphabet, t)
            engine.set_params(params)
            scores = engine.scores()
            g = engine.batch.weighted_grad(weights).to_dict()
            record[f"engine_threads_{t}"] = sha(scores, g["matrix"],
                                                [g[k] for k in ("gap_open_a", "gap_extend_a",
                                                                "gap_open_b", "gap_extend_b")])
        result = discrimalign(A, B, y, aligner_mode=mode, gap_mode=gap_mode,
                              substitution_mode=substitution_mode, initial_parameters=params,
                              max_iter=10, stepfunction=create_powerstep(1e-3),
                              return_alignments=False, backend="nwgrad", num_threads=1)
        keys = [k for k in ("substitution_matrix", "match_score", "mismatch_score",
                            "open_gap_score", "extend_gap_score", "gap_score") if k in result]
        record["fit"] = sha(*[result[k] for k in keys], result["alpha"])
        emit(record)


# --- main ---------------------------------------------------------------------

def default_threads():
    n = os.cpu_count() or 1
    threads, t = [], 1
    while t < n:
        threads.append(t)
        t *= 2
    return threads + [n, 0]


def summarize(records):
    lines = []
    for name in WORKLOADS:
        rows = [r for r in records if r["kind"] == "iteration" and r["workload"] == name]
        if not rows:
            continue
        bio = next((r for r in rows if r["backend"] == "biopython"), None)
        lines.append(f"\n{name}, {rows[0]['pairs']} pairs: seconds per iteration")
        lines.append(f"  {'backend':10s} {'threads':>7s} {'align':>8s} {'alpha':>8s} {'grad':>8s} "
                     f"{'other':>8s} {'total':>8s} {'vs Biopython':>13s}")
        for r in rows:
            label = "auto" if r["threads"] == 0 else str(r["threads"])
            speedup = f"{bio['total'] / r['total']:.1f}x" if bio else "-"
            lines.append(f"  {r['backend']:10s} {label:>7s} {r['align']:8.3f} {r['alpha']:8.3f} "
                         f"{r['grad']:8.3f} {r['other']:8.3f} {r['total']:8.3f} {speedup:>13s}")
        for r in records:
            if r["kind"] == "initial_estimate" and r["workload"] == name:
                bio_s = f"{r['biopython']:.2f} s" if r["biopython"] is not None else "-"
                lines.append(f"  initial estimate (1 thread): Biopython {bio_s}, nwgrad {r['nwgrad']:.2f} s")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--workloads", default=",".join(WORKLOADS),
                        help=f"comma-separated subset of: {', '.join(WORKLOADS)}")
    parser.add_argument("--pairs", type=int, help="pair count for every workload "
                        "(default: 10000 for miRNA-sized pairs, 2000 for proteins)")
    parser.add_argument("--threads", help="comma-separated nwgrad thread counts; 0 means "
                        "automatic (default: powers of 2 up to os.cpu_count(), that count, and 0)")
    parser.add_argument("--reps", type=int, default=5, help="timed iterations per measurement")
    parser.add_argument("--quick", action="store_true", help="a tenth of the pairs, 2 repetitions")
    parser.add_argument("--no-biopython", action="store_true", help="skip the Biopython backend")
    parser.add_argument("--fingerprint", action="store_true",
                        help="only print fingerprints for comparing results across machines")
    parser.add_argument("--json", help="also write the records to this file")
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    records = []
    sink = open(args.json, "w") if args.json else None

    def emit(record):
        records.append(record)
        line = json.dumps(record)
        print(line, flush=True)
        if sink:
            sink.write(line + "\n")
            sink.flush()

    emit({"kind": "environment", "cpu_count": os.cpu_count(), "machine": platform.machine(),
          "system": platform.system(), "python": platform.python_version(),
          "nwgrad_isa": nwgrad.simd_isa(), "nwgrad_compiled_with": nwgrad.compiled_with(),
          "numpy": np.__version__})
    if args.fingerprint:
        fingerprints(emit)
        return
    threads = [int(t) for t in args.threads.split(",")] if args.threads else default_threads()
    reps = 2 if args.quick else args.reps
    for name in args.workloads.split(","):
        n = args.pairs or WORKLOADS[name][5]
        if args.quick and not args.pairs:
            n //= 10
        bench_workload(name, n, threads, reps, not args.no_biopython, emit)
    print(summarize(records))
    if sink:
        sink.close()


if __name__ == "__main__":
    main()

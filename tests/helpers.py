"""
Shared helpers for the test suite: random sequence data, aligners built from
DiscrimAlign parameter dicts, and an independent reference for the
log-likelihood as a function of the alignment parameters.
"""
import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices
from scipy.optimize import brentq
from scipy.special import expit

from src.logit_link import logit_subgradient
from src.optimization import get_first_alignment

DNA = "ACGT"

MODES = ["local", "global"]
GAP_MODES = ["affine", "linear"]
SUBSTITUTION_MODES = ["simple", "symmetric", "general"]


def random_seq(rng, length, alphabet=DNA):
    return "".join(rng.choice(list(alphabet), size=length))


def mutate(rng, seq, sub_rate=0.1, indel_rate=0.05, alphabet=DNA):
    """Point substitutions plus single-residue insertions and deletions."""
    out = []
    for char in seq:
        u = rng.random()
        if u < indel_rate / 2:
            continue
        if u < indel_rate:
            out.append(rng.choice(list(alphabet)))
        if rng.random() < sub_rate:
            char = rng.choice([c for c in alphabet if c != char])
        out.append(char)
    if not out:
        out.append(seq[0])
    return "".join(out)


def make_pairs(rng, n_pos, n_neg, length, alphabet=DNA,
               sub_rate=0.1, indel_rate=0.05):
    """Positives are mutated copies, negatives are unrelated random sequences."""
    seqsA, seqsB, labels = [], [], []
    for _ in range(n_pos):
        seq = random_seq(rng, length, alphabet)
        seqsA.append(seq)
        seqsB.append(mutate(rng, seq, sub_rate, indel_rate, alphabet))
        labels.append(1)
    for _ in range(n_neg):
        seqsA.append(random_seq(rng, length, alphabet))
        seqsB.append(random_seq(rng, length, alphabet))
        labels.append(0)
    order = rng.permutation(len(labels))
    return ([seqsA[i] for i in order],
            [seqsB[i] for i in order],
            np.array([labels[i] for i in order]))


def random_params(rng, gap_mode, substitution_mode, alphabet=DNA, alpha=None):
    """
    Random, non-integer parameters in DiscrimAlign's dict format.

    Continuous values make ties between alignments with different count
    vectors a measure-zero event, so the log-likelihood is differentiable at
    these points. The extend score is always above the open score.
    """
    params = {"alpha": float(rng.normal(-1.0, 0.3)) if alpha is None else alpha}
    if gap_mode == "affine":
        params["open_gap_score"] = float(rng.uniform(-4.0, -2.5))
        params["extend_gap_score"] = float(rng.uniform(-1.5, -0.3))
    else:
        params["gap_score"] = float(rng.uniform(-3.0, -1.0))
    if substitution_mode == "simple":
        params["match_score"] = float(rng.uniform(1.5, 3.0))
        params["mismatch_score"] = float(rng.uniform(-2.0, -0.5))
    else:
        n = len(alphabet)
        data = rng.uniform(-2.0, -0.5, size=(n, n))
        data[np.diag_indices(n)] = rng.uniform(1.5, 3.0, size=n)
        if substitution_mode == "symmetric":
            data = (data + data.T) / 2
        params["substitution_matrix"] = substitution_matrices.Array(
            alphabet=alphabet, data=data)
    return params


def make_aligner(mode, params):
    """A PairwiseAligner configured from a DiscrimAlign parameter dict."""
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


def align_all(seqsA, seqsB, aligner):
    return [get_first_alignment(a, b, aligner) for a, b in zip(seqsA, seqsB)]


def scores_at(seqsA, seqsB, mode, params):
    aligner = make_aligner(mode, params)
    return np.array([aln.score for aln in align_all(seqsA, seqsB, aligner)])


def loglik_at(seqsA, seqsB, labels, mode, params):
    """
    Log-likelihood at params, with alpha held fixed at params['alpha'].

    Computed from the logits as sum(y*z - log(1 + e^z)), independently of
    logit_logL.
    """
    z = params["alpha"] + scores_at(seqsA, seqsB, mode, params)
    labels = np.asarray(labels, dtype=float)
    return float(np.sum(labels * z - np.logaddexp(0.0, z)))


def optimal_alpha(scores, labels, bracket=(-200.0, 200.0)):
    """Root of d logL / d alpha, by bracketing (independent of scipy.minimize)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return brentq(lambda a: np.sum(labels - expit(a + scores)), *bracket, xtol=1e-14)


def model_gradient(subgradient, gap_mode, substitution_mode):
    """
    Map logit_subgradient's raw counts onto the model's free parameters,
    in DiscrimAlign's dict format (without alpha).
    """
    grad = {}
    if gap_mode == "affine":
        grad["open_gap_score"] = subgradient["Gap opens"]
        grad["extend_gap_score"] = subgradient["Gap extends"]
    else:
        grad["gap_score"] = subgradient["Gap opens"] + subgradient["Gap extends"]
    G = np.asarray(subgradient["Substitutions"])
    if substitution_mode == "simple":
        grad["match_score"] = np.trace(G)
        grad["mismatch_score"] = G.sum() - np.trace(G)
    elif substitution_mode == "symmetric":
        grad["substitution_matrix"] = G + G.T - np.diag(np.diag(G))
    else:
        grad["substitution_matrix"] = G
    return grad


def subgradient_at(seqsA, seqsB, labels, mode, params, alphabet):
    aligner = make_aligner(mode, params)
    alns = align_all(seqsA, seqsB, aligner)
    scores = np.array([aln.score for aln in alns])
    logit_scores = expit(params["alpha"] + scores)
    return logit_subgradient(alns, logit_scores, labels, params["alpha"], alphabet)


def perturbed(params, key, delta, index=None, symmetric=False):
    """Copy of params with one scalar (or one matrix entry) shifted by delta."""
    new = dict(params)
    if index is None:
        new[key] = params[key] + delta
        return new
    M = substitution_matrices.Array(alphabet=params[key].alphabet,
                                    data=np.array(params[key]))
    i, j = index
    M[i, j] += delta
    if symmetric and i != j:
        M[j, i] += delta
    new[key] = M
    return new


# Edge-case pairs mixed into every random fixture.
EDGE_PAIRS = [
    ("A", "A"),
    ("A", "C"),
    ("ACGT", "ACGT"),
    ("AAAA", "CCCC"),
    ("ACGTACGT", "ACG"),
    ("G", "TTGTT"),
]


def small_fixture(seed, n=6, length=10):
    """A few random pairs plus EDGE_PAIRS, with random labels."""
    rng = np.random.default_rng(seed)
    seqsA, seqsB, labels = make_pairs(rng, n // 2, n - n // 2, length, sub_rate=0.2)
    seqsA += [a for a, _ in EDGE_PAIRS]
    seqsB += [b for _, b in EDGE_PAIRS]
    labels = np.concatenate([labels, rng.integers(0, 2, size=len(EDGE_PAIRS))])
    return rng, seqsA, seqsB, labels


def default_baseline(mode, gap_mode, substitution_mode, alphabet=DNA, perturb=None):
    """
    discrimalign()'s default baseline aligner. With perturb=rng, every score
    is shifted by uniform noise in [-0.05, 0.05]: the integer defaults tie
    often (two mismatches cost one gap open), and at a tie the backends may
    pick different, equally optimal alignments with different counts.
    """
    noise = (lambda: 0.0) if perturb is None else (lambda: float(perturb.uniform(-0.05, 0.05)))
    aligner = PairwiseAligner()
    aligner.mode = mode
    if gap_mode == "affine":
        aligner.open_gap_score = -8 + noise()
        aligner.extend_gap_score = -0.5 + noise()
    else:
        aligner.gap_score = -6 + noise()
    n = len(alphabet)
    if substitution_mode == "simple":
        aligner.match_score = 5 + noise()
        aligner.mismatch_score = -4 + noise()
    else:
        data = 9 * np.eye(n) - 4
        if perturb is not None:
            data = data + perturb.uniform(-0.05, 0.05, size=(n, n))
        aligner.substitution_matrix = substitution_matrices.Array(data=data, alphabet=alphabet)
    return aligner

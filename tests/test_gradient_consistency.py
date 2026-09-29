"""
Integration tests tying the subgradient to the alignment scores.

For a hard (Viterbi) alignment, the score is linear in the parameters with
the path's count vector as coefficients, and the log-likelihood gradient is
sum_i (y_i - p_i) * counts_i. These tests check both facts against
Biopython's scores, at random non-integer parameters where the optimal path
is locally unique.
"""
import itertools

import numpy as np
import pytest
from Bio.Align import PairwiseAligner, substitution_matrices

from src.logit_link import logit_subgradient
from tests.helpers import (DNA, GAP_MODES, MODES, SUBSTITUTION_MODES, align_all,
                           small_fixture,
                           loglik_at, make_aligner, make_pairs, model_gradient,
                           perturbed, random_params, subgradient_at)

SEEDS = range(5)

def _counts_dot_params(sg, params):
    """Score implied by a single alignment's counts (weight 1)."""
    G = np.asarray(sg["Substitutions"])
    if "substitution_matrix" in params:
        score = np.sum(G * np.asarray(params["substitution_matrix"]))
    else:
        score = np.trace(G) * params["match_score"] + (G.sum() - np.trace(G)) * params["mismatch_score"]
    if "gap_score" in params:
        score += (sg["Gap opens"] + sg["Gap extends"]) * params["gap_score"]
    else:
        score += sg["Gap opens"] * params["open_gap_score"] + sg["Gap extends"] * params["extend_gap_score"]
    return score


# --- score = counts . parameters --------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("substitution_mode", SUBSTITUTION_MODES)
@pytest.mark.parametrize("gap_mode", GAP_MODES)
@pytest.mark.parametrize("mode", MODES)
def test_score_equals_counts_dot_parameters(mode, gap_mode, substitution_mode, seed):
    rng, seqsA, seqsB, _ = small_fixture(seed)
    params = random_params(rng, gap_mode, substitution_mode)
    for aln in align_all(seqsA, seqsB, make_aligner(mode, params)):
        sg = logit_subgradient([aln], [0.0], [1], 0.0, DNA)
        assert _counts_dot_params(sg, params) == pytest.approx(aln.score, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_biopython_counts_match_subgradient_counts(mode, seed):
    """The initial estimate's features (aln.counts()) agree with logit_subgradient."""
    rng, seqsA, seqsB, _ = small_fixture(seed)
    params = random_params(rng, "affine", "simple")
    for aln in align_all(seqsA, seqsB, make_aligner(mode, params)):
        sg = logit_subgradient([aln], [0.0], [1], 0.0, DNA)
        G = np.asarray(sg["Substitutions"])
        counts = aln.counts()
        assert counts.identities == np.trace(G)
        assert counts.mismatches == G.sum() - np.trace(G)
        assert counts.open_gaps == sg["Gap opens"]
        assert counts.extend_gaps == sg["Gap extends"]
        assert counts.gaps == sg["Gap opens"] + sg["Gap extends"]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_biopython_counts_reproduce_score(mode, seed):
    rng, seqsA, seqsB, _ = small_fixture(seed)
    params = random_params(rng, "affine", "simple")
    for aln in align_all(seqsA, seqsB, make_aligner(mode, params)):
        c = aln.counts()
        implied = (c.identities * params["match_score"] + c.mismatches * params["mismatch_score"]
                   + c.open_gaps * params["open_gap_score"] + c.extend_gaps * params["extend_gap_score"])
        assert implied == pytest.approx(aln.score, rel=1e-12, abs=1e-12)


def test_local_score_is_nonnegative_and_global_is_not_bounded():
    rng = np.random.default_rng(0)
    params = random_params(rng, "affine", "simple")
    local = make_aligner("local", params)
    glob = make_aligner("global", params)
    for a, b in [("AAAA", "CCCC"), ("ACAC", "GTGT"), ("A", "G")]:
        assert align_all([a], [b], local)[0].score == 0.0
        assert align_all([a], [b], glob)[0].score < 0.0


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("gap_mode", GAP_MODES)
@pytest.mark.parametrize("substitution_mode", SUBSTITUTION_MODES)
def test_local_score_dominates_global_score(gap_mode, substitution_mode, seed):
    rng, seqsA, seqsB, _ = small_fixture(seed)
    params = random_params(rng, gap_mode, substitution_mode)
    local = align_all(seqsA, seqsB, make_aligner("local", params))
    glob = align_all(seqsA, seqsB, make_aligner("global", params))
    for l, g in zip(local, glob):
        assert l.score >= g.score - 1e-12


def test_score_decomposition_with_alternating_gaps():
    """With extend below open, Biopython returns -A-B / C-D-: four opens, no extends."""
    params = {"match_score": 1.0, "mismatch_score": -100.0,
              "open_gap_score": -1.0, "extend_gap_score": -10.0}
    aln = align_all(["AB"], ["CD"], make_aligner("global", params))[0]
    sg = logit_subgradient([aln], [0.0], [1], 0.0, "ABCD")
    assert _counts_dot_params(sg, params) == pytest.approx(aln.score)


# --- subgradient = d logL / d theta (central differences) -------------------

def _directions(params, substitution_mode, alphabet=DNA):
    """(key, index, symmetric) for every free parameter of the model."""
    for key in ("open_gap_score", "extend_gap_score", "gap_score",
                "match_score", "mismatch_score"):
        if key in params:
            yield key, None, False
    if "substitution_matrix" in params:
        n = len(alphabet)
        if substitution_mode == "symmetric":
            pairs = [(i, j) for i in range(n) for j in range(i, n)]
        else:
            pairs = list(itertools.product(range(n), repeat=2))
        for i, j in pairs:
            yield "substitution_matrix", (i, j), substitution_mode == "symmetric"


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("substitution_mode", SUBSTITUTION_MODES)
@pytest.mark.parametrize("gap_mode", GAP_MODES)
@pytest.mark.parametrize("mode", MODES)
def test_subgradient_matches_finite_differences(mode, gap_mode, substitution_mode, seed):
    rng, seqsA, seqsB, labels = small_fixture(seed)
    params = random_params(rng, gap_mode, substitution_mode)
    sg = subgradient_at(seqsA, seqsB, labels, mode, params, DNA)
    grad = model_gradient(sg, gap_mode, substitution_mode)
    h = 1e-6
    for key, index, symmetric in _directions(params, substitution_mode):
        up = loglik_at(seqsA, seqsB, labels, mode, perturbed(params, key, h, index, symmetric))
        down = loglik_at(seqsA, seqsB, labels, mode, perturbed(params, key, -h, index, symmetric))
        numeric = (up - down) / (2 * h)
        analytic = grad[key] if index is None else grad[key][index]
        assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-6), (key, index)


@pytest.mark.parametrize("mode", MODES)
def test_alpha_derivative_is_sum_of_residuals(mode):
    rng, seqsA, seqsB, labels = small_fixture(11)
    params = random_params(rng, "affine", "simple")
    h = 1e-6
    up = loglik_at(seqsA, seqsB, labels, mode, dict(params, alpha=params["alpha"] + h))
    down = loglik_at(seqsA, seqsB, labels, mode, dict(params, alpha=params["alpha"] - h))
    aligner = make_aligner(mode, params)
    scores = np.array([a.score for a in align_all(seqsA, seqsB, aligner)])
    residual = np.sum(labels - 1 / (1 + np.exp(-(params["alpha"] + scores))))
    assert residual == pytest.approx((up - down) / (2 * h), rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_score_is_convex_in_parameters(mode, seed):
    """Viterbi score is a max of linear functions, hence convex along any line."""
    rng, seqsA, seqsB, _ = small_fixture(seed)
    p0 = random_params(rng, "affine", "general")
    p1 = random_params(rng, "affine", "general")

    def mix(t):
        M = (1 - t) * np.asarray(p0["substitution_matrix"]) + t * np.asarray(p1["substitution_matrix"])
        out = {k: (1 - t) * p0[k] + t * p1[k] for k in ("open_gap_score", "extend_gap_score")}
        out["substitution_matrix"] = substitution_matrices.Array(alphabet=DNA, data=M)
        return out

    def scores(t):
        return np.array([a.score for a in align_all(seqsA, seqsB, make_aligner(mode, mix(t)))])

    s0, s_half, s1 = scores(0.0), scores(0.5), scores(1.0)
    assert np.all(s_half <= (s0 + s1) / 2 + 1e-9)


def test_symmetric_directional_derivative_counts_both_orientations():
    """For a symmetric matrix, d/dM[a,b] (a != b) sees A->B and B->A substitutions."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -10
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aln = align_all(["AC"], ["CA"], aligner)[0]
    sg = logit_subgradient([aln], [0.0], [1], 0.0, DNA)
    grad = model_gradient(sg, "affine", "symmetric")
    M = grad["substitution_matrix"]
    assert M[0, 1] == 2 and M[1, 0] == 2
    assert np.trace(M) == 0

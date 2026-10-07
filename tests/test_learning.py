"""
End-to-end learning on simulated data. These are the only tests that need
more than a handful of pairs, and they are marked slow.
"""
import warnings

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.discrimalign import discrimalign
from src.optimization import create_powerstep
from tests.helpers import (GAP_MODES, MODES, SUBSTITUTION_MODES, align_all, make_pairs,
                           mutate, random_seq)

pytestmark = pytest.mark.slow

ALL_MODES = [(m, g, s) for m in MODES for g in GAP_MODES for s in SUBSTITUTION_MODES]
BACKENDS = ["biopython", "nwgrad"]


def _fit(A, B, y, mode, gap_mode, substitution_mode, backend, max_iter=30, scale=1e-3):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return discrimalign(A, B, y, aligner_mode=mode, gap_mode=gap_mode,
                            substitution_mode=substitution_mode, backend=backend,
                            stepfunction=create_powerstep(scale), max_iter=max_iter)


def _auc(aligner, A, B, y):
    return roc_auc_score(y, [a.score for a in align_all(A, B, aligner)])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode, gap_mode, substitution_mode", ALL_MODES)
def test_learning_improves_fit_and_generalizes(mode, gap_mode, substitution_mode, backend):
    rng = np.random.default_rng(0)
    A, B, y = make_pairs(rng, 40, 40, 40, sub_rate=0.3, indel_rate=0.1)
    res = _fit(A, B, y, mode, gap_mode, substitution_mode, backend)

    traj = res["loglik_trajectory"]
    assert res["final_loglik"] > traj[0]
    assert np.all(np.isfinite(traj))
    assert _auc(res["aligner"], A, B, y) >= 0.95

    held_out = make_pairs(np.random.default_rng(1), 20, 20, 40, sub_rate=0.3, indel_rate=0.1)
    assert _auc(res["aligner"], *held_out) >= 0.9

    if substitution_mode == "simple":
        assert res["match_score"] > res["mismatch_score"]
    else:
        M = np.asarray(res["substitution_matrix"])
        assert np.min(np.diag(M)) > np.mean(M[~np.eye(4, dtype=bool)])
    # In global mode, adding c to every substitution score and c/2 to every
    # gap column shifts each score by c * (len A + len B) / 2. With equal
    # lengths alpha absorbs that, so absolute signs are not identified.
    # Local alignment breaks the symmetry.
    if mode == "local":
        if gap_mode == "affine":
            assert res["open_gap_score"] < 0
            assert res["extend_gap_score"] < 0
        else:
            assert res["gap_score"] < 0
        if substitution_mode == "simple":
            assert res["match_score"] > 0 > res["mismatch_score"]


_TRANSITION = {"A": "G", "G": "A", "C": "C", "T": "T"}


def _transitions_only(rng, seq, rate):
    return "".join(_TRANSITION[c] if c in "AG" and rng.random() < rate else c for c in seq)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("substitution_mode", ["symmetric", "general"])
def test_planted_substitution_is_recovered(substitution_mode, backend):
    """Homologs differ only by A<->G transitions, so A-G must outscore every other mismatch."""
    rng = np.random.default_rng(0)
    A, B, y = [], [], []
    for _ in range(40):
        seq = random_seq(rng, 40)
        A.append(seq)
        B.append(mutate(rng, _transitions_only(rng, seq, 0.5), sub_rate=0.0, indel_rate=0.05))
        y.append(1)
        A.append(random_seq(rng, 40))
        B.append(random_seq(rng, 40))
        y.append(0)
    res = _fit(A, B, np.array(y), "local", "affine", substitution_mode, backend)

    M = res["substitution_matrix"]
    others = [M[a, b] for a in "ACGT" for b in "ACGT"
              if a != b and {a, b} != {"A", "G"}]
    assert min(M["A", "G"], M["G", "A"]) > max(others) + 0.3
    assert min(M["A", "G"], M["G", "A"]) > 0

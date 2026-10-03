"""
Tests for src.nwgrad_params: parameters converted to nwgrad must reproduce
Biopython's scores, and nwgrad's gradients converted back must reproduce
logit_subgradient's counts.

The double-precision SeqPairDouble with pointer traceback is used
throughout, so scores are exact up to summation order and paths are optimal.
"""
from copy import deepcopy

import nwgrad
import numpy as np
import pytest
from Bio.Align import substitution_matrices

from src.logit_link import logit_subgradient
from src.nwgrad_params import (gap_penalties, grad_to_raw, substitution_data,
                               to_nwgrad)
from tests.helpers import (DNA, GAP_MODES, MODES, SUBSTITUTION_MODES, align_all,
                           make_aligner, model_gradient, random_params, small_fixture)

SEEDS = range(5)


def _nwgrad_pair(a, b, mode, gap_mode, substitution_mode, params, alphabet=DNA):
    pair = nwgrad.SeqPairDouble(a, b, to_nwgrad(params, gap_mode, substitution_mode, alphabet),
                                gap_model=gap_mode, mode=mode, traceback="pointers")
    pair.score_and_grad()
    return pair


def _simple(match=2.0, mismatch=-1.0, open_=-3.0, extend=-1.0):
    return {"match_score": match, "mismatch_score": mismatch,
            "open_gap_score": open_, "extend_gap_score": extend}


# --- forward: parameter conversion ------------------------------------------

def test_affine_gap_penalties():
    assert gap_penalties({"open_gap_score": -10.0, "extend_gap_score": -0.5}, "affine") == (9.5, 0.5)


def test_linear_gap_penalties():
    assert gap_penalties({"gap_score": -6.0}, "linear") == (0.0, 6.0)


def test_extend_below_open_gives_negative_surcharge():
    go, ge = gap_penalties({"open_gap_score": -1.0, "extend_gap_score": -10.0}, "affine")
    assert (go, ge) == (-9.0, 10.0)


def test_simple_mode_matrix():
    data, alphabet = substitution_data({"match_score": 5, "mismatch_score": -4}, "simple", "ACG")
    assert alphabet == "ACG"
    np.testing.assert_array_equal(data, 9 * np.eye(3) - 4)
    assert data.dtype == np.float64


def test_matrix_keeps_its_own_alphabet():
    M = substitution_matrices.Array(alphabet="TGCA", data=np.arange(16.0).reshape(4, 4))
    data, alphabet = substitution_data({"substitution_matrix": M}, "general", "ACGT")
    assert alphabet == "TGCA"
    np.testing.assert_array_equal(data, np.arange(16.0).reshape(4, 4))


def test_plain_array_matrix_uses_given_alphabet():
    data, alphabet = substitution_data({"substitution_matrix": np.eye(4)}, "general", "ACGT")
    assert alphabet == "ACGT"
    np.testing.assert_array_equal(data, np.eye(4))


@pytest.mark.parametrize("gap_mode", GAP_MODES)
@pytest.mark.parametrize("substitution_mode", SUBSTITUTION_MODES)
def test_to_nwgrad_fields(gap_mode, substitution_mode):
    params = random_params(np.random.default_rng(0), gap_mode, substitution_mode)
    nw = to_nwgrad(params, gap_mode, substitution_mode, DNA).to_dict()
    go, ge = gap_penalties(params, gap_mode)
    assert nw["gap_open_a"] == nw["gap_open_b"] == go
    assert nw["gap_extend_a"] == nw["gap_extend_b"] == ge
    assert nw["alphabet"] == DNA
    expected, _ = substitution_data(params, substitution_mode, DNA)
    np.testing.assert_array_equal(nw["matrix"], expected)


def test_to_nwgrad_does_not_mutate_params():
    params = random_params(np.random.default_rng(1), "affine", "general")
    before = deepcopy(params)
    to_nwgrad(params, "affine", "general", DNA)
    np.testing.assert_array_equal(np.asarray(params["substitution_matrix"]),
                                  np.asarray(before["substitution_matrix"]))
    assert params["open_gap_score"] == before["open_gap_score"]


@pytest.mark.parametrize("a, b, mode, expected", [
    ("ACGT", "ACGT", "global", 8.0),
    ("AGGGC", "AC", "global", 2 + 2 - 3 - 1 - 1),   # one gap of length 3
    ("AC", "AGGGC", "global", 2 + 2 - 3 - 1 - 1),   # same gap, in the other sequence
    ("ACGT", "AGT", "global", 2 + 2 + 2 - 3),       # one gap of length 1
    ("TTACGTT", "GGACGGG", "local", 6.0),           # ACG
    ("AAAA", "CCCC", "local", 0.0),                 # empty local alignment
])
def test_hand_computed_scores(a, b, mode, expected):
    pair = _nwgrad_pair(a, b, mode, "affine", "simple", _simple())
    assert pair.score == pytest.approx(expected)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("substitution_mode", SUBSTITUTION_MODES)
@pytest.mark.parametrize("gap_mode", GAP_MODES)
@pytest.mark.parametrize("mode", MODES)
def test_scores_match_biopython(mode, gap_mode, substitution_mode, seed):
    rng, seqsA, seqsB, _ = small_fixture(seed)
    params = random_params(rng, gap_mode, substitution_mode)
    bio = [aln.score for aln in align_all(seqsA, seqsB, make_aligner(mode, params))]
    nw = [_nwgrad_pair(a, b, mode, gap_mode, substitution_mode, params).score
          for a, b in zip(seqsA, seqsB)]
    assert nw == pytest.approx(bio, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("a, b", [("A", "C"), ("AT", "CG"), ("GAT", "GCT"), ("AB", "CD")])
def test_scores_match_biopython_when_gaps_beat_mismatches(a, b):
    """Adjacent gaps in different sequences, and extend below open."""
    for params in (_simple(1.0, -100.0, -1.0, -0.5), _simple(1.0, -100.0, -1.0, -10.0)):
        bio = align_all([a], [b], make_aligner("global", params))[0].score
        assert _nwgrad_pair(a, b, "global", "affine", "simple", params, "ABCDGT").score == \
            pytest.approx(bio)


# --- backward: gradient conversion ------------------------------------------

def _raw_counts(aln, alphabet=DNA):
    return logit_subgradient([aln], [0.0], [1], 0.0, alphabet)


def test_grad_to_raw_gap_of_length_three():
    raw = grad_to_raw(_nwgrad_pair("AGGGC", "AC", "global", "affine", "simple", _simple()).grad)
    assert raw["Gap opens"] == 1
    assert raw["Gap extends"] == 2
    assert raw["Substitutions"]["A", "A"] == 1
    assert raw["Substitutions"]["C", "C"] == 1
    assert np.asarray(raw["Substitutions"]).sum() == 2


def test_grad_to_raw_alternating_gaps_are_all_opens():
    params = _simple(1.0, -100.0, -1.0, -10.0)
    raw = grad_to_raw(_nwgrad_pair("AB", "CD", "global", "affine", "simple", params, "ABCD").grad)
    assert raw["Gap opens"] == 4
    assert raw["Gap extends"] == 0


def test_grad_to_raw_substitution_orientation():
    """Rows are residues of sequence A, columns residues of sequence B."""
    params = {"substitution_matrix": substitution_matrices.Array(
                  alphabet=DNA, data=np.where(np.eye(4) > 0, 2.0, -1.0)),
              "open_gap_score": -10.0, "extend_gap_score": -10.0}
    raw = grad_to_raw(_nwgrad_pair("AG", "CT", "global", "affine", "general", params).grad)
    assert raw["Substitutions"]["A", "C"] == 1
    assert raw["Substitutions"]["C", "A"] == 0
    assert raw["Substitutions"]["G", "T"] == 1


def test_grad_to_raw_accepts_dict_and_object():
    grad = _nwgrad_pair("ACGGT", "AT", "global", "affine", "simple", _simple()).grad
    a, b = grad_to_raw(grad), grad_to_raw(grad.to_dict())
    np.testing.assert_array_equal(np.asarray(a["Substitutions"]), np.asarray(b["Substitutions"]))
    assert (a["Gap opens"], a["Gap extends"]) == (b["Gap opens"], b["Gap extends"])


def test_grad_to_raw_is_linear():
    """Weighted sums of gradients may be converted after summing."""
    params = _simple()
    grads = [_nwgrad_pair(a, b, "global", "affine", "simple", params).grad.to_dict()
             for a, b in [("AGGGC", "AC"), ("ACGT", "AGT"), ("GATTACA", "GACA")]]
    weights = [0.3, -1.2, 0.7]
    summed = {k: sum(w * g[k] for w, g in zip(weights, grads))
              for k in ("matrix", "gap_open_a", "gap_extend_a", "gap_open_b", "gap_extend_b")}
    summed["alphabet"] = grads[0]["alphabet"]
    raw_sum = grad_to_raw(summed)
    parts = [grad_to_raw(g) for g in grads]
    np.testing.assert_allclose(np.asarray(raw_sum["Substitutions"]),
                               sum(w * np.asarray(p["Substitutions"]) for w, p in zip(weights, parts)))
    for key in ("Gap opens", "Gap extends"):
        assert raw_sum[key] == pytest.approx(sum(w * p[key] for w, p in zip(weights, parts)))


def test_empty_local_alignment_gradient_is_zero():
    raw = grad_to_raw(_nwgrad_pair("AAAA", "CCCC", "local", "affine", "simple", _simple()).grad)
    assert np.all(np.asarray(raw["Substitutions"]) == 0)
    assert raw["Gap opens"] == 0
    assert raw["Gap extends"] == 0


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("substitution_mode", SUBSTITUTION_MODES)
@pytest.mark.parametrize("gap_mode", GAP_MODES)
@pytest.mark.parametrize("mode", MODES)
def test_model_gradient_matches_logit_subgradient(mode, gap_mode, substitution_mode, seed):
    """
    Per pair, the gradient in the model's own parameters agrees.

    Raw counts can legitimately differ where the model has ties: in simple
    mode an A/C and an A/G mismatch score the same, and in linear mode so
    do different gap placements. Only the model's parameters must agree.
    """
    rng, seqsA, seqsB, _ = small_fixture(seed)
    params = random_params(rng, gap_mode, substitution_mode)
    alns = align_all(seqsA, seqsB, make_aligner(mode, params))
    for a, b, aln in zip(seqsA, seqsB, alns):
        nw = model_gradient(grad_to_raw(_nwgrad_pair(a, b, mode, gap_mode, substitution_mode,
                                                     params).grad), gap_mode, substitution_mode)
        bio = model_gradient(_raw_counts(aln), gap_mode, substitution_mode)
        assert nw.keys() == bio.keys()
        for key in nw:
            np.testing.assert_array_equal(np.asarray(nw[key]), np.asarray(bio[key]),
                                          err_msg=f"{key}: {a} / {b}")


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_raw_counts_match_logit_subgradient_without_ties(mode, seed):
    """General matrix and affine gaps at continuous parameters: the path is unique."""
    rng, seqsA, seqsB, _ = small_fixture(seed)
    params = random_params(rng, "affine", "general")
    alns = align_all(seqsA, seqsB, make_aligner(mode, params))
    for a, b, aln in zip(seqsA, seqsB, alns):
        nw = grad_to_raw(_nwgrad_pair(a, b, mode, "affine", "general", params).grad)
        bio = _raw_counts(aln)
        np.testing.assert_array_equal(np.asarray(nw["Substitutions"]),
                                      np.asarray(bio["Substitutions"]), err_msg=f"{a} / {b}")
        assert (nw["Gap opens"], nw["Gap extends"]) == (bio["Gap opens"], bio["Gap extends"])


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_linear_mode_reports_all_gap_columns_as_extends(mode, seed):
    rng, seqsA, seqsB, _ = small_fixture(seed)
    params = random_params(rng, "linear", "simple")
    for a, b in zip(seqsA, seqsB):
        nw = grad_to_raw(_nwgrad_pair(a, b, mode, "linear", "simple", params).grad)
        assert nw["Gap opens"] == 0

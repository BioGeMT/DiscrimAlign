"""
The nwgrad backend must follow the Biopython backend's parameter trajectory.

Both backends find optimal paths at the same parameters. Where raw counts can
differ (ties in simple or linear mode), the gradient in the model's own
parameters does not, so trajectories agree up to rounding.
"""
import warnings

import numpy as np
import pytest

from src.discrimalign import discrimalign, discrimalign_nwgrad
from src.optimization import EmptyLocalAlignment, create_constant_step, create_powerstep
from tests.helpers import (GAP_MODES, MODES, SUBSTITUTION_MODES, make_aligner, make_pairs,
                           random_params)

ALL_MODES = [(m, g, s) for m in MODES for g in GAP_MODES for s in SUBSTITUTION_MODES]

TRAJECTORY_KEYS = ("loglik_trajectory", "subgradient_l2_trajectory",
                   "alignment_logit_scores")


def _data(seed, n=12, length=12):
    rng = np.random.default_rng(seed)
    A, B, y = make_pairs(rng, n // 2, n - n // 2, length, sub_rate=0.25)
    y[np.flatnonzero(y == 1)[0]] = 0
    y[np.flatnonzero(y == 0)[0]] = 1
    return rng, A, B, y


def _param_keys(gap_mode, substitution_mode):
    keys = {"open_gap_score", "extend_gap_score"} if gap_mode == "affine" else {"gap_score"}
    keys |= {"match_score", "mismatch_score"} if substitution_mode == "simple" else {"substitution_matrix"}
    return keys | {"alpha", "final_loglik"}


def _both(A, B, y, mode, gap_mode, substitution_mode, **kwargs):
    kwargs.setdefault("stepfunction", create_constant_step(0.01))
    kwargs.setdefault("max_iter", 3)
    runs = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for backend in ("biopython", "nwgrad"):
            runs[backend] = discrimalign(A, B, y, aligner_mode=mode, gap_mode=gap_mode,
                                         substitution_mode=substitution_mode,
                                         backend=backend, **kwargs)
    return runs["biopython"], runs["nwgrad"]


def _assert_same_fit(bio, nw, gap_mode, substitution_mode, rtol=1e-9, atol=1e-9):
    for key in _param_keys(gap_mode, substitution_mode) | set(TRAJECTORY_KEYS):
        np.testing.assert_allclose(np.asarray(nw[key], dtype=float), np.asarray(bio[key], dtype=float),
                                   rtol=rtol, atol=atol, err_msg=key)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("mode, gap_mode, substitution_mode", ALL_MODES)
def test_trajectory_matches_biopython_from_given_parameters(mode, gap_mode, substitution_mode, seed):
    rng, A, B, y = _data(seed)
    p0 = random_params(rng, gap_mode, substitution_mode)
    bio, nw = _both(A, B, y, mode, gap_mode, substitution_mode, initial_parameters=p0)
    _assert_same_fit(bio, nw, gap_mode, substitution_mode)


_TERMINAL_GAP = pytest.mark.xfail(strict=True, reason=(
    "nwgrad's local mode lets an alignment end (though not start) with a gap "
    "column; Biopython allows neither. They agree while gap columns score "
    "negative, but here the linear-gap initial estimate is positive "
    "(gap_score = open + extend of an affine fit)."))


@pytest.mark.parametrize("mode, gap_mode, substitution_mode", [
    pytest.param(*combo, marks=_TERMINAL_GAP)
    if combo in {("local", "linear", "symmetric"), ("local", "linear", "general")} else combo
    for combo in ALL_MODES])
def test_trajectory_matches_biopython_from_initial_estimate(mode, gap_mode, substitution_mode):
    _, A, B, y = _data(2, n=20)
    bio, nw = _both(A, B, y, mode, gap_mode, substitution_mode, max_iter=3,
                    stepfunction=create_powerstep(1e-3))
    _assert_same_fit(bio, nw, gap_mode, substitution_mode)


@pytest.mark.parametrize("substitution_mode", SUBSTITUTION_MODES)
def test_trajectory_matches_with_seeded_noise(substitution_mode):
    rng, A, B, y = _data(3)
    p0 = random_params(rng, "affine", substitution_mode)
    runs = []
    for backend in ("biopython", "nwgrad"):
        np.random.seed(11)
        runs.append(discrimalign(A, B, y, aligner_mode="local", gap_mode="affine",
                                 substitution_mode=substitution_mode, initial_parameters=p0,
                                 stochastic_factor=0.3, max_iter=3,
                                 stepfunction=create_constant_step(0.01), backend=backend))
    _assert_same_fit(runs[0], runs[1], "affine", substitution_mode)


def test_baseline_aligner_mode_takes_precedence_in_both_backends():
    rng, A, B, y = _data(4)
    baseline = make_aligner("global", random_params(rng, "affine", "simple"))
    bio, nw = _both(A, B, y, "local", "affine", "simple", baseline_aligner=baseline)
    assert bio["aligner"].mode == nw["aligner"].mode == "global"
    _assert_same_fit(bio, nw, "affine", "simple")


@pytest.mark.parametrize("num_threads", [2, 5])
def test_nwgrad_thread_count_does_not_change_results(num_threads):
    rng, A, B, y = _data(5, n=20)
    p0 = random_params(rng, "affine", "general")
    kwargs = dict(aligner_mode="local", gap_mode="affine", substitution_mode="general",
                  initial_parameters=p0, max_iter=3, stepfunction=create_constant_step(0.01),
                  backend="nwgrad")
    one = discrimalign(A, B, y, num_threads=1, **kwargs)
    many = discrimalign(A, B, y, num_threads=num_threads, **kwargs)
    for key in _param_keys("affine", "general") | set(TRAJECTORY_KEYS):
        np.testing.assert_array_equal(np.asarray(many[key]), np.asarray(one[key]), err_msg=key)


def test_nwgrad_local_pairs_without_alignment():
    A = ["AAAA", "ACGT", "CCCC", "ACGTAC", "GGGG", "ACGA"]
    B = ["CCCC", "ACGT", "GGGG", "ACGTAC", "TTTT", "ACGT"]
    y = np.array([0, 1, 0, 1, 0, 1])
    p0 = random_params(np.random.default_rng(6), "affine", "general")
    bio, nw = _both(A, B, y, "local", "affine", "general", initial_parameters=p0)
    _assert_same_fit(bio, nw, "affine", "general")
    for i in (0, 2, 4):
        assert isinstance(nw["alignments"][i], EmptyLocalAlignment)


def test_nwgrad_returns_biopython_aligner_and_alignments():
    rng, A, B, y = _data(7)
    p0 = random_params(rng, "linear", "symmetric")
    bio, nw = _both(A, B, y, "global", "linear", "symmetric", initial_parameters=p0)
    assert type(nw["aligner"]) is type(bio["aligner"])
    assert [a.score for a in nw["alignments"]] == pytest.approx(
        [a.score for a in bio["alignments"]], rel=1e-9)


def test_nwgrad_without_alignments():
    rng, A, B, y = _data(8)
    _, nw = _both(A, B, y, "local", "affine", "simple",
                  initial_parameters=random_params(rng, "affine", "simple"),
                  return_alignments=False)
    assert "alignments" not in nw


def test_discrimalign_nwgrad_is_the_nwgrad_backend():
    rng, A, B, y = _data(9)
    p0 = random_params(rng, "affine", "symmetric")
    kwargs = dict(aligner_mode="local", gap_mode="affine", substitution_mode="symmetric",
                  initial_parameters=p0, max_iter=2, stepfunction=create_constant_step(0.01))
    wrapped = discrimalign_nwgrad(A, B, y, **kwargs)
    direct = discrimalign(A, B, y, backend="nwgrad", **kwargs)
    for key in _param_keys("affine", "symmetric") | set(TRAJECTORY_KEYS):
        np.testing.assert_array_equal(np.asarray(wrapped[key]), np.asarray(direct[key]), err_msg=key)


@pytest.mark.parametrize("gap_mode, params", [
    ("linear", {"gap_score": -1.0, "match_score": 2.0, "mismatch_score": -5.0}),
    ("affine", {"open_gap_score": -1.0, "extend_gap_score": -0.5,
                "match_score": 2.0, "mismatch_score": -5.0}),
])
@pytest.mark.parametrize("a, b", [("AAA", "AAAT"), ("AAA", "TAAA"), ("AAAT", "AAA")])
def test_local_terminal_gaps_agree_when_gaps_are_penalized(gap_mode, params, a, b):
    from src.nwgrad_backend import NwgradEngine
    engine = NwgradEngine([a], [b], "local", gap_mode, "simple", "ACGT", 1)
    engine.set_params(params)
    bio = make_aligner("local", params).score(a, b)
    assert engine.scores()[0] == pytest.approx(bio)


@pytest.mark.parametrize("a, b", [
    pytest.param("AAA", "AAAT", marks=pytest.mark.xfail(strict=True, reason=(
        "nwgrad's local mode lets an alignment end with a gap column; Biopython "
        "does not. Only reachable when a gap column scores positive."))),
    ("AAA", "TAAA"),  # neither engine starts a local alignment with a gap
])
def test_local_terminal_gaps_agree_when_gaps_score_positive(a, b):
    from src.nwgrad_backend import NwgradEngine
    params = {"gap_score": 1.0, "match_score": 2.0, "mismatch_score": -5.0}
    engine = NwgradEngine([a], [b], "local", "linear", "simple", "ACGT", 1)
    engine.set_params(params)
    assert engine.scores()[0] == pytest.approx(make_aligner("local", params).score(a, b))


def test_unknown_backend_is_rejected():
    _, A, B, y = _data(10)
    with pytest.raises(AssertionError):
        discrimalign(A, B, y, max_iter=0, backend="parasail")


def test_nwgrad_missing_stepfunction_raises():
    _, A, B, y = _data(11)
    with pytest.raises(ValueError, match="stepfunction is required"):
        discrimalign_nwgrad(A, B, y, max_iter=1)


@pytest.mark.slow
@pytest.mark.parametrize("mode, gap_mode, substitution_mode",
                         [("local", "affine", "symmetric"), ("global", "linear", "simple"),
                          ("local", "affine", "general")])
def test_long_trajectory_matches_biopython(mode, gap_mode, substitution_mode):
    """30 iterations on 80 pairs: rounding differences must not compound into divergence."""
    rng = np.random.default_rng(0)
    A, B, y = make_pairs(rng, 40, 40, 40, sub_rate=0.3, indel_rate=0.1)
    bio, nw = _both(A, B, y, mode, gap_mode, substitution_mode, max_iter=30,
                    stepfunction=create_powerstep(1e-3))
    _assert_same_fit(bio, nw, gap_mode, substitution_mode, rtol=1e-7, atol=1e-7)

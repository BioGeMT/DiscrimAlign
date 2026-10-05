"""Integration tests for src.discrimalign.discrimalign on small fixtures."""
import importlib
import sys
import warnings
from copy import deepcopy

import numpy as np
import pytest
from Bio.Align import PairwiseAligner, substitution_matrices
from scipy.special import expit

from src.discrimalign import _MAX_GAP_SCORE, discrimalign
from src.logit_link import logit_logL, logit_subgradient
from src.optimization import (EmptyLocalAlignment, create_constant_step,
                              create_powerstep, get_initial_estimate)
from tests.helpers import (DNA, GAP_MODES, MODES, SUBSTITUTION_MODES, align_all,
                           default_baseline, make_aligner, make_pairs, model_gradient,
                           optimal_alpha, random_params)

ALL_MODES = [(m, g, s) for m in MODES for g in GAP_MODES for s in SUBSTITUTION_MODES]


def _data(seed=0, n=12, length=12):
    rng = np.random.default_rng(seed)
    seqsA, seqsB, labels = make_pairs(rng, n // 2, n - n // 2, length, sub_rate=0.25)
    # Flip one label per class so the classes overlap and the intercept MLE exists.
    labels[np.flatnonzero(labels == 1)[0]] = 0
    labels[np.flatnonzero(labels == 0)[0]] = 1
    return rng, seqsA, seqsB, labels


def _param_keys(gap_mode, substitution_mode):
    keys = {"open_gap_score", "extend_gap_score"} if gap_mode == "affine" else {"gap_score"}
    keys |= {"match_score", "mismatch_score"} if substitution_mode == "simple" else {"substitution_matrix"}
    return keys


def _run(seqsA, seqsB, labels, mode, gap_mode, substitution_mode, **kwargs):
    kwargs.setdefault("stepfunction", create_constant_step(0.01))
    kwargs.setdefault("max_iter", 2)
    kwargs.setdefault("backend", "biopython")  # tests that need nwgrad ask for it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn convergence noise on tiny data
        return discrimalign(seqsA, seqsB, labels, aligner_mode=mode, gap_mode=gap_mode,
                            substitution_mode=substitution_mode, **kwargs)


GAP_KEYS = ("open_gap_score", "extend_gap_score", "gap_score")


def _clip_gaps(params):
    """The projection discrimalign applies: gap scores are capped."""
    for key in GAP_KEYS:
        if key in params:
            params[key] = np.minimum(params[key], _MAX_GAP_SCORE)


def _assert_alpha_stationary(alpha, scores, labels):
    """dlogL/dalpha vanishes at alpha: the default alpha solver finds the exact root."""
    assert abs(np.sum(labels - expit(alpha + scores))) < 1e-9
    assert alpha == pytest.approx(optimal_alpha(scores, labels), abs=1e-9)


BACKENDS = ["biopython", "nwgrad"]


def _run_on(backend, *args, **kwargs):
    return _run(*args, backend=backend, **kwargs)


def _assert_params_equal(result, expected, keys, **tol):
    for key in keys:
        np.testing.assert_allclose(np.asarray(result[key]), np.asarray(expected[key]),
                                   err_msg=key, **tol)


# --- argument validation ----------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"aligner_mode": "semiglobal"},
    {"gap_mode": "convex"},
    {"substitution_mode": "blosum"},
])
@pytest.mark.parametrize("backend", BACKENDS)
def test_rejects_unknown_modes(kwargs, backend):
    _, A, B, y = _data()
    with pytest.raises(AssertionError):
        discrimalign(A, B, y, backend=backend, stepfunction=create_constant_step(0.1), max_iter=1, **kwargs)


@pytest.mark.parametrize("backend", BACKENDS)
def test_missing_stepfunction_defaults_to_powerstep(backend):
    _, A, B, y = _data()
    kwargs = dict(backend=backend, aligner_mode="local", gap_mode="affine",
                  substitution_mode="general", max_iter=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = discrimalign(A, B, y, **kwargs)
        explicit = discrimalign(A, B, y, stepfunction=create_powerstep(1e-4), **kwargs)
    for key in ("loglik_trajectory", "subgradient_l2_trajectory", "alpha",
                "open_gap_score", "extend_gap_score"):
        assert default[key] == explicit[key], key
    np.testing.assert_array_equal(np.asarray(default["substitution_matrix"]),
                                  np.asarray(explicit["substitution_matrix"]))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("which", ["A", "B"])
def test_empty_sequences_raise_before_any_work(which, backend, monkeypatch):
    module = sys.modules["src.discrimalign"]

    def fail(*args, **kwargs):
        raise AssertionError("aligned before validating the sequences")

    monkeypatch.setattr(module, "_align_pairs", fail)
    _, A, B, y = _data()
    if which == "A":
        A[3] = ""
    else:
        B[1] = B[5] = ""
    with pytest.raises(ValueError, match=f"seqlist{which} contains empty sequences"):
        _run_on(backend, A, B, y, "local", "affine", "simple", max_iter=0, stepfunction=None)


@pytest.mark.parametrize("backend", BACKENDS)
def test_missing_stepfunction_is_fine_without_iterations(backend):
    _, A, B, y = _data()
    res = _run_on(backend, A, B, y, "local", "affine", "simple", max_iter=0, stepfunction=None)
    assert res["loglik_trajectory"] == [res["final_loglik"]]


@pytest.mark.parametrize("backend", BACKENDS)
def test_single_class_labels_without_initial_parameters_raise(backend):
    _, A, B, _ = _data()
    with pytest.raises(ValueError):
        _run_on(backend, A, B, np.ones(len(A), dtype=int), "local", "affine", "simple")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("value", [0, 1])
def test_single_class_labels_raise_before_any_work(backend, value, monkeypatch):
    """Checked once at the start: also with a warm start, which skips the initial
    estimate, and before anything is aligned."""
    rng, A, B, _ = _data()
    p0 = random_params(rng, "affine", "simple")
    module = sys.modules["src.discrimalign"]
    monkeypatch.setattr(module, "_align_pairs", lambda *a, **k: pytest.fail("aligned"))
    with pytest.raises(ValueError, match="both classes"):
        _run_on(backend, A, B, np.full(len(A), value), "local", "affine", "simple",
                initial_parameters=p0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_non_binary_labels_raise_before_any_work(backend, monkeypatch):
    rng, A, B, y = _data()
    p0 = random_params(rng, "affine", "simple")
    y = np.array(y, dtype=float)
    y[0] = 2
    module = sys.modules["src.discrimalign"]
    monkeypatch.setattr(module, "_align_pairs", lambda *a, **k: pytest.fail("aligned"))
    with pytest.raises(ValueError, match="0 or 1"):
        _run_on(backend, A, B, y, "local", "affine", "simple", initial_parameters=p0)


# --- result structure -------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode, gap_mode, substitution_mode", ALL_MODES)
def test_result_structure(mode, gap_mode, substitution_mode, backend):
    _, A, B, y = _data()
    res = _run_on(backend, A, B, y, mode, gap_mode, substitution_mode, max_iter=3)
    expected = _param_keys(gap_mode, substitution_mode) | {
        "loglik_trajectory", "subgradient_l2_trajectory", "final_loglik",
        "aligner", "alignments", "alignment_logit_scores", "alpha"}
    assert set(res) == expected
    assert len(res["loglik_trajectory"]) == 4
    assert len(res["subgradient_l2_trajectory"]) == 3
    assert res["final_loglik"] == res["loglik_trajectory"][-1]
    assert len(res["alignments"]) == len(A)
    assert len(res["alignment_logit_scores"]) == len(A)
    assert np.all(np.asarray(res["subgradient_l2_trajectory"]) >= 0)
    assert np.all(np.asarray(res["loglik_trajectory"]) <= 0)
    assert res["aligner"].mode == mode
    if substitution_mode != "simple":
        assert "".join(res["substitution_matrix"].alphabet) == DNA


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode, gap_mode, substitution_mode", ALL_MODES)
def test_final_outputs_are_consistent(mode, gap_mode, substitution_mode, backend):
    rng, A, B, y = _data(1)
    p0 = random_params(rng, gap_mode, substitution_mode)
    res = _run_on(backend, A, B, y, mode, gap_mode, substitution_mode, initial_parameters=p0)
    realigned = align_all(A, B, res["aligner"])
    scores = np.array([a.score for a in res["alignments"]])
    # nwgrad returns its own paths and scores, equal to Biopython's up to summation order.
    tol = 0 if backend == "biopython" else 1e-12
    np.testing.assert_allclose(scores, [a.score for a in realigned], rtol=tol, atol=tol)
    np.testing.assert_allclose(res["alignment_logit_scores"], expit(res["alpha"] + scores), rtol=1e-14)
    assert res["final_loglik"] == pytest.approx(logit_logL(scores, res["alpha"], y), rel=1e-14)
    # The returned aligner carries the returned parameters.
    reference = make_aligner(mode, {k: res[k] for k in _param_keys(gap_mode, substitution_mode)})
    assert [a.score for a in realigned] == [a.score for a in align_all(A, B, reference)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_return_alignments_false_omits_alignments(backend):
    _, A, B, y = _data()
    res = _run_on(backend, A, B, y, "local", "affine", "simple", return_alignments=False)
    assert "alignments" not in res
    assert len(res["alignment_logit_scores"]) == len(A)


# --- zero iterations and zero steps -----------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode, gap_mode, substitution_mode", ALL_MODES)
def test_zero_iterations_return_initial_parameters(mode, gap_mode, substitution_mode, backend):
    rng, A, B, y = _data(2)
    p0 = random_params(rng, gap_mode, substitution_mode)
    res = _run_on(backend, A, B, y, mode, gap_mode, substitution_mode, initial_parameters=p0,
               max_iter=0, stepfunction=None)
    _assert_params_equal(res, p0, _param_keys(gap_mode, substitution_mode))
    assert res["alpha"] == p0["alpha"]
    assert res["subgradient_l2_trajectory"] == []
    scores = np.array([a.score for a in align_all(A, B, make_aligner(mode, p0))])
    assert res["final_loglik"] == pytest.approx(logit_logL(scores, p0["alpha"], y))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode, gap_mode, substitution_mode", ALL_MODES)
def test_zero_step_keeps_parameters_and_fits_alpha(mode, gap_mode, substitution_mode, backend):
    rng, A, B, y = _data(3)
    p0 = random_params(rng, gap_mode, substitution_mode)
    res = _run_on(backend, A, B, y, mode, gap_mode, substitution_mode, initial_parameters=p0,
               max_iter=3, stepfunction=create_constant_step(0.0))
    _assert_params_equal(res, p0, _param_keys(gap_mode, substitution_mode))
    scores = np.array([a.score for a in res["alignments"]])
    _assert_alpha_stationary(res["alpha"], scores, y)
    traj = res["loglik_trajectory"]
    assert traj[1] >= traj[0] - 1e-12
    assert traj[1] == pytest.approx(traj[2], abs=1e-8)
    assert traj[2] == pytest.approx(traj[3], abs=1e-8)


# --- one iteration is exactly one subgradient step --------------------------

@pytest.mark.parametrize("backend", ["biopython", "nwgrad"])
@pytest.mark.parametrize("seed", [4, 5])
@pytest.mark.parametrize("mode, gap_mode, substitution_mode", ALL_MODES)
def test_one_iteration_is_one_subgradient_step(mode, gap_mode, substitution_mode, seed, backend):
    rng, A, B, y = _data(seed)
    p0 = random_params(rng, gap_mode, substitution_mode)
    eta = 0.013
    res = _run(A, B, y, mode, gap_mode, substitution_mode, initial_parameters=p0,
               max_iter=1, stepfunction=create_constant_step(eta), backend=backend)

    alns = align_all(A, B, make_aligner(mode, p0))
    scores = np.array([a.score for a in alns])
    alpha = res["alpha"]
    # alpha is the intercept MLE for the scores under p0 ...
    _assert_alpha_stationary(alpha, scores, y)
    # ... and the step is taken with the gradient at that alpha.
    logits = expit(alpha + scores)
    grad = model_gradient(logit_subgradient(alns, logits, y, alpha, DNA), gap_mode, substitution_mode)
    expected = {k: np.asarray(p0[k]) + eta * np.asarray(grad[k]) for k in grad}
    _clip_gaps(expected)
    _assert_params_equal(res, expected, grad.keys(), rtol=1e-12, atol=1e-12)

    norm = np.sqrt(sum(np.sum(np.asarray(g) ** 2) for g in grad.values()))
    assert res["subgradient_l2_trajectory"][0] == pytest.approx(norm, rel=1e-12)
    assert res["loglik_trajectory"][0] == pytest.approx(logit_logL(scores, p0["alpha"], y))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", MODES)
def test_small_step_increases_likelihood_at_fixed_alpha(mode, backend):
    """A short step along the subgradient is an ascent direction."""
    rng, A, B, y = _data(6, n=16)
    p0 = random_params(rng, "affine", "general")
    res = _run_on(backend, A, B, y, mode, "affine", "general", initial_parameters=p0,
               max_iter=1, stepfunction=create_constant_step(1e-4))
    p0_fitted = dict(p0, alpha=res["alpha"])
    p1 = {k: res[k] for k in _param_keys("affine", "general")}
    p1["alpha"] = res["alpha"]

    def loglik(p):
        s = np.array([a.score for a in align_all(A, B, make_aligner(mode, p))])
        return logit_logL(s, p["alpha"], y)

    assert loglik(p1) > loglik(p0_fitted)


# --- invariants over several iterations -------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode, gap_mode", [(m, g) for m in MODES for g in GAP_MODES])
def test_symmetric_mode_keeps_matrix_symmetric(mode, gap_mode, backend):
    _, A, B, y = _data(7)
    res = _run_on(backend, A, B, y, mode, gap_mode, "symmetric", max_iter=4,
               stepfunction=create_powerstep(0.05))
    M = np.asarray(res["substitution_matrix"])
    np.testing.assert_allclose(M, M.T, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_general_mode_can_become_asymmetric(backend):
    rng, A, B, y = _data(8)
    p0 = random_params(rng, "affine", "symmetric")
    res = _run_on(backend, A, B, y, "global", "affine", "general", initial_parameters=p0,
               max_iter=2, stepfunction=create_constant_step(0.05))
    M = np.asarray(res["substitution_matrix"])
    assert not np.allclose(M, M.T)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode, gap_mode, substitution_mode", ALL_MODES)
def test_inputs_are_not_mutated(mode, gap_mode, substitution_mode, backend):
    rng, A, B, y = _data(9)
    p0 = random_params(rng, gap_mode, substitution_mode)
    baseline = make_aligner(mode, random_params(rng, gap_mode, substitution_mode))
    snapshot = (deepcopy(p0), list(A), list(B), y.copy(),
                [a.score for a in align_all(A, B, baseline)])
    _run_on(backend, A, B, y, mode, gap_mode, substitution_mode, initial_parameters=p0,
         baseline_aligner=baseline, max_iter=2, stepfunction=create_constant_step(0.1))
    _assert_params_equal(p0, snapshot[0], p0.keys())
    assert (A, B) == (snapshot[1], snapshot[2])
    np.testing.assert_array_equal(y, snapshot[3])
    assert [a.score for a in align_all(A, B, baseline)] == snapshot[4]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode, gap_mode, substitution_mode",
                         [("local", "affine", "symmetric"), ("global", "linear", "simple")])
def test_thread_count_does_not_change_results(mode, gap_mode, substitution_mode, backend):
    _, A, B, y = _data(10, n=20)
    one = _run_on(backend, A, B, y, mode, gap_mode, substitution_mode, max_iter=3, num_threads=1)
    four = _run_on(backend, A, B, y, mode, gap_mode, substitution_mode, max_iter=3, num_threads=4)
    _assert_params_equal(four, one, _param_keys(gap_mode, substitution_mode), rtol=0, atol=0)
    assert four["loglik_trajectory"] == one["loglik_trajectory"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("gap_mode, substitution_mode",
                         [("affine", "simple"), ("linear", "general"), ("affine", "symmetric")])
def test_subgradient_scale_is_equivalent_to_scaled_step(gap_mode, substitution_mode, backend):
    rng, A, B, y = _data(11)
    p0 = random_params(rng, gap_mode, substitution_mode)
    scaled = _run_on(backend, A, B, y, "local", gap_mode, substitution_mode, initial_parameters=p0,
                  max_iter=3, subgradient_scale=0.5, stepfunction=create_constant_step(0.02))
    plain = _run_on(backend, A, B, y, "local", gap_mode, substitution_mode, initial_parameters=p0,
                 max_iter=3, stepfunction=create_constant_step(0.01))
    _assert_params_equal(scaled, plain, _param_keys(gap_mode, substitution_mode), rtol=1e-12)
    np.testing.assert_allclose(scaled["subgradient_l2_trajectory"],
                               0.5 * np.asarray(plain["subgradient_l2_trajectory"]), rtol=1e-12)


# --- stochastic_factor ------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("substitution_mode", SUBSTITUTION_MODES)
def test_stochastic_factor_is_reproducible_under_global_seed(substitution_mode, backend):
    rng, A, B, y = _data(12)
    p0 = random_params(rng, "affine", substitution_mode)
    runs = []
    for _ in range(2):
        np.random.seed(123)
        runs.append(_run_on(backend, A, B, y, "local", "affine", substitution_mode, initial_parameters=p0,
                         stochastic_factor=0.5, max_iter=2))
    keys = _param_keys("affine", substitution_mode)
    _assert_params_equal(runs[0], runs[1], keys, rtol=0, atol=0)
    quiet = _run_on(backend, A, B, y, "local", "affine", substitution_mode, initial_parameters=p0, max_iter=2)
    assert any(not np.allclose(np.asarray(runs[0][k]), np.asarray(quiet[k])) for k in keys)


@pytest.mark.parametrize("backend", BACKENDS)
def test_symmetric_mode_stays_symmetric_with_noise(backend):
    rng, A, B, y = _data(13)
    np.random.seed(7)
    res = _run_on(backend, A, B, y, "global", "affine", "symmetric",
               initial_parameters=random_params(rng, "affine", "symmetric"),
               stochastic_factor=1.0, max_iter=3)
    M = np.asarray(res["substitution_matrix"])
    np.testing.assert_allclose(M, M.T, atol=1e-12)


# --- alphabet handling ------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_alphabet_inferred_from_sequences_is_sorted_union(backend):
    A = ["GATTACA", "CAT", "TAG", "GGT"]
    B = ["GATACA", "CAT", "ACG", "TTT"]
    res = _run_on(backend, A, B, np.array([1, 1, 0, 0]), "local", "affine", "general", max_iter=1)
    assert "".join(res["substitution_matrix"].alphabet) == "ACGT"


@pytest.mark.parametrize("backend", BACKENDS)
def test_alphabet_taken_from_initial_parameters(backend):
    rng, A, B, y = _data(14)
    p0 = random_params(rng, "affine", "general", alphabet="TGCA")
    res = _run_on(backend, A, B, y, "local", "affine", "general", initial_parameters=p0, max_iter=1)
    assert "".join(res["substitution_matrix"].alphabet) == "TGCA"


@pytest.mark.parametrize("backend", BACKENDS)
def test_alphabet_taken_from_baseline_aligner(backend):
    rng, A, B, y = _data(15)
    baseline = make_aligner("local", random_params(rng, "affine", "general", alphabet="CATG"))
    res = _run_on(backend, A, B, y, "local", "affine", "general", baseline_aligner=baseline, max_iter=1)
    assert "".join(res["substitution_matrix"].alphabet) == "CATG"


@pytest.mark.parametrize("backend", BACKENDS)
def test_explicit_alphabet_wins(backend):
    _, A, B, y = _data(16)
    res = _run_on(backend, A, B, y, "local", "affine", "general", alphabet="GTAC", max_iter=1)
    assert "".join(res["substitution_matrix"].alphabet) == "GTAC"


@pytest.mark.parametrize("backend", BACKENDS)
def test_explicit_alphabet_missing_a_letter_fails(backend):
    _, A, B, y = _data(17)
    with pytest.raises((KeyError, IndexError, ValueError)):
        _run_on(backend, A, B, y, "local", "affine", "general", alphabet="ACG", max_iter=1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_protein_alphabet(backend):
    rng = np.random.default_rng(18)
    protein = "ACDEFGHIKLMNPQRSTVWY"
    A, B, y = make_pairs(rng, 5, 5, 15, alphabet=protein, sub_rate=0.2)
    res = _run_on(backend, A, B, y, "local", "affine", "symmetric", alphabet=protein, max_iter=2)
    assert np.asarray(res["substitution_matrix"]).shape == (20, 20)


# --- baselines used for the initial estimate --------------------------------

@pytest.mark.parametrize("mode, gap_mode, substitution_mode", ALL_MODES)
def test_initial_estimate_uses_default_baseline(mode, gap_mode, substitution_mode):
    _, A, B, y = _data(19, n=20)
    res = _run(A, B, y, mode, gap_mode, substitution_mode, max_iter=0, stepfunction=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = get_initial_estimate(
            align_all(A, B, default_baseline(mode, gap_mode, substitution_mode)), y,
            substitution_mode=substitution_mode, gap_mode=gap_mode, alphabet=DNA)
    _clip_gaps(expected)
    _assert_params_equal(res, expected, _param_keys(gap_mode, substitution_mode), rtol=1e-12)
    assert res["alpha"] == pytest.approx(expected["alpha"], rel=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_initial_estimate_uses_given_baseline_aligner(backend):
    rng, A, B, y = _data(20, n=20)
    baseline = make_aligner("global", random_params(rng, "linear", "simple"))
    res = _run_on(backend, A, B, y, "global", "linear", "simple", baseline_aligner=baseline,
               max_iter=0, stepfunction=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = get_initial_estimate(align_all(A, B, baseline), y, "simple", "linear")
    _clip_gaps(expected)
    _assert_params_equal(res, expected, {"match_score", "mismatch_score", "gap_score"}, rtol=1e-12)


# --- data edge cases --------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("gap_mode, substitution_mode", [("affine", "simple"), ("linear", "general")])
def test_local_pairs_without_alignment(gap_mode, substitution_mode, backend):
    A = ["AAAA", "ACGT", "CCCC", "ACGTAC", "GGGG", "ACGA"]
    B = ["CCCC", "ACGT", "GGGG", "ACGTAC", "TTTT", "ACGT"]
    y = np.array([0, 1, 0, 1, 0, 1])
    rng = np.random.default_rng(21)
    p0 = random_params(rng, gap_mode, substitution_mode)
    res = _run_on(backend, A, B, y, "local", gap_mode, substitution_mode, initial_parameters=p0,
               max_iter=2)
    for i in (0, 2, 4):
        assert isinstance(res["alignments"][i], EmptyLocalAlignment)
        assert res["alignment_logit_scores"][i] == pytest.approx(expit(res["alpha"]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_single_residue_sequences(backend):
    A = ["A", "C", "G", "T", "A", "C"]
    B = ["A", "C", "G", "A", "C", "T"]
    y = np.array([1, 1, 1, 0, 0, 0])
    rng = np.random.default_rng(22)
    res = _run_on(backend, A, B, y, "global", "affine", "general",
               initial_parameters=random_params(rng, "affine", "general"), max_iter=3)
    assert np.all(np.isfinite(np.asarray(res["substitution_matrix"])))


@pytest.mark.parametrize("backend", BACKENDS)
def test_list_labels_behave_like_array_labels(backend):
    rng, A, B, y = _data(23)
    p0 = random_params(rng, "affine", "simple")
    as_array = _run_on(backend, A, B, y, "local", "affine", "simple", initial_parameters=p0, max_iter=2)
    as_list = _run_on(backend, A, B, list(y), "local", "affine", "simple", initial_parameters=p0, max_iter=2)
    _assert_params_equal(as_list, as_array, _param_keys("affine", "simple"), rtol=0, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_verbose_prints_progress(capsys, backend):
    rng, A, B, y = _data(24)
    _run_on(backend, A, B, y, "local", "affine", "simple",
         initial_parameters=random_params(rng, "affine", "simple"), max_iter=1, verbose=True)
    out = capsys.readouterr().out
    assert "Alphabet:" in out
    assert "Start of iteration 0" in out
    assert "End of iteration 0" in out


# --- gap scores are projected onto <= 0 -------------------------------------

@pytest.mark.parametrize("backend", ["biopython", "nwgrad"])
@pytest.mark.parametrize("gap_mode, positive", [
    ("affine", {"open_gap_score": 0.4}),
    ("affine", {"extend_gap_score": 0.2}),
    ("affine", {"open_gap_score": 0.4, "extend_gap_score": 0.2}),
    ("linear", {"gap_score": 0.3}),
])
def test_positive_starting_gap_scores_are_projected(gap_mode, positive, backend):
    rng, A, B, y = _data(25)
    p0 = random_params(rng, gap_mode, "simple")
    negative = {k: p0[k] for k in GAP_KEYS if k in p0 and k not in positive}
    p0.update(positive)
    res = _run(A, B, y, "local", gap_mode, "simple", initial_parameters=p0,
               max_iter=0, stepfunction=None, backend=backend)
    for key in positive:
        assert res[key] == _MAX_GAP_SCORE
    for key, value in negative.items():
        assert res[key] == value
    assert p0[next(iter(positive))] > 0  # the caller's dict is not modified


@pytest.mark.parametrize("backend", ["biopython", "nwgrad"])
@pytest.mark.parametrize("mode, gap_mode", [(m, g) for m in MODES for g in GAP_MODES])
def test_gap_scores_never_become_positive(mode, gap_mode, backend):
    """Steps large enough to overshoot the cap are projected back onto it."""
    hits_cap = False
    for seed in range(4):
        rng, A, B, y = _data(26 + seed)
        p0 = random_params(rng, gap_mode, "simple")
        for key in GAP_KEYS:
            if key in p0:
                p0[key] = 2 * _MAX_GAP_SCORE
        res = _run(A, B, y, mode, gap_mode, "simple", initial_parameters=p0, max_iter=3,
                   stepfunction=create_constant_step(1.0), backend=backend)
        gaps = [res[k] for k in GAP_KEYS if k in res]
        assert all(g <= _MAX_GAP_SCORE for g in gaps)
        hits_cap |= any(g == _MAX_GAP_SCORE for g in gaps)
    assert hits_cap


@pytest.mark.parametrize("backend", BACKENDS)
def test_substitution_scores_are_not_projected(backend):
    rng, A, B, y = _data(30)
    p0 = random_params(rng, "affine", "simple")
    res = _run_on(backend, A, B, y, "local", "affine", "simple", initial_parameters=p0, max_iter=2)
    assert res["match_score"] > 0


def test_discrimalign_uses_default_stepfunction():
    result = discrimalign(
        seqlistA=["AUGCUA", "CUGA"],
        seqlistB=["AUGGUA", "CUGU"],
        labels=[1, 0],
        aligner_mode="local",
        gap_mode="affine",
        substitution_mode="symmetric",
        max_iter=1,
        num_threads=1,
    )

    assert "final_loglik" in result
    assert "alpha" in result


# --- alpha_solver -------------------------------------------------------------

def _record_alpha_fits(monkeypatch):
    """
    Record (scores, labels, alpha) for every alpha fit: discrimalign's Python
    _fit_alpha_unchecked, and nwgrad.logistic.step where the nwgrad backend uses it.
    """
    module = sys.modules["src.discrimalign"]
    calls = []
    original = module._fit_alpha_unchecked

    def recording(scores, labels, alpha0, **kwargs):
        alpha = original(scores, labels, alpha0, **kwargs)
        calls.append((np.array(scores, dtype=float), np.array(labels, dtype=float), alpha))
        return alpha

    monkeypatch.setattr(module, "_fit_alpha_unchecked", recording)

    backend = importlib.import_module("src.nwgrad_backend")
    logistic = backend.nwgrad_logistic

    class RecordingLogistic:
        @staticmethod
        def step(batch, labels, alpha0):
            st = logistic.step(batch, labels, alpha0)
            calls.append((np.array(batch.scores()), np.array(labels, dtype=float), st.alpha))
            return st

    monkeypatch.setattr(backend, "nwgrad_logistic", RecordingLogistic)
    return calls


def test_default_alpha_solver_is_safeguarded_newton():
    import inspect
    default = inspect.signature(discrimalign).parameters["alpha_solver"].default
    assert default == "safeguarded_newton"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode, gap_mode, substitution_mode",
                         [("local", "affine", "general"), ("global", "linear", "simple")])
def test_alpha_is_the_exact_optimum_at_every_iteration(backend, mode, gap_mode,
                                                       substitution_mode, monkeypatch):
    calls = _record_alpha_fits(monkeypatch)
    _, A, B, y = _data(20)
    res = _run_on(backend, A, B, y, mode, gap_mode, substitution_mode, max_iter=4)
    assert len(calls) == 4
    for scores, labels, alpha in calls:
        assert alpha == pytest.approx(optimal_alpha(scores, labels), abs=1e-10)
    assert res["alpha"] == calls[-1][2]


@pytest.mark.parametrize("backend", BACKENDS)
def test_alpha_stays_exact_when_the_optimum_jumps_far(backend, monkeypatch):
    """A large summed-gradient step moves every score far, so alpha* jumps between iterations."""
    calls = _record_alpha_fits(monkeypatch)
    _, A, B, y = _data(21)
    _run_on(backend, A, B, y, "local", "affine", "general", max_iter=4,
            stepfunction=create_constant_step(5.0))
    jumps = [abs(b[2] - a[2]) for a, b in zip(calls, calls[1:])]
    assert max(jumps) > 10
    for scores, labels, alpha in calls:
        ref = optimal_alpha(scores, labels, bracket=(alpha - 100.0, alpha + 100.0))
        assert alpha == pytest.approx(ref, abs=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_bfgs_alpha_solver_reaches_the_same_fit(backend):
    _, A, B, y = _data(22)
    newton = _run_on(backend, A, B, y, "local", "affine", "general", max_iter=3)
    bfgs = _run_on(backend, A, B, y, "local", "affine", "general", max_iter=3,
                   alpha_solver="bfgs")
    assert bfgs["alpha"] == pytest.approx(newton["alpha"], abs=1e-4)
    _assert_params_equal(bfgs, newton, _param_keys("affine", "general"), atol=1e-4)


def test_unknown_alpha_solver_is_rejected():
    _, A, B, y = _data(23)
    with pytest.raises(AssertionError):
        discrimalign(A, B, y, max_iter=0, alpha_solver="newton-cg")


@pytest.mark.parametrize("mode, gap_mode, substitution_mode", ALL_MODES)
def test_nwgrad_logistic_step_matches_the_python_path(mode, gap_mode, substitution_mode, monkeypatch):
    """nwgrad.logistic.step and the numpy path (the Biopython backend's) give the same fit."""
    backend = importlib.import_module("src.nwgrad_backend")
    _, A, B, y = _data(24)
    kwargs = dict(max_iter=5, stepfunction=create_constant_step(0.01))
    native = _run_on("nwgrad", A, B, y, mode, gap_mode, substitution_mode, **kwargs)
    # Without logistic_step, discrimalign() runs the numpy path on the nwgrad engine.
    monkeypatch.delattr(backend.NwgradEngine, "logistic_step")
    python = _run_on("nwgrad", A, B, y, mode, gap_mode, substitution_mode, **kwargs)
    assert native["alpha"] == pytest.approx(python["alpha"], rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(native["loglik_trajectory"], python["loglik_trajectory"], rtol=1e-12)
    _assert_params_equal(native, python, _param_keys(gap_mode, substitution_mode),
                         rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("fill", ["rowwise", "interpair"])
@pytest.mark.parametrize("mode, gap_mode, substitution_mode", ALL_MODES)
def test_fill_gives_the_same_fit(mode, gap_mode, substitution_mode, fill):
    """nwgrad_fill changes only the speed: the fit is bit-identical."""
    _, A, B, y = _data(25)
    kwargs = dict(max_iter=5, stepfunction=create_constant_step(0.01))
    striped = _run_on("nwgrad", A, B, y, mode, gap_mode, substitution_mode, **kwargs)
    other = _run_on("nwgrad", A, B, y, mode, gap_mode, substitution_mode,
                      nwgrad_fill=fill, **kwargs)
    assert other["alpha"] == striped["alpha"]
    assert other["loglik_trajectory"] == striped["loglik_trajectory"]
    _assert_params_equal(other, striped, _param_keys(gap_mode, substitution_mode),
                         rtol=0, atol=0)


def test_nwgrad_fill_is_validated():
    _, A, B, y = _data(26)
    with pytest.raises(ValueError, match="nwgrad_fill must be"):
        discrimalign(A, B, y, max_iter=0, backend="nwgrad", nwgrad_fill="diagonal")
    with pytest.raises(ValueError, match="nwgrad backend only"):
        discrimalign(A, B, y, max_iter=0, backend="biopython", nwgrad_fill="rowwise")

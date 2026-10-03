"""
The nwgrad backend must follow the Biopython backend's parameter trajectory.

Both backends find optimal paths at the same parameters. Where raw counts can
differ (ties in simple or linear mode), the gradient in the model's own
parameters does not, so trajectories agree up to rounding.
"""
import sys
import warnings

import numpy as np
import nwgrad
import pytest
from Bio.Align import PairwiseAligner, substitution_matrices

from src.discrimalign import _MAX_GAP_SCORE, discrimalign
from src.nwgrad_backend import baseline_parameters
from src.optimization import (EmptyLocalAlignment, create_constant_step, create_powerstep,
                              get_initial_estimate_from_counts)
from tests.helpers import (GAP_MODES, MODES, SUBSTITUTION_MODES, default_baseline,
                           make_aligner, make_pairs, random_params)

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


@pytest.mark.parametrize("mode, gap_mode, substitution_mode", ALL_MODES)
def test_trajectory_matches_biopython_from_initial_estimate(mode, gap_mode, substitution_mode):
    """From a tie-free baseline, both backends fit the same initial estimate."""
    rng, A, B, y = _data(2, n=20)
    baseline = default_baseline(mode, gap_mode, substitution_mode, perturb=rng)
    bio, nw = _both(A, B, y, mode, gap_mode, substitution_mode, max_iter=3,
                    stepfunction=create_powerstep(1e-3), baseline_aligner=baseline)
    _assert_same_fit(bio, nw, gap_mode, substitution_mode)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("mode, gap_mode, substitution_mode", ALL_MODES)
def test_initial_estimate_matches_biopython_from_tie_free_baseline(mode, gap_mode,
                                                                   substitution_mode, seed):
    rng, A, B, y = _data(20 + seed, n=20)
    baseline = default_baseline(mode, gap_mode, substitution_mode, perturb=rng)
    bio, nw = _both(A, B, y, mode, gap_mode, substitution_mode, max_iter=0,
                    stepfunction=None, baseline_aligner=baseline)
    _assert_same_fit(bio, nw, gap_mode, substitution_mode, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode, gap_mode, substitution_mode", ALL_MODES)
def test_nwgrad_initial_estimate_is_fitted_on_nwgrad_counts(mode, gap_mode, substitution_mode):
    """With the default (tie-prone) baseline: self-consistent, whatever ties nwgrad broke."""
    from src.nwgrad_backend import NwgradEngine, baseline_parameters
    _, A, B, y = _data(3, n=20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = discrimalign(A, B, y, aligner_mode=mode, gap_mode=gap_mode,
                           substitution_mode=substitution_mode, max_iter=0, backend="nwgrad")
        engine = NwgradEngine(A, B, mode, gap_mode, substitution_mode, "ACGT", 1)
        engine.set_params(baseline_parameters(default_baseline(mode, gap_mode, substitution_mode),
                                              gap_mode, "ACGT"), substitution_mode="general")
        engine.scores()
        expected = get_initial_estimate_from_counts(engine.raw_counts(), y, substitution_mode,
                                                    gap_mode, "ACGT")
    for key in _param_keys(gap_mode, substitution_mode) - {"alpha", "final_loglik"}:
        value = np.asarray(expected[key], dtype=float)
        if key in ("open_gap_score", "extend_gap_score", "gap_score"):
            value = np.minimum(value, _MAX_GAP_SCORE)
        np.testing.assert_allclose(np.asarray(res[key], dtype=float), value, rtol=1e-12, err_msg=key)
    assert res["alpha"] == pytest.approx(expected["alpha"], rel=1e-12)


NWGRAD_HAS_GRADS = hasattr(nwgrad.SeqPairBatchDouble, "grads")


@pytest.mark.parametrize("path", [
    "count_arrays", "_count_arrays_per_pair",
    pytest.param("_count_arrays_bulk", marks=pytest.mark.skipif(
        not NWGRAD_HAS_GRADS, reason="nwgrad without SeqPairBatch.grads()"))])
@pytest.mark.parametrize("mode, gap_mode, substitution_mode", ALL_MODES)
def test_count_arrays_equal_raw_counts(mode, gap_mode, substitution_mode, path):
    """count_arrays(), by either path, gives exactly the per-pair counts of raw_counts()."""
    from src.nwgrad_backend import NwgradEngine
    from src.optimization import _count_arrays_from_raw
    rng, A, B, _ = _data(5, n=20)
    engine = NwgradEngine(A, B, mode, gap_mode, substitution_mode, "ACGT", 2)
    baseline = baseline_parameters(default_baseline(mode, gap_mode, substitution_mode),
                                   gap_mode, "ACGT")
    for params, params_mode in ((baseline, "general"),
                                (random_params(rng, gap_mode, substitution_mode), None)):
        engine.set_params(params, substitution_mode=params_mode)
        engine.scores()
        counts = getattr(engine, path)()
        expected = _count_arrays_from_raw(engine.raw_counts())
        assert counts.alphabet == expected.alphabet
        for field in ("substitutions", "gap_opens", "gap_extends"):
            np.testing.assert_array_equal(getattr(counts, field), getattr(expected, field),
                                          err_msg=field)


@pytest.mark.parametrize("initial", [True, False])
def test_nwgrad_backend_needs_no_biopython_alignment(initial, monkeypatch):
    """Without returned alignments, the nwgrad path never aligns with Biopython."""
    module = sys.modules["src.discrimalign"]

    def fail(*args, **kwargs):
        raise AssertionError("Biopython alignment on the nwgrad path")

    monkeypatch.setattr(module, "_align_pairs", fail)
    rng, A, B, y = _data(4, n=20)
    kwargs = {"initial_parameters": random_params(rng, "affine", "general")} if initial else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = discrimalign(A, B, y, aligner_mode="local", gap_mode="affine",
                           substitution_mode="general", max_iter=2, return_alignments=False,
                           stepfunction=create_constant_step(0.01), backend="nwgrad", **kwargs)
    assert "alignments" not in res


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


def test_default_backend_is_nwgrad():
    rng, A, B, y = _data(9)
    p0 = random_params(rng, "affine", "symmetric")
    kwargs = dict(aligner_mode="local", gap_mode="affine", substitution_mode="symmetric",
                  initial_parameters=p0, max_iter=2, stepfunction=create_constant_step(0.01))
    default = discrimalign(A, B, y, **kwargs)
    explicit = discrimalign(A, B, y, backend="nwgrad", **kwargs)
    for key in _param_keys("affine", "symmetric") | set(TRAJECTORY_KEYS):
        np.testing.assert_array_equal(np.asarray(default[key]), np.asarray(explicit[key]), err_msg=key)


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
        "Positive gap scores are outside local alignment's domain: nwgrad lets a "
        "local alignment end with a gap column, and Biopython's align() and "
        "score() disagree with each other. discrimalign keeps gap scores negative."))),
    ("AAA", "TAAA"),
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


# --- baseline_parameters: strict conversion of a Biopython aligner ----------

@pytest.mark.parametrize("gap_mode", GAP_MODES)
@pytest.mark.parametrize("substitution_mode", SUBSTITUTION_MODES)
def test_baseline_parameters_of_default_aligner(gap_mode, substitution_mode):
    p = baseline_parameters(default_baseline("local", gap_mode, substitution_mode), gap_mode, "ACGT")
    if gap_mode == "affine":
        assert (p["open_gap_score"], p["extend_gap_score"]) == (-8.0, -0.5)
    else:
        assert p["gap_score"] == -6.0
    assert "".join(p["substitution_matrix"].alphabet) == "ACGT"
    np.testing.assert_array_equal(np.asarray(p["substitution_matrix"]), 9 * np.eye(4) - 4)


def test_baseline_parameters_reorders_matrix_to_alphabet():
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.Array(
        alphabet="TGCAN", data=np.arange(25.0).reshape(5, 5))
    M = baseline_parameters(aligner, "affine", "ACGT")["substitution_matrix"]
    assert "".join(M.alphabet) == "ACGT"
    src = aligner.substitution_matrix
    for a in "ACGT":
        for b in "ACGT":
            assert M[a, b] == src[a, b]


def test_baseline_parameters_affine_aligner_in_affine_mode_of_linear_scores():
    aligner = PairwiseAligner()
    aligner.gap_score = -3
    p = baseline_parameters(aligner, "affine", "ACGT")
    assert (p["open_gap_score"], p["extend_gap_score"]) == (-3.0, -3.0)


def _aligner(**scores):
    aligner = PairwiseAligner()
    aligner.match_score, aligner.mismatch_score = 2, -1
    aligner.open_gap_score, aligner.extend_gap_score = -5, -1
    for name, value in scores.items():
        setattr(aligner, name, value)
    return aligner


@pytest.mark.parametrize("gap_mode, scores, message", [
    ("affine", {"end_gap_score": 0}, "internal and end gaps"),
    ("affine", {"open_insertion_score": -2}, "same for both sequences"),
    ("linear", {}, "open must equal extend"),
    ("affine", {"wildcard": "N"}, "wildcard"),
    ("affine", {"mode": "fogsaa"}, "not supported"),
])
def test_baseline_parameters_rejects_what_the_model_cannot_express(gap_mode, scores, message):
    with pytest.raises(ValueError, match=message):
        baseline_parameters(_aligner(**scores), gap_mode, "ACGT")


def test_baseline_parameters_rejects_matrix_missing_letters():
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.Array(alphabet="ACG", data=np.eye(3))
    with pytest.raises(ValueError, match="lacks letters 'T'"):
        baseline_parameters(aligner, "affine", "ACGT")


def test_nwgrad_backend_rejects_nonuniform_baseline_aligner():
    _, A, B, y = _data(5)
    with pytest.raises(ValueError, match="internal and end gaps"):
        discrimalign(A, B, y, aligner_mode="global", gap_mode="affine", substitution_mode="simple",
                     baseline_aligner=_aligner(end_gap_score=0), max_iter=0, backend="nwgrad")


# --- inputs beyond short DNA ------------------------------------------------
#
# Continuous random starting parameters, so neither the start nor the fitted
# point has exactly equal scores that the backends could break differently.

PROTEIN = "ACDEFGHIKLMNPQRSTVWY"


@pytest.mark.parametrize("mode, gap_mode, substitution_mode",
                         [("local", "affine", "symmetric"), ("global", "linear", "general")])
def test_protein_trajectory_matches_biopython(mode, gap_mode, substitution_mode):
    rng = np.random.default_rng(50)
    A, B, y = make_pairs(rng, 5, 5, 60, alphabet=PROTEIN, sub_rate=0.3)
    p0 = random_params(rng, gap_mode, substitution_mode, alphabet=PROTEIN)
    bio, nw = _both(A, B, y, mode, gap_mode, substitution_mode, initial_parameters=p0,
                    alphabet=PROTEIN)
    _assert_same_fit(bio, nw, gap_mode, substitution_mode)


@pytest.mark.parametrize("mode, gap_mode, substitution_mode",
                         [("local", "affine", "general"), ("global", "linear", "simple"),
                          ("global", "affine", "symmetric")])
def test_long_sequence_trajectory_matches_biopython(mode, gap_mode, substitution_mode):
    rng = np.random.default_rng(51)
    A, B, y = make_pairs(rng, 3, 3, 400, sub_rate=0.2, indel_rate=0.05)
    p0 = random_params(rng, gap_mode, substitution_mode)
    bio, nw = _both(A, B, y, mode, gap_mode, substitution_mode, initial_parameters=p0)
    _assert_same_fit(bio, nw, gap_mode, substitution_mode)


def test_lowercase_sequences_match_biopython():
    rng, A, B, y = _data(52)
    A, B = [a.lower() for a in A], [b.lower() for b in B]
    p0 = random_params(rng, "affine", "general", alphabet="acgt")
    bio, nw = _both(A, B, y, "local", "affine", "general", initial_parameters=p0)
    _assert_same_fit(bio, nw, "affine", "general")


@pytest.mark.parametrize("backend", ["biopython", "nwgrad"])
def test_out_of_alphabet_letter_raises(backend):
    _, A, B, y = _data(53)
    A[2] = A[2][:3] + "X" + A[2][4:]
    with pytest.raises(ValueError):
        discrimalign(A, B, y, aligner_mode="local", gap_mode="affine", substitution_mode="general",
                     alphabet="ACGT", max_iter=1, stepfunction=create_constant_step(0.01),
                     backend=backend)


@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("backend", ["biopython", "nwgrad"])
def test_automatic_thread_count(backend, explicit, monkeypatch):
    """num_threads=0, the default: all logical cores with nwgrad, 1 with Biopython."""
    import src.nwgrad_backend as nwgrad_backend
    module = sys.modules["src.discrimalign"]
    seen = []
    engine_init = nwgrad_backend.NwgradEngine.__init__
    align_pairs = module._align_pairs

    def spy_engine(self, *args, **kwargs):
        seen.append(args[-1])
        engine_init(self, *args, **kwargs)

    def spy_align(seqsA, seqsB, aligner, num_threads):
        seen.append(num_threads)
        return align_pairs(seqsA, seqsB, aligner, num_threads)

    monkeypatch.setattr(nwgrad_backend.NwgradEngine, "__init__", spy_engine)
    monkeypatch.setattr(module, "_align_pairs", spy_align)
    monkeypatch.setattr(module.os, "cpu_count", lambda: 3)
    rng, A, B, y = _data(54)
    p0 = random_params(rng, "affine", "general")
    kwargs = dict(aligner_mode="local", gap_mode="affine", substitution_mode="general",
                  initial_parameters=p0, max_iter=2, stepfunction=create_constant_step(0.01),
                  backend=backend)
    auto = discrimalign(A, B, y, **({"num_threads": 0} if explicit else {}), **kwargs)
    expected = 3 if backend == "nwgrad" else 1
    assert seen and set(seen) == {expected}
    seen.clear()
    one = discrimalign(A, B, y, num_threads=1, **kwargs)
    for key in _param_keys("affine", "general") | set(TRAJECTORY_KEYS):
        np.testing.assert_array_equal(np.asarray(auto[key]), np.asarray(one[key]), err_msg=key)


@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("gap_mode", ["affine", "linear"])
@pytest.mark.parametrize("substitution_mode", ["general", "simple"])
def test_nwgrad_alignments_are_the_paths_nwgrad_scored(mode, gap_mode, substitution_mode):
    """The returned alignments are nwgrad's own paths: counting them gives exactly the
    per-pair counts nwgrad's gradient used at the final parameters, ties included
    (simple mode has many), and their scores are nwgrad's scores."""
    if not hasattr(nwgrad.SeqPairDouble, "coordinates"):
        pytest.skip("nwgrad without SeqPair.coordinates()")
    from src.logit_link import logit_subgradient
    from src.nwgrad_backend import NwgradEngine
    rng, A, B, y = _data(31, n=16, length=14)
    res = discrimalign(A, B, y, aligner_mode=mode, gap_mode=gap_mode,
                       substitution_mode=substitution_mode, backend="nwgrad", max_iter=3,
                       stepfunction=create_constant_step(0.01), alphabet="ACGT")
    engine = NwgradEngine(A, B, mode, gap_mode, substitution_mode, "ACGT", 2)
    engine.set_params(res)
    scores = engine.scores()
    counts = engine.count_arrays()
    for i, aln in enumerate(res["alignments"]):
        assert aln.score == scores[i]
        c = logit_subgradient([aln], [0.0], [1], res["alpha"], "ACGT")
        np.testing.assert_array_equal(np.asarray(c["Substitutions"]), counts.substitutions[i])
        if gap_mode == "affine":
            assert (c["Gap opens"], c["Gap extends"]) == (counts.gap_opens[i], counts.gap_extends[i])
        else:   # the linear model reports gap columns only
            assert c["Gap opens"] + c["Gap extends"] == counts.gap_opens[i] + counts.gap_extends[i]

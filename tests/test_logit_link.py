"""Unit tests for src.logit_link."""
import numpy as np
import pytest
from Bio.Align import PairwiseAligner

from src.logit_link import (d2lda2, dlda, fit_alpha, logit_logL, logit_partial_scores,
                            logit_subgradient)
from src.optimization import EmptyLocalAlignment
from tests.helpers import optimal_alpha


# --- logit_partial_scores ---------------------------------------------------

def test_partial_scores_is_sigmoid_of_shifted_scores():
    scores = [-3.0, -0.5, 0.0, 1.25, 4.0]
    alpha = 0.7
    expected = [1.0 / (1.0 + np.exp(-(alpha + s))) for s in scores]
    np.testing.assert_allclose(logit_partial_scores(scores, alpha), expected, rtol=1e-14)


def test_partial_scores_zero_gives_half():
    np.testing.assert_allclose(logit_partial_scores([0.0, -2.0], [0.0, 2.0]), [0.5, 0.5])


def test_partial_scores_accepts_scalar_and_returns_array():
    out = logit_partial_scores(1.0, 0.0)
    assert isinstance(out, (np.ndarray, np.floating))
    assert out == pytest.approx(1.0 / (1.0 + np.exp(-1.0)))


def test_partial_scores_accepts_integer_scores():
    out = logit_partial_scores([0, 1, 2], 0)
    assert out.dtype == float


def test_partial_scores_extreme_values_saturate_without_nan():
    out = logit_partial_scores([-1e4, 1e4], 0.0)
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0
    assert out[1] == 1.0


def test_partial_scores_monotone_in_score():
    scores = np.linspace(-10, 10, 41)
    out = logit_partial_scores(scores, -1.3)
    assert np.all(np.diff(out) > 0)


def test_partial_scores_empty_input():
    assert logit_partial_scores([], 0.0).shape == (0,)


# --- logit_logL -------------------------------------------------------------

def test_logL_matches_bernoulli_formula():
    p = np.array([0.1, 0.4, 0.8, 0.95])
    y = np.array([0, 1, 1, 0])
    expected = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert logit_logL(p, y) == pytest.approx(expected, rel=1e-14)


def test_logL_returns_python_float():
    assert type(logit_logL([0.3], [1])) is float


def test_logL_is_nonpositive():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, size=50)
    y = rng.integers(0, 2, size=50)
    assert logit_logL(p, y) <= 0.0


def test_logL_perfect_prediction_is_near_zero():
    assert logit_logL([1.0, 0.0], [1, 0]) == pytest.approx(0.0, abs=1e-12)


def test_logL_clips_certain_wrong_predictions_to_finite_value():
    value = logit_logL([0.0, 1.0], [1, 0])
    assert np.isfinite(value)
    assert value < -60  # 2 * log(eps) is about -72


def test_logL_accepts_lists():
    assert logit_logL([0.25, 0.75], [0, 1]) == pytest.approx(2 * np.log(0.75))


def test_logL_accepts_boolean_labels():
    assert logit_logL([0.25, 0.75], np.array([False, True])) == pytest.approx(2 * np.log(0.75))


@pytest.mark.parametrize("labels", [[0, 2], [-1, 1], [0.5, 1]])
def test_logL_rejects_non_binary_labels(labels):
    with pytest.raises(ValueError):
        logit_logL([0.3, 0.6], labels)


def test_logL_empty_is_zero():
    assert logit_logL([], []) == 0.0


# --- dlda and d2lda2 --------------------------------------------------------

def _logL_of_alpha(alpha, scores, labels):
    return logit_logL(logit_partial_scores(scores, alpha), labels)


@pytest.mark.parametrize("seed", range(5))
def test_dlda_matches_finite_difference(seed):
    rng = np.random.default_rng(seed)
    scores = rng.normal(0, 3, size=20)
    labels = rng.integers(0, 2, size=20)
    alpha, h = rng.normal(), 1e-6
    numeric = (_logL_of_alpha(alpha + h, scores, labels)
               - _logL_of_alpha(alpha - h, scores, labels)) / (2 * h)
    analytic = dlda(logit_partial_scores(scores, alpha), labels)
    assert analytic == pytest.approx(numeric, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("seed", range(5))
def test_d2lda2_matches_finite_difference(seed):
    rng = np.random.default_rng(seed)
    scores = rng.normal(0, 3, size=20)
    labels = rng.integers(0, 2, size=20)
    alpha, h = rng.normal(), 1e-5
    d = lambda a: dlda(logit_partial_scores(scores, a), labels)
    numeric = (d(alpha + h) - d(alpha - h)) / (2 * h)
    analytic = d2lda2(logit_partial_scores(scores, alpha), labels)
    assert analytic == pytest.approx(numeric, rel=1e-6)


def test_d2lda2_is_negative():
    p = np.array([0.2, 0.5, 0.9])
    assert d2lda2(p, [0, 1, 1]) < 0


def test_dlda_vanishes_when_mean_prediction_matches_mean_label():
    assert dlda(np.array([0.25, 0.75]), np.array([0, 1])) == pytest.approx(0.0)


# --- fit_alpha -------------------------------------------------------------

def _alpha_data(seed, n=500, offset=0.0):
    rng = np.random.default_rng(seed)
    scores = rng.normal(-3.0, 2.0, n) + offset
    labels = (rng.random(n) < 1.0 / (1.0 + np.exp(-(scores - offset + 1.5)))).astype(int)
    return scores, labels


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("start", [0.0, 1e-3, 0.5, -5.0, 20.0, -500.0, 500.0])
def test_fit_alpha_finds_the_root_from_any_start(seed, start):
    scores, labels = _alpha_data(seed)
    ref = optimal_alpha(scores, labels)
    alpha = fit_alpha(scores, labels, ref + start)
    assert alpha == pytest.approx(ref, abs=1e-10)
    assert abs(dlda(logit_partial_scores(scores, alpha), labels)) < 1e-8


@pytest.mark.parametrize("offset", [300.0, -300.0, 1600.0])
def test_fit_alpha_follows_a_large_jump_of_the_optimum(offset):
    """Every score shifted far, as after an unscaled step on large data: alpha* moves by -offset."""
    scores, labels = _alpha_data(1, offset=offset)
    ref = optimal_alpha(scores, labels, bracket=(-offset - 200.0, -offset + 200.0))
    assert fit_alpha(scores, labels, 0.0) == pytest.approx(ref, abs=1e-9)


def test_fit_alpha_does_not_depend_on_the_start():
    scores, labels = _alpha_data(2)
    results = [fit_alpha(scores, labels, start) for start in (-50.0, -1.0, 0.0, 3.0, 50.0)]
    assert max(results) - min(results) < 1e-10


def test_fit_alpha_matches_mean_label_when_scores_are_equal():
    labels = np.array([1, 0, 0, 1, 1, 0, 1, 1])
    alpha = fit_alpha(np.zeros(len(labels)), labels, 0.0)
    assert 1.0 / (1.0 + np.exp(-alpha)) == pytest.approx(labels.mean(), abs=1e-12)


def test_fit_alpha_returns_python_float_and_accepts_lists():
    alpha = fit_alpha([0.0, 1.0, -1.0, 2.0], [1, 0, 0, 1], 0)
    assert isinstance(alpha, float)


@pytest.mark.parametrize("labels", [[1, 1, 1], [0, 0, 0]])
def test_fit_alpha_needs_both_classes(labels):
    with pytest.raises(ValueError, match="both classes"):
        fit_alpha([0.0, 1.0, 2.0], labels, 0.0)


# --- logit_subgradient on hand-built alignments -----------------------------
#
# logit_subgradient only indexes aln[0] and aln[1], so a pair of gapped
# strings stands in for an Alignment object.

def _single(aln, alphabet="ACGT", weight=1.0):
    """Subgradient of one alignment with label - logit_score == weight."""
    return logit_subgradient([aln], [0.0], [weight], 0.0, alphabet)


def test_subgradient_counts_matches_and_mismatches():
    sg = _single(("ACGT", "ACCT"))
    G = np.asarray(sg["Substitutions"])
    assert G.sum() == 4
    assert sg["Substitutions"]["A", "A"] == 1
    assert sg["Substitutions"]["C", "C"] == 1
    assert sg["Substitutions"]["G", "C"] == 1
    assert sg["Substitutions"]["C", "G"] == 0  # direction matters
    assert sg["Substitutions"]["T", "T"] == 1
    assert sg["Gap opens"] == 0
    assert sg["Gap extends"] == 0


def test_subgradient_matrix_carries_alphabet_order():
    sg = _single(("AC", "AC"), alphabet="TGCA")
    assert "".join(sg["Substitutions"].alphabet) == "TGCA"
    assert np.asarray(sg["Substitutions"])[3, 3] == 1  # A is last


@pytest.mark.parametrize("aln, opens, extends", [
    (("A-C", "AGC"), 1, 0),
    (("A--C", "AGGC"), 1, 1),
    (("A---C", "AGGGC"), 1, 2),
    (("AGC", "A-C"), 1, 0),
    (("AGGGC", "A---C"), 1, 2),
    (("-AC", "GAC"), 1, 0),            # leading gap
    (("AC--", "ACGT"), 1, 1),          # trailing gap
    (("A-C-G", "ATCTG"), 2, 0),        # two gaps separated by a match
    (("A-CG", "AT-G"), 2, 0),          # gap in A, then gap in B
    (("A--C--", "AGTCGT"), 2, 2),
    (("--", "AC"), 1, 1),              # all gaps
])
def test_subgradient_gap_counts(aln, opens, extends):
    sg = _single(aln)
    assert sg["Gap opens"] == opens
    assert sg["Gap extends"] == extends


def test_subgradient_alternating_gaps_are_all_opens():
    """Switching between gap-in-A and gap-in-B opens a new gap each time."""
    sg = _single(("-A-", "C-G"))
    assert sg["Gap opens"] == 3
    assert sg["Gap extends"] == 0


def test_subgradient_gap_after_match_reopens():
    sg = _single(("A-CC-A", "AGCCTA"))
    assert sg["Gap opens"] == 2
    assert sg["Gap extends"] == 0


def test_subgradient_weight_is_label_minus_logit():
    sg = logit_subgradient([("AC", "AC")], [0.3], [1], 0.0, "ACGT")
    assert sg["Substitutions"]["A", "A"] == pytest.approx(0.7)
    sg = logit_subgradient([("AC", "AC")], [0.3], [0], 0.0, "ACGT")
    assert sg["Substitutions"]["A", "A"] == pytest.approx(-0.3)


def test_subgradient_ignores_alpha_argument():
    a = logit_subgradient([("A-C", "AGC")], [0.4], [1], -5.0, "ACGT")
    b = logit_subgradient([("A-C", "AGC")], [0.4], [1], 7.0, "ACGT")
    np.testing.assert_array_equal(np.asarray(a["Substitutions"]), np.asarray(b["Substitutions"]))
    assert a["Gap opens"] == b["Gap opens"]


def test_subgradient_sums_over_alignments_linearly():
    alns = [("A-C", "AGC"), ("GGT", "GAT"), ("AC--", "ACGT")]
    logits = [0.2, 0.9, 0.5]
    labels = [1, 0, 1]
    total = logit_subgradient(alns, logits, labels, 0.0, "ACGT")
    parts = [logit_subgradient([a], [p], [y], 0.0, "ACGT")
             for a, p, y in zip(alns, logits, labels)]
    np.testing.assert_allclose(np.asarray(total["Substitutions"]),
                               sum(np.asarray(p["Substitutions"]) for p in parts))
    assert total["Gap opens"] == pytest.approx(sum(p["Gap opens"] for p in parts))
    assert total["Gap extends"] == pytest.approx(sum(p["Gap extends"] for p in parts))


def test_subgradient_zero_when_predictions_are_exact():
    sg = logit_subgradient([("ACG", "A-G"), ("TT", "TA")], [1.0, 0.0], [1, 0], 0.0, "ACGT")
    assert np.all(np.asarray(sg["Substitutions"]) == 0)
    assert sg["Gap opens"] == 0
    assert sg["Gap extends"] == 0


def test_subgradient_of_empty_local_alignment_is_zero():
    sg = logit_subgradient([EmptyLocalAlignment()], [0.2], [1], 0.0, "ACGT")
    assert np.all(np.asarray(sg["Substitutions"]) == 0)
    assert sg["Gap opens"] == 0
    assert sg["Gap extends"] == 0


def test_subgradient_empty_list():
    sg = logit_subgradient([], [], [], 0.0, "AC")
    assert np.asarray(sg["Substitutions"]).shape == (2, 2)
    assert sg["Gap opens"] == 0


def test_subgradient_rejects_length_mismatch():
    with pytest.raises(AssertionError):
        logit_subgradient([("A", "A")], [0.1, 0.2], [1, 0], 0.0, "ACGT")
    with pytest.raises(AssertionError):
        logit_subgradient([("A", "A")], [0.1], [1, 0], 0.0, "ACGT")


def test_subgradient_rejects_character_outside_alphabet():
    with pytest.raises((KeyError, IndexError, ValueError)):
        _single(("AX", "AA"), alphabet="ACGT")


def test_subgradient_on_biopython_alignment():
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -3
    aligner.extend_gap_score = -1
    aln = next(aligner.align("ACGTTGCA", "ACGGCA"))
    sg = _single(aln)
    G = np.asarray(sg["Substitutions"])
    counts = aln.counts()
    assert np.trace(G) == counts.identities
    assert G.sum() - np.trace(G) == counts.mismatches
    assert sg["Gap opens"] == counts.open_gaps
    assert sg["Gap extends"] == counts.extend_gaps

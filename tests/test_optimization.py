"""Unit tests for src.optimization."""
import warnings

import numpy as np
import pytest
from Bio.Align import PairwiseAligner, substitution_matrices
from joblib import Parallel
from scipy.special import expit

from src.optimization import (EmptyAlignmentCounts, EmptyLocalAlignment,
                              _count_arrays_from_raw, _count_features, _fit_initial_estimate,
                              create_alignment_workers, create_constant_step,
                              create_powerstep, get_first_alignment,
                              get_initial_estimate, get_initial_estimate_from_counts)
from tests.helpers import DNA, align_all, make_aligner, make_pairs, random_params


# --- EmptyLocalAlignment ----------------------------------------------------

def test_empty_local_alignment_score_is_zero():
    assert EmptyLocalAlignment().score == 0.0


def test_empty_local_alignment_counts_are_zero():
    counts = EmptyLocalAlignment().counts()
    for field in ("identities", "mismatches", "open_gaps", "extend_gaps", "gaps"):
        assert getattr(counts, field) == 0


def test_empty_local_alignment_rows_are_empty_strings():
    aln = EmptyLocalAlignment()
    assert aln[0] == ""
    assert aln[1] == ""


@pytest.mark.parametrize("index", [2, -1, 5])
def test_empty_local_alignment_other_rows_raise(index):
    with pytest.raises(IndexError):
        EmptyLocalAlignment()[index]


def test_empty_alignment_counts_class_attributes():
    assert EmptyAlignmentCounts.gaps == 0


# --- get_first_alignment ----------------------------------------------------

def _simple_aligner(mode):
    aligner = PairwiseAligner()
    aligner.mode = mode
    aligner.match_score = 5
    aligner.mismatch_score = -4
    aligner.open_gap_score = -8
    aligner.extend_gap_score = -0.5
    return aligner


def test_first_alignment_global():
    aln = get_first_alignment("ACGT", "ACGT", _simple_aligner("global"))
    assert aln.score == 20
    assert aln[0] == "ACGT"


def test_first_alignment_local_substring():
    aln = get_first_alignment("TTTACGTTTT", "GGACGGG", _simple_aligner("local"))
    assert aln.score == 15  # ACG


@pytest.mark.parametrize("a, b", [("AAAA", "CCCC"), ("A", "C"), ("ACAC", "GTGT")])
def test_first_alignment_local_without_positive_score_is_empty(a, b):
    aln = get_first_alignment(a, b, _simple_aligner("local"))
    assert isinstance(aln, EmptyLocalAlignment)
    assert aln.score == 0.0


def test_first_alignment_global_always_exists_for_unrelated_sequences():
    aln = get_first_alignment("AAAA", "CCCC", _simple_aligner("global"))
    assert aln.score < 0


def test_first_alignment_reraises_stopiteration_outside_local_mode():
    class NoAlignments:
        mode = "global"

        def align(self, a, b):
            return iter(())

    with pytest.raises(StopIteration):
        get_first_alignment("A", "C", NoAlignments())


# --- create_alignment_workers -----------------------------------------------

@pytest.mark.parametrize("mode", ["local", "global"])
def test_alignment_workers_reproduce_serial_alignments(mode):
    rng = np.random.default_rng(1)
    seqsA, seqsB, _ = make_pairs(rng, 4, 4, 12)
    aligner = _simple_aligner(mode)
    parallel = Parallel(n_jobs=2, prefer="threads")
    alns = parallel(create_alignment_workers(seqsA, seqsB, aligner))
    expected = align_all(seqsA, seqsB, aligner)
    assert [a.score for a in alns] == [e.score for e in expected]


def test_alignment_workers_is_lazy_and_sized_by_input():
    gen = create_alignment_workers(["A", "C", "G"], ["A", "C", "G"], _simple_aligner("global"))
    assert len(list(gen)) == 3


# --- step functions ---------------------------------------------------------

@pytest.mark.parametrize("scale", [0.0, 0.01, 3.5])
def test_constant_step(scale):
    step = create_constant_step(scale)
    assert [step(i) for i in range(5)] == [scale] * 5


def test_powerstep_default_is_inverse_square_root():
    step = create_powerstep(2.0)
    for i in range(10):
        assert step(i) == pytest.approx(2.0 / np.sqrt(i + 1))


@pytest.mark.parametrize("power", [0.0, 0.5, 1.0, 2.0])
def test_powerstep_power(power):
    step = create_powerstep(1.0, power=power)
    assert step(3) == pytest.approx(4.0 ** -power)


def test_powerstep_burnin_is_constant_then_decays_from_scale():
    step = create_powerstep(1.0, power=1.0, burnin=3)
    assert [step(i) for i in range(3)] == [1.0, 1.0, 1.0]
    assert step(3) == pytest.approx(1.0)  # first post-burnin step is still the full scale
    assert step(4) == pytest.approx(0.5)
    assert step(12) == pytest.approx(0.1)


def test_powerstep_is_nonincreasing():
    step = create_powerstep(0.3, power=0.7, burnin=2)
    values = [step(i) for i in range(50)]
    assert all(a >= b for a, b in zip(values, values[1:]))


# --- initial estimate on synthetic count features ---------------------------
#
# Fake alignments with prescribed count features, labelled by a known logistic
# model. Fitting recovers the model, which pins down the mapping from
# regression coefficients to parameter names.

class FakeCounts:
    def __init__(self, identities, mismatches, open_gaps, extend_gaps):
        self.identities = identities
        self.mismatches = mismatches
        self.open_gaps = open_gaps
        self.extend_gaps = extend_gaps
        self.gaps = open_gaps + extend_gaps


class FakeAlignment:
    def __init__(self, row0, row1, counts):
        self.rows = (row0, row1)
        self._counts = counts

    def counts(self):
        return self._counts

    def __getitem__(self, index):
        return self.rows[index]


def _fake_simple_data(rng, n, coefs, alpha, linear=False):
    alns, labels = [], []
    for _ in range(n):
        ident, mism = rng.poisson(4), rng.poisson(3)
        opens, extends = rng.poisson(1.0), rng.poisson(1.5)
        if linear:
            x = np.array([ident, mism, opens + extends])
        else:
            x = np.array([ident, mism, opens, extends])
        labels.append(int(rng.random() < expit(alpha + x @ coefs)))
        alns.append(FakeAlignment("", "", FakeCounts(ident, mism, opens, extends)))
    return alns, np.array(labels)


def test_initial_estimate_simple_affine_recovers_coefficients():
    rng = np.random.default_rng(0)
    coefs = np.array([0.8, -0.6, -1.0, -0.3])
    alns, labels = _fake_simple_data(rng, 4000, coefs, alpha=-0.5)
    est = get_initial_estimate(alns, labels, "simple", "affine")
    assert set(est) == {"alpha", "match_score", "mismatch_score",
                        "open_gap_score", "extend_gap_score"}
    assert est["alpha"] == pytest.approx(-0.5, abs=0.25)
    assert est["match_score"] == pytest.approx(0.8, abs=0.1)
    assert est["mismatch_score"] == pytest.approx(-0.6, abs=0.1)
    assert est["open_gap_score"] == pytest.approx(-1.0, abs=0.15)
    assert est["extend_gap_score"] == pytest.approx(-0.3, abs=0.1)


def test_initial_estimate_simple_linear_recovers_coefficients():
    rng = np.random.default_rng(1)
    coefs = np.array([0.8, -0.6, -0.7])
    alns, labels = _fake_simple_data(rng, 4000, coefs, alpha=-0.5, linear=True)
    est = get_initial_estimate(alns, labels, "simple", "linear")
    assert set(est) == {"alpha", "match_score", "mismatch_score", "gap_score"}
    assert est["match_score"] == pytest.approx(0.8, abs=0.1)
    assert est["mismatch_score"] == pytest.approx(-0.6, abs=0.1)
    assert est["gap_score"] == pytest.approx(-0.7, abs=0.1)


def _fake_full_data(rng, n, matrix, open_coef, extend_coef, alpha, alphabet):
    alns, labels = [], []
    k = len(alphabet)
    for _ in range(n):
        pair_counts = rng.poisson(0.6, size=(k, k))
        opens, extends = rng.poisson(1.0), rng.poisson(1.5)
        row0 = "".join(alphabet[i] * pair_counts[i, j] for i in range(k) for j in range(k))
        row1 = "".join(alphabet[j] * pair_counts[i, j] for i in range(k) for j in range(k))
        # Gap columns in the rows must be ignored by the substitution features.
        row0 += "-" * opens
        row1 += alphabet[0] * opens
        eta = alpha + np.sum(pair_counts * matrix) + opens * open_coef + extends * extend_coef
        labels.append(int(rng.random() < expit(eta)))
        ident = int(np.trace(pair_counts))
        alns.append(FakeAlignment(row0, row1,
                                  FakeCounts(ident, int(pair_counts.sum()) - ident, opens, extends)))
    return alns, np.array(labels)


def test_initial_estimate_general_affine_matrix_orientation():
    rng = np.random.default_rng(2)
    alphabet = "ACG"
    matrix = np.full((3, 3), -0.3)
    matrix[np.diag_indices(3)] = 0.6
    matrix[0, 1] = 1.2    # A->C strongly positive, C->A not
    alns, labels = _fake_full_data(rng, 4000, matrix, -0.8, -0.2, -0.3, alphabet)
    est = get_initial_estimate(alns, labels, "general", "affine", alphabet)
    assert set(est) == {"alpha", "substitution_matrix", "open_gap_score", "extend_gap_score"}
    M = est["substitution_matrix"]
    assert "".join(M.alphabet) == alphabet
    # Default L2 penalty shrinks the fit a little, so the tolerance is loose.
    np.testing.assert_allclose(np.asarray(M), matrix, atol=0.2)
    assert M["A", "C"] > M["C", "A"] + 0.8
    assert est["open_gap_score"] == pytest.approx(-0.8, abs=0.2)
    assert est["extend_gap_score"] == pytest.approx(-0.2, abs=0.15)


def test_initial_estimate_symmetric_is_symmetrized_general():
    rng = np.random.default_rng(3)
    alphabet = "ACG"
    matrix = np.full((3, 3), -0.3)
    matrix[np.diag_indices(3)] = 0.6
    matrix[0, 1] = 1.2
    alns, labels = _fake_full_data(rng, 1000, matrix, -0.8, -0.2, -0.3, alphabet)
    general = get_initial_estimate(alns, labels, "general", "affine", alphabet)
    symmetric = get_initial_estimate(alns, labels, "symmetric", "affine", alphabet)
    G = np.asarray(general["substitution_matrix"])
    S = symmetric["substitution_matrix"]
    np.testing.assert_allclose(np.asarray(S), (G + G.T) / 2)
    assert "".join(S.alphabet) == alphabet
    assert symmetric["open_gap_score"] == pytest.approx(general["open_gap_score"])


@pytest.mark.parametrize("substitution_mode", ["general", "symmetric"])
def test_initial_estimate_full_linear_fits_one_gap_coefficient(substitution_mode):
    """Labels from a linear gap model: every gap column has the same coefficient."""
    rng = np.random.default_rng(4)
    gap = -0.6
    alns, labels = _fake_full_data(rng, 4000, np.eye(3) - 0.3, gap, gap, -0.3, "ACG")
    est = get_initial_estimate(alns, labels, substitution_mode, "linear", "ACG")
    assert set(est) == {"alpha", "substitution_matrix", "gap_score"}
    # The former open + extend of an affine fit would land near 2 * gap.
    assert est["gap_score"] == pytest.approx(gap, abs=0.1)


def test_initial_estimate_rejects_unknown_modes():
    alns = [FakeAlignment("", "", FakeCounts(1, 0, 0, 0))] * 2
    with pytest.raises(AssertionError):
        get_initial_estimate(alns, [0, 1], "simple", "convex")
    with pytest.raises(AssertionError):
        get_initial_estimate(alns, [0, 1], "banded", "affine")


@pytest.mark.parametrize("substitution_mode", ["general", "symmetric"])
def test_initial_estimate_full_requires_alphabet(substitution_mode):
    alns = [FakeAlignment("", "", FakeCounts(1, 0, 0, 0))] * 2
    with pytest.raises(AssertionError):
        get_initial_estimate(alns, [0, 1], substitution_mode, "affine")


def test_initial_estimate_needs_both_classes():
    alns = [FakeAlignment("", "", FakeCounts(i, 1, 0, 0)) for i in range(4)]
    with pytest.raises(ValueError):
        get_initial_estimate(alns, [1, 1, 1, 1], "simple", "affine")


# --- initial estimate on real alignments ------------------------------------

@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("gap_mode", ["affine", "linear"])
@pytest.mark.parametrize("substitution_mode", ["simple", "symmetric", "general"])
def test_initial_estimate_on_real_alignments(mode, gap_mode, substitution_mode):
    rng = np.random.default_rng(5)
    seqsA, seqsB, labels = make_pairs(rng, 20, 20, 25, sub_rate=0.2)
    params = random_params(rng, gap_mode, substitution_mode)
    alns = align_all(seqsA, seqsB, make_aligner(mode, params))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence warnings on tiny data
        est = get_initial_estimate(alns, labels, substitution_mode, gap_mode, DNA)
    for key, value in est.items():
        assert np.all(np.isfinite(np.asarray(value))), key
    if substitution_mode == "simple":
        assert est["match_score"] > est["mismatch_score"]
    else:
        M = np.asarray(est["substitution_matrix"])
        assert np.mean(np.diag(M)) > np.mean(M[~np.eye(4, dtype=bool)])
        if substitution_mode == "symmetric":
            np.testing.assert_allclose(M, M.T)


def test_initial_estimate_accepts_empty_local_alignments():
    alns = [EmptyLocalAlignment(), EmptyLocalAlignment(),
            FakeAlignment("AC", "AC", FakeCounts(2, 0, 0, 0)),
            FakeAlignment("AG", "AG", FakeCounts(2, 0, 0, 0))]
    labels = [0, 0, 1, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = get_initial_estimate(alns, labels, "general", "affine", "ACGT")
    assert np.all(np.isfinite(np.asarray(est["substitution_matrix"])))


# --- count features ----------------------------------------------------------

def _reference_count_features(raw_counts_list, substitution_mode, gap_mode, alphabet):
    """The per-row implementation that _count_features() replaced."""
    predictors = []
    for raw in raw_counts_list:
        subs = raw['Substitutions']
        G = np.asarray(subs)
        opens, extends = raw['Gap opens'], raw['Gap extends']
        gaps = [opens, extends] if gap_mode == 'affine' else [opens + extends]
        if substitution_mode == 'simple':
            predictors.append([np.trace(G), G.sum() - np.trace(G)] + gaps)
        else:
            order = [subs.alphabet.index(char) for char in alphabet]
            predictors.append(list(G[np.ix_(order, order)].ravel()) + gaps)
    return predictors


def _random_raw_counts(rng, n, count_alphabet):
    """Counts as grad_to_raw() returns them, over count_alphabet."""
    k = len(count_alphabet)
    return [{'Substitutions': substitution_matrices.Array(
                 alphabet=count_alphabet, data=rng.poisson(1.5, (k, k)).astype(float)),
             'Gap opens': float(rng.poisson(1)),
             'Gap extends': float(rng.poisson(2))} for _ in range(n)]


# Counts over the fit's alphabet, a permutation of it, and a superset of it.
COUNT_ALPHABETS = [DNA, "TGCA", "ACGTN"]


@pytest.mark.parametrize("count_alphabet", COUNT_ALPHABETS)
@pytest.mark.parametrize("gap_mode", ["affine", "linear"])
@pytest.mark.parametrize("substitution_mode", ["simple", "symmetric", "general"])
def test_count_features_match_per_row_reference(substitution_mode, gap_mode, count_alphabet):
    raws = _random_raw_counts(np.random.default_rng(0), 40, count_alphabet)
    expected = np.asarray(_reference_count_features(raws, substitution_mode, gap_mode, DNA),
                          dtype=float)
    for counts in (raws, _count_arrays_from_raw(raws)):
        features = _count_features(counts, substitution_mode, gap_mode, DNA)
        assert features.dtype == np.float64
        np.testing.assert_array_equal(features, expected)


def test_count_arrays_from_raw_reorders_to_the_first_alphabet():
    rng = np.random.default_rng(1)
    raws = _random_raw_counts(rng, 3, DNA) + _random_raw_counts(rng, 2, "TGCA")
    counts = _count_arrays_from_raw(raws)
    assert counts.alphabet == DNA
    for i, raw in enumerate(raws):
        for a, x in enumerate(DNA):
            for b, z in enumerate(DNA):
                assert counts.substitutions[i, a, b] == raw['Substitutions'][x, z]
    np.testing.assert_array_equal(counts.gap_opens, [r['Gap opens'] for r in raws])
    np.testing.assert_array_equal(counts.gap_extends, [r['Gap extends'] for r in raws])


@pytest.mark.parametrize("gap_mode", ["affine", "linear"])
@pytest.mark.parametrize("substitution_mode", ["simple", "symmetric", "general"])
def test_initial_estimate_from_counts_is_unchanged(substitution_mode, gap_mode):
    rng = np.random.default_rng(2)
    raws = _random_raw_counts(rng, 60, "TGCA")
    labels = rng.integers(0, 2, len(raws))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = _fit_initial_estimate(
            _reference_count_features(raws, substitution_mode, gap_mode, DNA),
            labels, substitution_mode, gap_mode, DNA)
        from_list = get_initial_estimate_from_counts(raws, labels, substitution_mode, gap_mode, DNA)
        from_arrays = get_initial_estimate_from_counts(_count_arrays_from_raw(raws), labels,
                                                       substitution_mode, gap_mode, DNA)
    for estimate in (from_list, from_arrays):
        assert estimate.keys() == reference.keys()
        for key in reference:
            np.testing.assert_array_equal(np.asarray(estimate[key]), np.asarray(reference[key]),
                                          err_msg=key)

"""Unit tests for the private helpers in src.discrimalign."""
import numpy as np
import pytest
from Bio.Align import PairwiseAligner, substitution_matrices

from src.discrimalign import (_align_pair_chunk, _align_pairs, _matrix_alphabet,
                              _pair_chunks, _resolve_alphabet, _warm_start_alphabet)
from src.optimization import EmptyLocalAlignment
from tests.helpers import make_aligner, make_pairs, random_params


# --- _pair_chunks -----------------------------------------------------------

@pytest.mark.parametrize("n, chunk_size", [(0, 3), (1, 1), (5, 1), (5, 2), (6, 3), (7, 10)])
def test_pair_chunks_cover_input_in_order(n, chunk_size):
    A = [f"a{i}" for i in range(n)]
    B = [f"b{i}" for i in range(n)]
    chunks = list(_pair_chunks(A, B, chunk_size))
    assert [p for chunk in chunks for p in chunk] == list(zip(A, B))
    assert all(1 <= len(c) <= chunk_size for c in chunks)
    assert len(chunks) == -(-n // chunk_size)


# --- _align_pair_chunk / _align_pairs ---------------------------------------

def _aligner(mode):
    rng = np.random.default_rng(0)
    return make_aligner(mode, random_params(rng, "affine", "simple"))


@pytest.mark.parametrize("mode", ["local", "global"])
def test_align_pair_chunk_matches_direct_alignment(mode):
    aligner = _aligner(mode)
    pairs = [("ACGT", "ACT"), ("GGGA", "GGA"), ("T", "T")]
    alns = _align_pair_chunk(pairs, aligner)
    assert [a.score for a in alns] == pytest.approx(
        [aligner.score(x, y) for x, y in pairs], rel=1e-12)


def test_align_pair_chunk_returns_empty_local_alignment():
    aligner = _aligner("local")
    alns = _align_pair_chunk([("AAA", "CCC")], aligner)
    assert isinstance(alns[0], EmptyLocalAlignment)


@pytest.mark.parametrize("num_threads", [1, 2, 3, 8, 64])
@pytest.mark.parametrize("mode", ["local", "global"])
def test_align_pairs_is_independent_of_thread_count(num_threads, mode):
    rng = np.random.default_rng(num_threads)
    seqsA, seqsB, _ = make_pairs(rng, 9, 8, 15)
    aligner = _aligner(mode)
    alns = _align_pairs(seqsA, seqsB, aligner, num_threads)
    serial = _align_pairs(seqsA, seqsB, aligner, 1)
    assert len(alns) == len(seqsA)
    assert [a.score for a in alns] == [a.score for a in serial]
    # aligner.score() sums in a different order than align(), so only approx.
    assert [a.score for a in alns] == pytest.approx(
        [aligner.score(a, b) for a, b in zip(seqsA, seqsB)], rel=1e-12)


@pytest.mark.parametrize("num_threads", [1, 4])
def test_align_pairs_empty_input(num_threads):
    assert _align_pairs([], [], _aligner("global"), num_threads) == []


def test_align_pairs_preserves_order_with_distinct_scores():
    aligner = _aligner("global")
    seqsA = ["A" * k for k in range(1, 21)]
    seqsB = ["A" * k for k in range(1, 21)]
    scores = [a.score for a in _align_pairs(seqsA, seqsB, aligner, 4)]
    assert scores == sorted(scores)
    assert len(set(scores)) == 20


# --- _matrix_alphabet / _warm_start_alphabet --------------------------------

def _matrix(alphabet):
    n = len(alphabet)
    return substitution_matrices.Array(alphabet=alphabet, data=np.eye(n))


def test_matrix_alphabet_of_array():
    assert _matrix_alphabet(_matrix("TGCA")) == "TGCA"


@pytest.mark.parametrize("matrix", [None, np.eye(2), 3.0])
def test_matrix_alphabet_without_alphabet_attribute(matrix):
    assert _matrix_alphabet(matrix) is None


def test_warm_start_alphabet_from_initial_parameters():
    assert _warm_start_alphabet(None, {"substitution_matrix": _matrix("CGAT")}) == "CGAT"


def test_warm_start_alphabet_from_baseline_aligner():
    aligner = PairwiseAligner()
    aligner.substitution_matrix = _matrix("TCAG")
    assert _warm_start_alphabet(aligner, None) == "TCAG"


def test_warm_start_alphabet_prefers_initial_parameters():
    aligner = PairwiseAligner()
    aligner.substitution_matrix = _matrix("TCAG")
    assert _warm_start_alphabet(aligner, {"substitution_matrix": _matrix("GATC")}) == "GATC"


def test_warm_start_alphabet_falls_back_to_aligner_without_matrix_in_params():
    aligner = PairwiseAligner()
    aligner.substitution_matrix = _matrix("TCAG")
    assert _warm_start_alphabet(aligner, {"match_score": 1.0}) == "TCAG"


def test_warm_start_alphabet_none_for_simple_scoring():
    aligner = PairwiseAligner()
    aligner.match_score = 2
    assert _warm_start_alphabet(aligner, {"match_score": 2.0}) is None
    assert _warm_start_alphabet(None, None) is None


# --- _resolve_alphabet -------------------------------------------------------

def test_resolve_alphabet_given_wins():
    aligner = PairwiseAligner()
    aligner.substitution_matrix = _matrix("TCAG")
    assert _resolve_alphabet("ACGT", ["XY"], ["Z"], aligner,
                             {"substitution_matrix": _matrix("GATC")}) == "ACGT"


def test_resolve_alphabet_from_warm_start():
    assert _resolve_alphabet(None, ["XY"], ["Z"], None,
                             {"substitution_matrix": _matrix("GATC")}) == "GATC"


def test_resolve_alphabet_from_sequences_is_sorted_union():
    assert _resolve_alphabet(None, ["GAT", "TTA"], ["CAN"], None, {"match_score": 1.0}) == "ACGNT"

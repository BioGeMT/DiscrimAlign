import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.discrimalign import discrimalign
from src.inference import load_model, predict_csv, predict_pairs, save_model


def _fit_model():
    return discrimalign(
        seqlistA=["AUGCUA", "CUGA"],
        seqlistB=["AUGGUA", "CUGU"],
        labels=[1, 0],
        max_iter=1,
        num_threads=1,
    )


def test_predict_pairs_returns_serializable_alignment_rows():
    rows = predict_pairs(["AUGCUA"], ["AUGGUA"], _fit_model())

    assert rows[0]["index"] == 0
    assert rows[0]["sequence_a"] == "AUGCUA"
    assert rows[0]["sequence_b"] == "AUGGUA"
    assert 0.0 <= rows[0]["probability"] <= 1.0
    assert rows[0]["aligned_sequence_a"]
    assert rows[0]["alignment_marks"]
    assert rows[0]["aligned_sequence_b"]
    assert len(rows[0]["operations"]) == len(rows[0]["alignment_marks"])


def test_predict_pairs_validates_pair_counts():
    try:
        predict_pairs(["A"], ["A", "C"], _fit_model())
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected pair-count validation error")


def test_save_load_model_and_predict_csv(tmp_path):
    model_path = tmp_path / "model.pkl"
    input_path = tmp_path / "pairs.csv"
    output_path = tmp_path / "predictions.csv"

    save_model(_fit_model(), model_path)
    model = load_model(model_path)

    input_path.write_text("id,sequence_a,sequence_b\nexample,AUGCUA,AUGGUA\n")
    rows = predict_csv(input_path, output_path, model)

    assert len(rows) == 1
    with output_path.open(newline="") as output_file:
        output_rows = list(csv.DictReader(output_file))
    assert output_rows[0]["id"] == "example"
    assert output_rows[0]["probability"]
    assert output_rows[0]["alignment_marks"]


def test_load_model_supports_case_study_schema():
    model = load_model("case_study_for_mirna/trained_models/manakov_best_model.pkl")
    rows = predict_pairs(["ATGCTA"], ["ATGGTA"], model)

    assert len(rows) == 1
    assert 0.0 <= rows[0]["probability"] <= 1.0
    assert rows[0]["alignment_marks"]


# --- backends -----------------------------------------------------------------

import numpy as np
import pytest
from Bio.Align import PairwiseAligner

from src import infer
from tests.helpers import make_pairs, random_seq

MANAKOV = "case_study_for_mirna/trained_models/manakov_best_model.pkl"


def _assert_same_scores(nwgrad_rows, biopython_rows):
    assert len(nwgrad_rows) == len(biopython_rows)
    for nw, bio in zip(nwgrad_rows, biopython_rows):
        assert nw["alignment_score"] == pytest.approx(bio["alignment_score"], rel=1e-12, abs=1e-12)
        assert nw["probability"] == pytest.approx(bio["probability"], rel=1e-12, abs=1e-15)
        assert nw["normalized_sequence_a"] == bio["normalized_sequence_a"]
        assert nw["normalized_sequence_b"] == bio["normalized_sequence_b"]
        assert len(nw["operations"]) == len(nw["alignment_marks"])


def test_predict_pairs_backends_agree_on_the_case_study_model():
    """Same scores and probabilities; an RNA miRNA, U/T normalization, reverse
    complement, and a pair with no positive local alignment included."""
    model = load_model(MANAKOV)
    rng = np.random.default_rng(28)
    A = [random_seq(rng, 22, "ACGU") for _ in range(40)] + ["AAAA"]
    B = [random_seq(rng, 50) for _ in range(40)] + ["CCCC"]
    rows = {backend: predict_pairs(A, B, model, backend=backend, reverse_complement_b=True)
            for backend in ("nwgrad", "biopython")}
    _assert_same_scores(rows["nwgrad"], rows["biopython"])
    assert rows["nwgrad"][-1]["alignment_score"] == 0.0
    assert rows["nwgrad"][-1]["aligned_sequence_a"] == ""


@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("gap_mode", ["affine", "linear"])
@pytest.mark.parametrize("substitution_mode", ["general", "simple"])
def test_predict_pairs_backends_agree_on_fitted_models(mode, gap_mode, substitution_mode):
    rng = np.random.default_rng(30)
    A, B, y = make_pairs(rng, 10, 10, 14, sub_rate=0.25)
    model = discrimalign(A, B, y, aligner_mode=mode, gap_mode=gap_mode,
                         substitution_mode=substitution_mode, max_iter=2, num_threads=1)
    A2, B2, _ = make_pairs(rng, 8, 8, 18, sub_rate=0.3)
    rows = {backend: predict_pairs(A2, B2, model, backend=backend)
            for backend in ("nwgrad", "biopython")}
    _assert_same_scores(rows["nwgrad"], rows["biopython"])


def test_predict_pairs_simple_model_scores_letters_outside_the_fit():
    """A match/mismatch model scores any characters, with either backend."""
    rng = np.random.default_rng(29)
    A, B, y = make_pairs(rng, 5, 5, 12)
    model = discrimalign(A, B, y, substitution_mode="simple", max_iter=1, num_threads=1)
    assert model["aligner"].substitution_matrix is None
    rows = {backend: predict_pairs(["ACGTX"], ["ACXT"], model, backend=backend)
            for backend in ("nwgrad", "biopython")}
    _assert_same_scores(rows["nwgrad"], rows["biopython"])


def test_predict_pairs_thread_count_does_not_change_results():
    model = load_model(MANAKOV)
    rng = np.random.default_rng(31)
    A = [random_seq(rng, 22) for _ in range(30)]
    B = [random_seq(rng, 50) for _ in range(30)]
    assert predict_pairs(A, B, model, num_threads=1) == predict_pairs(A, B, model, num_threads=3)


def test_predict_pairs_empty_input():
    assert predict_pairs([], [], load_model(MANAKOV)) == []


def test_predict_pairs_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        predict_pairs(["A"], ["A"], load_model(MANAKOV), backend="parasail")


def test_predict_pairs_nwgrad_rejects_an_aligner_it_cannot_express():
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.wildcard = "N"
    model = {"aligner": aligner, "alpha": 0.0}
    with pytest.raises(ValueError, match="backend='biopython'"):
        predict_pairs(["ACGT"], ["ACGT"], model, backend="nwgrad")
    assert predict_pairs(["ACGT"], ["ACGT"], model, backend="biopython")


@pytest.mark.parametrize("backend", ["nwgrad", "biopython"])
def test_infer_cli_backend_option(tmp_path, backend):
    model_path = tmp_path / "model.pkl"
    input_path = tmp_path / "pairs.csv"
    output_path = tmp_path / "predictions.csv"
    save_model(_fit_model(), model_path)
    input_path.write_text("id,sequence_a,sequence_b\nexample,AUGCUA,AUGGUA\n")
    infer.main(["--model", str(model_path), "--input", str(input_path),
                "--output", str(output_path), "--backend", backend, "--threads", "2"])
    with output_path.open(newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    expected = predict_pairs(["AUGCUA"], ["AUGGUA"], load_model(model_path), backend="biopython")
    assert float(rows[0]["alignment_score"]) == pytest.approx(expected[0]["alignment_score"],
                                                              rel=1e-12)


@pytest.mark.parametrize("backend", ["nwgrad", "biopython"])
def test_predict_pairs_rejects_empty_sequences(backend):
    model = load_model(MANAKOV)
    with pytest.raises(ValueError, match="seqlistB contains empty sequences"):
        predict_pairs(["ACGT", "ACG"], ["ACG", ""], model, backend=backend)

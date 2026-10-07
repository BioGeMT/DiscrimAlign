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

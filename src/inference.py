"""Inference helpers for trained DiscrimAlign models."""

import csv
import os
import pickle
from pathlib import Path

from Bio.Seq import Seq

from .logit_link import logit_partial_scores
from .optimization import get_first_alignment


def _model_parts(model):
    if not isinstance(model, dict):
        raise TypeError("model must be a discrimalign result dictionary")
    if "aligner" not in model:
        raise ValueError("model must contain an 'aligner' entry")
    if "alpha" in model:
        alpha = model["alpha"]
    elif isinstance(model.get("summary"), dict) and "alpha" in model["summary"]:
        alpha = model["summary"]["alpha"]
    else:
        raise ValueError("model must contain 'alpha' or 'summary[alpha]' entries")
    return model["aligner"], alpha


def _model_alphabet(aligner):
    matrix = getattr(aligner, "substitution_matrix", None)
    alphabet = getattr(matrix, "alphabet", None)
    if alphabet is None:
        return None
    return set(alphabet)


def _normalize_sequence(sequence, alphabet, normalize):
    if normalize in (None, False, "none"):
        return sequence
    if normalize != "auto":
        raise ValueError("normalize must be 'auto', 'none', None, or False")
    if alphabet and "T" in alphabet and "U" not in alphabet:
        return sequence.replace("U", "T").replace("u", "t")
    if alphabet and "U" in alphabet and "T" not in alphabet:
        return sequence.replace("T", "U").replace("t", "u")
    return sequence


def _reverse_complement(sequence):
    return str(Seq(sequence).reverse_complement())


def _alignment_rows(alignment):
    target = str(alignment[0])
    query = str(alignment[1])
    markers = []
    operations = []
    for target_char, query_char in zip(target, query):
        if target_char == "-" or query_char == "-":
            marker = "-"
            operation = "gap"
        elif target_char == query_char:
            marker = "|"
            operation = "match"
        else:
            marker = "."
            operation = "mismatch"
        markers.append(marker)
        operations.append(operation)
    return target, "".join(markers), query, operations


def summarize_alignment(alignment):
    """Return a serializable summary of a Biopython alignment."""
    target, markers, query, operations = _alignment_rows(alignment)
    return {
        "aligned_sequence_a": target,
        "alignment_marks": markers,
        "aligned_sequence_b": query,
        "operations": operations,
        "text": str(alignment),
    }


def _nwgrad_alignments(seqlistA, seqlistB, aligner, num_threads):
    """
    The pairs' optimal alignments under a fitted aligner, from nwgrad batches.
    A model with a substitution matrix is scored over that matrix's alphabet;
    a match/mismatch model scores any characters, so its alphabet is the one
    the pairs contain.
    """
    from .nwgrad_backend import baseline_parameters, nwgrad_alignments
    from .nwgrad_params import to_nwgrad

    if aligner.substitution_matrix is not None:
        alphabet = "".join(aligner.substitution_matrix.alphabet)
    else:
        alphabet = "".join(sorted(set("".join(seqlistA) + "".join(seqlistB))))
    try:
        linear = aligner.open_gap_score == aligner.extend_gap_score
    except ValueError:
        linear = False  # baseline_parameters() raises with the explanation
    gap_mode = "linear" if linear else "affine"
    try:
        params = baseline_parameters(aligner, gap_mode, alphabet)
    except ValueError as error:
        raise ValueError(f"backend='nwgrad' cannot express the model's aligner ({error}); "
                         "use backend='biopython'") from error
    nw_params = to_nwgrad(params, gap_mode, "general", alphabet)
    return nwgrad_alignments(seqlistA, seqlistB, nw_params, gap_mode, aligner.mode,
                             num_threads)


def predict_pairs(
    seqlistA,
    seqlistB,
    model,
    return_alignments=True,
    normalize="auto",
    reverse_complement_b=False,
    backend="nwgrad",
    num_threads=None,
):
    """Score sequence pairs with a fitted DiscrimAlign model.

    Parameters
    ----------
    seqlistA, seqlistB : iterable of str
        Sequence pairs to align and score.
    model : dict
        Result dictionary returned by ``discrimalign``.
    return_alignments : bool, default=True
        Include text and per-position alignment summaries in each row.
    normalize : {"auto", "none"}, default="auto"
        Convert U/T automatically when the fitted model alphabet requires it.
    reverse_complement_b : bool, default=False
        Reverse-complement each second sequence before normalization and scoring.
    backend : {"nwgrad", "biopython"}, default="nwgrad"
        Alignment engine. "nwgrad" aligns all pairs in parallel batches;
        "biopython" aligns them one at a time with the model's PairwiseAligner.
        Both give the same scores up to floating-point rounding; where several
        alignments of a pair are optimal, they may return different ones.
    num_threads : int, optional
        Threads for the nwgrad engine; all logical cores by default.
    """
    seqlistA = list(seqlistA)
    seqlistB = list(seqlistB)
    if len(seqlistA) != len(seqlistB):
        raise ValueError("seqlistA and seqlistB must have the same length")
    if backend not in ("nwgrad", "biopython"):
        raise ValueError(f"backend must be 'nwgrad' or 'biopython', got {backend!r}")
    for seqlist, name in ((seqlistA, "seqlistA"), (seqlistB, "seqlistB")):
        empty = [i for i, seq in enumerate(seqlist) if len(seq) == 0]
        if empty:
            raise ValueError(f"{name} contains empty sequences (at indices {empty[:10]}); "
                             "every sequence needs at least one residue")

    aligner, alpha = _model_parts(model)
    alphabet = _model_alphabet(aligner)
    normalized_seqlistA = []
    normalized_seqlistB = []
    for seqA, seqB in zip(seqlistA, seqlistB):
        transformed_seqB = _reverse_complement(seqB) if reverse_complement_b else seqB
        normalized_seqlistA.append(_normalize_sequence(seqA, alphabet, normalize))
        normalized_seqlistB.append(_normalize_sequence(transformed_seqB, alphabet, normalize))

    if backend == "nwgrad" and normalized_seqlistA:
        alignments = _nwgrad_alignments(normalized_seqlistA, normalized_seqlistB, aligner,
                                        num_threads or os.cpu_count() or 1)
    else:
        alignments = [get_first_alignment(seqA, seqB, aligner)
                      for seqA, seqB in zip(normalized_seqlistA, normalized_seqlistB)]

    rows = []
    for index, (seqA, seqB, normalized_seqA, normalized_seqB, alignment) in enumerate(
            zip(seqlistA, seqlistB, normalized_seqlistA, normalized_seqlistB, alignments)):
        probability = float(logit_partial_scores([alignment.score], alpha)[0])
        row = {
            "index": index,
            "sequence_a": seqA,
            "sequence_b": seqB,
            "normalized_sequence_a": normalized_seqA,
            "normalized_sequence_b": normalized_seqB,
            "alignment_score": float(alignment.score),
            "probability": probability,
        }
        if return_alignments:
            row.update(summarize_alignment(alignment))
        rows.append(row)
    return rows


def save_model(model, path):
    """Save a fitted DiscrimAlign result dictionary for later inference."""
    _model_parts(model)
    path = Path(path)
    with path.open("wb") as output_file:
        pickle.dump(model, output_file)


def load_model(path):
    """Load a fitted DiscrimAlign result dictionary saved with ``save_model``."""
    path = Path(path)
    with path.open("rb") as input_file:
        model = pickle.load(input_file)
    _model_parts(model)
    return model


def predict_csv(
    input_csv,
    output_csv,
    model,
    sequence_a_column="sequence_a",
    sequence_b_column="sequence_b",
    normalize="auto",
    reverse_complement_b=False,
    backend="nwgrad",
    num_threads=None,
):
    """Run inference from a CSV file and write prediction rows to another CSV.

    backend and num_threads are passed to ``predict_pairs``.
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    with input_csv.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError("input CSV must include a header row")
        missing_columns = {
            column
            for column in (sequence_a_column, sequence_b_column)
            if column not in reader.fieldnames
        }
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"input CSV is missing required columns: {missing}")
        input_rows = list(reader)

    predictions = predict_pairs(
        [row[sequence_a_column] for row in input_rows],
        [row[sequence_b_column] for row in input_rows],
        model,
        normalize=normalize,
        reverse_complement_b=reverse_complement_b,
        backend=backend,
        num_threads=num_threads,
    )
    output_rows = []
    for input_row, prediction in zip(input_rows, predictions):
        output_rows.append(
            {
                **input_row,
                "normalized_sequence_a": prediction["normalized_sequence_a"],
                "normalized_sequence_b": prediction["normalized_sequence_b"],
                "alignment_score": prediction["alignment_score"],
                "probability": prediction["probability"],
                "aligned_sequence_a": prediction["aligned_sequence_a"],
                "alignment_marks": prediction["alignment_marks"],
                "aligned_sequence_b": prediction["aligned_sequence_b"],
                "operations": ";".join(prediction["operations"]),
            }
        )

    fieldnames = list(input_rows[0].keys()) if input_rows else list(reader.fieldnames)
    for fieldname in (
        "normalized_sequence_a",
        "normalized_sequence_b",
        "alignment_score",
        "probability",
        "aligned_sequence_a",
        "alignment_marks",
        "aligned_sequence_b",
        "operations",
    ):
        if fieldname not in fieldnames:
            fieldnames.append(fieldname)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows

"""Command-line CSV inference for saved DiscrimAlign models."""

import argparse
from pathlib import Path

from .inference import load_model, predict_csv

MODEL_ALIASES = {
    "manakov": Path("case_study_for_mirna/trained_models/manakov_best_model.pkl"),
    "hejret": Path("case_study_for_mirna/trained_models/hejret_best_model.pkl"),
}


def _resolve_model_path(model):
    return MODEL_ALIASES.get(model, Path(model))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run DiscrimAlign inference on sequence-pair CSV data.")
    parser.add_argument(
        "--model",
        required=True,
        help="Model path, or one of the built-in miRNA aliases: manakov, hejret.",
    )
    parser.add_argument("--input", required=True, help="Input CSV containing sequence pairs.")
    parser.add_argument("--output", required=True, help="Output CSV for predictions.")
    parser.add_argument("--seq-a-column", default="sequence_a", help="Column name for first sequence.")
    parser.add_argument("--seq-b-column", default="sequence_b", help="Column name for second sequence.")
    parser.add_argument(
        "--normalize",
        default="auto",
        choices=["auto", "none"],
        help="Normalize U/T to match the trained model alphabet.",
    )
    parser.add_argument(
        "--reverse-complement-b",
        action="store_true",
        help="Reverse-complement the second sequence column before scoring.",
    )
    parser.add_argument(
        "--backend",
        default="nwgrad",
        choices=["nwgrad", "biopython"],
        help="Alignment engine (default: nwgrad).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Threads for the nwgrad engine (default: all logical cores).",
    )
    args = parser.parse_args(argv)

    model_path = _resolve_model_path(args.model)
    model = load_model(model_path)
    rows = predict_csv(
        input_csv=args.input,
        output_csv=args.output,
        model=model,
        sequence_a_column=args.seq_a_column,
        sequence_b_column=args.seq_b_column,
        normalize=args.normalize,
        reverse_complement_b=args.reverse_complement_b,
        backend=args.backend,
        num_threads=args.threads,
    )
    print(f"Loaded model from {model_path}")
    print(f"Wrote {len(rows)} prediction rows to {args.output}")


if __name__ == "__main__":
    main()

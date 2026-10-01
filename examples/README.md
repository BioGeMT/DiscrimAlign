# Example Inference Inputs

`mirna_pairs.csv` is a small ready-to-run input file for testing DiscrimAlign miRNA inference.

Run it with a bundled trained model:

```bash
uv run python -m src.infer \
  --model manakov \
  --input examples/mirna_pairs.csv \
  --output predictions_manakov.csv
```

The input schema is:

- `id`: optional identifier preserved in the output
- `sequence_a`: first sequence
- `sequence_b`: second sequence

The output adds probability, alignment score, normalized sequences, aligned sequences, alignment markers, and per-position operations.

# DiscrimAlign

**Learn substitution matrices and gap penalties from labeled pairs of biological sequences.**

DiscrimAlign learns pairwise alignment parameters for binary classification tasks, such as predicting sequence interactions or distinguishing related from unrelated sequences. Training requires positive and negative pairs of unaligned DNA, RNA, or protein sequences. It jointly learns substitution weights, gap penalties, and a logistic intercept; reference alignments are not required.

The repository provides a Python training and prediction API, simulation experiments, and case studies on miRNA–target interactions and protein sequence blocks. It accompanies the article *Discriminative Learning of Substitution Matrices and Gap Penalties for Pairwise Alignment of Biological Sequences*.

## How it works

For sequences $A$ and $B$, DiscrimAlign models the probability of a positive label as

$$
P(Y=1\mid A,B)=\frac{1}{1+\exp[-\alpha-S_\theta(A,B)]},
$$

where $S_\theta(A,B)$ is the optimal alignment score under substitution and gap parameters $\theta$, and $\alpha$ is an intercept. Training repeatedly realigns the sequence pairs and updates the parameters to maximize the label likelihood. The resulting model provides both a probability and an interpretable sequence alignment.

The implementation supports:

- **Local or global alignment.**
- **Simple match/mismatch scores, symmetric substitution matrices, or asymmetric substitution matrices.**
- **Linear or affine gap penalties.**
- **DNA, RNA, and protein alphabets**, inferred from the training sequences or specified explicitly.

The objective is binary classification of sequence pairs using alignment score, with high alignment score indicating a positive pair and low alignment score indicating a negative one. This includes for example homology search (homologous or random similarity), complementary binding (including miRNA-target interaction), etc. Learned matrices describe differences between positive and negative pairs; they are not necessarily estimates of evolutionary substitution rates or parameters that maximize structural alignment accuracy.

## Repository structure

```text
src/                         Core DiscrimAlign training and inference code
tests/                       Unit and integration tests (pytest)
examples/mirna_pairs.csv       Ready-to-run inference input example for the experimental miRNA-target prediction interface
case_study_for_mirna/         Bundled miRNA trained models and evaluation workflows
benchmarks/                   Backend benchmark script
RNA_simulation_experiments/  Simulation experiments for the manuscript
pyproject.toml                Package metadata and the uv project environment
.github/workflows/            CI tests and the PyPI release
```

## Installation

### Library

Requires Python 3.10 or later:

```bash
pip install discrimalign
```

This installs the training and prediction API and the `discrimalign-infer` command. Case-study data, notebooks, and bundled miRNA models are accessed through the repository checkout.

### Research environment and reproducing the case studies

For notebooks, case studies, and development, clone the repository and install the environment with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/BioGeMT/DiscrimAlign.git
cd DiscrimAlign
uv sync --python 3.12
```

The default environment includes pytest, JupyterLab, plotting tools, and miRBench. Currently, Python <=3.12 is needed for this environment because the current release of miRBench package requires NumPy below version 2.9

Run the repository examples below from the repository root with `uv run`. The checkout imports the code as `src`; the installed library imports it as `discrimalign`.

## Quick start 

### Learning substitution matrices and gap parameters

Prepare a CSV named `training_pairs.csv` with columns `sequence_a`, `sequence_b`, and `label`. Each row represents a sequence pair; labels must be `0` or `1`, with both classes represented. Sequences must be nonempty and use a common alphabet. Supply sequences in the orientation you want aligned; the training API does not reverse-complement them automatically.


```python
import csv

from discrimalign import discrimalign, predict_pairs, save_model
from discrimalign.optimization import create_constant_step

with open("training_pairs.csv", newline="") as handle:
    pairs = list(csv.DictReader(handle))

model = discrimalign(
    seqlistA=[row["sequence_a"] for row in pairs],
    seqlistB=[row["sequence_b"] for row in pairs],
    labels=[int(row["label"]) for row in pairs],
    aligner_mode="local",           # "local" or "global"
    substitution_mode="symmetric",  # "simple", "symmetric" or "general"
    gap_mode="affine",              # "affine" or "linear"
    stepfunction=create_constant_step(1e-5),   # select an appropriate learning rate 
    max_iter=500,
    return_alignments=False
)

print(model["substitution_matrix"])
print(model["open_gap_score"], model["extend_gap_score"])
print(model["alpha"], model["final_loglik"])
save_model(model, "model.pkl")

predictions = predict_pairs(["ACGTAC"], ["ACCTAC"], model=model)
print(predictions[0]["probability"])
```

For the `uv` research environment, replace `from discrimalign ...` with `from src ...` and import the step function from `src.optimization`.

The result contains the fitted Biopython aligner, learned parameters, predicted training probabilities, and log-likelihood and subgradient-norm trajectories. `return_alignments=False` skips returning the training alignments; prediction can still produce alignments for new pairs.

The step size and iteration count above are starting values, not tuned defaults for every dataset. Updates use a sum over pairs, so the appropriate step size depends on dataset size. Large datasets require smaller step sizes to avoid instability. Monitor the likelihood trajectory and evaluate classification performance on held-out data. The objective is generally nonconcave, and completely separable training data can lead to diverging parameter estimates.

The default alignment backend is **nwgrad**, which computes scores and alignment feature counts in parallel. Use `num_threads` to control its thread count, or `backend="biopython"` to use Biopython. Equally scoring alignments can be resolved differently by the two backends, affecting subsequent training updates. The current implementation constrains gap scores to be at most `-1e-4`.

### Predict with a saved model

Use a CSV with columns `sequence_a` and `sequence_b`; additional columns, such as an identifier, are preserved. A small input example is provided in [examples/mirna_pairs.csv](examples/mirna_pairs.csv).

```bash
uv run python -m src.infer \
  --model model.pkl \
  --input examples/mirna_pairs.csv \
  --output predictions.csv
```

Use a model trained for the input alphabet and task. To try a bundled miRNA model without training, replace `--model model.pkl` with `--model manakov` or `--model hejret`.

The output adds the positive-class probability, alignment score, normalized sequences, and a readable alignment with match, mismatch, and gap annotations. By default, `U`/`T` is normalized to match the model alphabet. Run `uv run python -m src.infer --help` for column-name, preprocessing, backend, and thread options.

For targets supplied in their original biological orientation, use `--reverse-complement-b` to match the miRNA case-study preprocessing. The bundled example is a small inference demonstration, not a biological benchmark.

With the installed package, the equivalent command is `discrimalign-infer --model path/to/model.pkl --input pairs.csv --output predictions.csv`. The `manakov` and `hejret` aliases resolve to files relative to the repository root; outside the checkout, pass the model file path explicitly.

## Advanced use

### Intercept fit

At every iteration the intercept α is refitted to the current alignment scores. `alpha_solver` selects how:

- `"safeguarded_newton"` (default): Newton's method on dl/dα, which is strictly decreasing in α, so its root is the unique optimum. Plain Newton steps are used while they are small, as they are when α changes little between iterations; otherwise the root is bracketed and Newton steps are combined with bisection. The result is solved to numerical tolerance from any starting value.
- `"bfgs"`: the previous `scipy.optimize.minimize` fit. When the optimum moves far between iterations, as with `subgradient_scale=1` on large datasets, it can stop far from the optimum.

On all 2.5 million Manakov training pairs, 300 iterations took 18.4 minutes with the default against 29.6 minutes with `"bfgs"`, and gave the same fit.

### Backends

`backend` selects where alignment scores and subgradients come from:

- `"nwgrad"` (default): the [nwgrad](https://github.com/michalsta/nwgrad) C++ library aligns all pairs in parallel and returns each alignment's score and gradient in one pass. The pairs are encoded once and reused across iterations.
- `"biopython"`: Biopython's `PairwiseAligner`, with the gradient counted in Python from the alignment strings. This backend supports using lists of Biopython Seq objects in `seqlistA` and `seqlistB` parameters. 

Both backends fit the same model and return the same kinds of objects: the returned `aligner` and `alignments` are Biopython objects with either backend. With `"nwgrad"`, the alignments are nwgrad's own paths at the final parameters, wrapped as `Bio.Align.Alignment` objects, so they are the alignments the final scores and subgradient come from, ties included. Recovering the paths takes one extra alignment pass at the end; pass `return_alignments=False` to skip it.

The backends agree up to tie-breaking. When two alignments of a pair score exactly the same but use different substitutions or gaps, the backends may pick different ones. Both are valid subgradients, but the fits then drift apart. Exact ties are common with integer-valued scores, such as the default baseline aligner, and with fitted substitution matrices that contain exactly equal entries, such as the zeros a ridge fit assigns to substitutions that never occur in the data. Away from ties, the two backends follow the same trajectory up to floating-point rounding.

With `"nwgrad"`, the initial estimate is fitted on nwgrad's own alignments of the baseline. A `baseline_aligner` must then be expressible in DiscrimAlign's model: the same gap scores for both sequences and for internal and end gaps, no wildcard, and `local` or `global` mode. Otherwise a `ValueError` explains what is unsupported.

On the x86-64 machines tested, one fitting iteration on 10,000 miRNA-sized pairs (22 × 50 nt, local alignment, affine gaps, full substitution matrix) was 13–22× faster with nwgrad on one thread than with Biopython, and 77–212× faster with all cores. On 300-residue proteins, where Biopython's C alignment is more competitive, it was 3–5× faster on one thread and 26–108× with all cores. See `benchmarks/` to measure your own machine.

`nwgrad_fill` selects nwgrad's vectorized DP fill: `"interpair"` (default, nwgrad's own default), `"striped"` or `"rowwise"`. All three give the same scores, gradients and fit, bit for bit; only the speed differs. `"interpair"` aligns several short pairs at once, one per vector lane, and sends pairs too long for that to a per-pair fill. On all 2.5 million Manakov training pairs (local/affine/general, 300 iterations, 12 threads on an i5-12500, nwgrad 0.5.2), the fit took 1072 s with `"striped"`, 469 s with `"rowwise"` and 291 s with `"interpair"`. Leave it at the default unless measuring.


## Experimental miRNA binding site prediction interface

Use this workflow when you want predictions from the bundled trained miRNA models without retraining.

### 1. Inspect the Example Input

A ready-to-run CSV is provided at:

```text
examples/mirna_pairs.csv
```

It uses this schema:

```csv
id,sequence_a,sequence_b
example_positive_like,AUGCUA,AUGGUA
example_short,CUGA,CUGU
```

Required columns:

- `sequence_a`: first miRNA/RNA sequence
- `sequence_b`: second miRNA/RNA sequence
- any extra columns, such as `id`, are preserved in the output

### 2. Run a Bundled Model

Two trained miRNA model aliases are available:

- `manakov`: `case_study_for_mirna/trained_models/manakov_best_model.pkl`
- `hejret`: `case_study_for_mirna/trained_models/hejret_best_model.pkl`

Run inference:

```bash
uv run python -m src.infer \
  --model manakov \
  --input examples/mirna_pairs.csv \
  --output predictions_manakov.csv
```

Or use the Hejret-trained model:

```bash
uv run python -m src.infer \
  --model hejret \
  --input examples/mirna_pairs.csv \
  --output predictions_hejret.csv
```

### 3. Read the Output

The output CSV includes:

- `probability`: logistic model probability for the positive class
- `alignment_score`: score assigned by the fitted aligner
- `aligned_sequence_a`, `alignment_marks`, `aligned_sequence_b`: readable alignment
- `operations`: per-position `match`, `mismatch`, or `gap`
- `normalized_sequence_a`, `normalized_sequence_b`: sequences actually scored by the model

By default, `--normalize auto` converts `U`/`T` to match the trained model alphabet. Use `--normalize none` only if you want to disable this behavior.

If your second sequence column contains target/gene sequences before reverse-complementing, pass `--reverse-complement-b` so the second sequence is reverse-complemented before normalization and scoring:

```bash
uv run python -m src.infer \
  --model manakov \
  --input my_pairs.csv \
  --output my_predictions.csv \
  --seq-a-column noncodingRNA \
  --seq-b-column gene \
  --reverse-complement-b
```

### 4. Use Your Own CSV Columns

If your input columns have different names, pass them explicitly:

```bash
uv run python -m src.infer \
  --model manakov \
  --input my_pairs.csv \
  --output my_predictions.csv \
  --seq-a-column mirna \
  --seq-b-column target
```



## Case studies and experiments

### miRNA–target interactions

[case_study_for_mirna/](case_study_for_mirna/) learns alignment parameters from experimentally labeled miRNA–target pairs in miRBench's Hejret and Manakov datasets. Targets are reverse-complemented before alignment. The workflow supports model selection, refitting, and evaluation using AUPRC and ROC-AUC, including evaluation across datasets and on the Manakov split containing held-out miRNAs.

Two fitted models are included in [trained_models/](case_study_for_mirna/trained_models/). For example, evaluate the Hejret-trained model on the manuscript evaluation splits:

```bash
uv run python case_study_for_mirna/case_study_mirna.py \
  --evaluate-only \
  --trained-model case_study_for_mirna/trained_models/hejret_best_model.pkl \
  --dataset hejret \
  --eval-splits hejret_test,manakov_test,manakov_leftout \
  --run-tag hejret_evaluation
```

This requires access to the miRBench datasets. Metrics and curve data are written under `results/case_study_for_mirna/`. See the [case-study instructions](case_study_for_mirna/README.md) for training, evaluating the Manakov model, and supplying external evaluation sets.

The article reports AUPRC values of approximately 0.85 on Hejret and 0.76–0.77 on Manakov for the full alignment model. These are competitive with several neural-network baselines, although the strongest retrained neural baseline performs better on Manakov.

### BLOCKS / BLOSUM case study

[BLOCKS_case_study/BLOCKS analysis.ipynb](BLOCKS_case_study/BLOCKS%20analysis.ipynb) explores learning a discriminative amino acid substitution matrix from the BLOCKS 5.0 database, the conserved-block setting associated with BLOSUM matrices.

The notebook constructs positive pairs from sequences within the same block and an equal number of negative pairs sampled from different blocks. It then fits a **symmetric substitution matrix with local alignment and affine gap penalties**. This is an exploratory training notebook; it does not currently report a held-out comparison against BLOSUM62.

The directory contains `blocks_5.0.fa` and the original `blocks_5.0.dat`; the [directory README](BLOCKS_case_study/README.md) records their source. Run the notebook with `BLOCKS_case_study/` as its working directory, as its data paths are relative to that folder. Pair generation enumerates all within-block pairs and can require substantial memory.

### Simulation experiments

The RNA notebooks generate binary labels from known alignment parameters, then study parameter recovery, learning-rate sensitivity, and variability across replicated datasets:

- [RNA_simulated_simple_scheme.ipynb](RNA_simulated_simple_scheme.ipynb): match/mismatch weights with affine gaps.
- [RNA_simulated_general_scheme.ipynb](RNA_simulated_general_scheme.ipynb): a full asymmetric nucleotide matrix with affine gaps.

Both use miRBench sequences with simulated labels. Launch JupyterLab from the repository root:

```bash
uv run jupyter lab
```

The older [protein estimation notebook](Accuracy%20of%20learning%20on%20protein%20sequences.ipynb) investigates statistical estimation error using a scaled BLOSUM62 matrix and fixed ground-truth alignments. It requires local protein data and contains a legacy `src.estimalign` import, so it needs adaptation before running with the current API.

## Tests

```bash
uv run pytest -m "not slow"
uv run pytest
```

The suite covers alignment feature counts, gradients, training, and inference. Large miRBench end-to-end tests require an explicit opt-in with `DISCRIMALIGN_RUN_MIRBENCH=1`; they download datasets and fit models on full training sets.

## Citation

If you use DiscrimAlign in research, please cite:

Michał Aleksander Ciach, Elissavet Zacharopoulou, Michał Piotr Startek, Błażej Miasojedow, and Panagiotis Alexiou. **Discriminative Learning of Substitution Matrices and Gap Penalties for Pairwise Alignment of Biological Sequences.** WABI 2026, LIPIcs, Article 17, pp. 17:1–17:23. DOI: [10.4230/LIPIcs.WABI.2026.17](https://doi.org/10.4230/LIPIcs.WABI.2026.17).

## License

The DiscrimAlign code is released under the [MIT License](LICENSE). External datasets retain their respective terms of use.

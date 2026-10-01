# TODO: follow-ups to the nwgrad backend

Items deliberately left out of the nwgrad backend work, roughly in the order
they become relevant.

## At the nwgrad 0.5.0 release

- **Dependency:** in `pyproject.toml`, require `nwgrad>=0.5.0`.
  `SeqPairBatch.scores()` and `SeqPairBatch.weighted_grad()` first appear in
  0.5.0.
- **CI:** in `.github/workflows/tests.yml`, remove the step that installs
  nwgrad from a pinned GitHub commit (`NWGRAD_REV`), so CI tests the released
  wheel, and confirm one green run against it.
- Until then, a local `uv sync` installs nwgrad 0.4.0 from PyPI, which lacks
  the API this code uses.

## Validation

- **Real data:** run both backends on the miRNA case study and compare the
  fitted models. So far, equivalence is established on simulated data only:
  DNA and protein, up to 400 residues, at most 80 pairs per fit in the test
  suite.
- **Notebooks:** rerun the simulation notebooks under the new defaults
  (`backend="nwgrad"`, gap cap, linear-gap estimator fix). Their results will
  shift slightly.

## Known issues in DiscrimAlign

- **scikit-learn 1.10** removes `LogisticRegression(penalty=None)`, used by the
  simple-mode initial estimators in `src/optimization.py`. They will stop
  working; the replacement is `C=np.inf`. It is deprecated, with a warning,
  since 1.8.
- **`logit_logL` precision** (`src/logit_link.py`): it computes `log(p)` and
  `log1p(-p)` from `p = expit(z)`, which loses precision on confident
  predictions, and clips each pair's loss at about 36. A stable form computes
  `sum(y*z - logaddexp(0, z))` from the logits.
- **The intercept fit** (`scipy.optimize.minimize`, default BFGS tolerance)
  leaves α about 1e-4 from the optimum where the likelihood is flat. It is
  also the largest per-iteration cost at high thread counts: the alignment
  itself drops to under half of an iteration. A one-dimensional Newton or
  bracketing solve on dL/dα would be exact and cheaper.
- **`src.discrimalign` is shadowed:** `src/__init__.py` imports the function
  under the module's name, so `import src.discrimalign` binds the function.
  Use `sys.modules["src.discrimalign"]` to reach the module.
- **`tol` is accepted but unused** (TODO in `discrimalign()`).

## The gap-score cap

`discrimalign()` keeps gap scores at or below `_MAX_GAP_SCORE = -1e-4`
(`src/discrimalign.py`). The cap should be 0. It is not, because at a gap score
of exactly 0, zero-cost gap columns tie with no gap: the backends return equal
scores but different gap counts, i.e. different valid subgradients, and the
fits diverge. Reset the cap to 0 once that kink is handled consistently, e.g.
by making both backends break such ties the same way.

## nwgrad-side follow-ups

- **Local mode with positive gap scores.** nwgrad lets a local alignment end
  with a gap column but not start with one. That only differs from the
  alternatives when a gap column scores positive, which the cap prevents.
  Biopython's local mode is itself inconsistent there, so there is no reference
  to match. One strict xfail in `tests/test_nwgrad_backend.py` documents it. A
  clean definition would require local alignments to start and end with an
  aligned column, in every DP fill.
- **Default thread count.** nwgrad's `n_threads=0` picks physical cores, which
  is about 15–25% slower than all logical cores for short pairs, where the DP
  tables stay in cache. See `TODO.md` in nwgrad. DiscrimAlign therefore resolves
  `num_threads=0` to `os.cpu_count()` itself; long-sequence (protein) workloads
  may do better with physical cores (TODO in `discrimalign()`).
- **Per-pair gradients in bulk.** The initial estimate reads every pair's
  gradient through `batch[i].grad.to_dict()`, about 1.4 s per 50,000 pairs. A
  batch call returning all per-pair gradients as one array would remove that.
- **float32.** The backend uses nwgrad's double-precision classes with pointer
  traceback. The float32 classes would be faster, but nwgrad's default float32
  traceback can return a suboptimal path, and Biopython works in double.

## Pending upstream work

- **PR #10** (`theta_trajectory`) edits the optimization loop this work
  refactored. It needs to be ported onto the shared loop in `discrimalign()`.

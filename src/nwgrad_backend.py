"""
nwgrad backend for discrimalign(): alignment scores and the log-likelihood
subgradient from one nwgrad batch, built on first use and reused across
iterations.
"""
import numpy as np
import nwgrad
import nwgrad.logistic as nwgrad_logistic
from Bio.Align import substitution_matrices

from .nwgrad_params import gap_counts, grad_to_raw, to_nwgrad
from .optimization import CountArrays, EmptyLocalAlignment


def baseline_parameters(aligner, gap_mode, alphabet):
    """
    The scores of a Biopython PairwiseAligner as a DiscrimAlign parameter
    dict, with the substitutions always as a matrix over alphabet. Raises
    ValueError for anything the model cannot express: an unsupported mode, a
    wildcard, gap scores that differ between the two sequences or between
    internal and end gaps, affine gaps when gap_mode is 'linear', or a
    matrix missing letters of alphabet.
    """
    if aligner.mode not in ('local', 'global'):
        raise ValueError(f"nwgrad backend: baseline_aligner mode {aligner.mode!r} "
                         "is not supported (expected 'local' or 'global')")
    if aligner.wildcard is not None:
        raise ValueError("nwgrad backend: baseline_aligner wildcard is not supported")
    try:
        if gap_mode == 'affine':
            params = {'open_gap_score': aligner.open_gap_score,
                      'extend_gap_score': aligner.extend_gap_score}
        else:
            params = {'gap_score': aligner.gap_score}
    except ValueError as error:
        raise ValueError(
            "nwgrad backend: baseline_aligner gap scores must be the same for both "
            "sequences and for internal and end gaps"
            + (", and open must equal extend for gap_mode='linear'"
               if gap_mode == 'linear' else "")) from error
    matrix = aligner.substitution_matrix
    if matrix is None:
        data = np.full((len(alphabet), len(alphabet)), float(aligner.mismatch_score))
        data[np.diag_indices(len(alphabet))] = aligner.match_score
    else:
        missing = sorted(set(alphabet) - set(matrix.alphabet))
        if missing:
            raise ValueError("nwgrad backend: baseline_aligner substitution matrix "
                             f"lacks letters {''.join(missing)!r}")
        data = np.array([[matrix[a, b] for b in alphabet] for a in alphabet], dtype=float)
    params['substitution_matrix'] = substitution_matrices.Array(alphabet=alphabet, data=data)
    return params


class NwgradEngine:
    """
    Same interface as the Biopython engine in discrimalign: set_params(),
    then scores(), then raw_subgradient() for the scores just computed.

    Double precision with pointer traceback, so paths are exactly optimal and
    scores agree with Biopython's up to summation order.
    """

    def __init__(self, seqlistA, seqlistB, mode, gap_mode, substitution_mode,
                 alphabet, num_threads, fill='striped'):
        self.seqlistA = list(seqlistA)
        self.seqlistB = list(seqlistB)
        self.mode = mode
        self.gap_mode = gap_mode
        self.substitution_mode = substitution_mode
        self.alphabet = alphabet
        self.batch = nwgrad.SeqPairBatchDouble(n_threads=int(num_threads),
                                               traceback='pointers')
        self.batch.fill = fill
        self._built = False

    def set_params(self, params, substitution_mode=None):
        """
        substitution_mode overrides the engine's own for this call, e.g. to
        align a baseline given as a full matrix in a simple-mode fit.
        """
        nw_params = to_nwgrad(params, self.gap_mode,
                              substitution_mode or self.substitution_mode, self.alphabet)
        self._nw_params = nw_params
        if self._built:
            self.batch.set_params(nw_params)
        else:
            self.batch.add_many(self.seqlistA, self.seqlistB, nw_params,
                                gap_model=self.gap_mode, mode=self.mode,
                                grad_mode='hard')
            self._built = True

    def alignments(self, chunk_size=100_000):
        """
        Every pair's alignment at the parameters last set: a Bio.Align.Alignment
        with .score, or EmptyLocalAlignment for a local pair with no
        positive-scoring alignment. These are nwgrad's own paths, the ones the
        scores and subgradients come from, so on ties they agree with them, which
        a Biopython realignment need not. Recovering a path needs pair-owned DP
        tables, so the pairs are aligned chunk_size at a time to bound memory.
        """
        from Bio.Align import Alignment
        out = []
        for start in range(0, len(self.seqlistA), chunk_size):
            seqs_a = self.seqlistA[start:start + chunk_size]
            seqs_b = self.seqlistB[start:start + chunk_size]
            batch = nwgrad.SeqPairBatchDouble(n_threads=self.batch.n_threads,
                                              traceback='pointers')
            batch.add_many(seqs_a, seqs_b, self._nw_params, gap_model=self.gap_mode,
                           mode=self.mode, grad_mode='none')
            batch.alloc_dp()
            batch.align_full()
            for i, (seq_a, seq_b) in enumerate(zip(seqs_a, seqs_b)):
                pair = batch[i]
                coordinates = pair.coordinates()
                if (coordinates[:, 0] == coordinates[:, -1]).all():
                    out.append(EmptyLocalAlignment())
                    continue
                alignment = Alignment([seq_a, seq_b], coordinates)
                alignment.score = pair.score
                out.append(alignment)
        return out

    def scores(self):
        self.batch.score_and_grad()
        return self.batch.scores()

    def logistic_step(self, labels, alpha0):
        """
        Align, then one iteration's logistic work in nwgrad (C++, parallel):
        the log-likelihood at alpha0, the fitted alpha (as fit_alpha()), and
        the raw subgradient at that alpha, in logit_subgradient's format.
        labels: float64 array of 0s and 1s.
        """
        self.batch.score_and_grad()
        step = nwgrad_logistic.step(self.batch, labels, alpha0)
        return step.loglik_at_alpha0, step.alpha, grad_to_raw(step.grad)

    def raw_counts(self):
        """Per-pair counts in logit_subgradient's format, for the scores just computed."""
        return [grad_to_raw(self.batch[i].grad) for i in range(len(self.batch))]

    def count_arrays(self):
        """
        The same per-pair counts as raw_counts(), as CountArrays, from one
        SeqPairBatch.grads() call.
        """
        matrices, gaps = self.batch.grads()
        gap_opens, gap_extends = gap_counts(*gaps.T)
        return CountArrays(matrices, gap_opens, gap_extends, self.batch.alphabet)

    def raw_subgradient(self, logit_scores, labels, alpha):
        """
        sum_i (label_i - logit_score_i) * counts_i, in logit_subgradient's
        format. nwgrad sums the cached per-pair gradients in its own
        parametrization, and the sum is converted once, which is exact
        because the conversion is linear. alpha is unused, as in
        logit_subgradient.
        """
        weights = np.asarray(labels, dtype=float) - np.asarray(logit_scores, dtype=float)
        return grad_to_raw(self.batch.weighted_grad(weights))

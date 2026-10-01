"""
nwgrad backend for discrimalign(): alignment scores and the log-likelihood
subgradient from one nwgrad batch, built on first use and reused across
iterations.
"""
import numpy as np
import nwgrad
from Bio.Align import substitution_matrices

from .nwgrad_params import GAP_FIELDS, gap_counts, grad_to_raw, to_nwgrad
from .optimization import CountArrays


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
                 alphabet, num_threads):
        self.seqlistA = list(seqlistA)
        self.seqlistB = list(seqlistB)
        self.mode = mode
        self.gap_mode = gap_mode
        self.substitution_mode = substitution_mode
        self.alphabet = alphabet
        self.batch = nwgrad.SeqPairBatchDouble(n_threads=int(num_threads),
                                               traceback='pointers')
        self._built = False

    def set_params(self, params, substitution_mode=None):
        """
        substitution_mode overrides the engine's own for this call, e.g. to
        align a baseline given as a full matrix in a simple-mode fit.
        """
        nw_params = to_nwgrad(params, self.gap_mode,
                              substitution_mode or self.substitution_mode, self.alphabet)
        if self._built:
            self.batch.set_params(nw_params)
        else:
            self.batch.add_many(self.seqlistA, self.seqlistB, nw_params,
                                gap_model=self.gap_mode, mode=self.mode,
                                grad_mode='hard')
            self._built = True

    def scores(self):
        self.batch.score_and_grad()
        return self.batch.scores()

    def raw_counts(self):
        """Per-pair counts in logit_subgradient's format, for the scores just computed."""
        return [grad_to_raw(self.batch[i].grad) for i in range(len(self.batch))]

    def count_arrays(self):
        """
        The same per-pair counts as raw_counts(), as CountArrays: from one
        SeqPairBatch.grads() call where nwgrad has it, else pair by pair.
        """
        if hasattr(self.batch, 'grads') and len(self.batch):
            return self._count_arrays_bulk()
        return self._count_arrays_per_pair()

    def _count_arrays_bulk(self):
        matrices, gaps = self.batch.grads()
        gap_opens, gap_extends = gap_counts(*gaps.T)
        return CountArrays(matrices, gap_opens, gap_extends, self.batch.alphabet)

    def _count_arrays_per_pair(self):
        n = len(self.batch)
        substitutions = None
        gap_opens = np.empty(n)
        gap_extends = np.empty(n)
        for i in range(n):
            grad = self.batch[i].grad.to_dict()
            if substitutions is None:
                alphabet = grad['alphabet']
                substitutions = np.empty((n, len(alphabet), len(alphabet)))
            substitutions[i] = grad['matrix']
            gap_opens[i], gap_extends[i] = gap_counts(*(grad[field] for field in GAP_FIELDS))
        if substitutions is None:
            alphabet = self.alphabet
            substitutions = np.empty((0, len(alphabet), len(alphabet)))
        return CountArrays(substitutions, gap_opens, gap_extends, alphabet)

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

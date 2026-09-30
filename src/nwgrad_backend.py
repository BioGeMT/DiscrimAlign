"""
nwgrad backend for discrimalign(): alignment scores and the log-likelihood
subgradient from one nwgrad batch, built on first use and reused across
iterations.
"""
import numpy as np
import nwgrad

from .nwgrad_params import grad_to_raw, to_nwgrad


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

    def set_params(self, params):
        nw_params = to_nwgrad(params, self.gap_mode, self.substitution_mode, self.alphabet)
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

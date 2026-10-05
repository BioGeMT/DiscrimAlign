"""
Functions to calculate the logistic function of alignment score
and its subgradients
"""

import numpy as np
from Bio.Align import substitution_matrices
from scipy.special import expit


def logit_partial_scores(alignment_scores, alpha):
    """
    Calculate the logistic function of the alignment scores.
    This is the function \varsigma(score; alpha) in the paper.
    """
    alignment_scores = np.asarray(alignment_scores, dtype=float)
    return expit(alpha + alignment_scores)


def logit_logL(alignment_scores, alpha, labels):
    """
    Calculate the log likelihood in the logistic model:
    sum_i y_i z_i - log(1 + e^{z_i}), with z_i = alpha + alignment_scores[i].
    Computed from the logits rather than from the probabilities, so it needs
    no clipping and loses no precision on confident predictions, and
    log(1 + e^z) is evaluated without overflow.
    labels are a 1D array with values 0 or 1.
    Raises FloatingPointError if the result is not finite.
    """
    labels = np.asarray(labels)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Labels can only be 0 or 1')
    return _logit_logL_unchecked(np.asarray(alignment_scores, dtype=float), alpha,
                                 labels.astype(float))


def _logit_logL_unchecked(alignment_scores, alpha, labels):
    """
    logit_logL() for a float array of alignment scores and float labels
    already known to be 0 or 1. For loops that evaluate it on the same labels
    many times.
    """
    z = alignment_scores + alpha
    with np.errstate(invalid='ignore'):   # a non-finite result raises below
        logL = float(np.dot(labels, z) - np.sum(np.logaddexp(0.0, z)))
    if not np.isfinite(logL):
        raise FloatingPointError(f'Log-likelihood is {logL} at alpha={alpha}: '
                                 'alignment scores or alpha are not finite')
    return logL


def dlda(logit_scores, labels):
    """
    Calculate the derivative of the likelihood function with respect to
    the intercept alpha, d_l/d_alpha.
    The variable labels needs to be a binary iterable with values 0 or 1
    """
    return np.sum(labels - logit_scores)


def d2lda2(logit_scores, labels):
    """
    Calculate the second derivative of the likelihood function with respect to
    the intercept alpha, d^2_l/d_alpha^2.
    """
    return -np.sum(logit_scores*(1-logit_scores))


def fit_alpha(alignment_scores, labels, alpha0, tol=1e-12, max_newton=8, maxiter=200,
              logit_scores0=None):
    """
    The intercept alpha that maximises the likelihood for fixed alignment
    scores, starting from alpha0.

    This is the root of dlda, which is strictly decreasing in alpha, so it is
    unique whenever both classes are present. Plain Newton steps are taken from
    alpha0 while they stay small (|step| <= 1), which is the usual case when
    alpha0 is the previous iteration's alpha. Otherwise the root is bracketed
    and found by Newton steps inside the bracket, with bisection whenever a
    Newton step would leave it or converge too slowly (Numerical Recipes'
    rtsafe). The log-likelihood itself is never evaluated.

    logit_scores0, if given, must be logit_partial_scores(alignment_scores,
    alpha0); a caller that already has them saves one pass over the data.
    """
    labels = np.asarray(labels, dtype=float)
    positives = np.sum(labels)
    if positives == 0 or positives == len(labels):
        raise ValueError('Fitting alpha needs labels of both classes: with one class '
                         'the likelihood has no finite maximum')
    return _fit_alpha_unchecked(np.asarray(alignment_scores, dtype=float), labels, alpha0,
                                tol, max_newton, maxiter, logit_scores0)


def _fit_alpha_unchecked(alignment_scores, labels, alpha0, tol=1e-12, max_newton=8,
                         maxiter=200, logit_scores0=None):
    """
    fit_alpha() for float arrays of scores and labels, the labels already
    known to be 0 or 1 with both classes present. For loops that fit alpha on
    the same labels many times.
    """
    def derivatives(alpha, logit_scores=None):
        if logit_scores is None:
            logit_scores = logit_partial_scores(alignment_scores, alpha)
        return float(dlda(logit_scores, labels)), float(-d2lda2(logit_scores, labels))

    alpha = float(alpha0)
    for k in range(max_newton):
        g, h = derivatives(alpha, logit_scores0 if k == 0 else None)
        step = g / h if h > 0 else np.inf
        if not np.isfinite(step) or abs(step) > 1.0:
            break
        alpha += step
        if abs(step) <= tol * max(1.0, abs(alpha)):
            return alpha

    # Bracket the root: dlda(lo) >= 0 >= dlda(hi).
    lo, hi, width = alpha - 1.0, alpha + 1.0, 1.0
    for _ in range(maxiter):
        if derivatives(lo)[0] >= 0:
            break
        lo -= width
        width *= 2
    else:
        raise RuntimeError('fit_alpha: could not bracket the root')
    width = 1.0
    for _ in range(maxiter):
        if derivatives(hi)[0] <= 0:
            break
        hi += width
        width *= 2
    else:
        raise RuntimeError('fit_alpha: could not bracket the root')

    alpha = min(max(alpha, lo), hi)
    dx_old = dx = hi - lo
    g, h = derivatives(alpha)
    for _ in range(maxiter):
        if g == 0.0:
            return alpha
        if h > 0 and lo <= alpha + g / h <= hi and abs(2 * g) <= abs(dx_old * h):
            dx_old, dx = dx, g / h
            alpha += dx
        else:
            dx_old, dx = dx, 0.5 * (hi - lo)
            alpha = lo + dx
        if abs(dx) <= tol * max(1.0, abs(alpha)):
            return alpha
        g, h = derivatives(alpha)
        if g > 0:
            lo = alpha
        else:
            hi = alpha
    raise RuntimeError('fit_alpha: did not converge')


def logit_subgradient(alignment_list, logit_scores,
                      labels, alpha,
                      alphabet):
    """
    Calculate a (random) subgradient of the log-likelihood function
    with respect to the alignment scoring matrix and gap open and
    extend penalties.
    """
    assert len(alignment_list) == len(logit_scores)
    assert len(logit_scores) == len(labels)
    substitution_counts = substitution_matrices.Array(alphabet=alphabet,
                                                data = np.zeros((len(alphabet), len(alphabet))))
    subgradient = {'Substitutions': substitution_counts,
                   'Gap opens': 0,
                   'Gap extends': 0}
    for aln, lab, lscore in zip(alignment_list, labels, logit_scores):
        ingap1 = False
        ingap2 = False
        weight = lab - lscore
        for char1, char2 in zip(aln[0], aln[1]):
            if char1 == '-':
                if ingap1:
                    subgradient['Gap extends'] += weight
                else:
                    subgradient['Gap opens'] += weight
                ingap1 = True
                ingap2 = False
            elif char2 == '-':
                if ingap2:
                    subgradient['Gap extends'] += weight
                else:
                    subgradient['Gap opens'] += weight
                ingap1 = False
                ingap2 = True
            else:
                ingap1 = False
                ingap2 = False
                subgradient['Substitutions'][char1, char2] += weight
    return subgradient

if __name__ == '__main__':
    seq1 = 'CCTTTCCCGGGGTCTAAGGGTT'
    seq2 = 'TTACCCAAAATCTGGGCC'
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.open_gap_score = -6
    aligner.extend_gap_score = -0.5
    aligner.match_score = 5
    aligner.mismatch_score = -4
    aln = next(aligner.align(seq1, seq2))
    lsub = logit_subgradient([aln], [0], [1], 0.5, 'ACTG')
    print(aln)
    print(lsub)


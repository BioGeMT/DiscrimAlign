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


def logit_logL(logit_scores, labels):
    """
    Calculate the log likelihood in the logistic model.
    logit_scores are a list of values of the logistic function
    of the alignment score, calculated with logit_partial_scores.
    labels are a 1D numpy array with values 0 or 1.
    """
    logit_scores = np.asarray(logit_scores, dtype=float)
    labels = np.asarray(labels)
    eps = np.finfo(float).eps
    logit_scores = np.clip(logit_scores, eps, 1.0 - eps)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Labels can only be 0 or 1')
    return float(np.sum(labels * np.log(logit_scores) + (1 - labels) * np.log1p(-logit_scores)))


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
            elif char2 == '-':
                if ingap2:
                    subgradient['Gap extends'] += weight
                else:
                    subgradient['Gap opens'] += weight
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


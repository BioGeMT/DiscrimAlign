"""
Functions to calculate the logistic function of alignment score
and its subgradients
"""

import numpy as np
from Bio.Align import substitution_matrices


def logit_partial_scores(alignment_scores, alpha):
    """
    Calculate the logistic function of the alignment scores.
    This is the function \varsigma(score; alpha) in the paper
    """
    return 1/(1 + np.exp(-alpha - alignment_scores))

def logit_logL(logit_scores, labels):
    """
    Calculate the log likelihood in the logistic model.
    logit_scores are a list of values of the logistic function
    of the alignment score, calculated with logit_partial_scores.
    labels are a 1D numpy array with values 0 or 1.   
    """
    logl = 0
    for p, lab in zip(logit_scores, labels):
        if lab == 1:
            logl += np.log(p)
        elif lab == 0:
            logl += np.log(1-p)
        else:
            raise ValueError('Labels can only be 0 or 1')
    return logl

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
        ingap = False
        weight = lab - lscore
        for char1, char2 in zip(aln[0], aln[1]):
            if char1 == '-' or char2 == '-':
                if ingap:
                    subgradient['Gap extends'] += weight
                else:
                    subgradient['Gap opens'] += weight
                ingap = True
            else:
                ingap = False
                subgradient['Substitutions'][char1, char2] += weight
    return subgradient
    




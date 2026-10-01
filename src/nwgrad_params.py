"""
Conversion between DiscrimAlign's alignment parameters and nwgrad's.

DiscrimAlign keeps parameters in Biopython's convention: gap values are
negative scores, and a gap of length k scores open + (k-1)*extend.
nwgrad takes positive penalties, a gap of length k costs open + k*extend,
and gaps in sequence A (*_a) and sequence B (*_b) are priced separately.
DiscrimAlign's model has one gap cost for both sequences.
"""
import numpy as np
import nwgrad
from Bio.Align import substitution_matrices


def substitution_data(params, substitution_mode, alphabet):
    """
    The substitution matrix as a float64 array, and its alphabet.

    In simple mode it is built from match_score and mismatch_score over
    alphabet. Otherwise the matrix's own alphabet is used, if it has one.
    """
    if substitution_mode == 'simple':
        n = len(alphabet)
        data = np.full((n, n), float(params['mismatch_score']))
        data[np.diag_indices(n)] = params['match_score']
        return data, alphabet
    matrix = params['substitution_matrix']
    matrix_alphabet = getattr(matrix, 'alphabet', None)
    if matrix_alphabet is not None:
        alphabet = ''.join(matrix_alphabet)
    return np.ascontiguousarray(matrix, dtype=np.float64), alphabet


def gap_penalties(params, gap_mode):
    """(gap_open, gap_extend) as nwgrad penalties, for either sequence."""
    if gap_mode == 'affine':
        return (params['extend_gap_score'] - params['open_gap_score'],
                -params['extend_gap_score'])
    return 0.0, -params['gap_score']


def to_nwgrad(params, gap_mode, substitution_mode, alphabet):
    """
    nwgrad.AlignParams equivalent to a DiscrimAlign parameter dict.

    alphabet is used in simple mode, and in the other modes when the
    substitution matrix does not carry its own alphabet.
    """
    data, alphabet = substitution_data(params, substitution_mode, alphabet)
    gap_open, gap_extend = gap_penalties(params, gap_mode)
    return nwgrad.AlignParams(nwgrad.SubstMatrix(data, alphabet=alphabet),
                              gap_open_a=gap_open, gap_extend_a=gap_extend,
                              gap_open_b=gap_open, gap_extend_b=gap_extend)


def grad_to_raw(grad):
    """
    An nwgrad gradient in logit_subgradient's output format.

    grad is an nwgrad.AlignParams gradient, or a dict shaped like its
    to_dict(). nwgrad reports gap gradients as minus the counts: gap_open is
    minus the number of gaps and gap_extend minus the number of gap columns.
    Biopython counts the first column of each gap as its open, and the rest
    as extends. In linear mode nwgrad only reports gap columns, so all of
    them land in 'Gap extends' and only the sum of the two is meaningful.
    The conversion is linear, so weighted sums of gradients may be
    converted after summing.
    """
    if hasattr(grad, 'to_dict'):
        grad = grad.to_dict()
    gap_opens, gap_extends = gap_counts(grad)
    substitutions = substitution_matrices.Array(alphabet=grad['alphabet'],
                                                data=np.array(grad['matrix'], dtype=float))
    return {'Substitutions': substitutions,
            'Gap opens': gap_opens,
            'Gap extends': gap_extends}


def gap_counts(grad):
    """
    (gap opens, gap extends) in logit_subgradient's sense, from an nwgrad
    gradient's to_dict(). See grad_to_raw().
    """
    gap_open = grad['gap_open_a'] + grad['gap_open_b']
    gap_extend = grad['gap_extend_a'] + grad['gap_extend_b']
    return -gap_open, gap_open - gap_extend

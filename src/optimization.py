from typing import NamedTuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from Bio.Align import substitution_matrices


class EmptyAlignmentCounts:
    identities = 0
    mismatches = 0
    open_gaps = 0
    extend_gaps = 0
    gaps = 0


class EmptyLocalAlignment:
    """Representation of the empty local alignment."""

    score = 0.0

    def counts(self):
        return EmptyAlignmentCounts()

    def __getitem__(self, index):
        if index in (0, 1):
            return ""
        raise IndexError(index)


def get_first_alignment(seqA, seqB, aligner):
    """Return the first alignment, or the empty local alignment when none exists."""
    try:
        return next(aligner.align(seqA, seqB))
    except StopIteration:
        if getattr(aligner, "mode", None) == "local":
            return EmptyLocalAlignment()
        raise

### Starting point
def _alignment_features(alignment_list, substitution_mode, gap_mode, alphabet):
    """
    Predictors for the initial estimate, one row per alignment: numbers of
    matches and mismatches (simple) or of each substitution (full), then the
    numbers of gap opens and gap extends (affine) or of gap columns (linear).
    """
    predictors = []
    if substitution_mode != 'simple':
        Asize = len(alphabet)
        pair_to_id = {(char1, char2): Asize*i + j for i, char1 in enumerate(alphabet) for j, char2 in enumerate(alphabet)}
    for aln in alignment_list:
        counts = aln.counts()
        if gap_mode == 'affine':
            gaps = [counts.open_gaps,
                    counts.extend_gaps]
        else:
            gaps = [counts.gaps]
        if substitution_mode == 'simple':
            predictors.append([counts.identities, counts.mismatches] + gaps)
        else:
            substitutions = [0]*(Asize**2)
            for char1, char2 in zip(aln[0], aln[1]):
                if char1 != '-' and char2 != '-':
                    substitutions[pair_to_id[(char1, char2)]] += 1
            predictors.append(substitutions+gaps)
    return predictors


class CountArrays(NamedTuple):
    """
    Per-alignment counts in logit_subgradient's sense, one row per alignment:
    substitutions (N, A, A) with rows and columns in the order of alphabet,
    and gap_opens and gap_extends (N,).
    """
    substitutions: np.ndarray
    gap_opens: np.ndarray
    gap_extends: np.ndarray
    alphabet: str


def _count_arrays_from_raw(raw_counts_list):
    """
    CountArrays from a list of counts in logit_subgradient's format, in the
    alphabet order of the first one.
    """
    n = len(raw_counts_list)
    alphabet = ''.join(raw_counts_list[0]['Substitutions'].alphabet) if n else ''
    substitutions = np.empty((n, len(alphabet), len(alphabet)))
    gap_opens = np.empty(n)
    gap_extends = np.empty(n)
    for i, raw in enumerate(raw_counts_list):
        subs = raw['Substitutions']
        order = [subs.alphabet.index(char) for char in alphabet]
        substitutions[i] = np.asarray(subs)[np.ix_(order, order)]
        gap_opens[i] = raw['Gap opens']
        gap_extends[i] = raw['Gap extends']
    return CountArrays(substitutions, gap_opens, gap_extends, alphabet)


def _count_features(counts, substitution_mode, gap_mode, alphabet):
    """
    The same predictors as _alignment_features(), as an (N, p) array, from
    per-alignment counts: CountArrays, or a list in logit_subgradient's format
    ({'Substitutions', 'Gap opens', 'Gap extends'}), such as nwgrad's per-pair
    gradients converted by grad_to_raw(). In linear mode only the total number
    of gap columns is used, which is all that nwgrad reports there.
    """
    if not isinstance(counts, CountArrays):
        counts = _count_arrays_from_raw(counts)
    S = counts.substitutions
    gaps = ([counts.gap_opens, counts.gap_extends] if gap_mode == 'affine'
            else [counts.gap_opens + counts.gap_extends])
    if substitution_mode == 'simple':
        matches = np.trace(S, axis1=1, axis2=2)
        columns = [matches, S.sum(axis=(1, 2)) - matches] + gaps
        return np.column_stack(columns) if len(S) else np.empty((0, len(columns)))
    # Rows and columns in the order of alphabet, whatever the order of the counts.
    if alphabet != counts.alphabet:
        order = [counts.alphabet.index(char) for char in alphabet]
        S = S[:, order][:, :, order]
    features = np.empty((len(S), len(alphabet)**2 + len(gaps)))
    features[:, :len(alphabet)**2] = S.reshape(len(S), -1)
    for k, column in enumerate(gaps, start=len(alphabet)**2):
        features[:, k] = column
    return features


def _fit_initial_estimate(predictors, labels, substitution_mode, gap_mode, alphabet):
    """
    Logistic regression of the labels on the predictors. Simple mode is
    unpenalized; the full matrix uses the default ridge penalty.
    """
    if substitution_mode == 'simple':
        logit = LogisticRegression(fit_intercept=True, penalty=None)
        logit.fit(predictors, labels)
        estimates = {'alpha': logit.intercept_[0],
                     'match_score': logit.coef_[0][0],
                     'mismatch_score': logit.coef_[0][1]}
    else:
        Asize = len(alphabet)
        pair_to_id = {(char1, char2): Asize*i + j for i, char1 in enumerate(alphabet) for j, char2 in enumerate(alphabet)}
        logit = LogisticRegression(fit_intercept=True, solver='newton-cg')
        logit.fit(predictors, labels)
        substitution_matrix = substitution_matrices.Array(data=np.zeros((Asize, Asize)),
                                                          alphabet=alphabet)
        for char1 in alphabet:
            for char2 in alphabet:
                substitution_matrix[char1, char2] = logit.coef_[0][pair_to_id[(char1, char2)]]
        if substitution_mode == 'symmetric':
            # estimation of symmetric matrix should be implemented in
            # a separate function, for now we use this trick
            substitution_matrix = (substitution_matrix.T + substitution_matrix)/2
        estimates = {'alpha': logit.intercept_[0],
                     'substitution_matrix': substitution_matrix}
    if gap_mode == 'affine':
        estimates['open_gap_score'] = logit.coef_[0][-2]
        estimates['extend_gap_score'] = logit.coef_[0][-1]
    else:
        estimates['gap_score'] = logit.coef_[0][-1]
    return estimates


def _check_estimate_modes(substitution_mode, gap_mode, alphabet):
    assert gap_mode in {'affine', 'linear'}, 'Only linear and affine gap modes are supported'
    assert substitution_mode in {'simple', 'symmetric', 'general'}
    if substitution_mode != 'simple':
        assert alphabet is not None, 'General and symmetric substitution mode require to specify the alphabet'


def get_initial_estimate(alignment_list, labels,
                         substitution_mode = 'simple',
                         gap_mode = 'affine',
                         alphabet=None):
    """
    Returns an initial estimator of alignment parameters
    using a simple logistic models with intercept and
    summary predictors: numbers of matches, mismatches, and gaps.
    """
    _check_estimate_modes(substitution_mode, gap_mode, alphabet)
    predictors = _alignment_features(alignment_list, substitution_mode, gap_mode, alphabet)
    return _fit_initial_estimate(predictors, labels, substitution_mode, gap_mode, alphabet)


def get_initial_estimate_from_counts(counts, labels,
                                     substitution_mode='simple',
                                     gap_mode='affine',
                                     alphabet=None):
    """
    get_initial_estimate() from per-alignment counts instead of alignment
    objects: CountArrays, or a list in logit_subgradient's format.
    """
    _check_estimate_modes(substitution_mode, gap_mode, alphabet)
    predictors = _count_features(counts, substitution_mode, gap_mode, alphabet)
    return _fit_initial_estimate(predictors, labels, substitution_mode, gap_mode, alphabet)

### Parallel processing
def create_alignment_workers(seqlistA, seqlistB, aligner):
    """
    Create joblib parallel workers
    """
    from joblib import delayed
    def return_alignment(seqA, seqB, aligner):
        return get_first_alignment(seqA, seqB, aligner)
    for seqA, seqB in zip(seqlistA, seqlistB):
        yield delayed(return_alignment)(seqA, seqB, aligner)
        
### Subgradient method stepfunctions
def create_constant_step(scale):
    def step(niter):
        return scale
    return step


def create_powerstep(scale, power=0.5, burnin=0):
    """
    Function to create a step function in which the step
    scale is equal to scale/iteration_number**power.
    Typically power == 0.5.
    The power scaling kicks in after a burnin number of steps, before
    which it's equal to scale. 
    """
    def step(niter):
        if niter >= burnin:
            return scale/(niter - burnin + 1)**power
        else:
            return scale
    return step

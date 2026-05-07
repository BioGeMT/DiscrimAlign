import numpy as np
from sklearn.linear_model import LogisticRegression
from Bio.Align import substitution_matrices

### Starting point
def _initial_estim_affinegap_simplesubs(alignment_list, labels):
    """
    Returns an initial estimator of alignment parameters
    using a simple logistic model with intercept and
    summary predictors: numbers of matches, mismatches, gap opens and gap extends.
    """
    predictors = []
    for aln in alignment_list:
        counts = aln.counts()
        predictors.append(
            [
                counts.identities,
                counts.mismatches,
                counts.open_gaps,
                counts.extend_gaps
            ]
                )
    logit = LogisticRegression(fit_intercept=True, penalty=None)
    logit.fit(predictors, labels)
    estimates = {'alpha': logit.intercept_[0],
                 'match_score': logit.coef_[0][0],
                 'mismatch_score': logit.coef_[0][1],
                 'open_gap_score': logit.coef_[0][2],
                 'extend_gap_score': logit.coef_[0][3]}
    return estimates

def _initial_estim_lineargap_simplesubs(alignment_list, labels):
    """
    Returns an initial estimator of alignment parameters
    using a simple logistic model with intercept and
    summary predictors: numbers of matches, mismatches, gap opens and gaps.
    """
    predictors = []
    for aln in alignment_list:
        counts = aln.counts()
        predictors.append(
            [
                counts.identities,
                counts.mismatches,
                counts.gaps
            ]
                )
    logit = LogisticRegression(fit_intercept=True, penalty=None)
    logit.fit(predictors, labels)
    estimates = {'alpha': logit.intercept_[0],
                 'match_score': logit.coef_[0][0],
                 'mismatch_score': logit.coef_[0][1],
                 'gap_score': logit.coef_[0][2]}
    return estimates

def _initial_estim_affinegap_fullsubs(alignment_list, labels,
                                      alphabet):
    """
    Returns an initial estimator of alignment parameters
    using a Ridge logistic model with intercept and
    a full set of predictors: numbers of gap extends and a substitution matrix
    """
    Asize = len(alphabet)
    predictors = []
    pair_to_id = {(char1, char2): Asize*i + j for i, char1 in enumerate(alphabet) for j, char2 in enumerate(alphabet)}
    for aln in alignment_list:
        counts = aln.counts()
        gaps = [counts.open_gaps,
                counts.extend_gaps]
        substitutions = [0]*(Asize**2)
        for char1, char2 in zip(aln[0], aln[1]):
            if char1 != '-' and char2 != '-':
                substitutions[pair_to_id[(char1, char2)]] += 1
        predictors.append(substitutions+gaps)
    
    logit = LogisticRegression(fit_intercept=True, solver='newton-cg')
    logit.fit(predictors, labels)
    substitution_matrix = substitution_matrices.Array(data=np.zeros((Asize, Asize)),
                                                      alphabet=alphabet)
    for char1 in alphabet:
        for char2 in alphabet:
            substitution_matrix[char1, char2] = logit.coef_[0][pair_to_id[(char1, char2)]]
    open_gap_score = logit.coef_[0][-2]
    extend_gap_score = logit.coef_[0][-1]
    estimates = {'alpha': logit.intercept_[0],
                 'substitution_matrix': substitution_matrix,
                 'open_gap_score': open_gap_score,
                 'extend_gap_score': extend_gap_score}
    return estimates


def get_initial_estimate(alignment_list, labels,
                         substitution_mode = 'simple',
                         gap_mode = 'affine',
                         alphabet=None):
    """
    Returns an initial estimator of alignment parameters
    using a simple logistic models with intercept and
    summary predictors: numbers of matches, mismatches, and gaps.
    """
    assert gap_mode in {'affine', 'linear'}, 'Only linear and affine gap modes are supported'
    assert substitution_mode in {'simple', 'symmetric', 'general'}
    if substitution_mode != 'simple':
        assert alphabet is not None, 'General and symmetric substitution mode require to specify the alphabet'
    if substitution_mode == 'simple':
        if gap_mode == 'affine':
            estimates = _initial_estim_affinegap_simplesubs(alignment_list, labels)
        elif gap_mode == 'linear':
            estimates = _initial_estim_lineargap_simplesubs(alignment_list, labels)
    else:
        estimates = _initial_estim_affinegap_fullsubs(alignment_list, labels, alphabet)
        if substitution_mode == 'symmetric':
            # estimation of symmetric matrix should be implemented in
            # a separate function, for now we use this trick
            subsM = estimates['substitution_matrix']
            subsM = (subsM.T + subsM)/2
            estimates['substitution_matrix'] = subsM
        if gap_mode == 'linear':
            estimates['gap_score'] = estimates['open_gap_score']+ estimates['extend_gap_score']
            del estimates['open_gap_score']
            del estimates['extend_gap_score']
    return estimates

### Parallel processing
def create_alignment_workers(seqlistA, seqlistB, aligner):
    """
    Create joblib parallel workers
    """
    from joblib import delayed
    def return_alignment(seqA, seqB, aligner):
        aln = aligner.align(seqA, seqB)
        return next(aln)
    for seqA, seqB in zip(seqlistA, seqlistB):
        yield delayed(return_alignment)(seqA, seqB, aligner)


def get_first_alignment(seqA, seqB, aligner):
    """
    Return the first optimal alignment for a pair of sequences.
    This helper is used by the miRNA case-study workflow.
    """
    return next(aligner.align(seqA, seqB))
        
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

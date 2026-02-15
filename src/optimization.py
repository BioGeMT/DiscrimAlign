import numpy as np
from sklearn.linear_model import LogisticRegression

### Starting point
def get_initial_estimate(alignment_list, labels, gap_mode = 'affine'):
    """
    Returns an initial estimator of alignment parameters
    using a simple logistic models with intercept and
    summary predictors: numbers of matches, mismatches, and gaps.
    """
    predictors = []
    for aln in alignment_list:
        counts = aln.counts()
        if gap_mode == 'affine':
            predictors.append(
                [
                    counts.identities,
                    counts.mismatches,
                    counts.open_gaps,
                    counts.extend_gaps
                ]
                    )
        elif gap_mode == 'linear':
            predictors.append(
                [
                    counts.identities,
                    counts.mismatches,
                    counts.gaps
                ]
                    )
        else:
            raise ValueError('Only linear and affine gap modes are supported')
    logit = LogisticRegression(fit_intercept=True, penalty=None)
    logit.fit(predictors, labels)
    estimates = {'alpha': logit.intercept_[0],
                 'match_score': logit.coef_[0][0],
                 'mismatch_score': logit.coef_[0][1]}
    if gap_mode == 'affine':
        estimates['open_gap_score'] = logit.coef_[0][2]
        estimates['extend_gap_score'] = logit.coef_[0][3]
    elif gap_mode == 'linear':
        estimates['gap_score'] = logit.coef_[0][2]
    else:
        raise ValueError('Only linear and affine gap modes are supported')
    return estimates

### Parallel processing
def create_alignment_workers(seqlistA, seqlistB, aligner):
    """
    Create joblib parallel workers
    """
    from joblib import delayed
    def return_alignment(seqA, seqB, aligner):
        return next(aligner.align(seqA, seqB))
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

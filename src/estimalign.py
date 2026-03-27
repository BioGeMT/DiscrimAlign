"""
Main functions of the package used for estimating
alignment parameters from labeled pairs of sequences.
"""
from Bio.Align import PairwiseAligner, substitution_matrices
import numpy as np
from numpy import random as rd
from scipy.optimize import minimize
from copy import deepcopy
from .optimization import get_initial_estimate
from .logit_link import logit_partial_scores, logit_logL, logit_subgradient

def estimalign(seqlistA, seqlistB,
               labels,
               baseline_aligner=None,
               aligner_mode = 'local',
               gap_mode = 'affine',
               substitution_mode = 'symmetric',
               alphabet = None,
               stochastic_factor=None,
               stepfunction = None,
               max_iter=1000, tol=1e-3,
               num_threads = 1,
               verbose=False):
    ### TOOD: Implement tol, 
    # and stepfunctions (put them in optimization.py)
    assert aligner_mode in {'local', 'global'}
    assert gap_mode in {'affine', 'linear'}
    assert substitution_mode in {'general', 'symmetric', 'simple'}
    if num_threads > 1:
        from joblib import Parallel
        from .optimization import create_alignment_workers
        parallel = Parallel(n_jobs=num_threads,
                            return_as = 'list')

        
    if alphabet is None:
        charsetA = set(char for seq in seqlistA for char in seq)
        charsetB = set(char for seq in seqlistB for char in seq)
        alphabet = charsetA | charsetB
        alphabet = ''.join(alphabet)

    if verbose:
        print('Alphabet:')
        print(alphabet)

    # Stochastic trick function
    if stochastic_factor is None:
        def add_noise(shape, niter):
            if shape == 1: 
                return 0.
            else:
                return np.zeros(shape)
            # Note: For consistency with numpy, returning a number should
            # happen when shape == None, shape 1 should return an array.
            # This would require modifying the code elsewhere
    else:
        def add_noise(shape, niter):
            if shape == 1:
                return rd.normal(scale=stochastic_factor/(niter+1))
            else:
                return rd.normal(size=shape, scale=stochastic_factor/(niter+1))

    # Initial alignments
    if baseline_aligner is not None:
        aligner = deepcopy(baseline_aligner)
    else:
        aligner = PairwiseAligner()
        aligner.mode = aligner_mode
        if gap_mode == 'affine':
            aligner.open_gap_score = -8
            aligner.extend_gap_score = -0.5
        elif gap_mode == 'linear':
            aligner.gap_score = -6
        if substitution_mode == 'simple':
            aligner.match_score = 5
            aligner.mismatch_score = -4
        else:
            aligner.substitution_matrix = substitution_matrices.Array(data=9*np.eye(len(alphabet))-4,
                                                                      alphabet=alphabet)

    if num_threads == 1:
        alnlist = [aligner.align(seqA, seqB) for seqA, seqB in zip(seqlistA, seqlistB)]
        alnlist = [next(aln) for aln in alnlist]
    else:
        workers = create_alignment_workers(seqlistA, seqlistB, aligner)
        alnlist = parallel(workers)
    alignment_scores = [aln.score for aln in alnlist]
    
    # Initial logistic estimation
    updated_parameters = get_initial_estimate(alnlist, labels,
                                              substitution_mode = substitution_mode,
                                              gap_mode =gap_mode,
                                              alphabet=alphabet)
    if verbose:
        print('Initial parameters:')
        print(updated_parameters)
    if gap_mode == 'affine':
        aligner.open_gap_score = updated_parameters['open_gap_score']
        aligner.extend_gap_score = updated_parameters['extend_gap_score']
    else:
        aligner.gap_score = updated_parameters['gap_score']

    if substitution_mode == 'simple':
        aligner.match_score = updated_parameters['match_score']
        aligner.mismatch_score = updated_parameters['mismatch_score']
    else:
        aligner.substitution_matrix = updated_parameters['substitution_matrix']
    
    # Subgradient refinement
    loglik_trajectory = []
    subgradient_l2_trajectory = []
    loglik_expectation = []
    loglik_sd = []
    for iternb in range(max_iter):
        if verbose:
            print('Start of iteration', iternb)
        # Set new aligner parameters
        if gap_mode == 'affine':
            aligner.open_gap_score = updated_parameters['open_gap_score']
            aligner.extend_gap_score = updated_parameters['extend_gap_score']
        elif gap_mode == 'linear':
            aligner.gap_score = updated_parameters['gap_score']

        if substitution_mode == 'simple':
            aligner.match_score = updated_parameters['match_score']
            aligner.mismatch_score = updated_parameters['mismatch_score']
        else:
            aligner.substitution_matrix = updated_parameters['substitution_matrix']

        # Realign with the new parameters
        if num_threads == 1:
            alnlist = [aligner.align(seqA, seqB) for seqA, seqB in zip(seqlistA, seqlistB)]
            alnlist = [next(aln) for aln in alnlist]
        else:
            workers = create_alignment_workers(seqlistA, seqlistB, aligner)
            alnlist = parallel(workers)
        alignment_scores = [aln.score for aln in alnlist]
        logit_scores = logit_partial_scores(alignment_scores,
                                                updated_parameters['alpha'])
        new_logL = logit_logL(logit_scores, labels)
        loglik_trajectory.append(new_logL)
        if verbose:
            print("Current alpha:", updated_parameters['alpha'])
            print('Current logL:', new_logL)
##        EL = 0
##        VL = 0
##        for ls in logit_scores:
##            if 1e-30 < ls < 1-1e-30:
##                EL += ls*np.log(ls) + (1-ls)*np.log(1-ls)
##                VL += ls*(1-ls)*(np.log(ls)**2 + np.log(1-ls)**2)
##        SDL = np.sqrt(VL)
##        loglik_expectation.append(EL)
##        loglik_sd.append(SDL)
        
        # Optimize the logistic intercept (alpha)
        def alpha_target(alpha):
            logit_scores = logit_partial_scores(alignment_scores, alpha)
            return -logit_logL(logit_scores, labels)

        def alpha_fprime(alpha):
            logit_scores = logit_partial_scores(alignment_scores, alpha)
            return  -np.sum(labels - logit_scores)

        def alpha_fsec(alpha):
            logit_scores = logit_partial_scores(alignment_scores, alpha)
            return  np.sum(logit_scores*(1 - logit_scores))
        
        new_alpha = minimize(alpha_target,
                             updated_parameters['alpha'],
                             jac=alpha_fprime,
                             # hess=alpha_fsec,
                             #method='Newton-CG'
                             )['x'][0]
        
        logit_scores = logit_partial_scores(alignment_scores, new_alpha)
        if verbose:
            new_logL = logit_logL(logit_scores, labels)
            print("Updated alpha:", new_alpha)
            print('Updated logL:', new_logL)

        updated_parameters['alpha'] = new_alpha

        # Make a subgradient step
        subgradient = logit_subgradient(alnlist, logit_scores,
                                        labels, new_alpha,
                                        alphabet)
        
        stepsize = stepfunction(iternb)
        if verbose:
            print('New subgradient:')
            print(subgradient)
            print('Stepsize:', stepsize)

        subgradient_square_norm = 0
        if gap_mode == 'affine':
            updated_parameters['open_gap_score'] += stepsize*subgradient['Gap opens'] + add_noise(1, iternb)
            updated_parameters['extend_gap_score'] += stepsize*subgradient['Gap extends'] + add_noise(1, iternb)
            subgradient_square_norm += subgradient['Gap opens']**2
            subgradient_square_norm += subgradient['Gap extends']**2
            if verbose:
                print('Gap open step:', stepsize*subgradient['Gap opens'])
                print('Gap extend step:', stepsize*subgradient['Gap extends'])
        elif gap_mode == 'linear':
            gapnb = subgradient['Gap opens'] + subgradient['Gap extends'] 
            updated_parameters['gap_score'] += stepsize*gapnb + add_noise(1, iternb)
            subgradient_square_norm += gapnb**2
            if verbose:
                print('Linear gap step:', stepsize*gapnb)

        if substitution_mode == 'simple':
            match_subgradient = np.sum(np.diag(subgradient['Substitutions']))
            mismatch_subgradient = np.sum(subgradient['Substitutions']) - match_subgradient
            updated_parameters['match_score'] += stepsize*match_subgradient + add_noise(1, iternb)
            updated_parameters['mismatch_score'] += stepsize*mismatch_subgradient + add_noise(1, iternb)
            subgradient_square_norm += match_subgradient**2
            subgradient_square_norm += mismatch_subgradient**2
            if verbose:
                print('Match step:', stepsize*match_subgradient)
                print('Mismatch step:', stepsize*mismatch_subgradient)
        elif substitution_mode == 'symmetric':
            subsM = subgradient['Substitutions']
            subsM = subsM + add_noise(subsM.shape, iternb)
            subsM = (subsM + subsM.T) # divide by 2?
            updated_parameters['substitution_matrix'] += stepsize*subsM 
            subgradient_square_norm += np.sum(subsM**2)
        else:
            subsM = subgradient['Substitutions']
            subsM = subsM + add_noise(subsM.shape, iternb)
            updated_parameters['substitution_matrix'] += stepsize*subsM
            subgradient_square_norm += np.sum(subsM**2)
        subgradient_l2_trajectory.append(np.sqrt(subgradient_square_norm))
        if verbose:
            print('New parameters:')
            print(updated_parameters)
            print('Subgradient norm:', subgradient_l2_trajectory[-1])
        if verbose:
            print('End of iteration', iternb)
            print()
        
    # Set final parameters
    results = {}
    if gap_mode == 'affine':
        aligner.open_gap_score = updated_parameters['open_gap_score']
        aligner.extend_gap_score = updated_parameters['extend_gap_score']
        results['open_gap_score'] = updated_parameters['open_gap_score']
        results['extend_gap_score'] = updated_parameters['extend_gap_score']
    elif gap_mode == 'linear':
        aligner.gap_score = updated_parameters['gap_score']
        results['gap_score'] = updated_parameters['gap_score']

    if substitution_mode == 'simple':
        aligner.match_score = updated_parameters['match_score']
        aligner.mismatch_score = updated_parameters['mismatch_score']
        results['match_score'] = updated_parameters['match_score']
        results['mismatch_score'] = updated_parameters['mismatch_score']
    else:
        aligner.substitution_matrix = updated_parameters['substitution_matrix']
        results['substitution_matrix'] = updated_parameters['substitution_matrix']
    
    # Realign with the new parameters
    alnlist = [next(aligner.align(seqA, seqB)) for seqA, seqB in zip(seqlistA, seqlistB)]
    alignment_scores = [aln.score for aln in alnlist]
    logit_scores = logit_partial_scores(alignment_scores,
                                            updated_parameters['alpha'])
    new_logL = logit_logL(logit_scores, labels)
##    EL = 0
##    VL = 0
##    for ls in logit_scores:
##        if 1e-30 < ls < 1-1e-30:
##            EL += ls*np.log(ls) + (1-ls)*np.log(1-ls)
##            VL += ls*(1-ls)*(np.log(ls)**2 + np.log(1-ls)**2)
##    SDL = np.sqrt(VL)
##    loglik_expectation.append(EL)
##    loglik_sd.append(SDL)
    loglik_trajectory.append(new_logL)
    results['loglik_trajectory'] = loglik_trajectory
    results['subgradient_l2_trajectory'] = subgradient_l2_trajectory
    results['final_loglik'] = new_logL
    results['aligner'] = aligner
    results['alignments'] = alnlist
    results['alignment_logit_scores'] = logit_scores
    results['alpha'] = updated_parameters['alpha']
    # results['loglik_expectation_trajectory'] = loglik_expectation
    # results['loglik_sd_trajectory'] = loglik_sd
    return results

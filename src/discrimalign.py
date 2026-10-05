"""
Main functions of DiscrimAlign for learning alignment parameters from
labeled pairs of biological sequences.
"""
from Bio.Align import PairwiseAligner, substitution_matrices
import numpy as np
from numpy import random as rd
from scipy.optimize import minimize
from copy import deepcopy
from math import ceil
import os
from .optimization import (create_powerstep, get_initial_estimate,
                           get_initial_estimate_from_counts, get_first_alignment)
from .logit_link import (_logit_logL_unchecked, fit_alpha, logit_partial_scores, logit_logL,
                         logit_subgradient)


def _align_pair_chunk(pair_chunk, aligner):
    """Align a chunk of sequence pairs in one joblib task.

    Dispatching one pair per task creates substantial scheduling overhead when
    this function is called at every optimization iteration. Chunking keeps the
    returned alignment order stable while reducing per-iteration joblib overhead.
    """
    return [get_first_alignment(seqA, seqB, aligner) for seqA, seqB in pair_chunk]


def _pair_chunks(seqlistA, seqlistB, chunk_size):
    pair_count = len(seqlistA)
    for start in range(0, pair_count, chunk_size):
        stop = min(start + chunk_size, pair_count)
        yield list(zip(seqlistA[start:stop], seqlistB[start:stop]))


def _align_pairs(seqlistA, seqlistB, aligner, num_threads):
    pair_count = len(seqlistA)
    if num_threads == 1 or pair_count == 0:
        return [get_first_alignment(seqA, seqB, aligner) for seqA, seqB in zip(seqlistA, seqlistB)]

    from joblib import Parallel, delayed

    n_jobs = min(int(num_threads), pair_count)
    # Aim for a few chunks per worker so the work is balanced, without creating
    # one joblib task per sequence pair at every fitting iteration.
    chunk_size = max(1, ceil(pair_count / (n_jobs * 4)))
    parallel = Parallel(n_jobs=n_jobs, prefer="threads", return_as="list")
    chunked_alignments = parallel(
        delayed(_align_pair_chunk)(pair_chunk, aligner)
        for pair_chunk in _pair_chunks(seqlistA, seqlistB, chunk_size)
    )
    return [alignment for chunk in chunked_alignments for alignment in chunk]


def _matrix_alphabet(matrix):
    alphabet = getattr(matrix, 'alphabet', None)
    if alphabet is None:
        return None
    return ''.join(alphabet)


def _warm_start_alphabet(baseline_aligner=None, initial_parameters=None):
    if initial_parameters is not None:
        matrix = initial_parameters.get('substitution_matrix')
        alphabet = _matrix_alphabet(matrix)
        if alphabet:
            return alphabet
    if baseline_aligner is not None:
        try:
            matrix = getattr(baseline_aligner, 'substitution_matrix')
        except Exception:
            matrix = None
        alphabet = _matrix_alphabet(matrix)
        if alphabet:
            return alphabet
    return None


def _configure_aligner(aligner, params, gap_mode, substitution_mode):
    if gap_mode == 'affine':
        aligner.open_gap_score = params['open_gap_score']
        aligner.extend_gap_score = params['extend_gap_score']
    elif gap_mode == 'linear':
        aligner.gap_score = params['gap_score']

    if substitution_mode == 'simple':
        aligner.match_score = params['match_score']
        aligner.mismatch_score = params['mismatch_score']
    else:
        aligner.substitution_matrix = params['substitution_matrix']


# Gap scores are costs and are kept at or below this value. A positive gap
# score makes local alignment ill-posed: Biopython's align() and score()
# disagree there, and the two backends end local alignments differently.
# TODO: reset to 0.0 once the backends handle the kink at a gap score of 0.
# There, zero-cost gap columns tie with none, and Biopython and nwgrad break
# the tie differently, so they return different (both valid) subgradients.
# Out of scope for the nwgrad integration. -1e-4 keeps the fit off the kink
# and well above PairwiseAligner.epsilon (1e-6), Biopython's tie tolerance.
_MAX_GAP_SCORE = -1e-4


def _clip_gap_scores(params, gap_mode):
    """Project gap scores onto <= _MAX_GAP_SCORE."""
    keys = ('open_gap_score', 'extend_gap_score') if gap_mode == 'affine' else ('gap_score',)
    for key in keys:
        params[key] = min(params[key], _MAX_GAP_SCORE)


class _BiopythonEngine:
    """
    Scores and subgradients from Biopython alignments: set_params(), then
    scores(), then raw_subgradient() for the alignments just computed.
    """

    def __init__(self, seqlistA, seqlistB, aligner, gap_mode, substitution_mode,
                 alphabet, num_threads):
        self.seqlistA = seqlistA
        self.seqlistB = seqlistB
        self.aligner = aligner
        self.gap_mode = gap_mode
        self.substitution_mode = substitution_mode
        self.alphabet = alphabet
        self.num_threads = num_threads
        self.alignments = None

    def set_params(self, params):
        _configure_aligner(self.aligner, params, self.gap_mode, self.substitution_mode)

    def scores(self):
        self.alignments = _align_pairs(self.seqlistA, self.seqlistB, self.aligner, self.num_threads)
        return [aln.score for aln in self.alignments]

    def raw_subgradient(self, logit_scores, labels, alpha):
        return logit_subgradient(self.alignments, logit_scores, labels, alpha, self.alphabet)


def _python_logistic_step(engine, updated_parameters, labels, labels_float, one_minus_labels,
                          alpha_solver, loglik_trajectory, verbose):
    """
    Align, then one iteration's logistic work in numpy: the log-likelihood at
    the current alpha (appended to loglik_trajectory), the alpha fit, and the
    raw subgradient at the new alpha. Returns (new_alpha, subgradient).
    """
    alignment_scores = np.asarray(engine.scores(), dtype=float)
    logit_scores = logit_partial_scores(alignment_scores,
                                        updated_parameters['alpha'])
    new_logL = _logit_logL_unchecked(logit_scores, labels_float, one_minus_labels)
    loglik_trajectory.append(new_logL)
    if verbose:
        print("Current alpha:", updated_parameters['alpha'])
        print('Current logL:', new_logL)
##    EL = 0
##    VL = 0
##    for ls in logit_scores:
##        if 1e-30 < ls < 1-1e-30:
##            EL += ls*np.log(ls) + (1-ls)*np.log(1-ls)
##            VL += ls*(1-ls)*(np.log(ls)**2 + np.log(1-ls)**2)
##    SDL = np.sqrt(VL)
##    loglik_expectation.append(EL)
##    loglik_sd.append(SDL)

    # Optimize the logistic intercept (alpha)
    if alpha_solver == 'safeguarded_newton':
        new_alpha = fit_alpha(alignment_scores, labels_float, updated_parameters['alpha'],
                              logit_scores0=logit_scores)
    else:
        def alpha_target(alpha):
            logit_scores = logit_partial_scores(alignment_scores, alpha)
            return -logit_logL(logit_scores, labels)

        def alpha_fprime(alpha):
            logit_scores = logit_partial_scores(alignment_scores, alpha)
            return -np.sum(labels - logit_scores)

        new_alpha = minimize(alpha_target,
                             updated_parameters['alpha'],
                             jac=alpha_fprime)['x'][0]

    logit_scores = logit_partial_scores(alignment_scores, new_alpha)
    if verbose:
        new_logL = _logit_logL_unchecked(logit_scores, labels_float, one_minus_labels)
        print("Updated alpha:", new_alpha)
        print('Updated logL:', new_logL)

    # The subgradient at the new alpha
    subgradient = engine.raw_subgradient(logit_scores, labels_float, new_alpha)
    return new_alpha, subgradient


def discrimalign(seqlistA, seqlistB,
                 labels,
                 baseline_aligner=None,
                 aligner_mode='local',
                 gap_mode='affine',
                 substitution_mode='symmetric',
                 alphabet=None,
                 stochastic_factor=None,
                 stepfunction=None,
                 max_iter=1000, tol=1e-3,
                 num_threads=0,
                 subgradient_scale=1.0,
                 initial_parameters=None,
                 return_alignments=True,
                 verbose=False,
                 backend='nwgrad',
                 alpha_solver='safeguarded_newton',
                 nwgrad_fill='striped'):
    """
    backend selects where alignment scores and subgradients come from:
    'nwgrad' (the default) or 'biopython' (PairwiseAligner). With 'nwgrad', the initial
    estimate is fitted on nwgrad's alignments of the baseline too, and a
    baseline_aligner must have uniform gap scores and no wildcard. The
    returned aligner is a Biopython PairwiseAligner, and the returned alignments
    are Biopython Alignment objects; with 'nwgrad' they are nwgrad's own paths.

    num_threads=0 (the default) picks the thread count automatically: all
    logical cores with 'nwgrad', 1 with 'biopython', whose threads contend
    for the GIL and only slow it down.

    alpha_solver selects how the intercept alpha is fitted at each iteration:
    'safeguarded_newton' (the default) finds the exact optimum with fit_alpha()
    in logit_link; 'bfgs' is the previous scipy.optimize.minimize (BFGS) fit,
    which can stop far from the optimum when alpha moves a long way between
    iterations, e.g. with subgradient_scale=1 on large data.

    nwgrad_fill selects nwgrad's vectorized DP fill: 'striped' (the default),
    'rowwise' or 'interpair'. All three give the same scores, gradients and fit,
    bit for bit. On short pairs such as miRNA-target sites, 'rowwise' is about 2x
    faster than 'striped', and 'interpair' (several pairs per vector) faster
    still.
    """
    # TODO: Implement tol and additional stepfunctions.
    assert backend in {'biopython', 'nwgrad'}
    assert alpha_solver in {'safeguarded_newton', 'bfgs'}
    assert aligner_mode in {'local', 'global'}
    assert gap_mode in {'affine', 'linear'}
    assert substitution_mode in {'general', 'symmetric', 'simple'}
    if nwgrad_fill not in {'striped', 'rowwise', 'interpair'}:
        raise ValueError("nwgrad_fill must be 'striped', 'rowwise' or 'interpair', "
                         f"got {nwgrad_fill!r}")
    if nwgrad_fill != 'striped' and backend != 'nwgrad':
        raise ValueError("nwgrad_fill applies to the nwgrad backend only")
    if stepfunction is None:
        stepfunction = create_powerstep(1e-4)
    for seqlist, name in ((seqlistA, 'seqlistA'), (seqlistB, 'seqlistB')):
        empty = [i for i, seq in enumerate(seqlist) if len(seq) == 0]
        if empty:
            raise ValueError(f'{name} contains empty sequences (at indices {empty[:10]}); '
                             'every sequence needs at least one residue')
    if num_threads == 0 and backend == 'biopython':
        num_threads = 1
    elif num_threads == 0:
        # All logical cores, passed to nwgrad explicitly rather than its own
        # n_threads=0, which picks physical cores: short pairs such as
        # miRNA-target sites gain about 25% from SMT on hosts that have it.
        # TODO: long-pair (e.g. protein) workflows can be faster on physical
        # cores once their DP tables outgrow the cache; see TODO.md in nwgrad.
        num_threads = os.cpu_count() or 1
    if alphabet is None:
        alphabet = _warm_start_alphabet(baseline_aligner, initial_parameters)
    if alphabet is None:
        charsetA = set(char for seq in seqlistA for char in seq)
        charsetB = set(char for seq in seqlistB for char in seq)
        alphabet = charsetA | charsetB
        alphabet = ''.join(sorted(alphabet))

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

    # The baseline aligner's mode, when given, takes precedence over aligner_mode.
    if backend == 'nwgrad':
        from .nwgrad_backend import NwgradEngine, baseline_parameters
        engine = NwgradEngine(seqlistA, seqlistB, aligner.mode, gap_mode,
                              substitution_mode, alphabet, num_threads, fill=nwgrad_fill)
    else:
        engine = _BiopythonEngine(seqlistA, seqlistB, aligner, gap_mode,
                                  substitution_mode, alphabet, num_threads)

    # Initial logistic estimation from the baseline alignments, unless the
    # caller provides fitted parameters for a warm start.
    if initial_parameters is not None:
        updated_parameters = deepcopy(initial_parameters)
    elif backend == 'nwgrad':
        engine.set_params(baseline_parameters(aligner, gap_mode, alphabet),
                          substitution_mode='general')
        engine.scores()
        updated_parameters = get_initial_estimate_from_counts(engine.count_arrays(), labels,
                                                              substitution_mode=substitution_mode,
                                                              gap_mode=gap_mode,
                                                              alphabet=alphabet)
    else:
        alnlist = _align_pairs(seqlistA, seqlistB, aligner, num_threads)
        updated_parameters = get_initial_estimate(alnlist, labels,
                                                  substitution_mode=substitution_mode,
                                                  gap_mode=gap_mode,
                                                  alphabet=alphabet)
    if verbose:
        print('Initial parameters:')
        print(updated_parameters)
    _clip_gap_scores(updated_parameters, gap_mode)
    _configure_aligner(aligner, updated_parameters, gap_mode, substitution_mode)

    # The loop uses the labels as floats, checked once.
    labels_float = np.asarray(labels, dtype=float)
    if not np.isin(labels_float, [0, 1]).all():
        raise ValueError('Labels can only be 0 or 1')
    one_minus_labels = 1 - labels_float
    # The nwgrad engine does the logistic part of each iteration in C++ (the
    # same alpha fit, same probabilities); the Biopython engine has no such step.
    use_logistic_step = (alpha_solver == 'safeguarded_newton'
                         and hasattr(engine, 'logistic_step'))

    # Subgradient refinement
    loglik_trajectory = []
    subgradient_l2_trajectory = []
    loglik_expectation = []
    loglik_sd = []
    for iternb in range(max_iter):
        if verbose:
            print('Start of iteration', iternb)
        # Realign with the new parameters
        engine.set_params(updated_parameters)
        if use_logistic_step:
            # Alignment, log-likelihood, alpha fit and subgradient in nwgrad.
            new_logL, new_alpha, subgradient = engine.logistic_step(
                labels_float, updated_parameters['alpha'])
            loglik_trajectory.append(new_logL)
            if verbose:
                print("Current alpha:", updated_parameters['alpha'])
                print('Current logL:', new_logL)
                print("Updated alpha:", new_alpha)
            updated_parameters['alpha'] = new_alpha
        else:
            new_alpha, subgradient = _python_logistic_step(
                engine, updated_parameters, labels, labels_float, one_minus_labels,
                alpha_solver, loglik_trajectory, verbose)
            updated_parameters['alpha'] = new_alpha
        if subgradient_scale != 1.0:
            subgradient['Gap opens'] *= subgradient_scale
            subgradient['Gap extends'] *= subgradient_scale
            subgradient['Substitutions'] *= subgradient_scale

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
            # Summing mismatch counts to keep the matrix symmetric
            # with off-diagonal elements corresponding to mismatches,
            # subtracting the diagonal to avoid counting matches twice
            subsM = subsM + subsM.T - np.diag(np.diag(subsM))
            # subsM = subsM + subsM.T
            updated_parameters['substitution_matrix'] += stepsize*subsM
            subgradient_square_norm += np.sum(subsM**2)
        else:
            subsM = subgradient['Substitutions']
            subsM = subsM + add_noise(subsM.shape, iternb)
            updated_parameters['substitution_matrix'] += stepsize*subsM
            subgradient_square_norm += np.sum(subsM**2)
        _clip_gap_scores(updated_parameters, gap_mode)
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
    engine.set_params(updated_parameters)
    alignment_scores = engine.scores()
    if return_alignments:
        if backend == 'nwgrad':
            alnlist = (_align_pairs(seqlistA, seqlistB, aligner, num_threads)
                       if return_alignments else None)
        else:
            alnlist = engine.alignments
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
    if return_alignments:
        results['alignments'] = alnlist
    results['alignment_logit_scores'] = logit_scores
    results['alpha'] = updated_parameters['alpha']
    # results['loglik_expectation_trajectory'] = loglik_expectation
    # results['loglik_sd_trajectory'] = loglik_sd
    return results

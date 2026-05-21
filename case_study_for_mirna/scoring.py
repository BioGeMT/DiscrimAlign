from __future__ import annotations

from math import ceil

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.optimization import get_first_alignment


def _score_pair(seq_a, seq_b, aligner):
    return get_first_alignment(seq_a, seq_b, aligner).score


def _score_pair_chunk(pair_chunk, aligner):
    return [_score_pair(seq_a, seq_b, aligner) for seq_a, seq_b in pair_chunk]


def _pair_chunks(seqlist_a, seqlist_b, chunk_size):
    pair_count = len(seqlist_a)
    for start in range(0, pair_count, chunk_size):
        stop = min(start + chunk_size, pair_count)
        yield list(zip(seqlist_a[start:stop], seqlist_b[start:stop]))


def score_pairs_with_model(seqlist_a, seqlist_b, aligner, alpha, num_threads=1):
    pair_count = len(seqlist_a)
    if num_threads == 1 or pair_count == 0:
        scores = [_score_pair(seq_a, seq_b, aligner) for seq_a, seq_b in zip(seqlist_a, seqlist_b)]
    else:
        from joblib import Parallel, delayed

        n_jobs = min(int(num_threads), pair_count)
        chunk_size = max(1, ceil(pair_count / (n_jobs * 4)))
        chunked_scores = Parallel(n_jobs=n_jobs, prefer="threads", return_as="list")(
            delayed(_score_pair_chunk)(pair_chunk, aligner)
            for pair_chunk in _pair_chunks(seqlist_a, seqlist_b, chunk_size)
        )
        scores = [score for chunk in chunked_scores for score in chunk]
    # sklearn ranking metrics accept raw decision scores. Avoid sigmoid here:
    # large fitted alignment scores can saturate to identical probabilities.
    return np.asarray(scores, dtype=float) + float(alpha)


def empty_evaluation():
    return {
        "average_precision": np.nan,
        "roc_auc": np.nan,
        "precision": np.array([]),
        "recall": np.array([]),
        "pr_thresholds": np.array([]),
        "fpr": np.array([]),
        "tpr": np.array([]),
        "roc_thresholds": np.array([]),
    }


def evaluate_probabilities(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.size == 0 or not np.isfinite(probabilities).all():
        return empty_evaluation()
    precision, recall, pr_thresholds = precision_recall_curve(labels, probabilities)
    average_precision = average_precision_score(labels, probabilities)
    try:
        roc_auc = roc_auc_score(labels, probabilities)
        fpr, tpr, roc_thresholds = roc_curve(labels, probabilities)
    except ValueError:
        roc_auc = np.nan
        fpr = np.array([])
        tpr = np.array([])
        roc_thresholds = np.array([])
    return {
        "average_precision": float(average_precision),
        "roc_auc": float(roc_auc) if np.isfinite(roc_auc) else np.nan,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
    }

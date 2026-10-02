"""
Short run on the miRBench datasets. Similar to a standard analysis.
"""
import pytest
from Bio.Seq import Seq 
from Bio.SeqRecord import SeqRecord

from miRBench.dataset import list_datasets, get_dataset_df
from src.discrimalign import discrimalign
from tests.helpers import (GAP_MODES, MODES, SUBSTITUTION_MODES, align_all, make_pairs,
                           mutate, random_seq)

ALL_MODES = [(m, g, s) for m in MODES for g in GAP_MODES for s in SUBSTITUTION_MODES]
BACKENDS = ["biopython", "nwgrad"]
DATASET_IDs = [0, 2]

def _get_data_and_params(dset_id):
    assert dset_id in {0, 2}
    train = get_dataset_df(list_datasets()[dset_id], split="train")
    mirlist = train['noncodingRNA']
    mirlist = [str(Seq(seq)) for seq in mirlist]
    genelist = train['gene']
    genelist = [str(Seq(seq).reverse_complement()) for seq in genelist]
    if dset_id == 0:
        stepfunction = create_constant_step(0.00005)
    else:
        stepfunction = create_constant_step(0.0000005)
    return (mirlist, genelist, stepfunction)
    
NITER = 20

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dset_id", DATASET_IDs)
@pytest.mark.parametrize("mode, gap_mode, substitution_mode", ALL_MODES)
def test_mirbench_run(dset_id, mode, gap_mode, substitution_mode, backend):
    """
    Check if discrimalign runs properly using data preprocessed with BioPython.
    """
    mirlist, genelist, stepfunction = _get_data_and_params(dset_id)
    res = discrimalign(mirlist, genelist, labels, 
                    stepfunction=stepfunction,
                    aligner_mode=mode,
                    substitution_mode=substitution_mode,
                    gap_mode=gap_mode, 
                    stochastic_factor=0.01,
                    verbose=True, max_iter=NITER,
                    backend=backend)
    traj = res["loglik_trajectory"]
    assert res["final_loglik"] > traj[0]
    assert np.all(np.isfinite(traj))
    if substitution_mode == "simple":
        assert res["match_score"] > res["mismatch_score"]
    else:
        M = np.asarray(res["substitution_matrix"])
        assert np.min(np.diag(M)) > np.mean(M[~np.eye(4, dtype=bool)])
    # In global mode, adding c to every substitution score and c/2 to every
    # gap column shifts each score by c * (len A + len B) / 2. With equal
    # lengths alpha absorbs that, so absolute signs are not identified.
    # Local alignment breaks the symmetry.
    if mode == "local":
        if gap_mode == "affine":
            assert res["open_gap_score"] < 0
            assert res["extend_gap_score"] < 0
        else:
            assert res["gap_score"] < 0
        if substitution_mode == "simple":
            assert res["match_score"] > 0 > res["mismatch_score"]

    

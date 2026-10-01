import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.discrimalign import discrimalign


def test_discrimalign_uses_default_stepfunction():
    result = discrimalign(
        seqlistA=["AUGCUA", "CUGA"],
        seqlistB=["AUGGUA", "CUGU"],
        labels=[1, 0],
        aligner_mode="local",
        gap_mode="affine",
        substitution_mode="symmetric",
        max_iter=1,
        num_threads=1,
    )

    assert "final_loglik" in result
    assert "alpha" in result

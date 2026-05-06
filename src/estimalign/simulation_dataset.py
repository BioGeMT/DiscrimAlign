from __future__ import annotations

from Bio.Seq import Seq
from miRBench.dataset import get_dataset_df, list_datasets


def load_mirbench_sequences(
    *,
    dataset_index: int,
    split: str,
    max_records: int | None,
) -> tuple[list[Seq], list[Seq]]:
    dataset_name = list_datasets()[dataset_index]
    dataset = get_dataset_df(dataset_name, split=split)

    if max_records is not None:
        dataset = dataset.head(max_records)

    first_sequences = [Seq(seq) for seq in dataset["noncodingRNA"]]
    second_sequences = [Seq(seq).reverse_complement() for seq in dataset["gene"]]
    return first_sequences, second_sequences

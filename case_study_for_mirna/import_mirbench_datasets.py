from __future__ import annotations

import pandas as pd
from miRBench.dataset import get_dataset_df

DATASET_ALIASES = {
    "hejret_train": ("AGO2_CLASH_Hejret2023", "train"),
    "hejret_test": ("AGO2_CLASH_Hejret2023", "test"),
    "klimentova_test": ("AGO2_eCLIP_Klimentova2022", "test"),
    "manakov_train": ("AGO2_eCLIP_Manakov2022", "train"),
    "manakov_test": ("AGO2_eCLIP_Manakov2022", "test"),
    "manakov_leftout": ("AGO2_eCLIP_Manakov2022", "leftout"),
}


def resolve_dataset_alias(alias: str) -> tuple[str, str]:
    if alias not in DATASET_ALIASES:
        valid = ", ".join(sorted(DATASET_ALIASES))
        raise ValueError(f"Unknown dataset alias {alias!r}. Valid aliases: {valid}")
    return DATASET_ALIASES[alias]


def get_dataset_dataframe(alias: str) -> pd.DataFrame:
    dataset_name, split = resolve_dataset_alias(alias)
    return get_dataset_df(dataset_name, split=split)

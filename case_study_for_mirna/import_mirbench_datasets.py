from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from miRBench.dataset import get_dataset_df, get_dataset_path, list_datasets

DATASET_ALIASES = {
    "hejret_train": ("AGO2_CLASH_Hejret2023", "train"),
    "hejret_test": ("AGO2_CLASH_Hejret2023", "test"),
    "klimentova_test": ("AGO2_eCLIP_Klimentova2022", "test"),
    "manakov_train": ("AGO2_eCLIP_Manakov2022", "train"),
    "manakov_test": ("AGO2_eCLIP_Manakov2022", "test"),
    "manakov_leftout": ("AGO2_eCLIP_Manakov2022", "leftout"),
}

DATASET_GROUPS = {
    "hejret": ["hejret_train", "hejret_test"],
    "manakov": ["manakov_train", "manakov_test"],
    "manakov_with_leftout": ["manakov_train", "manakov_test", "manakov_leftout"],
    "table": ["hejret_train", "hejret_test", "manakov_train", "manakov_test", "manakov_leftout"],
    "all_balanced": [
        "hejret_train",
        "hejret_test",
        "manakov_train",
        "manakov_test",
        "manakov_leftout",
        "klimentova_test",
    ],
    "all": list(DATASET_ALIASES),
}

OUTPUT_FILENAMES = {
    "hejret_train": "AGO2_CLASH_Hejret2023_train.tsv.gz",
    "hejret_test": "AGO2_CLASH_Hejret2023_test.tsv.gz",
    "klimentova_test": "AGO2_eCLIP_Klimentova2022_test.tsv.gz",
    "manakov_train": "AGO2_eCLIP_Manakov2022_train.tsv.gz",
    "manakov_test": "AGO2_eCLIP_Manakov2022_test.tsv.gz",
    "manakov_leftout": "AGO2_eCLIP_Manakov2022_leftout.tsv.gz",
}


def _parse_csv(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def available_datasets() -> dict:
    return list_datasets(full=True)


def resolve_dataset_alias(alias: str) -> tuple[str, str]:
    if alias not in DATASET_ALIASES:
        valid = ", ".join(sorted(DATASET_ALIASES))
        raise ValueError(f"Unknown dataset alias {alias!r}. Valid aliases: {valid}")
    return DATASET_ALIASES[alias]


def output_filename(alias: str) -> str:
    if alias not in OUTPUT_FILENAMES:
        valid = ", ".join(sorted(OUTPUT_FILENAMES))
        raise ValueError(f"Unknown dataset alias {alias!r}. Valid aliases: {valid}")
    return OUTPUT_FILENAMES[alias]


def expand_dataset_groups(group_names: list[str]) -> list[str]:
    aliases: list[str] = []
    for group_name in group_names:
        if group_name not in DATASET_GROUPS:
            valid = ", ".join(sorted(DATASET_GROUPS))
            raise ValueError(f"Unknown dataset group {group_name!r}. Valid groups: {valid}")
        aliases.extend(DATASET_GROUPS[group_name])
    return aliases


def normalize_aliases(aliases: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        resolve_dataset_alias(alias)
        if alias not in seen:
            normalized.append(alias)
            seen.add(alias)
    return normalized


def get_dataset_dataframe(alias: str) -> pd.DataFrame:
    dataset_name, split = resolve_dataset_alias(alias)
    return get_dataset_df(dataset_name, split=split)


def get_dataset_cache_path(alias: str) -> Path:
    dataset_name, split = resolve_dataset_alias(alias)
    return Path(get_dataset_path(dataset_name, split=split))


def is_valid_gzip_tsv(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        pd.read_csv(path, sep="\t", compression="gzip", nrows=1)
    except Exception:
        return False
    return True


def write_dataset_gzip(alias: str, out_path: Path) -> None:
    df = get_dataset_dataframe(alias)
    df.to_csv(out_path, sep="\t", index=False, compression="gzip")


def download_named_dataset(
    alias: str,
    output_dir: str | Path = "data/raw",
    overwrite: bool = False,
) -> Path:
    """Fetch a miRBench dataset and store it locally as a valid gzipped TSV file."""
    alias = normalize_aliases([alias])[0]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_filename(alias)

    if out_path.exists() and not overwrite and is_valid_gzip_tsv(out_path):
        return out_path

    write_dataset_gzip(alias, out_path)
    return out_path


def download_datasets(
    aliases: list[str],
    output_dir: str | Path = "data/raw",
    overwrite: bool = False,
) -> list[Path]:
    return [
        download_named_dataset(alias, output_dir=output_dir, overwrite=overwrite)
        for alias in normalize_aliases(aliases)
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download or reuse miRBench datasets for the EstimAlign miRNA case study."
    )
    parser.add_argument("--datasets", default="", help="Comma-separated aliases such as hejret_train,manakov_leftout")
    parser.add_argument("--groups", default="hejret", help=f"Comma-separated groups. Supported: {', '.join(sorted(DATASET_GROUPS))}")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list", action="store_true", help="List available aliases and miRBench datasets, then exit.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.list:
        print("miRBench datasets:")
        print(available_datasets())
        print("\nEstimAlign miRNA case-study aliases:")
        for alias, target in sorted(DATASET_ALIASES.items()):
            print(f"  {alias}: {target[0]} / {target[1]}")
        raise SystemExit(0)

    requested = []
    requested.extend(expand_dataset_groups(_parse_csv(args.groups)))
    requested.extend(_parse_csv(args.datasets))
    for path in download_datasets(requested, output_dir=args.output_dir, overwrite=args.overwrite):
        print(f"Available locally: {path}")

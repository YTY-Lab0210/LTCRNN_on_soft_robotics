from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


KEEP_COLUMNS = ["Time_ms", "Thumb", "Middle", "Pinky"]
DEFAULT_SEED = 20260729


def label_from_path(path: Path) -> str:
    return path.stem.rsplit("_", 1)[0]


def read_selected_columns(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [column for column in KEEP_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        return [{column: row[column] for column in KEEP_COLUMNS} for row in reader]


def write_selected_columns(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEEP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def create_3ch_dataset(source_dir: Path, output_dir: Path, manifest_path: Path, seed: int) -> None:
    source_files = sorted(source_dir.glob("*.csv"))
    if not source_files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")

    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in source_files:
        grouped[label_from_path(path)].append(path)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    manifest_rows = []

    for label in sorted(grouped):
        files = grouped[label][:]
        rng.shuffle(files)

        for new_index, source_path in enumerate(files, start=1):
            output_name = f"{label}_{new_index:03d}.csv"
            output_path = output_dir / output_name
            rows = read_selected_columns(source_path)
            write_selected_columns(output_path, rows)
            manifest_rows.append(
                {
                    "label": label,
                    "new_file": output_name,
                    "source_file": source_path.name,
                    "channels": "Thumb,Middle,Pinky",
                    "seed": str(seed),
                }
            )

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "new_file", "source_file", "channels", "seed"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a randomized 3-channel CSV dataset.")
    parser.add_argument("--source", type=Path, default=Path("data/dataset_new_new_new"))
    parser.add_argument("--output", type=Path, default=Path("data/dataset_new_new_new_3ch"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/dataset_new_new_new_3ch_manifest.csv"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    create_3ch_dataset(args.source, args.output, args.manifest, args.seed)
    print(f"Created 3-channel dataset at {args.output}")
    print(f"Manifest written to {args.manifest}")


if __name__ == "__main__":
    main()

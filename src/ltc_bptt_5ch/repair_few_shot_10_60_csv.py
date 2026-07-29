from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = (
    ROOT
    / "\u8ad6\u6587"
    / "paper_figures"
    / "few_shot_10_60"
    / "paper_ltc4_few_shot_10_60_raw.csv"
)

NEW_COLUMNS = [
    "Channel",
    "ChannelLabel",
    "Model",
    "Samples_Per_Class",
    "Run",
    "MinSelectedSamplesPerClass",
    "TotalSelectedTrainingSamples",
    "TrainableParams",
    "Accuracy",
    "MacroF1",
    "TestLoss",
    "BestValAccuracy",
    "BestValLoss",
    "EpochsTrained",
    "InputColumns",
]


def repair() -> None:
    backup_path = RAW_PATH.with_suffix(".csv.bak")
    backup_path.write_bytes(RAW_PATH.read_bytes())

    rows = []
    with RAW_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            if not row:
                continue
            if len(row) == 13:
                sample_count = int(row[3])
                total_count = sample_count * 10
                row = row[:5] + [str(sample_count), str(total_count)] + row[5:]
            elif len(row) != 15:
                raise ValueError(f"Unexpected field count {len(row)}: {row}")
            rows.append(row)

    with RAW_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(NEW_COLUMNS)
        writer.writerows(rows)

    print(f"Repaired rows: {len(rows)}")
    print(f"Backup: {backup_path}")


if __name__ == "__main__":
    repair()

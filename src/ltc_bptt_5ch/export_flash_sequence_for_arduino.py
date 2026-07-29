import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_COLUMNS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def pick_columns(df, requested):
    if requested:
        missing = [name for name in requested if name not in df.columns]
        if missing:
            raise ValueError(f"Missing requested columns: {missing}")
        return requested

    if all(name in df.columns for name in DEFAULT_COLUMNS):
        return DEFAULT_COLUMNS

    numeric_columns = []
    for column in df.columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().all():
            numeric_columns.append(column)

    if len(numeric_columns) < 5:
        raise ValueError(
            "Could not find 5 numeric sensor columns. "
            "Pass --columns with the exact 5 column names."
        )
    return numeric_columns[:5]


def format_header(values):
    rows = []
    for row in values:
        rows.append("  {" + ", ".join(str(int(v)) for v in row) + "}")

    return "\n".join(
        [
            "#pragma once",
            "#include <Arduino.h>",
            "#include <avr/pgmspace.h>",
            "",
            "// Generated from a 400-step raw ADC sequence.",
            "const uint16_t FLASH_RAW_SEQUENCE[SEQ_LEN][INPUT_DIM] PROGMEM = {",
            ",\n".join(rows),
            "};",
            "",
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert one 400-step raw ADC CSV into flash_sequence.h for Arduino "
            "flash replay benchmarking."
        )
    )
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path, normally the sketch folder's flash_sequence.h.",
    )
    parser.add_argument("--columns", nargs=5, default=None)
    parser.add_argument("--samples", type=int, default=400)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    columns = pick_columns(df, args.columns)
    if len(df) < args.samples:
        raise ValueError(f"Need at least {args.samples} rows, found {len(df)}.")

    raw = df.loc[: args.samples - 1, columns].apply(pd.to_numeric, errors="raise")
    values = np.rint(raw.to_numpy(dtype=float))
    values = np.clip(values, 0, 1023).astype(np.uint16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_header(values), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"Columns: {', '.join(columns)}")
    print(f"Shape: {values.shape[0]} samples x {values.shape[1]} channels")


if __name__ == "__main__":
    main()

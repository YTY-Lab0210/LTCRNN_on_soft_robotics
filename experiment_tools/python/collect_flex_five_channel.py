"""Record five-channel flex-sensor ADC values into one CSV file."""

from __future__ import annotations

import argparse
import time

from serial_utils import (
    add_serial_arguments,
    open_serial,
    parse_five_adc,
    read_csv_rows,
    timestamped_path,
    write_stream,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_serial_arguments(parser)
    parser.add_argument("--label", default="sample", help="Object/action label used in the filename.")
    parser.add_argument("--samples", type=int, default=400, help="Number of rows to record.")
    parser.add_argument("--countdown", type=int, default=3)
    args = parser.parse_args()

    output_path = timestamped_path(args.output_dir, f"flex_{args.label}")
    connection = open_serial(args.port, args.baud)
    try:
        for remaining in range(args.countdown, 0, -1):
            print(f"Recording starts in {remaining}...")
            time.sleep(1)
        connection.reset_input_buffer()
        rows = (
            (index * 10, *values)
            for index, values in enumerate(read_csv_rows(connection, parse_five_adc))
        )
        saved = write_stream(
            connection,
            output_path,
            ["time_ms", "thumb", "index", "middle", "ring", "pinky"],
            rows,
            limit=args.samples,
        )
        print(f"Saved {saved} samples to {output_path}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()


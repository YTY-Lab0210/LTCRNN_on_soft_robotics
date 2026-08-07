"""Record ADC and calculated flex-sensor resistance from Arduino."""

from __future__ import annotations

import argparse
import time

from serial_utils import (
    add_serial_arguments,
    open_serial,
    parse_flex_resistance,
    read_csv_rows,
    timestamped_path,
    write_stream,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_serial_arguments(parser)
    args = parser.parse_args()

    output_path = timestamped_path(args.output_dir, "flex_resistance")
    connection = open_serial(args.port, args.baud)
    start = time.perf_counter()
    try:
        parsed_rows = read_csv_rows(connection, parse_flex_resistance)
        rows = ((f"{time.perf_counter() - start:.6f}", *row) for row in parsed_rows)
        saved = write_stream(
            connection,
            output_path,
            ["computer_time_s", "sample_index", "arduino_time_ms", "adc", "resistance_kohm"],
            rows,
        )
        print(f"Saved {saved} samples to {output_path}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()


"""Record three FSR ADC channels and a host-computer timestamp."""

from __future__ import annotations

import argparse
import time

from serial_utils import (
    add_serial_arguments,
    open_serial,
    parse_three_adc,
    read_csv_rows,
    timestamped_path,
    write_stream,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_serial_arguments(parser)
    args = parser.parse_args()

    output_path = timestamped_path(args.output_dir, "fsr")
    connection = open_serial(args.port, args.baud)
    start = time.perf_counter()
    try:
        parsed_rows = read_csv_rows(connection, parse_three_adc)
        rows = ((f"{(time.perf_counter() - start) * 1000:.3f}", *row) for row in parsed_rows)
        saved = write_stream(
            connection,
            output_path,
            ["computer_time_ms", "fsr_3", "fsr_4", "fsr_5"],
            rows,
        )
        print(f"Saved {saved} samples to {output_path}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()


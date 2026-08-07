"""Print the single-channel flex ADC stream without creating a file."""

from __future__ import annotations

import argparse

from serial_utils import add_serial_arguments, open_serial, parse_flex_adc, read_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_serial_arguments(parser)
    args = parser.parse_args()
    connection = open_serial(args.port, args.baud)
    try:
        for sample_index, arduino_time_us, adc in read_csv_rows(connection, parse_flex_adc):
            print(
                f"sample={sample_index:6d} "
                f"time={arduino_time_us / 1_000_000:9.3f}s "
                f"adc={adc:4d}"
            )
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        connection.close()


if __name__ == "__main__":
    main()


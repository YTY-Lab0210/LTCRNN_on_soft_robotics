"""Parse drop/catch timing messages from Arduino into a CSV file."""

from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import datetime

from serial_utils import add_serial_arguments, open_serial, timestamped_path


METRICS = {
    "Speed (Delta AB)": "delta_ab_ms",
    "Catch Delay": "catch_delay_ms",
    "Total Time": "total_time_ms",
}


def parse_metric(line: str) -> tuple[str, int] | None:
    for marker, key in METRICS.items():
        if marker not in line:
            continue
        match = re.search(r":\s*(\d+)", line)
        if match:
            return key, int(match.group(1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_serial_arguments(parser)
    args = parser.parse_args()

    output_path = timestamped_path(args.output_dir, "drop_timing")
    connection = open_serial(args.port, args.baud)
    pending: dict[str, int] = {}
    drop_index = 1

    try:
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["timestamp", "drop_index", "delta_ab_ms", "catch_delay_ms", "total_time_ms"]
            )
            while True:
                raw = connection.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                print(f"Arduino: {line}")

                metric = parse_metric(line)
                if metric is None:
                    continue
                key, value = metric
                pending[key] = value

                if all(key in pending for key in METRICS.values()):
                    writer.writerow(
                        [
                            datetime.now().isoformat(timespec="seconds"),
                            drop_index,
                            pending["delta_ab_ms"],
                            pending["catch_delay_ms"],
                            pending["total_time_ms"],
                        ]
                    )
                    handle.flush()
                    print(f"Saved drop {drop_index}")
                    drop_index += 1
                    pending.clear()
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        connection.close()
        print(f"Output: {output_path}")


if __name__ == "__main__":
    main()


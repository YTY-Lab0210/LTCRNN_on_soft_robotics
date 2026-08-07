"""Shared serial-port and CSV parsing helpers."""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

import serial
from serial.tools import list_ports


Row = TypeVar("Row", bound=Sequence[object])


def add_serial_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--port",
        help="Serial port. If omitted, automatically select a connected Arduino-like device.",
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for generated files (default: ./output).",
    )


def resolve_serial_port(requested_port: str | None) -> str:
    if requested_port:
        return requested_port

    devices = [port.device for port in list_ports.comports()]
    preferred = [
        device
        for device in devices
        if re.search(r"(usbmodem|usbserial|ttyACM|ttyUSB)", device, re.IGNORECASE)
    ]
    candidates = preferred or devices

    if not candidates:
        raise RuntimeError("No serial device found. Connect the board or pass --port.")
    if len(candidates) > 1:
        choices = "\n  ".join(candidates)
        raise RuntimeError(f"Multiple serial devices found; pass --port:\n  {choices}")
    return candidates[0]


def open_serial(port: str | None, baud: int, timeout: float = 1.0) -> serial.Serial:
    resolved = resolve_serial_port(port)
    connection = serial.Serial(resolved, baudrate=baud, timeout=timeout)
    time.sleep(2.0)  # Most Arduino boards reset when the port opens.
    connection.reset_input_buffer()
    print(f"Connected to {resolved} at {baud} baud")
    return connection


def timestamped_path(output_dir: Path, prefix: str, suffix: str = ".csv") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{prefix}_{timestamp}{suffix}"


def read_csv_rows(
    connection: serial.Serial,
    parse_line: Callable[[str], Row | None],
) -> Iterable[Row]:
    while True:
        raw = connection.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        row = parse_line(line)
        if row is not None:
            yield row


def write_stream(
    connection: serial.Serial,
    output_path: Path,
    header: Sequence[str],
    rows: Iterable[Row],
    limit: int | None = None,
) -> int:
    saved = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        try:
            for row in rows:
                writer.writerow(row)
                saved += 1
                if saved % 100 == 0:
                    handle.flush()
                    print(f"Saved {saved} samples")
                if limit is not None and saved >= limit:
                    break
        except KeyboardInterrupt:
            print("\nStopped by user")
    return saved


def parse_five_adc(line: str) -> tuple[int, int, int, int, int] | None:
    parts = line.split(",")
    if len(parts) != 5:
        return None
    try:
        values = tuple(int(part) for part in parts)
    except ValueError:
        return None
    if not all(0 <= value <= 1023 for value in values):
        return None
    return values  # type: ignore[return-value]


def parse_flex_resistance(line: str) -> tuple[int, int, int, float] | None:
    parts = line.split(",")
    if len(parts) != 4 or line.startswith("sample_index"):
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
    except ValueError:
        return None


def parse_flex_adc(line: str) -> tuple[int, int, int] | None:
    parts = line.split(",")
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def parse_three_adc(line: str) -> tuple[int, int, int] | None:
    parts = line.split(",")
    if len(parts) != 3:
        return None
    try:
        values = tuple(int(part) for part in parts)
    except ValueError:
        return None
    if not all(0 <= value <= 1023 for value in values):
        return None
    return values  # type: ignore[return-value]

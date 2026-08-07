"""Tests that do not require an Arduino, relay board, or camera."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_DIR))

from serial_utils import (  # noqa: E402
    parse_five_adc,
    parse_flex_adc,
    parse_flex_resistance,
    parse_three_adc,
)
from collect_drop_timing import parse_metric  # noqa: E402


class ParserTests(unittest.TestCase):
    def test_five_adc(self) -> None:
        self.assertEqual(parse_five_adc("1,2,3,4,1023"), (1, 2, 3, 4, 1023))
        self.assertIsNone(parse_five_adc("1,2,3,4"))
        self.assertIsNone(parse_five_adc("1,2,3,4,1200"))

    def test_flex_adc(self) -> None:
        self.assertEqual(parse_flex_adc("4,50000,721"), (4, 50000, 721))
        self.assertIsNone(parse_flex_adc("sample_index,time_us,adc"))

    def test_flex_resistance(self) -> None:
        self.assertEqual(parse_flex_resistance("3,30,512,20.0391"), (3, 30, 512, 20.0391))
        self.assertIsNone(parse_flex_resistance("sample_index,time_ms,adc,resistance_kohm"))

    def test_three_adc(self) -> None:
        self.assertEqual(parse_three_adc("100,200,300"), (100, 200, 300))
        self.assertIsNone(parse_three_adc("100,broken,300"))

    def test_drop_metric(self) -> None:
        self.assertEqual(parse_metric("1. Speed (Delta AB) : 150"), ("delta_ab_ms", 150))
        self.assertEqual(parse_metric("Catch Delay: 42"), ("catch_delay_ms", 42))
        self.assertIsNone(parse_metric("unrelated log line"))


if __name__ == "__main__":
    unittest.main()

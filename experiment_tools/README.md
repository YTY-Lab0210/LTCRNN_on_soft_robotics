# Experiment Tools

This folder contains data-collection, sensor-checking, and relay-test tools used in the soft robotic hand experiments. Python scripts avoid hard-coded local paths and fixed USB ports; if `--port` is omitted, they try to automatically detect an Arduino-like serial device.

## Contents

| Function | Arduino firmware | Python script |
|---|---|---|
| Five-finger flex sensors, 100 Hz, 400 samples | `arduino/flex_sensor_5ch/flex_sensor_5ch.ino` | `python/collect_flex_five_channel.py` |
| Single flex ADC stream | `arduino/flex_adc/flex_adc.ino` | `python/read_flex_adc.py` |
| Single flex resistance measurement | `arduino/flex_resistance/flex_resistance.ino` | `python/collect_flex_resistance.py` |
| Five-relay cycling test | `arduino/relay_test/relay_test.ino` | Not required |

## Installation

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r experiment_tools/requirements.txt
```

On Windows, activate the virtual environment with:

```powershell
.venv\Scriptsctivate
```

List available serial ports:

```bash
python -m serial.tools.list_ports
```

## Common Commands

Collect one 400-sample five-finger flex-sensor trial:

```bash
python experiment_tools/python/collect_flex_five_channel.py   --port /dev/cu.usbmodem101   --label bottle   --samples 400
```

If only one Arduino is connected, `--port` can usually be omitted:

```bash
python experiment_tools/python/collect_flex_five_channel.py --label bottle
```

Collect single flex-sensor resistance values:

```bash
python experiment_tools/python/collect_flex_resistance.py --port /dev/cu.usbmodem101
```

View a single flex ADC stream in the terminal:

```bash
python experiment_tools/python/read_flex_adc.py --port /dev/cu.usbmodem101
```

Outputs are saved to `output/` by default. Use `--output-dir` to choose another location.

## Relay Notes

`relay_test.ino` assumes an active-HIGH relay module by default:

```cpp
const uint8_t RELAY_ON = HIGH;
const uint8_t RELAY_OFF = LOW;
```

If your relay module is active LOW, swap these two constants. Before powering the system, verify the voltage ratings of the relay module, pump, and solenoid valve. The Arduino should only handle low-voltage control signals; the relay control side must share ground with Arduino, while high-voltage or 110 V load wiring must remain isolated.

## Tests

Parser tests can be run without hardware:

```bash
python -m unittest discover -s experiment_tools/tests -v
```

Arduino compilation checks require Arduino CLI and the relevant board core. Before running real hardware tests, confirm the board type, pin assignments, and relay active level.

## Not Included

- CSV files, MP4 files, and experimental datasets
- `.venv`, Python cache, and IDE settings
- Scripts with hard-coded personal paths or fixed USB ports
- Duplicate legacy files

Sources were consolidated from earlier flex-sensor data collectors, single-flex readers, resistance readers, and relay sketches.

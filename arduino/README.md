# Arduino Deployment

This folder contains Arduino Uno sketches for on-board inference. The main goal is to let Arduino read flex sensors, control five relays for grasping, and run LTC-4 inference directly on the board.

## Pin Assignment

Flex-sensor analog inputs:

```text
A0, A1, A2, A3, A4
```

Relay control outputs:

```text
D2, D3, D4, D5, D6
```

Serial Monitor baud rate:

```text
115200
```

## LTC-4 Relay Variants

Two variants are provided for different relay-module trigger logic:

```text
deployment_candidates/ltc4_best_low_active/ltc4_best_low_active.ino
deployment_candidates/ltc4_best_high_active/ltc4_best_high_active.ino
```

Low-active:

```cpp
RELAY_ON_LEVEL = LOW
RELAY_OFF_LEVEL = HIGH
```

High-active:

```cpp
RELAY_ON_LEVEL = HIGH
RELAY_OFF_LEVEL = LOW
```

These sketches do not automatically detect the relay type. Flash the version that matches the relay module used in the experiment.

## Serial Monitor Commands

```text
g  Start one 400-sample grasp window
r  Release all relays
b  Run inference benchmark on a flash-stored 400-sample window
t  Print one task-level timing breakdown
m  Repeat the flash benchmark 100 times
d  Repeat the timing breakdown 100 times
```

## `g` Command Flow

```text
Enter g
  -> sample flex sensors every 10 ms
  -> keep the first 1.5 s as the pre-grasp baseline period
  -> activate five relays after 1.5 s to start grasping
  -> collect 400 samples
  -> run z-score normalization
  -> run LTC-4 Euler updates
  -> run dense layer and softmax
  -> print the predicted class
  -> turn off relays and release
```

## Flash Replay Benchmark

The `b`, `t`, `m`, and `d` commands use a flash-stored `flash_sequence.h` 400-step sequence. This mode does not wait for real 10 ms sampling intervals, so it can estimate pure on-board inference computation time.

Timing breakdown stages:

```text
State reset
Flash read
Z-score normalize
LTC Euler update
Dense + softmax
Argmax
Total
```

## Wiring Notes

Arduino handles only low-voltage logic signals:

```text
Arduino analog pins -> flex-sensor voltage-divider outputs
Arduino digital pins -> relay module input pins
Arduino GND -> relay control-side GND
```

The relay module needs an appropriate DC supply. The pump, solenoid valve, and 110 V load side must not be connected directly to Arduino logic pins. The high-voltage/load side should remain isolated and controlled only through relay contacts.

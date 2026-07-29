# Arduino Deployment Notes

The deployment sketches target Arduino Uno.

## Pins

Flex sensor analog inputs:

```text
A0, A1, A2, A3, A4
```

Relay control output pins:

```text
D2, D3, D4, D5, D6
```

## LTC-4 Relay Variants

Two versions are provided for testing different relay trigger types:

```text
deployment_candidates/ltc4_best_low_active/
deployment_candidates/ltc4_best_high_active/
```

Low-active version:

```cpp
RELAY_ON_LEVEL = LOW
RELAY_OFF_LEVEL = HIGH
```

High-active version:

```cpp
RELAY_ON_LEVEL = HIGH
RELAY_OFF_LEVEL = LOW
```

There is no automatic relay-type detection in these two sketches.

## Serial Commands

Open Serial Monitor at `115200` baud.

```text
g  Start one 400-sample grasp window
r  Release all relays
b  Benchmark the flash-stored 400-sample window
t  Print task-level timing breakdown
m  Run flash replay benchmark 100 times
d  Repeat timing breakdown 100 times
```

## Grasp Window

The online grasp command collects 400 samples at 10 ms per sample.

The relays turn on after 1.5 s:

```text
0.0 s to 1.5 s    baseline / pre-grasp samples
1.5 s to 4.0 s    grasping dynamics
after 400 steps   LTC-4 inference result is printed
```

The sketch releases the relays after inference.

## Wiring Reminder

The Arduino handles only the low-voltage control side.

The relay module should use its own suitable DC power supply. The Arduino digital pins connect to the relay input pins, and Arduino GND must share a common reference with the relay control-side GND.

The pump and valve power wiring should remain isolated from Arduino logic wiring.

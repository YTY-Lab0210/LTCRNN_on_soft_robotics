# Dataset Notes

## Main Dataset

The main dataset is:

```text
dataset_new_new_new/
```

This is the renamed raw ADC dataset. File names use the display labels used in the confusion-matrix figures.

Each CSV file is a 5-channel time-series recording from flex sensors mounted on the soft robotic hand.

## File Naming

Example:

```text
Baseball_001.csv
Bottle_001.csv
Sponge Dice_001.csv
3D-Printed Part_001.csv
```

## Class Counts

```text
3D-Printed Part   100
Baseball           99
Bottle            100
Optical Mouse     100
Plush Toy         100
Rubik's Cube      100
Smartphone        100
Sponge Dice       100
Stuffed Ball      100
Tape              100
```

Total:

```text
999 CSV files
```

The `Baseball` class currently has 99 samples because one valid baseball sample was missing during final dataset cleanup.

## Preprocessing

This folder stores the raw renamed dataset. Z-score normalized data can be regenerated from the raw CSV files using the preprocessing utilities in:

```text
src/utils/z_score.py
```

## Input Channels

The 5-channel setting uses all five flex sensors.

The 3-channel experiments use a reduced sensor subset, implemented in the corresponding 3-channel experiment scripts.

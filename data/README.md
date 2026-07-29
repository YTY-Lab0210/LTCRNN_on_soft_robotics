# Dataset Notes

## Main Dataset

The main dataset is:

```text
dataset_xx2020_new_new_new/
```

It contains two versions:

```text
dataset_602020/          Raw ADC data
dataset_602020_zscore/   Z-score normalized data
```

Each sequence is a 5-channel time series from flex sensors mounted on the soft robotic hand.

## Split

The naming `602020` means the dataset is split into training, validation, and test sets:

```text
training/
validation/
test/
```

The current raw split has:

```text
training:   59 or 60+ samples per class, depending on class availability
validation: 20 samples per class
test:       20 samples per class
```

The `ball` class currently has 59 training samples because one valid baseball sample was missing during the final data cleanup.

## Class Mapping

The file prefixes map to the paper/display labels as follows:

```text
ball        -> Baseball
bottle      -> Bottle
cube        -> Sponge Dice
cylinder    -> Tape
doll        -> Plush Toy
mouse       -> Optical Mouse
phone       -> Smartphone
rubik_cube  -> Rubik's Cube
small_ball  -> Stuffed Ball
support     -> 3D-Printed Part
```

## Input Channels

The 5-channel setting uses all five flex sensors.

The 3-channel experiments use a reduced sensor subset, implemented in the corresponding 3-channel experiment scripts.

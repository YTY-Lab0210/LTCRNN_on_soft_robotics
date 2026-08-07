# Dataset

This folder stores the cleaned raw ADC datasets collected from a five-finger soft robotic hand. Each finger is equipped with a flex sensor, and each CSV records finger-bending signals during one grasping trial.

## 5-Channel Dataset

Main dataset:

```text
dataset_new_new_new/
```

Each CSV is one grasping sequence with the following columns:

```text
Time_ms, Thumb, Index, Middle, Ring, Pinky
```

Each sample contains 400 time steps sampled every 10 ms. Including the header, each CSV has 401 rows.

## 3-Channel Dataset

Reduced-sensor dataset:

```text
dataset_new_new_new_3ch/
```

Kept columns:

```text
Time_ms, Thumb, Middle, Pinky
```

This dataset is generated from the 5-channel dataset by selecting the specified fingers and randomly renumbering samples within each class. The source mapping is stored in:

```text
dataset_new_new_new_3ch_manifest.csv
```

## File Naming

File names follow the object labels used in the confusion matrix:

```text
Baseball_001.csv
Bottle_001.csv
Sponge Dice_001.csv
3D-Printed Part_001.csv
```

## Classes and Counts

```text
3D-Printed Part   100
Baseball          100
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
1000 CSV files
```

## Preprocessing

The files in this folder store raw ADC values, not normalized values. During training, compute the mean and standard deviation from the training set only, then apply z-score normalization to the train, validation, and test splits.

Related utilities:

```text
src/utils/z_score.py
src/utils/create_3ch_dataset.py
src/utils/split_data.py
```

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow import keras

import run_extended_rnn_lstm_ltc_benchmark as bench


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "dataset_xx2020_new_new_new" / "dataset_602020_zscore"
DEFAULT_OUT_DIR = ROOT / "LTCRNN" / "BPTT" / "deployment_candidates"
DEFAULT_ARDUINO_DIR = ROOT / "LTCRNN" / "arduino" / "deployment_candidates"

TARGET_MODELS = ["LTC-4", "LSTM-8", "SimpleRNN-8"]

SENSOR_MEAN_BY_COLUMN = {
    "Thumb": 795.53795191,
    "Index": 784.61776534,
    "Middle": 801.80461028,
    "Ring": 759.27079602,
    "Pinky": 840.59891376,
}

SENSOR_STD_BY_COLUMN = {
    "Thumb": 62.00574361,
    "Index": 50.61546292,
    "Middle": 39.76813518,
    "Ring": 39.13091513,
    "Pinky": 40.79949670,
}

DISPLAY_LABELS = {
    "ball": "Baseball",
    "bottle": "Bottle",
    "cube": "Sponge Dice",
    "cylinder": "Tape",
    "doll": "Plush Toy",
    "mouse": "Optical Mouse",
    "phone": "Smartphone",
    "rubik_cube": "Rubik's Cube",
    "small_ball": "Stuffed Ball",
    "support": "3D-Printed Part",
}


def c_string(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def c_float(value):
    return f"{float(value): .8f}f"


def format_vector(name, values):
    values = np.asarray(values).reshape(-1)
    body = ", ".join(c_float(v) for v in values)
    return f"const float {name}[{len(values)}] PROGMEM = {{\n  {body}\n}};"


def format_matrix(name, values):
    values = np.asarray(values)
    rows = []
    for row in values:
        rows.append("  {" + ", ".join(c_float(v) for v in row) + "}")
    return (
        f"const float {name}[{values.shape[0]}][{values.shape[1]}] PROGMEM = {{\n"
        + ",\n".join(rows)
        + "\n};"
    )


def display_class_names(class_names):
    return [DISPLAY_LABELS.get(name, name) for name in class_names]


def append_result(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    pd.DataFrame([row]).to_csv(
        csv_path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )


def existing_keys(raw_path):
    if not raw_path.exists():
        return set()
    df = pd.read_csv(raw_path)
    if df.empty:
        return set()
    return set(zip(df["Channel"], df["Model"], df["Run"]))


def train_one(dataset, channel, model_name, run_index, args):
    keras.backend.clear_session()
    model_offset = TARGET_MODELS.index(model_name) * 1000 if model_name in TARGET_MODELS else 0
    keras.utils.set_random_seed(args.seed + model_offset + run_index)

    input_shape = (bench.TARGET_LINES, len(dataset["columns"]))
    model = bench.build_model(model_name, input_shape, dataset["num_classes"])
    trainable_params = int(model.count_params())

    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=args.clipnorm)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=args.jit_compile,
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        dataset["x_train"],
        dataset["y_train"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(dataset["x_val"], dataset["y_val"]),
        callbacks=[early_stopping],
        verbose=args.verbose,
    )

    test_loss, test_accuracy = model.evaluate(dataset["x_test"], dataset["y_test"], verbose=0)
    probabilities = model.predict(dataset["x_test"], verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    macro_f1 = f1_score(dataset["y_test_encoded"], y_pred, average="macro", zero_division=0)

    history_df = pd.DataFrame(history.history)
    best_idx = int(history_df["val_loss"].idxmin())

    checkpoint_dir = args.out_dir / "checkpoints" / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / f"{model_name}_{channel}_run{run_index:02d}.keras"
    history_path = checkpoint_dir / f"{model_name}_{channel}_run{run_index:02d}_history.csv"

    model.save(model_path)
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    return {
        "Channel": channel,
        "Model": model_name,
        "Run": run_index,
        "TrainableParams": trainable_params,
        "Accuracy(%)": float(test_accuracy * 100.0),
        "Macro-F1(%)": float(macro_f1 * 100.0),
        "TestLoss": float(test_loss),
        "BestValAccuracy(%)": float(history_df.loc[best_idx, "val_accuracy"] * 100.0),
        "BestValLoss": float(history_df.loc[best_idx, "val_loss"]),
        "EpochsTrained": int(len(history_df)),
        "InputColumns": ",".join(dataset["columns"]),
        "ClassNames": "|".join(dataset["class_names"]),
        "ModelPath": str(model_path),
        "HistoryPath": str(history_path),
    }


def select_best_models(raw_path, best_path, channel, models):
    raw = pd.read_csv(raw_path)
    selected = []
    for model_name in models:
        subset = raw[(raw["Channel"] == channel) & (raw["Model"] == model_name)].copy()
        if subset.empty:
            raise RuntimeError(f"No finished runs found for {channel} {model_name}.")
        subset = subset.sort_values(
            ["Accuracy(%)", "Macro-F1(%)", "TestLoss", "Run"],
            ascending=[False, False, True, True],
        )
        selected.append(subset.iloc[0].to_dict())

    best = pd.DataFrame(selected)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(best_path, index=False, encoding="utf-8-sig")
    return best


def find_recurrent_and_dense(model):
    recurrent = None
    dense = None
    for layer in model.layers:
        if isinstance(layer, (keras.layers.SimpleRNN, keras.layers.LSTM)):
            recurrent = layer
        elif isinstance(layer, keras.layers.RNN) and isinstance(layer.cell, bench.LTCNeuron):
            recurrent = layer
        elif isinstance(layer, keras.layers.Dense):
            dense = layer

    if recurrent is None:
        raise RuntimeError("Could not find recurrent layer in saved model.")
    if dense is None:
        raise RuntimeError("Could not find Dense output layer in saved model.")
    return recurrent, dense


def sensor_array_text(columns):
    means = [SENSOR_MEAN_BY_COLUMN[col] for col in columns]
    stds = [SENSOR_STD_BY_COLUMN[col] for col in columns]
    return "\n\n".join(
        [
            format_vector("SENSOR_MEAN", means),
            format_vector("SENSOR_STD", stds),
        ]
    )


def class_names_text(class_names):
    labels = display_class_names(class_names)
    lines = []
    for idx, label in enumerate(labels):
        lines.append(f'const char CLASS_{idx}[] PROGMEM = "{c_string(label)}";')
    table = ",\n  ".join(f"CLASS_{idx}" for idx in range(len(labels)))
    lines.append(f"const char *const CLASS_NAMES[NUM_CLASSES] PROGMEM = {{\n  {table}\n}};")
    return "\n".join(lines)


def default_flash_sequence_text(columns):
    baseline = [int(round(SENSOR_MEAN_BY_COLUMN[col])) for col in columns]
    row_text = "  {" + ", ".join(str(v) for v in baseline) + "}"
    rows = ",\n".join(row_text for _ in range(400))
    return "\n".join(
        [
            "#pragma once",
            "#include <Arduino.h>",
            "#include <avr/pgmspace.h>",
            "",
            "// Default flat baseline sequence. Replace this file with a real 400-step CSV export.",
            "const uint16_t FLASH_RAW_SEQUENCE[SEQ_LEN][INPUT_DIM] PROGMEM = {",
            rows,
            "};",
            "",
        ]
    )


def common_code(model_name, row, columns, class_names, hidden_units, arrays_text, state_step_text):
    sensor_pins = ", ".join(f"A{i}" for i in range(len(columns)))
    relay_pins = ", ".join(str(i) for i in range(2, 2 + len(columns)))
    return "\n".join(
        [
            "#include <Arduino.h>",
            "#include <math.h>",
            "#include <avr/pgmspace.h>",
            "#include <string.h>",
            "",
            f"// Trained model path: {c_string(row['ModelPath'])}",
            f'const char MODEL_NAME[] PROGMEM = "{c_string(model_name)}";',
            f"constexpr uint8_t INPUT_DIM = {len(columns)};",
            f"constexpr uint8_t HIDDEN_UNITS = {hidden_units};",
            f"constexpr uint8_t NUM_CLASSES = {len(class_names)};",
            "constexpr uint16_t SEQ_LEN = 400;",
            "constexpr uint16_t SAMPLE_INTERVAL_MS = 10;",
            "constexpr uint16_t RELAY_ON_DELAY_MS = 1500;",
            "constexpr uint16_t BENCHMARK_REPEAT_COUNT = 100;",
            "constexpr uint8_t TIMING_TASK_COUNT = 7;",
            "constexpr uint8_t TIMING_RESET = 0;",
            "constexpr uint8_t TIMING_FLASH_READ = 1;",
            "constexpr uint8_t TIMING_NORMALIZE = 2;",
            "constexpr uint8_t TIMING_MODEL_STEP = 3;",
            "constexpr uint8_t TIMING_DENSE_SOFTMAX = 4;",
            "constexpr uint8_t TIMING_ARGMAX = 5;",
            "constexpr uint8_t TIMING_TOTAL = 6;",
            "constexpr bool RELEASE_AFTER_INFERENCE = true;",
            "constexpr bool RELAY_ACTIVE_LOW = true;",
            f"constexpr uint8_t SELECTED_RUN = {int(row['Run'])};",
            f"constexpr float SELECTED_TEST_ACCURACY = {float(row['Accuracy(%)']):.6f}f;",
            f"constexpr float SELECTED_MACRO_F1 = {float(row['Macro-F1(%)']):.6f}f;",
            "",
            '#include "flash_sequence.h"',
            "",
            f"const uint8_t SENSOR_PINS[INPUT_DIM] = {{{sensor_pins}}};",
            f"const uint8_t RELAY_PINS[INPUT_DIM] = {{{relay_pins}}};",
            "",
            "// z = (raw_adc - training_mean) / training_std",
            sensor_array_text(columns),
            "",
            class_names_text(class_names),
            "",
            arrays_text,
            "",
            r"""float readFloat(const float *address) {
  return pgm_read_float(address);
}

float sigmoidClamped(float x);
""",
            "",
            state_step_text,
            "",
            r"""float probabilities[NUM_CLASSES];
char labelBuffer[32];

struct BenchmarkTiming {
  unsigned long resetUs;
  unsigned long flashReadUs;
  unsigned long normalizeUs;
  unsigned long modelStepUs;
  unsigned long denseSoftmaxUs;
  unsigned long argmaxUs;
  unsigned long totalUs;
};

BenchmarkTiming timingBreakdown;
float repeatedTimingMeanUs[TIMING_TASK_COUNT];
float repeatedTimingM2Us[TIMING_TASK_COUNT];
unsigned long repeatedTimingMinUs[TIMING_TASK_COUNT];
unsigned long repeatedTimingMaxUs[TIMING_TASK_COUNT];

void printProgmemString(const char *address) {
  char buffer[32];
  strncpy_P(buffer, address, sizeof(buffer) - 1);
  buffer[sizeof(buffer) - 1] = '\0';
  Serial.print(buffer);
}

void loadClassName(uint8_t classIndex, char *buffer, size_t bufferSize) {
  const char *classAddress = reinterpret_cast<const char *>(pgm_read_ptr(&CLASS_NAMES[classIndex]));
  strncpy_P(buffer, classAddress, bufferSize - 1);
  buffer[bufferSize - 1] = '\0';
}

void setRelay(uint8_t relayIndex, bool on) {
  const uint8_t activeLevel = RELAY_ACTIVE_LOW ? LOW : HIGH;
  const uint8_t inactiveLevel = RELAY_ACTIVE_LOW ? HIGH : LOW;
  digitalWrite(RELAY_PINS[relayIndex], on ? activeLevel : inactiveLevel);
}

void setAllRelays(bool on) {
  for (uint8_t i = 0; i < INPUT_DIM; ++i) {
    setRelay(i, on);
  }
}

void setupRelays() {
  const uint8_t inactiveLevel = RELAY_ACTIVE_LOW ? HIGH : LOW;
  for (uint8_t i = 0; i < INPUT_DIM; ++i) {
    digitalWrite(RELAY_PINS[i], inactiveLevel);
    pinMode(RELAY_PINS[i], OUTPUT);
  }
}

float sigmoidClamped(float x) {
  if (x > 20.0f) {
    return 1.0f;
  }
  if (x < -20.0f) {
    return 0.0f;
  }
  return 1.0f / (1.0f + expf(-x));
}

void readRawSensors(float raw[INPUT_DIM]) {
  for (uint8_t i = 0; i < INPUT_DIM; ++i) {
    raw[i] = static_cast<float>(analogRead(SENSOR_PINS[i]));
  }
}

void readFlashRawSample(uint16_t sampleIndex, float raw[INPUT_DIM]) {
  for (uint8_t i = 0; i < INPUT_DIM; ++i) {
    raw[i] = static_cast<float>(pgm_read_word(&FLASH_RAW_SEQUENCE[sampleIndex][i]));
  }
}

void zScoreNormalize(const float raw[INPUT_DIM], float z[INPUT_DIM]) {
  for (uint8_t i = 0; i < INPUT_DIM; ++i) {
    z[i] = (raw[i] - readFloat(&SENSOR_MEAN[i])) / readFloat(&SENSOR_STD[i]);
  }
}

void denseSoftmax(float probs[NUM_CLASSES]) {
  float logits[NUM_CLASSES];
  float maxLogit = -3.4028235e38f;

  for (uint8_t c = 0; c < NUM_CLASSES; ++c) {
    float logit = readFloat(&DENSE_B[c]);
    for (uint8_t u = 0; u < HIDDEN_UNITS; ++u) {
      logit += hiddenState[u] * readFloat(&DENSE_W[u][c]);
    }
    logits[c] = logit;
    if (logit > maxLogit) {
      maxLogit = logit;
    }
  }

  float sumExp = 0.0f;
  for (uint8_t c = 0; c < NUM_CLASSES; ++c) {
    probs[c] = expf(logits[c] - maxLogit);
    sumExp += probs[c];
  }

  if (sumExp <= 0.0f) {
    for (uint8_t c = 0; c < NUM_CLASSES; ++c) {
      probs[c] = 1.0f / NUM_CLASSES;
    }
    return;
  }

  for (uint8_t c = 0; c < NUM_CLASSES; ++c) {
    probs[c] /= sumExp;
  }
}

uint8_t argmaxClass(const float probs[NUM_CLASSES]) {
  uint8_t bestClass = 0;
  float bestValue = probs[0];

  for (uint8_t c = 1; c < NUM_CLASSES; ++c) {
    if (probs[c] > bestValue) {
      bestValue = probs[c];
      bestClass = c;
    }
  }

  return bestClass;
}

uint8_t runOneInferenceWindow(float probs[NUM_CLASSES]) {
  resetModelState();
  setAllRelays(false);

  float raw[INPUT_DIM];
  float z[INPUT_DIM];
  bool relaysActivated = false;
  const uint16_t relayOnSample = RELAY_ON_DELAY_MS / SAMPLE_INTERVAL_MS;

  for (uint16_t t = 0; t < SEQ_LEN; ++t) {
    const unsigned long startMs = millis();

    if (!relaysActivated && t >= relayOnSample) {
      setAllRelays(true);
      relaysActivated = true;
    }

    readRawSensors(raw);
    zScoreNormalize(raw, z);
    modelStep(z);

    const unsigned long elapsed = millis() - startMs;
    if (elapsed < SAMPLE_INTERVAL_MS) {
      delay(SAMPLE_INTERVAL_MS - elapsed);
    }
  }

  denseSoftmax(probs);
  return argmaxClass(probs);
}

uint8_t runFlashReplayBenchmark(float probs[NUM_CLASSES], unsigned long *elapsedUs) {
  resetModelState();

  float raw[INPUT_DIM];
  float z[INPUT_DIM];

  const unsigned long startUs = micros();
  for (uint16_t t = 0; t < SEQ_LEN; ++t) {
    readFlashRawSample(t, raw);
    zScoreNormalize(raw, z);
    modelStep(z);
  }

  denseSoftmax(probs);
  const uint8_t pred = argmaxClass(probs);
  *elapsedUs = micros() - startUs;
  return pred;
}

uint8_t runFlashReplayTimingBreakdown(float probs[NUM_CLASSES]) {
  timingBreakdown.resetUs = 0;
  timingBreakdown.flashReadUs = 0;
  timingBreakdown.normalizeUs = 0;
  timingBreakdown.modelStepUs = 0;
  timingBreakdown.denseSoftmaxUs = 0;
  timingBreakdown.argmaxUs = 0;
  timingBreakdown.totalUs = 0;

  unsigned long segmentStartUs = micros();
  resetModelState();
  timingBreakdown.resetUs = micros() - segmentStartUs;

  float raw[INPUT_DIM];
  float z[INPUT_DIM];

  const unsigned long totalStartUs = micros();
  for (uint16_t t = 0; t < SEQ_LEN; ++t) {
    segmentStartUs = micros();
    readFlashRawSample(t, raw);
    timingBreakdown.flashReadUs += micros() - segmentStartUs;

    segmentStartUs = micros();
    zScoreNormalize(raw, z);
    timingBreakdown.normalizeUs += micros() - segmentStartUs;

    segmentStartUs = micros();
    modelStep(z);
    timingBreakdown.modelStepUs += micros() - segmentStartUs;
  }

  segmentStartUs = micros();
  denseSoftmax(probs);
  timingBreakdown.denseSoftmaxUs = micros() - segmentStartUs;

  segmentStartUs = micros();
  const uint8_t pred = argmaxClass(probs);
  timingBreakdown.argmaxUs = micros() - segmentStartUs;

  timingBreakdown.totalUs = micros() - totalStartUs;
  return pred;
}

void printProbabilities(const float probs[NUM_CLASSES]) {
  for (uint8_t c = 0; c < NUM_CLASSES; ++c) {
    loadClassName(c, labelBuffer, sizeof(labelBuffer));
    Serial.print(labelBuffer);
    Serial.print(F(": "));
    Serial.println(probs[c], 6);
  }
}

void printTimingLine(const __FlashStringHelper *label, unsigned long totalUs, bool perStep) {
  Serial.print(label);
  Serial.print(F(" total (us): "));
  Serial.println(totalUs);
  if (perStep) {
    Serial.print(label);
    Serial.print(F(" per step (us): "));
    Serial.println(static_cast<float>(totalUs) / static_cast<float>(SEQ_LEN), 3);
  }
}

void resetRepeatedTimingStats() {
  for (uint8_t i = 0; i < TIMING_TASK_COUNT; ++i) {
    repeatedTimingMeanUs[i] = 0.0f;
    repeatedTimingM2Us[i] = 0.0f;
    repeatedTimingMinUs[i] = 4294967295UL;
    repeatedTimingMaxUs[i] = 0;
  }
}

void updateRepeatedTimingStat(uint8_t taskIndex, unsigned long valueUs, uint16_t count) {
  if (valueUs < repeatedTimingMinUs[taskIndex]) {
    repeatedTimingMinUs[taskIndex] = valueUs;
  }
  if (valueUs > repeatedTimingMaxUs[taskIndex]) {
    repeatedTimingMaxUs[taskIndex] = valueUs;
  }

  const float x = static_cast<float>(valueUs);
  const float delta = x - repeatedTimingMeanUs[taskIndex];
  repeatedTimingMeanUs[taskIndex] += delta / static_cast<float>(count);
  const float delta2 = x - repeatedTimingMeanUs[taskIndex];
  repeatedTimingM2Us[taskIndex] += delta * delta2;
}

void updateRepeatedTimingStats(uint16_t count) {
  updateRepeatedTimingStat(TIMING_RESET, timingBreakdown.resetUs, count);
  updateRepeatedTimingStat(TIMING_FLASH_READ, timingBreakdown.flashReadUs, count);
  updateRepeatedTimingStat(TIMING_NORMALIZE, timingBreakdown.normalizeUs, count);
  updateRepeatedTimingStat(TIMING_MODEL_STEP, timingBreakdown.modelStepUs, count);
  updateRepeatedTimingStat(TIMING_DENSE_SOFTMAX, timingBreakdown.denseSoftmaxUs, count);
  updateRepeatedTimingStat(TIMING_ARGMAX, timingBreakdown.argmaxUs, count);
  updateRepeatedTimingStat(TIMING_TOTAL, timingBreakdown.totalUs, count);
}

void printRepeatedTimingStat(const __FlashStringHelper *label, uint8_t taskIndex, bool perStep) {
  const float varianceUs = repeatedTimingM2Us[taskIndex] / static_cast<float>(BENCHMARK_REPEAT_COUNT - 1);
  const float stdUs = sqrtf(varianceUs);

  Serial.print(label);
  Serial.print(F(" total us mean/std/min/max: "));
  Serial.print(repeatedTimingMeanUs[taskIndex], 3);
  Serial.print(F(", "));
  Serial.print(stdUs, 3);
  Serial.print(F(", "));
  Serial.print(repeatedTimingMinUs[taskIndex]);
  Serial.print(F(", "));
  Serial.println(repeatedTimingMaxUs[taskIndex]);

  if (perStep) {
    Serial.print(label);
    Serial.print(F(" per-step us mean/std/min/max: "));
    Serial.print(repeatedTimingMeanUs[taskIndex] / static_cast<float>(SEQ_LEN), 3);
    Serial.print(F(", "));
    Serial.print(stdUs / static_cast<float>(SEQ_LEN), 3);
    Serial.print(F(", "));
    Serial.print(static_cast<float>(repeatedTimingMinUs[taskIndex]) / static_cast<float>(SEQ_LEN), 3);
    Serial.print(F(", "));
    Serial.println(static_cast<float>(repeatedTimingMaxUs[taskIndex]) / static_cast<float>(SEQ_LEN), 3);
  }
}

void runRepeatedFlashBenchmark() {
  float meanUs = 0.0f;
  float m2Us = 0.0f;
  unsigned long minUs = 4294967295UL;
  unsigned long maxUs = 0;
  uint8_t lastPred = 0;

  for (uint16_t i = 1; i <= BENCHMARK_REPEAT_COUNT; ++i) {
    unsigned long elapsedUs = 0;
    lastPred = runFlashReplayBenchmark(probabilities, &elapsedUs);

    if (elapsedUs < minUs) {
      minUs = elapsedUs;
    }
    if (elapsedUs > maxUs) {
      maxUs = elapsedUs;
    }

    const float x = static_cast<float>(elapsedUs);
    const float delta = x - meanUs;
    meanUs += delta / static_cast<float>(i);
    const float delta2 = x - meanUs;
    m2Us += delta * delta2;
  }

  const float varianceUs = m2Us / static_cast<float>(BENCHMARK_REPEAT_COUNT - 1);
  const float stdUs = sqrtf(varianceUs);

  Serial.print(F("Repeat count: "));
  Serial.println(BENCHMARK_REPEAT_COUNT);
  Serial.print(F("Mean inference time (us): "));
  Serial.println(meanUs, 3);
  Serial.print(F("Std inference time (us): "));
  Serial.println(stdUs, 3);
  Serial.print(F("Min inference time (us): "));
  Serial.println(minUs);
  Serial.print(F("Max inference time (us): "));
  Serial.println(maxUs);
  Serial.print(F("Mean inference time per step (us): "));
  Serial.println(meanUs / static_cast<float>(SEQ_LEN), 3);
  Serial.print(F("Std inference time per step (us): "));
  Serial.println(stdUs / static_cast<float>(SEQ_LEN), 3);
  Serial.print(F("Last predicted class index: "));
  Serial.println(lastPred);
  Serial.print(F("Last predicted class label: "));
  loadClassName(lastPred, labelBuffer, sizeof(labelBuffer));
  Serial.println(labelBuffer);
  Serial.print(F("Last confidence: "));
  Serial.println(probabilities[lastPred], 6);
}

void runRepeatedTimingBreakdown() {
  resetRepeatedTimingStats();
  uint8_t lastPred = 0;

  for (uint16_t i = 1; i <= BENCHMARK_REPEAT_COUNT; ++i) {
    lastPred = runFlashReplayTimingBreakdown(probabilities);
    updateRepeatedTimingStats(i);
  }

  Serial.print(F("Repeat count: "));
  Serial.println(BENCHMARK_REPEAT_COUNT);
  Serial.println(F("Format: mean, std, min, max"));
  printRepeatedTimingStat(F("State reset"), TIMING_RESET, false);
  printRepeatedTimingStat(F("Flash read"), TIMING_FLASH_READ, true);
  printRepeatedTimingStat(F("Z-score normalize"), TIMING_NORMALIZE, true);
  printRepeatedTimingStat(F("LTC Euler update"), TIMING_MODEL_STEP, true);
  printRepeatedTimingStat(F("Dense + softmax"), TIMING_DENSE_SOFTMAX, false);
  printRepeatedTimingStat(F("Argmax"), TIMING_ARGMAX, false);
  printRepeatedTimingStat(F("Breakdown total"), TIMING_TOTAL, true);
  Serial.print(F("Last predicted class index: "));
  Serial.println(lastPred);
  Serial.print(F("Last predicted class label: "));
  loadClassName(lastPred, labelBuffer, sizeof(labelBuffer));
  Serial.println(labelBuffer);
  Serial.print(F("Last confidence: "));
  Serial.println(probabilities[lastPred], 6);
}

void setup() {
  setupRelays();

  Serial.begin(115200);
  while (!Serial) {
    ;
  }

  printProgmemString(MODEL_NAME);
  Serial.println(F(" on-board inference"));
  Serial.print(F("Selected run: "));
  Serial.println(SELECTED_RUN);
  Serial.print(F("Test accuracy (%): "));
  Serial.println(SELECTED_TEST_ACCURACY, 4);
  Serial.print(F("Macro F1 (%): "));
  Serial.println(SELECTED_MACRO_F1, 4);
  Serial.println(F("Send 'g' to start one 400-sample grasp window."));
  Serial.println(F("Send 'b' to benchmark the flash-stored 400-sample window."));
  Serial.println(F("Send 't' to print a task-level timing breakdown."));
  Serial.println(F("Send 'm' to run the flash benchmark 100 times."));
  Serial.println(F("Send 'd' to repeat the timing breakdown 100 times."));
  Serial.println(F("Send 'r' to release all relays."));
}

void loop() {
  if (Serial.available() <= 0) {
    return;
  }

  const char command = Serial.read();

  if (command == 'r' || command == 'R') {
    setAllRelays(false);
    Serial.println(F("Relays released."));
    return;
  }

  if (command == 'b' || command == 'B') {
    setAllRelays(false);
    unsigned long elapsedUs = 0;
    Serial.println(F("Start flash replay benchmark..."));
    const uint8_t pred = runFlashReplayBenchmark(probabilities, &elapsedUs);

    Serial.print(F("Inference time total (us): "));
    Serial.println(elapsedUs);
    Serial.print(F("Inference time per step (us): "));
    Serial.println(static_cast<float>(elapsedUs) / static_cast<float>(SEQ_LEN), 3);
    Serial.print(F("Predicted class index: "));
    Serial.println(pred);
    Serial.print(F("Predicted class label: "));
    loadClassName(pred, labelBuffer, sizeof(labelBuffer));
    Serial.println(labelBuffer);
    Serial.print(F("Confidence: "));
    Serial.println(probabilities[pred], 6);
    Serial.println(F("Done."));
    return;
  }

  if (command == 't' || command == 'T') {
    setAllRelays(false);
    Serial.println(F("Start flash replay timing breakdown..."));
    const uint8_t pred = runFlashReplayTimingBreakdown(probabilities);

    Serial.println(F("Timing breakdown includes measurement overhead."));
    printTimingLine(F("State reset"), timingBreakdown.resetUs, false);
    printTimingLine(F("Flash read"), timingBreakdown.flashReadUs, true);
    printTimingLine(F("Z-score normalize"), timingBreakdown.normalizeUs, true);
    printTimingLine(F("LTC Euler update"), timingBreakdown.modelStepUs, true);
    printTimingLine(F("Dense + softmax"), timingBreakdown.denseSoftmaxUs, false);
    printTimingLine(F("Argmax"), timingBreakdown.argmaxUs, false);
    printTimingLine(F("Breakdown total"), timingBreakdown.totalUs, true);
    Serial.print(F("Predicted class index: "));
    Serial.println(pred);
    Serial.print(F("Predicted class label: "));
    loadClassName(pred, labelBuffer, sizeof(labelBuffer));
    Serial.println(labelBuffer);
    Serial.print(F("Confidence: "));
    Serial.println(probabilities[pred], 6);
    Serial.println(F("Done."));
    return;
  }

  if (command == 'm' || command == 'M') {
    setAllRelays(false);
    Serial.println(F("Start repeated flash replay benchmark..."));
    runRepeatedFlashBenchmark();
    Serial.println(F("Done."));
    return;
  }

  if (command == 'd' || command == 'D') {
    setAllRelays(false);
    Serial.println(F("Start repeated timing breakdown..."));
    Serial.println(F("Timing breakdown includes measurement overhead."));
    runRepeatedTimingBreakdown();
    Serial.println(F("Done."));
    return;
  }

  if (command != 'g' && command != 'G') {
    return;
  }

  Serial.println(F("Start grasp window..."));
  const uint8_t pred = runOneInferenceWindow(probabilities);

  if (RELEASE_AFTER_INFERENCE) {
    setAllRelays(false);
  }

  Serial.print(F("Predicted class index: "));
  Serial.println(pred);
  Serial.print(F("Predicted class label: "));
  loadClassName(pred, labelBuffer, sizeof(labelBuffer));
  Serial.println(labelBuffer);
  Serial.println(F("Probabilities:"));
  printProbabilities(probabilities);
  Serial.println(F("Done."));
}
""",
        ]
    )


def simple_rnn_code(recurrent, dense, model_name, row, columns, class_names):
    kernel, recurrent_kernel, bias = recurrent.get_weights()
    dense_w, dense_b = dense.get_weights()
    arrays = "\n\n".join(
        [
            format_matrix("RNN_KERNEL", kernel),
            format_matrix("RNN_RECURRENT", recurrent_kernel),
            format_vector("RNN_BIAS", bias),
            format_matrix("DENSE_W", dense_w),
            format_vector("DENSE_B", dense_b),
        ]
    )
    state_step = r"""float hiddenState[HIDDEN_UNITS];

void resetModelState() {
  for (uint8_t u = 0; u < HIDDEN_UNITS; ++u) {
    hiddenState[u] = 0.0f;
  }
}

void modelStep(const float inputZ[INPUT_DIM]) {
  float nextState[HIDDEN_UNITS];

  for (uint8_t u = 0; u < HIDDEN_UNITS; ++u) {
    float value = readFloat(&RNN_BIAS[u]);
    for (uint8_t i = 0; i < INPUT_DIM; ++i) {
      value += inputZ[i] * readFloat(&RNN_KERNEL[i][u]);
    }
    for (uint8_t j = 0; j < HIDDEN_UNITS; ++j) {
      value += hiddenState[j] * readFloat(&RNN_RECURRENT[j][u]);
    }
    nextState[u] = tanhf(value);
  }

  for (uint8_t u = 0; u < HIDDEN_UNITS; ++u) {
    hiddenState[u] = nextState[u];
  }
}"""
    return common_code(
        model_name, row, columns, class_names, int(recurrent.units), arrays, state_step
    )


def lstm_code(recurrent, dense, model_name, row, columns, class_names):
    kernel, recurrent_kernel, bias = recurrent.get_weights()
    dense_w, dense_b = dense.get_weights()
    arrays = "\n\n".join(
        [
            format_matrix("LSTM_KERNEL", kernel),
            format_matrix("LSTM_RECURRENT", recurrent_kernel),
            format_vector("LSTM_BIAS", bias),
            format_matrix("DENSE_W", dense_w),
            format_vector("DENSE_B", dense_b),
        ]
    )
    state_step = r"""float hiddenState[HIDDEN_UNITS];
float cellState[HIDDEN_UNITS];

void resetModelState() {
  for (uint8_t u = 0; u < HIDDEN_UNITS; ++u) {
    hiddenState[u] = 0.0f;
    cellState[u] = 0.0f;
  }
}

void modelStep(const float inputZ[INPUT_DIM]) {
  float gates[4 * HIDDEN_UNITS];

  for (uint8_t k = 0; k < 4 * HIDDEN_UNITS; ++k) {
    float value = readFloat(&LSTM_BIAS[k]);
    for (uint8_t i = 0; i < INPUT_DIM; ++i) {
      value += inputZ[i] * readFloat(&LSTM_KERNEL[i][k]);
    }
    for (uint8_t j = 0; j < HIDDEN_UNITS; ++j) {
      value += hiddenState[j] * readFloat(&LSTM_RECURRENT[j][k]);
    }
    gates[k] = value;
  }

  for (uint8_t u = 0; u < HIDDEN_UNITS; ++u) {
    const float inputGate = sigmoidClamped(gates[u]);
    const float forgetGate = sigmoidClamped(gates[HIDDEN_UNITS + u]);
    const float cellCandidate = tanhf(gates[2 * HIDDEN_UNITS + u]);
    const float outputGate = sigmoidClamped(gates[3 * HIDDEN_UNITS + u]);

    cellState[u] = forgetGate * cellState[u] + inputGate * cellCandidate;
    hiddenState[u] = outputGate * tanhf(cellState[u]);
  }
}"""
    return common_code(
        model_name, row, columns, class_names, int(recurrent.units), arrays, state_step
    )


def ltc_code(recurrent, dense, model_name, row, columns, class_names):
    ltc_w, ltc_r, ltc_mu = recurrent.get_weights()
    dense_w, dense_b = dense.get_weights()
    arrays = "\n\n".join(
        [
            format_matrix("LTC_W", ltc_w),
            format_matrix("LTC_R", ltc_r),
            format_matrix("LTC_MU", ltc_mu),
            format_matrix("DENSE_W", dense_w),
            format_vector("DENSE_B", dense_b),
        ]
    )
    state_step = r"""constexpr float DELTA_T = 0.01f;
float hiddenState[HIDDEN_UNITS];

void resetModelState() {
  for (uint8_t u = 0; u < HIDDEN_UNITS; ++u) {
    hiddenState[u] = 0.0f;
  }
}

void modelStep(const float inputZ[INPUT_DIM]) {
  float nextState[HIDDEN_UNITS];

  for (uint8_t u = 0; u < HIDDEN_UNITS; ++u) {
    float damping = 1.0f;
    float driving = 0.0f;

    for (uint8_t i = 0; i < INPUT_DIM; ++i) {
      const float w = readFloat(&LTC_W[i][u]);
      const float r = readFloat(&LTC_R[i][u]);
      const float mu = readFloat(&LTC_MU[i][u]);
      const float sigma = sigmoidClamped(inputZ[i] * r + mu);
      damping += fabsf(w) * sigma;
      driving += w * sigma;
    }

    const float dx = -damping * hiddenState[u] + driving;
    nextState[u] = hiddenState[u] + DELTA_T * dx;
  }

  for (uint8_t u = 0; u < HIDDEN_UNITS; ++u) {
    hiddenState[u] = nextState[u];
  }
}"""
    return common_code(
        model_name, row, columns, class_names, int(recurrent.cell.units), arrays, state_step
    )


def export_arduino_code(best_rows, dataset, arduino_dir):
    outputs = []
    custom_objects = {"LTCNeuron": bench.LTCNeuron}
    columns = dataset["columns"]
    class_names = dataset["class_names"]

    for _, row in best_rows.iterrows():
        model_name = row["Model"]
        model_path = Path(row["ModelPath"])
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        recurrent, dense = find_recurrent_and_dense(model)

        if model_name.startswith("SimpleRNN"):
            code = simple_rnn_code(recurrent, dense, model_name, row, columns, class_names)
        elif model_name.startswith("LSTM"):
            code = lstm_code(recurrent, dense, model_name, row, columns, class_names)
        elif model_name.startswith("LTC"):
            code = ltc_code(recurrent, dense, model_name, row, columns, class_names)
        else:
            raise ValueError(f"Unsupported model for Arduino export: {model_name}")

        sketch_name = model_name.lower().replace("-", "") + "_best"
        sketch_dir = arduino_dir / sketch_name
        sketch_dir.mkdir(parents=True, exist_ok=True)
        sketch_path = sketch_dir / f"{sketch_name}.ino"
        sketch_path.write_text(code, encoding="utf-8")
        flash_header_path = sketch_dir / "flash_sequence.h"
        if not flash_header_path.exists():
            flash_header_path.write_text(default_flash_sequence_text(columns), encoding="utf-8")
        outputs.append(sketch_path)

    return outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train LTC-4, LSTM-8, and SimpleRNN-8 deployment candidates, "
            "select the highest-accuracy run, and export Arduino sketches."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--arduino-dir", type=Path, default=DEFAULT_ARDUINO_DIR)
    parser.add_argument("--channel", choices=["5ch", "3ch"], default="5ch")
    parser.add_argument("--models", nargs="+", default=TARGET_MODELS)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--clipnorm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--jit-compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.arduino_dir.mkdir(parents=True, exist_ok=True)

    raw_path = args.out_dir / f"deployment_candidates_{args.channel}_raw.csv"
    best_path = args.out_dir / f"deployment_candidates_{args.channel}_best.csv"

    dataset = bench.load_channel_dataset(args.dataset, args.channel)
    print(
        f"[{args.channel}] samples: train={len(dataset['x_train'])}, "
        f"val={len(dataset['x_val'])}, test={len(dataset['x_test'])}, "
        f"classes={dataset['num_classes']}"
    )
    print("Class order:", ", ".join(display_class_names(dataset["class_names"])))

    if not args.export_only:
        done = existing_keys(raw_path) if args.resume else set()

        for model_name in args.models:
            for run_index in range(1, args.runs + 1):
                key = (args.channel, model_name, run_index)
                if key in done:
                    print(f"Skip {args.channel} {model_name} run {run_index:02d}")
                    continue

                print(f"Run {args.channel} {model_name} {run_index:02d}/{args.runs} ...", end=" ")
                row = train_one(dataset, args.channel, model_name, run_index, args)
                append_result(raw_path, row)
                print(
                    f"Acc={row['Accuracy(%)']:.2f}%, "
                    f"F1={row['Macro-F1(%)']:.2f}%, "
                    f"epochs={row['EpochsTrained']}"
                )

    if not raw_path.exists():
        raise FileNotFoundError(f"No raw results found: {raw_path}")

    best_rows = select_best_models(raw_path, best_path, args.channel, args.models)
    print(f"Saved best-run table: {best_path}")
    for _, row in best_rows.iterrows():
        print(
            f"Best {row['Model']}: run {int(row['Run'])}, "
            f"Acc={row['Accuracy(%)']:.2f}%, F1={row['Macro-F1(%)']:.2f}%, "
            f"params={int(row['TrainableParams'])}"
        )

    if not args.no_export:
        sketches = export_arduino_code(best_rows, dataset, args.arduino_dir)
        for sketch in sketches:
            print(f"Saved Arduino sketch: {sketch}")


if __name__ == "__main__":
    main()

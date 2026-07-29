#include <Arduino.h>
#include <math.h>
#include <avr/pgmspace.h>
#include <string.h>

// Trained model path: C:\\Users\\HAO\\Desktop\\YTY_from_macbook\\LTCRNN\\BPTT\\deployment_candidates\\checkpoints\\LTC-4\\LTC-4_5ch_run02.keras
const char MODEL_NAME[] PROGMEM = "LTC-4";
constexpr uint8_t INPUT_DIM = 5;
constexpr uint8_t HIDDEN_UNITS = 4;
constexpr uint8_t NUM_CLASSES = 10;
constexpr uint16_t SEQ_LEN = 400;
constexpr uint16_t SAMPLE_INTERVAL_MS = 10;
constexpr uint16_t RELAY_ON_DELAY_MS = 1500;
constexpr uint16_t BENCHMARK_REPEAT_COUNT = 100;
constexpr uint8_t TIMING_TASK_COUNT = 7;
constexpr uint8_t TIMING_RESET = 0;
constexpr uint8_t TIMING_FLASH_READ = 1;
constexpr uint8_t TIMING_NORMALIZE = 2;
constexpr uint8_t TIMING_MODEL_STEP = 3;
constexpr uint8_t TIMING_DENSE_SOFTMAX = 4;
constexpr uint8_t TIMING_ARGMAX = 5;
constexpr uint8_t TIMING_TOTAL = 6;
constexpr bool RELEASE_AFTER_INFERENCE = true;
constexpr uint8_t RELAY_ON_LEVEL = LOW;
constexpr uint8_t RELAY_OFF_LEVEL = HIGH;
constexpr uint8_t SELECTED_RUN = 2;
constexpr float SELECTED_TEST_ACCURACY = 91.000003f;
constexpr float SELECTED_MACRO_F1 = 90.933781f;

#include "flash_sequence.h"

const uint8_t SENSOR_PINS[INPUT_DIM] = {A0, A1, A2, A3, A4};
const uint8_t RELAY_PINS[INPUT_DIM] = {2, 3, 4, 5, 6};

// z = (raw_adc - training_mean) / training_std
const float SENSOR_MEAN[5] PROGMEM = {
   795.53795191f,  784.61776534f,  801.80461028f,  759.27079602f,  840.59891376f
};

const float SENSOR_STD[5] PROGMEM = {
   62.00574361f,  50.61546292f,  39.76813518f,  39.13091513f,  40.79949670f
};

const char CLASS_0[] PROGMEM = "Baseball";
const char CLASS_1[] PROGMEM = "Bottle";
const char CLASS_2[] PROGMEM = "Sponge Dice";
const char CLASS_3[] PROGMEM = "Tape";
const char CLASS_4[] PROGMEM = "Plush Toy";
const char CLASS_5[] PROGMEM = "Optical Mouse";
const char CLASS_6[] PROGMEM = "Smartphone";
const char CLASS_7[] PROGMEM = "Rubik's Cube";
const char CLASS_8[] PROGMEM = "Stuffed Ball";
const char CLASS_9[] PROGMEM = "3D-Printed Part";
const char *const CLASS_NAMES[NUM_CLASSES] PROGMEM = {
  CLASS_0,
  CLASS_1,
  CLASS_2,
  CLASS_3,
  CLASS_4,
  CLASS_5,
  CLASS_6,
  CLASS_7,
  CLASS_8,
  CLASS_9
};

const float LTC_W[5][4] PROGMEM = {
  { 4.93142033f, -7.18577290f, -10.52399540f, -8.91330147f},
  {-7.11571836f, -1.66463006f, -0.23919778f, -2.02559876f},
  { 0.38490209f, -18.88042831f,  10.43723297f,  30.36274719f},
  { 0.01425037f,  11.36738396f, -6.79484177f, -19.66677856f},
  { 0.53132248f, -0.00103223f,  2.29109120f, -48.87559509f}
};

const float LTC_R[5][4] PROGMEM = {
  { 11.67281818f,  19.83379555f,  8.37863541f, -9.60912132f},
  { 2.35052085f, -2.84820342f, -23.93206978f,  11.55571651f},
  {-0.10304331f, -11.93898010f,  7.90272713f,  8.86442471f},
  {-0.90995282f,  5.26199150f, -6.31933880f,  5.49018192f},
  {-4.73187160f, -2.13582301f, -4.58517170f, -29.38112640f}
};

const float LTC_MU[5][4] PROGMEM = {
  {-10.13881493f, -8.30846882f, -11.19579983f,  7.04399204f},
  {-3.08786154f, -12.08648109f, -8.25056076f, -7.73417807f},
  {-14.08142281f, -13.45602226f, -6.87218714f, -11.44804382f},
  {-14.43230343f, -6.32752228f, -1.01304400f, -26.56407928f},
  { 5.75351048f, -12.19529343f, -3.02892375f,  2.94893193f}
};

const float DENSE_W[4][10] PROGMEM = {
  {-0.09151290f, -15.26660442f, -7.71976185f, -21.79019547f,  1.85485017f,  21.33733749f, -34.84183121f, -23.25146484f,  18.88271523f,  3.62421036f},
  {-2.67621732f,  15.61618614f, -8.26868725f,  16.67878342f,  16.72279167f, -8.95269108f,  4.12242270f, -13.65349007f,  5.44837332f,  3.40011239f},
  {-8.82331753f,  31.80079460f,  5.91204071f, -66.06320953f,  20.75394440f, -0.22975010f,  5.94428635f,  16.40366745f, -12.61448193f,  9.27567577f},
  {-4.65786028f,  8.27192783f, -10.75760555f, -7.27414894f,  15.29831696f,  33.98694992f,  47.93236542f, -11.04071712f, -18.82514191f,  4.87954140f}
};

const float DENSE_B[10] PROGMEM = {
   10.11334038f, -11.04694557f,  1.29809403f, -4.95734930f,  2.16634989f, -3.49991393f, -7.33320951f, -12.06916046f, -4.10922146f,  10.99566841f
};

float readFloat(const float *address) {
  return pgm_read_float(address);
}

float sigmoidClamped(float x);


constexpr float DELTA_T = 0.01f;
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
}

float probabilities[NUM_CLASSES];
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
  digitalWrite(RELAY_PINS[relayIndex], on ? RELAY_ON_LEVEL : RELAY_OFF_LEVEL);
}

void setAllRelays(bool on) {
  for (uint8_t i = 0; i < INPUT_DIM; ++i) {
    setRelay(i, on);
  }
}

void setupRelays() {
  for (uint8_t i = 0; i < INPUT_DIM; ++i) {
    digitalWrite(RELAY_PINS[i], RELAY_OFF_LEVEL);
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

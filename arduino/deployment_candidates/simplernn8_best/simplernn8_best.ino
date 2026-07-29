#include <Arduino.h>
#include <math.h>
#include <avr/pgmspace.h>
#include <string.h>

// Trained model path: C:\\Users\\HAO\\Desktop\\YTY_from_macbook\\LTCRNN\\BPTT\\deployment_candidates\\checkpoints\\SimpleRNN-8\\SimpleRNN-8_5ch_run06.keras
const char MODEL_NAME[] PROGMEM = "SimpleRNN-8";
constexpr uint8_t INPUT_DIM = 5;
constexpr uint8_t HIDDEN_UNITS = 8;
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
constexpr bool RELAY_ACTIVE_LOW = true;
constexpr uint8_t SELECTED_RUN = 6;
constexpr float SELECTED_TEST_ACCURACY = 92.000002f;
constexpr float SELECTED_MACRO_F1 = 91.856100f;

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

const float RNN_KERNEL[5][8] PROGMEM = {
  { 3.19924998f,  0.91191143f,  2.29862690f,  4.30356312f, -5.12825871f, -0.81159484f,  3.22898412f, -1.69665980f},
  {-3.18155861f,  1.03134513f, -1.56349516f, -1.31815350f,  0.28290269f, -0.07029435f,  1.13623309f,  2.47764921f},
  { 0.61093187f, -3.42484498f,  0.70484012f, -2.76536870f, -0.25349474f,  2.51318002f, -3.96192980f,  4.96864128f},
  {-2.14768767f, -1.31616342f,  0.84779888f, -0.62803513f,  1.78738129f,  3.62704921f, -0.10999674f, -2.71016073f},
  { 0.32247058f, -1.29813886f, -3.42000270f,  0.71154737f,  0.34493050f, -1.60501099f,  1.30430079f, -0.48194712f}
};

const float RNN_RECURRENT[8][8] PROGMEM = {
  { 0.76131374f,  0.52176291f,  0.07321552f, -0.69065386f, -0.38215467f, -0.09836037f,  0.06868856f, -0.44970405f},
  { 0.36082014f,  0.69068128f, -1.35445511f, -2.56574273f, -0.32923818f,  0.17934227f, -0.77533931f, -2.15813804f},
  { 2.47283483f,  0.74355143f, -0.25533134f,  0.69144160f, -0.57483953f, -0.16160844f, -0.26401800f, -0.02519700f},
  {-0.06592466f, -0.67428613f,  1.16234708f,  1.07871258f,  0.12490850f, -0.36093244f, -0.04165267f, -0.16423619f},
  {-0.72171259f, -1.96282101f, -1.68969691f, -1.39566040f,  0.75806206f, -0.29303440f,  1.21786261f,  0.63221562f},
  {-0.15765245f, -2.57927322f,  2.34956384f, -0.52061272f,  0.64527142f, -0.22888337f,  0.55033922f,  0.01474419f},
  { 0.27528533f,  0.13381220f, -0.19230117f,  1.01208806f, -0.13467970f,  0.32695451f,  0.67825776f, -1.35618174f},
  { 0.47491407f, -2.15335512f,  0.65102136f, -0.66141754f,  1.28036380f, -0.37326324f,  0.74386823f,  0.48013201f}
};

const float RNN_BIAS[8] PROGMEM = {
  -1.37772644f,  3.93958259f,  1.91716218f, -0.37425640f,  1.10153961f, -3.14319611f, -1.22509789f, -0.26050836f
};

const float DENSE_W[8][10] PROGMEM = {
  {-1.68454826f,  1.35182345f, -2.65644169f,  0.64882630f,  3.55895090f,  3.97414851f, -0.30854458f, -1.25012100f,  0.72088665f, -0.07965399f},
  { 1.57077360f, -1.26199055f,  0.22092241f,  2.85866547f, -2.30679822f, -4.83727217f, -1.29192817f,  2.67358446f,  0.92808282f, -0.85645229f},
  {-0.35236719f, -3.28597021f, -0.88468403f, -3.50143003f,  1.56694889f,  0.52427649f,  5.03625870f, -1.30556631f,  1.34867752f,  0.20345829f},
  { 0.76968956f, -3.39625072f, -1.35787904f,  2.48728061f,  2.10136127f,  2.32154012f, -1.99478304f, -0.07165312f,  0.08610857f, -0.59215999f},
  {-3.17581677f,  4.12736893f, -3.02628827f, -0.11702964f,  2.94905210f, -0.99385846f,  7.64051962f,  1.13988268f,  2.74975467f, -3.15052152f},
  {-1.95395565f,  6.39262056f, -4.05278254f,  0.99651802f,  7.08668613f,  3.06018972f,  5.83884001f, -4.93780231f, -2.02260971f,  0.35184994f},
  {-0.31116116f,  1.67407870f,  0.42443880f,  8.41784286f, -2.06077218f, -1.85180008f, -7.47453403f,  0.47050989f, -0.93904114f, -0.53410935f},
  {-2.18429089f,  1.74830437f,  2.63598061f, -5.43345881f,  1.44403350f,  1.60523677f, -0.97026676f,  4.16313648f, -4.11709738f,  5.38270330f}
};

const float DENSE_B[10] PROGMEM = {
   0.52266920f, -2.97507834f,  1.48788464f, -1.21534765f, -2.11312413f, -1.83987927f, -9.58500195f,  0.52965087f,  3.32995820f, -0.70533651f
};

float readFloat(const float *address) {
  return pgm_read_float(address);
}

float sigmoidClamped(float x);


float hiddenState[HIDDEN_UNITS];

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

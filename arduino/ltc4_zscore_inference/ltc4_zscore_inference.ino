#include <Arduino.h>
#include <math.h>

// LTC-4 Arduino inference template.
// Preprocessing matches z_score.py:
//   z = (raw_adc - training_mean) / training_std
// The mean/std below were fitted on the valid training waveforms in:
//   dataset_xx2020_new_new_new/dataset_602020/training
//
// Important:
// - Do not enable baseline subtraction here unless the model was trained with it.
// - Replace all weight arrays with one trained LTC-4 checkpoint before use.

constexpr uint8_t INPUT_DIM = 5;
constexpr uint8_t LTC_UNITS = 4;
constexpr uint8_t NUM_CLASSES = 10;
constexpr uint16_t SEQ_LEN = 400;

// Keras LTCNeuron used delta_t = 0.01.
constexpr float DELTA_T = 0.01f;

// 400 samples over about 4 seconds -> 100 Hz.
constexpr uint16_t SAMPLE_INTERVAL_MS = 10;

const uint8_t SENSOR_PINS[INPUT_DIM] = {A0, A1, A2, A3, A4};

// Relay pins for Thumb, Index, Middle, Ring, Pinky.
// Many 5 V relay modules are active-low. If your module is active-high,
// change RELAY_ACTIVE_LOW to false.
const uint8_t RELAY_PINS[INPUT_DIM] = {2, 3, 4, 5, 6};
constexpr bool RELAY_ACTIVE_LOW = true;

// Start sampling first, then actuate the relays after this delay.
// 1.5 s at 10 ms/sample gives 150 pre-grasp samples in the 400-point window.
constexpr uint16_t RELAY_ON_DELAY_MS = 1500;
constexpr bool RELEASE_AFTER_INFERENCE = true;

// Order: Thumb, Index, Middle, Ring, Pinky.
const float SENSOR_MEAN[INPUT_DIM] = {
  795.53795191f,
  784.61776534f,
  801.80461028f,
  759.27079602f,
  840.59891376f
};

const float SENSOR_STD[INPUT_DIM] = {
  62.00574361f,
  50.61546292f,
  39.76813518f,
  39.13091513f,
  40.79949670f
};

// LabelEncoder order from the dataset file names.
// PPT-friendly mapping, if needed:
// ball -> Baseball, cube -> Sponge Dice, cylinder -> Tape,
// doll -> Plush Toy, mouse -> Optical Mouse, phone -> Smartphone,
// small_ball -> Stuffed Ball, support -> 3D-Printed Part.
const char *CLASS_NAMES[NUM_CLASSES] = {
  "Baseball",
  "Bottle",
  "Sponge Dice",
  "Tape",
  "Plush Toy",
  "Optical Mouse",
  "Smartphone",
  "Rubik's Cube",
  "Stuffed Ball",
  "3D-Printed Part"
};

// TODO: Replace these placeholders with trained LTC-4 weights.
// Shape follows TensorFlow/Keras:
//   ltc_w[input_channel][ltc_unit]
//   ltc_r[input_channel][ltc_unit]
//   ltc_mu[input_channel][ltc_unit]
const float LTC_W[INPUT_DIM][LTC_UNITS] = {
  {0.0f, 0.0f, 0.0f, 0.0f},
  {0.0f, 0.0f, 0.0f, 0.0f},
  {0.0f, 0.0f, 0.0f, 0.0f},
  {0.0f, 0.0f, 0.0f, 0.0f},
  {0.0f, 0.0f, 0.0f, 0.0f}
};

const float LTC_R[INPUT_DIM][LTC_UNITS] = {
  {1.0f, 1.0f, 1.0f, 1.0f},
  {1.0f, 1.0f, 1.0f, 1.0f},
  {1.0f, 1.0f, 1.0f, 1.0f},
  {1.0f, 1.0f, 1.0f, 1.0f},
  {1.0f, 1.0f, 1.0f, 1.0f}
};

const float LTC_MU[INPUT_DIM][LTC_UNITS] = {
  {0.0f, 0.0f, 0.0f, 0.0f},
  {0.0f, 0.0f, 0.0f, 0.0f},
  {0.0f, 0.0f, 0.0f, 0.0f},
  {0.0f, 0.0f, 0.0f, 0.0f},
  {0.0f, 0.0f, 0.0f, 0.0f}
};

// Dense layer after LTC:
//   logits[class] = dense_b[class] + sum_u state[u] * dense_w[u][class]
const float DENSE_W[LTC_UNITS][NUM_CLASSES] = {
  {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
  {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
  {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
  {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
};

const float DENSE_B[NUM_CLASSES] = {
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

float ltcState[LTC_UNITS];
float probabilities[NUM_CLASSES];

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

void resetLTCState() {
  for (uint8_t u = 0; u < LTC_UNITS; ++u) {
    ltcState[u] = 0.0f;
  }
}

void readRawSensors(float raw[INPUT_DIM]) {
  for (uint8_t i = 0; i < INPUT_DIM; ++i) {
    raw[i] = static_cast<float>(analogRead(SENSOR_PINS[i]));
  }
}

void zScoreNormalize(const float raw[INPUT_DIM], float z[INPUT_DIM]) {
  for (uint8_t i = 0; i < INPUT_DIM; ++i) {
    z[i] = (raw[i] - SENSOR_MEAN[i]) / SENSOR_STD[i];
  }
}

void ltcEulerStep(const float inputZ[INPUT_DIM]) {
  float nextState[LTC_UNITS];

  for (uint8_t u = 0; u < LTC_UNITS; ++u) {
    float damping = 1.0f;
    float driving = 0.0f;

    for (uint8_t i = 0; i < INPUT_DIM; ++i) {
      const float sigma = sigmoidClamped(inputZ[i] * LTC_R[i][u] + LTC_MU[i][u]);
      damping += fabsf(LTC_W[i][u]) * sigma;
      driving += LTC_W[i][u] * sigma;
    }

    const float dx = -damping * ltcState[u] + driving;
    nextState[u] = ltcState[u] + DELTA_T * dx;
  }

  for (uint8_t u = 0; u < LTC_UNITS; ++u) {
    ltcState[u] = nextState[u];
  }
}

void denseSoftmax(const float state[LTC_UNITS], float probs[NUM_CLASSES]) {
  float logits[NUM_CLASSES];
  float maxLogit = -3.4028235e38f;

  for (uint8_t c = 0; c < NUM_CLASSES; ++c) {
    float logit = DENSE_B[c];
    for (uint8_t u = 0; u < LTC_UNITS; ++u) {
      logit += state[u] * DENSE_W[u][c];
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
  resetLTCState();
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
    ltcEulerStep(z);

    const unsigned long elapsed = millis() - startMs;
    if (elapsed < SAMPLE_INTERVAL_MS) {
      delay(SAMPLE_INTERVAL_MS - elapsed);
    }
  }

  denseSoftmax(ltcState, probs);
  return argmaxClass(probs);
}

void printProbabilities(const float probs[NUM_CLASSES]) {
  for (uint8_t c = 0; c < NUM_CLASSES; ++c) {
    Serial.print(CLASS_NAMES[c]);
    Serial.print(": ");
    Serial.println(probs[c], 6);
  }
}

void setup() {
  setupRelays();

  Serial.begin(115200);
  while (!Serial) {
    ; // Needed by boards with native USB. Safe on boards that ignore it.
  }

  Serial.println("LTC-4 z-score inference template");
  Serial.println("Send 'g' in Serial Monitor to start one 400-sample grasp window.");
  Serial.println("Send 'r' to release all relays.");
  Serial.println("Reminder: replace placeholder weights before real inference.");
}

void loop() {
  if (Serial.available() <= 0) {
    return;
  }

  const char command = Serial.read();

  if (command == 'r' || command == 'R') {
    setAllRelays(false);
    Serial.println("Relays released.");
    return;
  }

  if (command != 'g' && command != 'G') {
    return;
  }

  Serial.println("Start grasp window...");
  const uint8_t pred = runOneInferenceWindow(probabilities);

  if (RELEASE_AFTER_INFERENCE) {
    setAllRelays(false);
  }

  Serial.print("Predicted class index: ");
  Serial.println(pred);
  Serial.print("Predicted class label: ");
  Serial.println(CLASS_NAMES[pred]);
  Serial.println("Probabilities:");
  printProbabilities(probabilities);
  Serial.println("Done.");
}

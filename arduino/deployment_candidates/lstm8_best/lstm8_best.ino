#include <Arduino.h>
#include <math.h>
#include <avr/pgmspace.h>
#include <string.h>

// Trained model path: C:\\Users\\HAO\\Desktop\\YTY_from_macbook\\LTCRNN\\BPTT\\deployment_candidates\\checkpoints\\LSTM-8\\LSTM-8_5ch_run10.keras
const char MODEL_NAME[] PROGMEM = "LSTM-8";
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
constexpr uint8_t SELECTED_RUN = 10;
constexpr float SELECTED_TEST_ACCURACY = 96.499997f;
constexpr float SELECTED_MACRO_F1 = 96.496868f;

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

const float LSTM_KERNEL[5][32] PROGMEM = {
  { 0.49916452f, -2.83661509f,  0.51393330f, -3.27619076f,  2.98265982f, -2.64473820f, -2.23858953f,  4.97866011f,  1.17685163f,  1.79498541f, -0.72805458f, -2.52315140f,  0.05857633f, -1.80491436f, -0.82885081f,  1.45444012f, -0.79074776f, -4.62030745f,  3.60583425f, -2.31781578f, -2.38446999f,  4.00076008f, -2.02612948f, -0.44102588f, -0.59623647f, -5.15220785f,  2.79581809f,  1.70948577f, -1.09869754f,  0.86590493f,  5.94250154f,  3.33926487f},
  { 1.99462569f,  0.14706309f,  0.93808520f,  0.33081290f,  0.12287296f,  0.84300411f, -0.79212213f,  1.92583454f,  1.01936173f,  3.88397598f, -0.71273226f,  0.31664503f, -1.47363508f,  1.39169300f,  0.46955612f,  0.97936863f,  1.23800802f,  1.22636998f, -0.55505985f,  0.11234551f,  0.60185313f, -0.76347375f,  3.65097356f,  1.23871005f,  2.18586516f,  0.67219472f,  1.01391947f,  0.25541615f, -1.28924155f, -1.85965788f,  1.06241179f,  2.09171629f},
  {-0.26259580f,  1.68103063f,  6.79524279f,  2.31025648f, -2.81238890f,  1.75285399f,  0.97879130f,  4.80981588f, -0.86853892f,  0.96532971f, -2.38246536f,  3.01487470f, -2.27237201f,  0.31461951f,  0.49979371f, -1.36752021f, -4.66034937f,  0.84917557f, -1.32516289f, -0.80563563f,  4.32358503f, -0.23905645f,  2.31520796f, -4.75881338f,  2.78746009f,  0.82807744f,  0.04242478f,  1.75509202f,  0.01826843f, -2.01597619f,  3.41540742f, -1.07355285f},
  {-0.20733917f,  0.25397715f, -0.07086153f, -3.73994565f,  0.87671548f, -1.15101898f,  1.28049064f,  1.22970474f,  0.43609571f,  2.96827888f,  1.08251989f, -2.40365839f, -0.40341887f,  1.58646894f,  2.28012490f,  0.55294943f,  0.49353939f,  3.12154269f,  0.25922269f,  3.65998244f, -0.95795530f,  0.45789179f, -2.63031006f,  0.07231794f,  1.88909280f,  2.03351068f,  3.66426945f,  3.45045495f,  2.77658272f, -0.56518757f,  2.15893316f,  1.79613996f},
  { 0.85325998f,  1.90351880f,  2.21539307f,  3.15834975f,  0.26100254f, -1.37570369f, -0.94755334f,  0.77178508f,  1.19299209f,  2.50161839f, -0.14893173f,  3.13389516f, -1.04236472f,  0.50987524f, -0.60371310f, -2.31629777f,  1.44402027f, -0.84554702f,  0.69816649f,  1.48469126f,  0.79148000f, -2.32341385f, -0.28669241f,  0.68813682f,  2.64856982f,  0.01488386f, -0.42623532f, -1.26036882f,  0.56492364f,  2.47501755f,  3.20788050f, -1.31677675f}
};

const float LSTM_RECURRENT[8][32] PROGMEM = {
  {-0.64957249f, -0.36222568f,  0.91367686f, -0.07271925f,  4.98028421f, -1.68530488f, -0.97239882f,  0.63783526f, -0.92138112f,  1.67377579f,  3.73648000f, -0.75494188f,  0.92741632f,  0.90431261f,  0.21586151f,  0.67777991f, -0.51305628f, -0.44113177f,  1.48623931f,  2.65078354f, -1.99797630f,  0.53752244f,  1.00666189f,  3.20815301f,  1.20240176f, -2.13787508f,  3.80909467f,  0.70660043f,  1.05022502f,  6.85078382f, -3.32635498f,  1.63568294f},
  { 2.44011426f, -0.69180226f, -0.14351955f, -1.28517318f,  0.30836263f, -0.03578817f,  1.10332668f,  1.16317296f, -0.48761374f, -0.13647874f, -0.76497352f, -0.17302655f,  0.54439783f, -4.91115665f,  1.48807597f,  0.60529065f, -0.22560306f, -0.51496899f, -2.05737805f,  1.08481085f,  0.99874097f,  0.31273749f, -0.69042438f,  0.89613080f, -0.10397848f, -1.31225514f,  1.03643703f, -1.74899209f, -1.88638866f, -3.12179065f, -4.22216988f,  3.34344268f},
  {-1.35806918f, -2.09864068f, -0.46470565f, -0.09481605f,  0.14504503f, -1.68748558f, -2.48543382f,  1.74444234f,  2.06620598f,  1.06080985f, -1.26704931f,  0.94902575f,  0.07312445f,  1.73603821f,  1.97146237f,  0.92565304f, -0.00286423f,  0.59805125f, -0.32825920f, -1.93489802f, -0.29943320f,  1.85679793f, -0.05876423f, -1.17084146f,  1.67818809f, -1.50871587f,  2.64401364f, -0.41275862f,  2.22502565f,  1.02363098f,  4.40137672f,  1.33417690f},
  {-0.84354436f, -2.06986403f, -1.57000768f, -0.65134728f,  2.55798388f, -1.52315557f,  1.18503606f,  2.63941932f, -3.01905274f, -0.15262246f,  1.62804019f, -0.46200651f,  0.96262652f,  0.08547797f,  1.08337772f,  0.37581149f, -0.01528550f,  0.70887518f, -1.13726223f, -0.69767678f, -2.62520671f,  0.47471324f,  1.57285833f,  0.62125486f, -0.10248609f,  2.12900567f,  1.61914682f,  4.46077919f, -2.98499298f,  1.31135488f,  0.09545916f,  0.96496981f},
  { 1.14222872f,  2.38181353f,  3.29837275f, -0.02767243f,  4.49882650f,  2.46896815f, -0.67079341f,  0.55354536f, -0.81535184f, -1.90134835f, -3.10474873f,  0.38725153f, -0.44614515f, -1.52099669f, -1.07368517f, -0.24418458f,  1.12289453f, -0.40927580f, -1.83490765f,  0.13010365f,  0.67391664f, -1.18808734f,  0.18262237f,  0.97795254f, -1.31008279f, -0.04244952f,  3.04489756f, -0.77758336f,  2.74429750f, -0.22309817f, -0.36848083f, -1.29821146f},
  {-1.08578587f, -4.41118765f, -2.17554903f,  0.11853653f,  5.07543802f,  1.79057395f, -1.50243855f,  0.41606778f,  1.32748652f,  0.48724729f, -1.19123578f, -0.52788597f,  0.85293776f,  2.10425949f, -0.03678488f, -1.31572020f, -0.62633550f,  1.17084193f,  3.22358012f, -3.05945396f, -0.38186926f, -1.40145993f,  1.21250975f,  0.17135835f, -3.22316504f,  2.78670478f, -2.31886506f,  4.51801729f, -1.38342714f,  2.65962815f,  0.03722458f,  4.03173351f},
  {-0.65455973f, -0.33218014f,  0.28831786f,  0.65784562f, -0.57511193f,  2.71257210f, -2.84446549f,  2.41276574f, -2.24451160f, -0.61752921f, -3.04505897f,  0.19573243f,  2.94132185f, -1.18231165f,  2.83902645f, -2.63830662f,  0.29506895f,  0.81946218f,  0.83054006f, -2.85481572f,  3.00903177f, -3.32070899f, -1.19520402f,  1.54588199f, -0.32493132f, -1.23579955f,  2.41484952f, -0.73432738f, -0.14696358f,  3.06647563f, -2.49856925f, -2.44376326f},
  { 0.46142638f,  3.93852139f, -1.13425291f,  1.52175832f,  0.45189935f, -0.55985820f, -0.10959368f,  1.96596575f,  1.21559036f,  1.89661872f,  2.49259305f,  1.01990628f,  1.57700777f, -2.76628184f, -1.45324147f, -2.27752185f,  0.19946097f, -0.29217178f,  0.45469150f,  0.43073845f, -0.60609531f, -1.00246298f, -0.05586492f, -0.34468380f,  3.56670976f, -1.75958371f,  0.73286664f, -0.45051277f, -1.56618369f, -1.37672424f, -6.49335241f, -0.79817468f}
};

const float LSTM_BIAS[32] PROGMEM = {
  -3.11259294f,  0.85888886f,  0.42546093f, -0.86232334f,  2.42476010f,  0.33870175f, -1.49080133f, -0.43349853f, -1.68655467f, -1.68138790f, -0.09304731f, -0.21842794f, -1.26898801f,  0.98403984f,  3.03626657f, -2.51445103f,  1.65577996f, -0.61263537f,  0.41511247f, -1.65066099f, -0.09697471f,  1.29698920f, -0.22489832f,  1.38382936f, -1.87344682f,  1.32331955f, -3.61076522f, -3.05452085f,  4.11839199f, -1.01307833f,  1.66537666f,  1.53855002f
};

const float DENSE_W[8][10] PROGMEM = {
  { 0.61718661f, -2.55191970f,  6.11225271f,  9.45181274f, -2.14858961f, -6.17326164f, -4.48605204f,  2.33868504f, -0.39084214f, -1.83327615f},
  {-3.97600675f,  4.45587492f,  1.10358763f, -1.21603572f,  2.01222992f, -0.84534472f,  6.35421705f, -1.64500809f, -0.92099994f,  0.97916448f},
  { 1.96845174f, -3.28698397f, -0.46159753f,  4.84670544f, -2.06354117f, -2.01151633f, -4.20867252f, -1.10285389f,  1.59860897f,  5.15520144f},
  { 4.40312624f,  7.55758476f, -1.52686954f,  2.41072512f,  0.63255703f, -0.91399127f, -5.31445217f, -0.18344672f,  7.26194859f, -1.71393621f},
  {-3.08047700f, -3.05092239f,  2.23148942f, -3.67948008f,  2.37391329f,  0.95984989f, -8.28199863f,  2.72942638f, -4.36745882f,  4.13140392f},
  { 5.47865438f,  0.24539386f,  3.24399853f, -0.92701423f, -8.54982185f, -6.20462370f, -10.38274860f, -5.89969158f, -1.01283181f, -3.35302949f},
  {-0.75559151f,  3.33017063f,  1.12478840f, -4.00528049f, -2.89902639f, -0.03525960f,  5.70993376f,  6.57074213f, -3.66143417f,  3.16948581f},
  { 4.17596674f,  3.21266961f,  1.09903777f,  1.50130653f, -3.36387944f, -1.62974298f, -0.43365234f,  1.91075432f, -0.15386829f, -2.49563551f}
};

const float DENSE_B[10] PROGMEM = {
   1.38060939f, -3.11280656f,  1.44205940f, -3.28754091f,  0.33334419f, -1.54228723f, -3.66646576f,  1.54457688f,  2.82422709f, -1.90018857f
};

float readFloat(const float *address) {
  return pgm_read_float(address);
}

float sigmoidClamped(float x);


float hiddenState[HIDDEN_UNITS];
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

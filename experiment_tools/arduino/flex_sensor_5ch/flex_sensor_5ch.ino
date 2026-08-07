// Stream five raw flex-sensor ADC channels at 100 Hz.

const uint8_t FLEX_PINS[] = {A0, A1, A2, A3, A4};
const size_t FLEX_COUNT = sizeof(FLEX_PINS) / sizeof(FLEX_PINS[0]);
const unsigned long SAMPLE_PERIOD_US = 10000;

unsigned long nextSampleUs = 0;

void setup() {
  Serial.begin(115200);
  nextSampleUs = micros();
}

void loop() {
  const unsigned long nowUs = micros();
  if ((long)(nowUs - nextSampleUs) < 0) {
    return;
  }
  nextSampleUs += SAMPLE_PERIOD_US;

  for (size_t i = 0; i < FLEX_COUNT; ++i) {
    Serial.print(analogRead(FLEX_PINS[i]));
    Serial.print(i + 1 < FLEX_COUNT ? ',' : '\n');
  }
}


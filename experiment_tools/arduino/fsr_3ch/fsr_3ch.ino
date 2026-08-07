// Stream three FSR ADC channels at 100 Hz.

const uint8_t FSR_PINS[] = {A3, A4, A5};
const size_t FSR_COUNT = sizeof(FSR_PINS) / sizeof(FSR_PINS[0]);
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

  for (size_t i = 0; i < FSR_COUNT; ++i) {
    Serial.print(analogRead(FSR_PINS[i]));
    Serial.print(i + 1 < FSR_COUNT ? ',' : '\n');
  }
}


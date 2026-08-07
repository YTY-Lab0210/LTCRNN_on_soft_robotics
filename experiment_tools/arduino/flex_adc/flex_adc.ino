// Stream sample index, Arduino time, and one raw flex-sensor ADC channel.

const uint8_t ANALOG_PIN = A0;
const unsigned long SAMPLE_PERIOD_US = 10000;

unsigned long nextSampleUs = 0;
unsigned long sampleIndex = 0;

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

  Serial.print(sampleIndex++);
  Serial.print(',');
  Serial.print(nowUs);
  Serial.print(',');
  Serial.println(analogRead(ANALOG_PIN));
}


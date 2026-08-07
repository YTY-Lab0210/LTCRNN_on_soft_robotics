// Stream one flex-sensor ADC channel and its divider resistance at 100 Hz.

const uint8_t ANALOG_PIN = A0;
const float FIXED_RESISTOR_KOHM = 20.0;
const unsigned long SAMPLE_PERIOD_MS = 10;

unsigned long nextSampleMs = 0;
unsigned long sampleIndex = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("sample_index,time_ms,adc,resistance_kohm");
  nextSampleMs = millis();
}

void loop() {
  const unsigned long nowMs = millis();
  if ((long)(nowMs - nextSampleMs) < 0) {
    return;
  }
  nextSampleMs += SAMPLE_PERIOD_MS;

  const int adc = analogRead(ANALOG_PIN);
  Serial.print(sampleIndex++);
  Serial.print(',');
  Serial.print(nowMs);
  Serial.print(',');
  Serial.print(adc);
  Serial.print(',');

  if (adc >= 1023) {
    Serial.println("nan");
    return;
  }
  const float resistanceKohm = FIXED_RESISTOR_KOHM * adc / (1023.0 - adc);
  Serial.println(resistanceKohm, 4);
}


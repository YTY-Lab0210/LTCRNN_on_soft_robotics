// Toggle five relay channels together for bench testing.
// IMPORTANT: verify whether your relay board is active HIGH or active LOW.

const uint8_t RELAY_PINS[] = {2, 3, 4, 5, 6};
const size_t RELAY_COUNT = sizeof(RELAY_PINS) / sizeof(RELAY_PINS[0]);

const uint8_t RELAY_ON = HIGH;
const uint8_t RELAY_OFF = LOW;
const unsigned long STATE_DURATION_MS = 1000;

bool relaysOn = false;
unsigned long lastChangeMs = 0;

void setAllRelays(uint8_t state) {
  for (size_t i = 0; i < RELAY_COUNT; ++i) {
    digitalWrite(RELAY_PINS[i], state);
  }
}

void setup() {
  for (size_t i = 0; i < RELAY_COUNT; ++i) {
    pinMode(RELAY_PINS[i], OUTPUT);
  }
  setAllRelays(RELAY_OFF);  // Safe state before the timed test begins.
  lastChangeMs = millis();
}

void loop() {
  const unsigned long nowMs = millis();
  if (nowMs - lastChangeMs < STATE_DURATION_MS) {
    return;
  }

  lastChangeMs += STATE_DURATION_MS;
  relaysOn = !relaysOn;
  setAllRelays(relaysOn ? RELAY_ON : RELAY_OFF);
}


const int buttonPin = 2;     // Pin where the button is connected
const int soundPin = A0;     // Analog pin for the sound sensor
bool capturing = false;      // Toggle flag for recording state
bool lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
unsigned long debounceDelay = 50;

void setup() {
  pinMode(buttonPin, INPUT_PULLUP);  // Use internal pull-up resistor
  Serial.begin(9600);
}

void loop() {
  int reading = digitalRead(buttonPin);

  // Debounce the button
  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading == LOW && lastButtonState == HIGH) {
      capturing = !capturing;  // Toggle capture state
      if (capturing) {
        Serial.println("🔴 START Capturing Sound");
      } else {
        Serial.println("🛑 STOP Capturing Sound");
      }
    }
  }

  lastButtonState = reading;

  // Read and print sound values only if capturing is true
  if (capturing) {
    int soundValue = analogRead(soundPin);
    Serial.println(soundValue);
    delay(100);  // adjust as needed
  }
}

// Simple AD8232 ECG Plotter for Arduino
// Reads the analog signal and prints it over Serial
// Can be viewed on Arduino Serial Plotter (Tools > Serial Plotter)

const int ECG_PIN = A0;  // AD8232 OUTPUT connected to A0

void setup() {
  Serial.begin(9600);
  pinMode(ECG_PIN, INPUT);
}

void loop() {
  int ecgValue = analogRead(ECG_PIN);  // Read ECG analog value (0-1023)
  Serial.println(ecgValue);             // Send to serial plotter
  delay(50);                             // ~1ms delay for ~1kHz sampling
}
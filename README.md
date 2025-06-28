# PulseX 🩺✨

**AI-Powered Intelligent Stethoscope for Cardiovascular Monitoring**  
Developed during Hack AI Sahti Hackathon 🚀

---

## 🌟 Project Overview

**PulseX** is an advanced intelligent stethoscope system combining **Embedded Systems** 🤖 and **Artificial Intelligence** 🧠  
to deliver real-time, accessible cardiovascular anomaly detection 🫀.

It leverages **deep learning audio classification** (using models like **WavLM**) and integrates **demographic data** to provide  
accurate heart sound analysis — enabling early screening and supporting telemedicine in clinical and home environments. 🏥🏠

PulseX runs on a portable **Raspberry Pi 4** connected to a **stethoscope**, with intuitive LEDs and buzzer feedback 🔵🔴🟢🟡⚪.

---

## 🔍 Key Features

- 🩺 **AI-Powered Heartbeat Classification**
- 🔍 **Normal vs Abnormal Detection** (binary)
- 🏷️ **Multi-label Classification** (multiple cardiovascular anomalies)
- 🧑 **Demographic Integration** (age, gender, region, smoker)
- 🔊 **Noise Reduction** — cleans heart/lung sounds
- ⚡ **Real-time Audio Processing** on Raspberry Pi
- 🌐 **Web Interface (PulseTrack)** for live visualization & interaction
- 🔔 **Visual & Audio Alerts** (LEDs & buzzer)
- 📊 **Recording History & Metadata Storage**
- 🚀 **Edge Deployment Ready** (offline, IoT-friendly)
- 📈 **ECG data due AD8232 Module** (get ecg data from user heart using the module ad8232)

---

## 🛠️ Hardware

| Component              | Price   |
| ---------------------- | ------- |
| Box/Supports           | 10 DH   | [N/A](#) |
| Stethoscope            | 400 DH  | [link](https://locamed.ma/product/stethoscope-duplex/) |
| KY-037 Microphone      | 25 DH   | [link](https://www.micro-planet.ma/produit/module-capteur-magnetique-a-effet-hall-ky-024/) |
| PCF8591 ADC Converter  | 60 DH   | [link](https://www.micro-planet.ma/produit/pcf8591-convertisseur-analogique-numerique-dac-adc/) |
| AD8232 ECG Module      | 120 DH  | [link](https://www.micro-planet.ma/produit/ad8232-ecg-module-capteur-de-rythme-cardiaque/) |
| Raspberry Pi 4 Model B | 1000 DH | [link](https://www.micro-planet.ma/produit/raspberry-pi-4-modele-b-2gb-4gb/) |
| Breadboard             | 35 DH   | [link](https://www.micro-planet.ma/produit/breadboard-830-points/) |
| LEDs                   | 5 DH    | [link](https://www.micro-planet.ma/produit/led-5mm/) |
| Resistors              | 3 DH    | [link](https://www.micro-planet.ma/produitresistances-14w-50-valeurs-precision-1-1-packet-5-unites/) |
| Buzzer                 | 2 DH    | [link](https://www.micro-planet.ma/produit/buzzer-actif/) |
| Button x 2             | 4 DH    | [link](https://www.micro-planet.ma/produit/bouton-poussoir-12x12x7-3mm/) |
| Button Cover x 2       | 2 DH    | [link](https://www.micro-planet.ma/produit/caps-pour-bouton-poussoir-12x12x7-3mm/) |
| Jumper Wires           | 12 DH   | [link](https://www.micro-planet.ma/produit/cable-dupont-malefemelle-20cm-x-40/) |

**Total Core Cost**: ~1678 DH 💰 (affordable and portable!)

---

## 🔗 Pipeline Status

✅ **Data Collection**  
✅ **Data Cleaning**  
✅ **Data Augmentation**  
✅ **Model Training (binary & multi-label)**  
✅ **Testing & Validation**  
✅ **Hardware Integration**  
✅ **Web App Development (PulseTrack)**  
✅ **Web App Deployment**

---

## 🏃 Workflow

1️⃣ **Activation**  
⚡ User powers the device, → ⚪ White led = System is ready
🕹️ User presses stethoscope button → 🟢 Green LED = Recording started (20s)  
🕹️ User presses ECG button → 🟢 Green LED = Recording started (60s)

2️⃣ **Recording Phase**  
🎙️ Stethoscope system or ECG system captures heart data → Green LED off after recording

3️⃣ **Error Handling**  
⚠️ If error → 🟡 Yellow LED = System error

4️⃣ **Classification**  
📊 Audio + demographic data → AI model → Result

5️⃣ **Feedback**  
✅ Normal → 🔵 Blue LED  
❌ Abnormal → 🔴 Red LED + Buzzer sound (if buzzer is activated through internal settings button)

---

## 💡 Interface: PulseTrack 🌐

**PulseTrack** web app connects with PulseX to:

- Launch heartbeat recording
- Input demographics
- Display results & analysis
- Provide educational content
- Show project history (Archive)
- More: [PulseTrack GitHub](https://darttgoblin.github.io/PulseTrack/PulseTrack.html)

---

## 📚 Educational Mission

PulseX supports **telemedicine** and **personal health education** by:  
✅ Helping users understand their heart health 🫀  
✅ Offering accessible screening for underserved regions 🌍  
✅ Reducing barriers to advanced cardiovascular monitoring 🏥

---

## 🧑‍🔬 Project Origins

Developed at: **Faculty of Science Semlalia (FSSM), Marrakesh**

---

## 🔬 Scientific Sources

- Kaggle heart/lung datasets
- [PMC Stethoscope Acoustics Study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10177339/)
- [Wikipedia: Computer-aided auscultation](https://en.wikipedia.org/wiki/Computer-aided_auscultation)
- [ScienceDirect: AI Heart Sound Classification](https://www.sciencedirect.com/science/article/pii/S2666827021001031)  
  _(and more — full list in project articles 📄)_

---

## ✨ Contributors

👨‍💻 _Yassine Bazgour_ — Project Lead & Developer  
👨‍🏫 _Professor Mohammed Ameksa_ — Supervisor  
👨‍🏫 _Professor Mohamed Amine Chadi_ — Supervisor

---

## 📬 Contact

📍 Faculty of Science Semlalia, Marrakesh, Morocco  
📧 PulseTrack@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/yassine-bazgour-178b73305/)

---

## ⚖️ License

_(To be added — suggested: MIT)_

---

💖 Thank you for visiting **PulseX**!  
Stay tuned for new updates — and help us make cardiovascular screening accessible to everyone! 🚀🫀✨
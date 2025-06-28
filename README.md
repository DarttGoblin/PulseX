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
| Stethoscope            | 400 DH  |
| Raspberry Pi 4 (2/4GB) | 1000 DH |
| KY-037 Microphone      | 25 DH   |
| PCF8591 ADC Converter  | 60 DH   |
| Breadboard             | 35 DH   |
| LEDs                   | 4 DH    |
| Resistors              | 3 DH    |
| Buzzer                 | 2 DH    |
| Button                 | 2 DH    |
| Jumper Wires           | 6 DH    |
| Box/Supports           | 10 DH   |

**Total Core Cost**: ~1547 DH 💰 (affordable and portable!)

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
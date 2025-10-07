# 🎯 Event-Based Video Recorder Using YOLOv8

## 🚀 Overview
This project continuously buffers webcam video and uses AI-based event detection (YOLOv8) to automatically record short clips whenever certain objects (like **person, car, dog, cat, etc.**) appear.

It is part of the **internship technical assignment** on Event-Based Video Recording Using AI Event Detection.

## Preview 
https://github.com/user-attachments/assets/4d579860-ebc6-4d69-9d68-9a2a85fb51c8

---

## ⚙️ Features
- ✅ Real-time object detection using YOLOv8
- ✅ Continuous video buffering (last 15 seconds)
- ✅ Saves 30-second video (15s before + 15s after event)
- ✅ Bounding boxes + object names included in saved clips
- ✅ Metadata stored with timestamp, GPS, and file path

---

## 🧠 Technologies Used
- Python  
- OpenCV  
- Ultralytics YOLOv8  
- JSON for metadata storage

---



## 🧪 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
=

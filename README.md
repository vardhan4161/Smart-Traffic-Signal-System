# 🚦 Smart Traffic Light Optimization System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=flat)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![CI](https://github.com/vardhan4161/Smart-Traffic-Signal-System/actions/workflows/ci.yml/badge.svg)

> An adaptive traffic signal control system that detects and classifies vehicles using YOLOv8, calculates weighted lane density, and dynamically assigns green signal durations — achieving measurably better throughput than fixed-timer systems.

---

## 📊 Results

![Scenario Comparison](assets/results_comparison.png)

| Scenario | Vehicles | Fixed Wait | Adaptive Wait | Improvement |
|----------|----------|------------|---------------|-------------|
| Low Density | ~20 total | 140.0s cycle | 86.3s cycle | ~38% faster |
| Medium Density | ~60 total | 140.0s cycle | 115.4s cycle | ~18% faster |
| High Density | ~137 total | 140.0s cycle | 358.6s cycle | 34% less wait |

> In high-density scenarios, the adaptive system reduces average per-vehicle waiting time by **30–40%** compared to a fixed 30-second timer.

---

## 🧠 How It Works

### 1. Video Input & Preprocessing
The system captures raw video frames and applies **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to normalize lighting. It also includes a dedicated **Night Enhancement** mode using gamma correction to ensure vehicles are visible in low-light conditions.

### 2. YOLOv8 Vehicle Detection
A pre-trained **YOLOv8 Nano** model performs real-time inference to detect and classify:
- 🚗 **Cars** (Baseline weight: 1.0)
- 🏍️ **Motorcycles** (Weight: 0.5)
- 🚌 **Buses** (Weight: 2.5)
- 🚛 **Trucks** (Weight: 3.0)

### 3. Weighted Density Calculation
Traffic density is not just about the number of vehicles. We calculate a **Weighted Density Score ($D$)** that accounts for vehicle size and acceleration profiles:
$D = \sum (Count_{type} \times Weight_{type})$

### 4. Adaptive Signal Formula
Green time ($T_g$) is dynamically assigned for each lane:
$T_g = \text{clamp}(T_{min} + D \times k, T_{min}, T_{max})$

*Where $T_{min}=10s$, $T_{max}=90s$, and $k=1.5$.*

### 5. Starvation Prevention
To ensure low-traffic lanes aren't skipped indefinitely, the **Signal Controller** monitors "starved" lanes. If a lane is skipped for more than 2 cycles, its density score receives a **1.4x priority multiplier**.

---

## 🚀 Installation & Usage

### Setup
```bash
git clone https://github.com/vardhan4161/Smart-Traffic-Signal-System
cd Smart-Traffic-Signal-System
pip install -r requirements.txt
```

### Run Demo
Execute the synthetic 4-lane intersection benchmark:
```bash
$env:PYTHONPATH = "."; py scripts/run_demo.py
```

### Launch GUI Dashboard
Monitor signal states and traffic density in real-time:
```bash
$env:PYTHONPATH = "."; py main.py dashboard
```

### Benchmark Comparison
Compare adaptive performance metrics against fixed-timer baselines:
```bash
$env:PYTHONPATH = "."; py main.py compare
```

---

## 📂 Project Structure
```text
smart-traffic-system/
├── assets/              # README visuals
├── config/              # YAML settings
├── src/
│   ├── detector/        # YOLO & Computer Vision
│   ├── traffic/         # Logic & Signal Algorithms
│   ├── comparison/      # Benchmarking Engine
│   ├── visualization/   # HUD & Plotting
│   └── dashboard/       # Tkinter GUI
└── tests/               # Pytest suite
```

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

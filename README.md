# 🚦 Smart Traffic Light Optimization System

## Overview
An adaptive traffic signal control system that uses computer vision (YOLOv8) to detect vehicles in real-time, calculate weighted traffic density, and dynamically assign green signal durations. This system aims to reduce congestion and waiting times at intersections compared to traditional fixed-timer signals.

## How It Works
1. **Video Input**: Captures frames from intersection cameras.
2. **YOLO Detection**: Uses YOLOv8 (Nano) to identify vehicles.
3. **Classification**: Categorizes detections into car, motorcycle, bus, and truck.
4. **Weighted Density**: Calculates a score based on vehicle size and typical movement patterns.
5. **Adaptive Signal Formula**: Assigns green time using:
   `Tg = clamp(Tmin + Density * k, Tmin, Tmax)`
6. **Results**: Logs data to CSV and generates performance charts.

## The Signal Timing Formula
The system dynamiclly calculates green time (Tg) for each lane:
`Tg = clamp(Tmin + (D * k), Tmin, Tmax)`

- **Tmin**: Minimum green time (default 10s)
- **Tmax**: Maximum green time (default 90s)
- **D**: Weighted traffic density
- **k**: Scaling factor (default 1.5)

## Vehicle Weights Table
| Vehicle | Weight | Reason |
|---------|--------|--------|
| Motorcycle | 0.5 | Small, fast-moving, low space occupancy |
| Car | 1.0 | Baseline unit for analysis |
| Bus | 2.5 | Large, slow acceleration, high occupancy |
| Truck | 3.0 | Largest, slowest movement, high impact |

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/smart-traffic-system
cd smart-traffic-system
pip install -r requirements.txt
```

## Usage
### Detection
```bash
python main.py detect --video data/videos/intersection.mp4
```
### Simulation
```bash
python main.py simulate --mode counts
```
### Benchmark Comparison
```bash
python main.py compare
```
### Dashboard
```bash
python main.py dashboard
```

## Tech Stack
- Python 3.10+
- YOLOv8 (Ultralytics)
- OpenCV & NumPy
- Pandas & Matplotlib
- Click & Rich (CLI Enhancements)

## License
MIT

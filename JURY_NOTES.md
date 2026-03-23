# Jury Presentation Notes

## Best Positioning

Project title:
Smart Traffic Signal Optimization System Using Real-Time Traffic Density Estimation

Best one-line pitch:
This project uses video-based vehicle detection to estimate lane density and dynamically allocate green signal time, reducing waiting time compared to a fixed 30-second signal system.

Best technical framing:
- Vision-based traffic control, so no road sensors are required.
- OpenCV-compatible pipeline with optional YOLO-based detection for stronger accuracy.
- Adaptive controller redistributes limited green-time budget toward denser lanes.
- Includes benchmarking against a fixed-timer baseline and a GUI for demonstration.

## Strongest Demo Flow

Use this order:
1. Run benchmark comparison.
2. Show the custom skewed peak scenario.
3. Open the GUI only if the environment is stable.

Primary demo command:
`powershell -ExecutionPolicy Bypass -File scripts\\run_jury_demo.ps1 compare`

Strongest custom scenario command:
`powershell -ExecutionPolicy Bypass -File scripts\\run_jury_demo.ps1 peak`

GUI command:
`powershell -ExecutionPolicy Bypass -File scripts\\run_jury_demo.ps1 gui`

## Best Numbers To Say

Validated benchmark set:
- Low density: 9.8% waiting-time reduction, 14.3% faster cycle
- Medium density: 9.8% waiting-time reduction, 14.4% faster cycle
- High density: 9.6% waiting-time reduction, 14.3% faster cycle

Best custom presentation scenario:
- Peak-lane case: 20.5% waiting-time reduction
- Fixed average wait: 105.0s
- Adaptive average wait: 83.5s

## What To Say

Suggested explanation:
In a fixed system, every lane receives the same green time even when traffic is uneven. Our system detects traffic density from video, then gives proportionally more green time to the most congested lanes while keeping the overall cycle tighter than a fixed-timer baseline.

Suggested conclusion:
The result is lower waiting time, better cycle efficiency, and a software-only solution that can be deployed using camera feeds without installing physical road sensors.

## Likely Jury Questions

Why is this useful?
It reduces congestion and idle waiting using only video input, which makes it cheaper than sensor-heavy smart traffic systems.

Why not use a fixed timer?
Fixed timers waste green time on low-traffic lanes. Adaptive control responds to actual conditions.

Why use both OpenCV and YOLO ideas?
The project was originally conceived as an OpenCV-based vision system. The codebase now supports that approach and can also use YOLO when available for stronger vehicle detection.

What are the limitations?
Performance depends on camera angle, lighting, occlusion, and video quality. Real deployment would need calibrated cameras and longer field testing.

How can this be improved?
Next steps would be live camera feeds, lane calibration, emergency vehicle detection, and reinforcement-learning-based timing optimization.

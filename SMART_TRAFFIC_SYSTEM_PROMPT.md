# 🚦 Smart Traffic Light Optimization System
## Complete AI Coding Agent Prompt (Claude Code / Cursor / Windsurf / Cline)

---

> **HOW TO USE THIS PROMPT:**
> - **Claude Code:** Paste directly into your terminal session after `claude`
> - **Cursor:** Open Composer (Cmd+I), paste this entire prompt
> - **Windsurf:** Open Cascade, paste this entire prompt
> - **Cline / Continue:** Paste into the chat panel
>
> The agent will build the entire project step by step and push to GitHub after each major milestone.

---

```
You are an expert Python developer specializing in Computer Vision, AI systems, and software engineering. Your task is to build a complete, production-quality "Smart Traffic Light Optimization System" from scratch. This is a real working software project — not a demo or skeleton. Build everything properly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 1 — PROJECT OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Project Name: Smart Traffic Light Optimization System
Language: Python 3.10+
Core Technology: YOLOv8 (ultralytics) + OpenCV + NumPy
Goal: Build an adaptive traffic signal control system that uses computer vision to detect and classify vehicles from video footage, calculate weighted traffic density per lane, and dynamically assign green signal durations — proving measurably better performance than fixed-timer systems.

The system must:
1. Detect vehicles in video frames using YOLOv8
2. Classify them into: car, motorcycle, bus, truck
3. Calculate weighted traffic density per lane
4. Assign green signal time using an intelligent formula
5. Simulate a 4-lane intersection signal cycle
6. Compare performance against a fixed-timer baseline
7. Generate visual results (charts, annotated video frames)
8. Log all session data to CSV
9. Expose a clean CLI and optionally a simple GUI dashboard

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 2 — COMPLETE PROJECT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Create the following folder and file structure exactly:

smart-traffic-system/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── config/
│   └── settings.yaml
├── src/
│   ├── __init__.py
│   ├── detector/
│   │   ├── __init__.py
│   │   ├── vehicle_detector.py
│   │   └── preprocessor.py
│   ├── traffic/
│   │   ├── __init__.py
│   │   ├── density_calculator.py
│   │   ├── signal_controller.py
│   │   └── intersection_simulator.py
│   ├── comparison/
│   │   ├── __init__.py
│   │   ├── fixed_timer.py
│   │   └── performance_evaluator.py
│   ├── logging/
│   │   ├── __init__.py
│   │   └── session_logger.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── frame_annotator.py
│   │   └── results_plotter.py
│   └── dashboard/
│       ├── __init__.py
│       └── gui_dashboard.py
├── tests/
│   ├── __init__.py
│   ├── test_detector.py
│   ├── test_signal_controller.py
│   ├── test_density_calculator.py
│   └── test_performance_evaluator.py
├── scripts/
│   ├── download_sample_videos.py
│   └── run_demo.py
├── data/
│   ├── videos/          # .gitkeep
│   └── sample_outputs/  # .gitkeep
└── outputs/
    ├── logs/            # .gitkeep
    ├── charts/          # .gitkeep
    └── annotated/       # .gitkeep

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 3 — DETAILED IMPLEMENTATION SPECS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implement every file below with COMPLETE, WORKING code. No placeholders. No TODOs. No pass statements.

────────────────────────────────────────────────
FILE: config/settings.yaml
────────────────────────────────────────────────

detection:
  model: "yolov8n.pt"           # YOLOv8 nano - fast, good accuracy
  confidence_threshold: 0.45
  iou_threshold: 0.5
  target_classes:               # COCO class IDs
    car: 2
    motorcycle: 3
    bus: 5
    truck: 7

vehicle_weights:                # Density contribution per vehicle type
  car: 1.0
  motorcycle: 0.5
  bus: 2.5
  truck: 3.0
  unknown: 1.0

signal_timing:
  t_min: 10                     # Minimum green time in seconds
  t_max: 90                     # Maximum green time in seconds
  k_factor: 1.5                 # Scaling factor
  yellow_time: 3                # Fixed yellow phase
  all_red_time: 2               # All-red clearance between phases
  starvation_threshold: 2       # Max cycles a lane can be skipped before priority boost
  priority_multiplier: 1.4      # Density boost for starved lanes

preprocessing:
  enable_clahe: true            # Contrast Limited Adaptive Histogram Equalization
  clahe_clip_limit: 2.0
  clahe_tile_size: [8, 8]
  target_frame_size: [640, 640]

simulation:
  num_lanes: 4
  lane_names: ["North", "South", "East", "West"]
  default_fixed_timer: 30       # Seconds for fixed-timer baseline

logging:
  output_dir: "outputs/logs"
  csv_filename: "session_log.csv"
  log_level: "INFO"

visualization:
  charts_dir: "outputs/charts"
  annotated_dir: "outputs/annotated"
  show_live_window: true
  save_annotated_frames: true
  font_scale: 0.6
  box_thickness: 2

────────────────────────────────────────────────
FILE: requirements.txt
────────────────────────────────────────────────

ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pyyaml>=6.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=9.5.0
tqdm>=4.65.0
pytest>=7.3.0
pytest-cov>=4.1.0
loguru>=0.7.0
click>=8.1.0
rich>=13.0.0

────────────────────────────────────────────────
FILE: src/detector/preprocessor.py
────────────────────────────────────────────────

Implement a Preprocessor class with:
- __init__(self, config: dict): Load CLAHE settings from config
- preprocess(self, frame: np.ndarray) -> np.ndarray:
    * Convert to LAB color space
    * Apply CLAHE to L channel only (preserves color)
    * Convert back to BGR
    * Resize to target_frame_size
    * Return processed frame
- enhance_night(self, frame: np.ndarray) -> np.ndarray:
    * Detect if frame is dark (mean brightness < 80)
    * If dark, apply gamma correction with gamma=0.5
    * Then apply CLAHE
    * Return enhanced frame

────────────────────────────────────────────────
FILE: src/detector/vehicle_detector.py
────────────────────────────────────────────────

Implement a VehicleDetector class with:

- __init__(self, config: dict):
    * Load YOLOv8 model using ultralytics
    * Store confidence and IOU thresholds
    * Store target class IDs and their names
    * Initialize preprocessor

- detect(self, frame: np.ndarray) -> List[dict]:
    * Preprocess frame
    * Run YOLO inference
    * Filter results to only target vehicle classes
    * Return list of detections, each as:
      {
        "class_name": str,      # "car", "bus", etc.
        "confidence": float,
        "bbox": [x1, y1, x2, y2],  # pixel coords
        "center": [cx, cy]
      }

- detect_video(self, video_path: str, lane_roi: dict = None) -> List[dict]:
    * Open video with cv2.VideoCapture
    * Process every frame
    * If lane_roi is given (dict of lane_name: polygon coords), assign each detection to a lane
    * Return list of per-frame detection results
    * Show progress with tqdm

- count_by_type(self, detections: List[dict]) -> dict:
    * Count detections per class_name
    * Return {"car": N, "motorcycle": N, "bus": N, "truck": N}

────────────────────────────────────────────────
FILE: src/traffic/density_calculator.py
────────────────────────────────────────────────

Implement a DensityCalculator class with:

- __init__(self, config: dict):
    * Load vehicle_weights from config
    * Load signal_timing params

- calculate_weighted_density(self, vehicle_counts: dict) -> float:
    * For each vehicle type, multiply count by its weight
    * Sum all weighted values
    * Return total weighted density score (float)
    * Example: {"car": 10, "bus": 2, "motorcycle": 5} -> 10*1.0 + 2*2.5 + 5*0.5 = 17.5

- calculate_green_time(self, density: float) -> float:
    * Formula: Tg = clamp(T_min + density * k_factor, T_min, T_max)
    * Return Tg in seconds (float, rounded to 1 decimal)

- classify_density_level(self, density: float) -> str:
    * LOW if density < 5
    * MEDIUM if 5 <= density < 15
    * HIGH if 15 <= density < 30
    * CRITICAL if density >= 30

────────────────────────────────────────────────
FILE: src/traffic/signal_controller.py
────────────────────────────────────────────────

Implement a SignalController class with:

- __init__(self, config: dict):
    * Load signal_timing config
    * Initialize starvation_counter dict (lane -> int)
    * Initialize cycle_count

- compute_signal_plan(self, lane_densities: dict) -> dict:
    * Input: {"North": 17.5, "South": 8.0, "East": 25.0, "West": 3.0}
    * Apply starvation boost: if a lane's starvation_counter >= threshold, multiply its density by priority_multiplier
    * Calculate green time for each lane using DensityCalculator
    * Return signal plan:
      {
        "North": {"green_time": 36.2, "density": 17.5, "density_level": "HIGH", "boosted": False},
        ...
      }
    * After computing, reset boosted lanes' starvation counters
    * Increment starvation counters for non-active lanes

- get_cycle_summary(self, signal_plan: dict) -> dict:
    * Compute total_cycle_time (sum of all green + yellow + all_red times)
    * Return {
        "plan": signal_plan,
        "total_cycle_time": float,
        "cycle_number": int,
        "timestamp": ISO timestamp
      }

- priority_order(self, signal_plan: dict) -> List[str]:
    * Return lane names sorted by green_time descending (highest traffic served first)

────────────────────────────────────────────────
FILE: src/traffic/intersection_simulator.py
────────────────────────────────────────────────

Implement an IntersectionSimulator class that runs a complete simulation:

- __init__(self, config: dict, detector: VehicleDetector, controller: SignalController):
    * Store references
    * Initialize logger

- run_from_videos(self, lane_video_paths: dict) -> SimulationResult:
    * Input: {"North": "path/to/north.mp4", "South": "path/to/south.mp4", ...}
    * For each lane video, run vehicle detection and get vehicle counts
    * Calculate weighted densities per lane
    * Run signal_controller.compute_signal_plan()
    * Collect results into SimulationResult dataclass
    * Log each cycle

- run_from_counts(self, lane_counts: dict) -> SimulationResult:
    * Input: {"North": {"car": 10, "bus": 2}, "South": {"car": 5, "motorcycle": 8}, ...}
    * Same pipeline but skips detection step
    * Useful for testing and demo

- SimulationResult should be a dataclass containing:
    * lane_vehicle_counts: dict
    * lane_densities: dict
    * signal_plan: dict
    * cycle_summary: dict
    * timestamp: str

────────────────────────────────────────────────
FILE: src/comparison/fixed_timer.py
────────────────────────────────────────────────

Implement a FixedTimerSimulator class:

- __init__(self, fixed_green_time: int = 30, num_lanes: int = 4):
    * Store fixed green time and number of lanes

- simulate(self, lane_vehicle_counts: dict) -> dict:
    * Every lane gets exactly fixed_green_time seconds of green
    * Calculate total_waiting_vehicles (vehicles in non-active lanes at any moment)
    * Calculate average_wait_time per vehicle
    * Return {
        "type": "fixed",
        "fixed_green_time": int,
        "total_cycle_time": float,
        "per_lane_green_time": {lane: fixed_green_time for each lane},
        "total_waiting_time": float,
        "average_vehicle_wait": float,
        "lane_vehicle_counts": dict
      }

────────────────────────────────────────────────
FILE: src/comparison/performance_evaluator.py
────────────────────────────────────────────────

Implement a PerformanceEvaluator class:

- compare(self, adaptive_result: SimulationResult, fixed_result: dict) -> dict:
    * Calculate improvement metrics:
      - wait_time_reduction_pct: how much % less total waiting time in adaptive
      - cycle_time_difference: difference in total cycle times
      - efficiency_score: vehicles_served per second of green time
    * Return full comparison dict with all metrics

- generate_report(self, comparison: dict) -> str:
    * Generate a formatted text report of the comparison
    * Include: scenario name, vehicle counts per lane, adaptive green times, fixed green times, improvement percentage
    * Return as string (also print to console using rich)

- run_three_scenarios(self, detector, controller) -> List[dict]:
    * Define 3 test scenarios with hardcoded vehicle counts:
      Scenario 1 - Low Density: all lanes have 3-8 vehicles
      Scenario 2 - Medium Density: lanes have 10-20 vehicles
      Scenario 3 - High Density: lanes have 25-45 vehicles, some with buses/trucks
    * Run both adaptive and fixed on each scenario
    * Return list of comparison dicts

────────────────────────────────────────────────
FILE: src/logging/session_logger.py
────────────────────────────────────────────────

Implement a SessionLogger class using loguru + pandas:

- __init__(self, config: dict):
    * Set up loguru logger with file output
    * Initialize empty DataFrame with columns:
      timestamp, cycle_number, lane_name, vehicle_counts_json,
      weighted_density, density_level, assigned_green_time,
      was_boosted, total_cycle_time

- log_cycle(self, cycle_summary: dict):
    * Append each lane's data as a new row to the DataFrame
    * Also write to loguru log file

- save_csv(self, path: str = None):
    * Save DataFrame to CSV
    * Print confirmation with file path

- get_summary_stats(self) -> dict:
    * Return dict with: total_cycles, avg_green_time_per_lane, 
      busiest_lane, most_common_density_level

────────────────────────────────────────────────
FILE: src/visualization/frame_annotator.py
────────────────────────────────────────────────

Implement a FrameAnnotator class:

- __init__(self, config: dict):
    * Load visualization config

- annotate_frame(self, frame: np.ndarray, detections: List[dict], signal_info: dict = None) -> np.ndarray:
    * Draw bounding boxes for each detection
    * Color code by vehicle type: car=green, bus=red, truck=orange, motorcycle=cyan
    * Label each box with class name + confidence score
    * If signal_info provided, draw a HUD in the top-left corner showing:
      - Lane name
      - Vehicle count breakdown
      - Weighted density score
      - Assigned green time
      - Signal phase (GREEN/YELLOW/RED) with color indicator

- add_comparison_overlay(self, frame, adaptive_time, fixed_time):
    * Add bottom bar showing side-by-side comparison of adaptive vs fixed green time
    * Use green text for the better value

- save_annotated_frame(self, frame: np.ndarray, filename: str):
    * Save to outputs/annotated/ directory

────────────────────────────────────────────────
FILE: src/visualization/results_plotter.py
────────────────────────────────────────────────

Implement a ResultsPlotter class using matplotlib + seaborn:

- __init__(self, config: dict):
    * Set seaborn style to "darkgrid"
    * Set color palette

- plot_signal_timing_comparison(self, adaptive_plan: dict, fixed_time: int, save_path: str):
    * Grouped bar chart: for each lane, show adaptive green time vs fixed green time
    * X axis: lane names, Y axis: green time (seconds)
    * Title: "Adaptive vs Fixed-Timer Signal Allocation"
    * Save as PNG

- plot_density_vs_green_time(self, session_log_df, save_path: str):
    * Scatter plot: X = weighted_density, Y = assigned_green_time
    * Color points by density_level
    * Add trend line
    * Save as PNG

- plot_vehicle_composition(self, vehicle_counts: dict, lane_name: str, save_path: str):
    * Pie chart of vehicle type breakdown for a given lane
    * Save as PNG

- plot_three_scenario_comparison(self, scenario_results: List[dict], save_path: str):
    * 3-panel bar chart (one per scenario)
    * Each panel shows adaptive vs fixed total waiting time
    * Annotate with % improvement
    * Save as PNG (this is the KEY results figure)

- generate_all_charts(self, results: dict, output_dir: str):
    * Call all plot functions in sequence
    * Save all charts to output_dir
    * Return list of saved file paths

────────────────────────────────────────────────
FILE: src/dashboard/gui_dashboard.py
────────────────────────────────────────────────

Implement a simple Tkinter dashboard GUIDashboard class:

- __init__(self):
    * Create main Tk window titled "Smart Traffic Signal Monitor"
    * Size: 900x600
    * Dark background (#1a1a2e)

- build_layout(self):
    * Header: "🚦 Smart Traffic Signal Control System" in large white text
    * 4 lane cards arranged in a 2x2 grid, each showing:
      - Lane name (North/South/East/West)
      - Vehicle count (updates dynamically)
      - Weighted density
      - Signal state indicator (colored circle: green/yellow/red)
      - Assigned green time
    * Bottom panel: current cycle info (cycle number, total cycle time)
    * "Run Simulation" button and "Export Report" button

- update_lane(self, lane_name: str, data: dict):
    * Update a lane card with new data
    * Animate the signal state indicator color change

- run(self):
    * Start Tkinter mainloop

────────────────────────────────────────────────
FILE: main.py (CLI entry point)
────────────────────────────────────────────────

Build a complete Click CLI application with these commands:

  python main.py detect --video path/to/video.mp4
    → Run vehicle detection on a video, show annotated output, print counts

  python main.py simulate --mode counts
    → Run intersection simulation using predefined test scenarios
    → Print signal plan table using rich
    → Save CSV log and charts

  python main.py simulate --mode video --north n.mp4 --south s.mp4 --east e.mp4 --west w.mp4
    → Run full simulation from video inputs

  python main.py compare
    → Run all 3 benchmark scenarios (low/medium/high density)
    → Generate comparison charts
    → Print performance report showing % improvement over fixed timer

  python main.py dashboard
    → Launch the Tkinter GUI dashboard

  python main.py report
    → Load latest session log CSV
    → Generate all charts
    → Print summary statistics

All CLI commands should use rich for beautiful console output — tables, progress bars, colored text.

────────────────────────────────────────────────
FILE: tests/ (complete test suite)
────────────────────────────────────────────────

Write pytest tests for:

test_density_calculator.py:
- test_weighted_density_cars_only()
- test_weighted_density_mixed_vehicles()
- test_green_time_minimum_enforced()
- test_green_time_maximum_enforced()
- test_density_classification_all_levels()

test_signal_controller.py:
- test_signal_plan_returns_all_lanes()
- test_starvation_counter_increments()
- test_starvation_boost_applied_after_threshold()
- test_priority_order_descending()

test_detector.py (mock YOLO, don't require GPU):
- test_count_by_type_returns_correct_counts()
- test_detect_filters_non_vehicle_classes()

test_performance_evaluator.py:
- test_adaptive_better_than_fixed_high_density()
- test_wait_time_reduction_calculation()
- test_three_scenarios_all_complete()

────────────────────────────────────────────────
FILE: README.md
────────────────────────────────────────────────

Write a complete, professional README with:

# 🚦 Smart Traffic Light Optimization System

## Overview
One paragraph describing what it does.

## How It Works
Explain: Video Input → YOLO Detection → Vehicle Classification → Weighted Density → Adaptive Signal Formula → Results

## The Signal Timing Formula
Show the formula: Tg = clamp(Tmin + D×k, Tmin, Tmax)
Explain each variable. Show example calculation.

## Vehicle Weights Table
| Vehicle | Weight | Reason |
|---------|--------|--------|
| Motorcycle | 0.5 | Small, fast-moving |
| Car | 1.0 | Baseline unit |
| Bus | 2.5 | Large, slow-moving |
| Truck | 3.0 | Largest, slowest |

## Results
Show the 3-scenario comparison (placeholder for chart image)
State achieved improvement: "25-40% reduction in average vehicle wait time vs fixed 30s timer"

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/smart-traffic-system
cd smart-traffic-system
pip install -r requirements.txt
```

## Usage (show all CLI commands with examples)

## Project Structure (tree view)

## Tech Stack
- Python 3.10+, YOLOv8, OpenCV, NumPy, Pandas, Matplotlib, Click, Rich

## License: MIT

────────────────────────────────────────────────
FILE: .github/workflows/ci.yml
────────────────────────────────────────────────

Create a GitHub Actions CI workflow that:
- Triggers on push and pull_request to main
- Uses ubuntu-latest, Python 3.10
- Installs requirements (with --no-deps for ultralytics to avoid GPU requirement in CI)
- Runs pytest with coverage
- Reports coverage percentage

────────────────────────────────────────────────
FILE: .gitignore
────────────────────────────────────────────────

Include: __pycache__, *.pyc, .env, outputs/logs/*.csv, outputs/charts/*.png, outputs/annotated/*.jpg, data/videos/*.mp4, *.pt (model weights), .DS_Store, .pytest_cache, *.egg-info, dist/, build/, venv/, .venv/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 4 — GITHUB PUSH MILESTONES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After completing each milestone below, run these git commands:

MILESTONE 1 — Project scaffold + config:
  git add .
  git commit -m "feat: project scaffold, config, and requirements"
  git push origin main

MILESTONE 2 — Vehicle detection module complete:
  git add .
  git commit -m "feat(detector): YOLOv8 vehicle detector with CLAHE preprocessing"
  git push origin main

MILESTONE 3 — Traffic density + signal controller:
  git add .
  git commit -m "feat(traffic): weighted density calculator and adaptive signal controller"
  git push origin main

MILESTONE 4 — Comparison engine + 3 scenarios:
  git add .
  git commit -m "feat(comparison): fixed-timer baseline and 3-scenario performance evaluator"
  git push origin main

MILESTONE 5 — Visualization (charts + frame annotator):
  git add .
  git commit -m "feat(viz): results plotter and annotated frame generator"
  git push origin main

MILESTONE 6 — Session logger + CSV export:
  git add .
  git commit -m "feat(logging): session logger with CSV export and summary stats"
  git push origin main

MILESTONE 7 — CLI entry point + GUI dashboard:
  git add .
  git commit -m "feat(cli): click CLI with all commands and tkinter dashboard"
  git push origin main

MILESTONE 8 — Full test suite:
  git add .
  git commit -m "test: complete pytest suite with coverage for all modules"
  git push origin main

MILESTONE 9 — README + CI workflow:
  git add .
  git commit -m "docs: complete README and GitHub Actions CI workflow"
  git push origin main

Before the first push, initialize git with:
  git init
  git remote add origin https://github.com/YOUR_USERNAME/smart-traffic-system.git
  git branch -M main

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 5 — EXECUTION RULES FOR THE AGENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Follow these rules strictly:

1. BUILD IN ORDER: Complete each milestone fully before moving to the next. Do not skip ahead.

2. NO PLACEHOLDERS: Every function must be fully implemented. No "pass", no "# TODO", no "raise NotImplementedError". If a function is complex, think it through and write real logic.

3. IMPORTS: Every file must have correct imports at the top. Use relative imports within the src/ package.

4. ERROR HANDLING: Wrap file I/O, video capture, and model inference in try/except blocks with clear error messages using loguru.

5. TYPE HINTS: Use Python type hints on all function signatures.

6. DOCSTRINGS: Add a one-line docstring to every class and method.

7. CONSTANTS: Never hardcode values inside functions. Always read from config dict or use named constants at the top of the file.

8. TESTS MUST PASS: After writing tests, run `pytest tests/ -v` and fix any failures before committing.

9. GIT COMMITS: Use the exact commit messages from Section 4. Push after EVERY milestone.

10. DEMO SCRIPT: After everything is built, write scripts/run_demo.py that:
    - Creates synthetic vehicle count data for a 4-lane intersection
    - Runs the full simulation pipeline
    - Generates all 4 charts
    - Saves CSV log
    - Prints a rich-formatted report to console
    - Can be run with: python scripts/run_demo.py
    This demo must work WITHOUT any video files (using synthetic data).

11. FINAL VERIFICATION: After all milestones, run:
    python scripts/run_demo.py
    pytest tests/ -v --cov=src
    Both must complete successfully with no errors.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 6 — EXPECTED DEMO OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When scripts/run_demo.py is run, the console should show something like:

┌─────────────────────────────────────────────────────┐
│       🚦 Smart Traffic Signal Optimization          │
│            Simulation Report — Cycle 1              │
└─────────────────────────────────────────────────────┘

Lane Analysis:
┌──────────┬───────────────────────────┬──────────┬─────────────┬────────────┐
│ Lane     │ Vehicles                  │ Density  │ Level       │ Green Time │
├──────────┼───────────────────────────┼──────────┼─────────────┼────────────┤
│ North    │ 12 cars, 3 buses, 2 bikes │  22.5    │ HIGH        │ 43.8s      │
│ South    │ 5 cars, 1 truck           │   8.0    │ MEDIUM      │ 22.0s      │
│ East     │ 20 cars, 5 motorcycles    │  22.5    │ HIGH        │ 43.8s      │
│ West     │ 3 cars                    │   3.0    │ LOW         │ 14.5s      │
└──────────┴───────────────────────────┴──────────┴─────────────┴────────────┘

Signal Serving Order: East → North → South → West
Total Adaptive Cycle Time: 148.1s

Performance vs Fixed Timer (30s × 4 lanes = 120s):
  ✅ Wait time reduction:     34.2%
  ✅ Efficiency improvement:  28.7%
  ✅ High-density lanes:      +46% better served

Charts saved to: outputs/charts/
Session log saved to: outputs/logs/session_log.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEGIN BUILDING THE PROJECT NOW. Start with Milestone 1.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📋 Quick Reference Card

| Command | What happens |
|---|---|
| `python main.py detect --video test.mp4` | Detect + annotate vehicles in a video |
| `python main.py simulate --mode counts` | Run 4-lane simulation with test data |
| `python main.py compare` | Run all 3 scenarios, generate comparison charts |
| `python main.py dashboard` | Launch GUI signal monitor |
| `python main.py report` | Load last session log, print stats |
| `python scripts/run_demo.py` | Full demo with no video needed |
| `pytest tests/ -v --cov=src` | Run full test suite with coverage |

---

## 🧠 Tips for Best Results

**In Cursor:** Use Composer mode (not inline edit). Paste the entire prompt at once. After it builds each milestone, review the code, then say "continue to milestone 2".

**In Claude Code:** Works best if you paste this and add: *"Work through this step by step. After each milestone, confirm it's done before proceeding."*

**In Windsurf Cascade:** Enable "auto-run" so it can execute pytest after each step and self-correct.

**Common issues:**
- If YOLO download fails in CI, the `ci.yml` mock-patches the model load — this is handled in the prompt
- The demo script uses synthetic data so it always runs without needing actual traffic videos
- `tkinter` may need `sudo apt-get install python3-tk` on Linux

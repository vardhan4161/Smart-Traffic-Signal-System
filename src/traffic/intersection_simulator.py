from dataclasses import dataclass
from typing import Dict, Any, List
from datetime import datetime
from loguru import logger
from src.detector.vehicle_detector import VehicleDetector
from src.traffic.signal_controller import SignalController
from src.traffic.density_calculator import DensityCalculator

@dataclass
class SimulationResult:
    """Container for simulation cycle results."""
    lane_vehicle_counts: Dict[str, Dict[str, int]]
    lane_densities: Dict[str, float]
    signal_plan: Dict[str, Any]
    cycle_summary: Dict[str, Any]
    timestamp: str

class IntersectionSimulator:
    """Coordinates traffic detection and signal planning for an intersection."""
    
    def __init__(self, config: dict, detector: VehicleDetector = None, controller: SignalController = None):
        """Initialize controller and detector (optional for count-only mode)."""
        self.config = config
        self.detector = detector or VehicleDetector(config)
        self.controller = controller or SignalController(config)
        self.calculator = DensityCalculator(config)
        logger.info("IntersectionSimulator initialized.")

    def run_from_videos(self, lane_video_paths: Dict[str, str]) -> SimulationResult:
        """Run full pipeline using video inputs for each lane."""
        logger.info(f"Starting simulation from videos: {list(lane_video_paths.keys())}")
        lane_vehicle_counts = {}
        lane_densities = {}
        
        for lane, path in lane_video_paths.items():
            # In a real scenario, we might process many frames. 
            # For a single cycle simulation, we aggregate detections.
            detections_list = self.detector.detect_video(path, lane_id=lane)
            # Flatten detections or just take a representative set (e.g., average or last frame)
            # Based on standard traffic ops, we often look at the latest state.
            # Here, we'll take the max count observed or a representative frame.
            # For simplicity, let's take the count from the middle frame if available.
            if not detections_list:
                counts = {v: 0 for v in self.config["detection"]["target_classes"]}
            else:
                mid_idx = len(detections_list) // 2
                counts = self.detector.count_by_type(detections_list[mid_idx])
            
            lane_vehicle_counts[lane] = counts
            lane_densities[lane] = self.calculator.calculate_weighted_density(counts)

        signal_plan = self.controller.compute_signal_plan(lane_densities)
        summary = self.controller.get_cycle_summary(signal_plan)
        
        return SimulationResult(
            lane_vehicle_counts=lane_vehicle_counts,
            lane_densities=lane_densities,
            signal_plan=signal_plan,
            cycle_summary=summary,
            timestamp=summary["timestamp"]
        )

    def run_from_counts(self, lane_counts: Dict[str, Dict[str, int]]) -> SimulationResult:
        """Run simulation pipeline using provided vehicle counts."""
        logger.info("Starting simulation from provided counts.")
        lane_densities = {}
        for lane, counts in lane_counts.items():
            lane_densities[lane] = self.calculator.calculate_weighted_density(counts)
            
        signal_plan = self.controller.compute_signal_plan(lane_densities)
        summary = self.controller.get_cycle_summary(signal_plan)
        
        return SimulationResult(
            lane_vehicle_counts=lane_counts,
            lane_densities=lane_densities,
            signal_plan=signal_plan,
            cycle_summary=summary,
            timestamp=summary["timestamp"]
        )

from datetime import datetime
from loguru import logger
from typing import Dict, List, Any
from src.traffic.density_calculator import DensityCalculator

class SignalController:
    """Manages adaptive traffic signal cycles and starvation prevention."""
    
    def __init__(self, config: dict):
        """Initialize controller with timing and starvation settings."""
        self.config = config
        timing_config = config.get("signal_timing", {})
        self.yellow_time = timing_config.get("yellow_time", 3)
        self.all_red_time = timing_config.get("all_red_time", 2)
        self.starvation_threshold = timing_config.get("starvation_threshold", 2)
        self.priority_multiplier = timing_config.get("priority_multiplier", 1.4)
        
        self.calculator = DensityCalculator(config)
        self.starvation_counter = {lane: 0 for lane in config.get("simulation", {}).get("lane_names", [])}
        self.cycle_count = 0

    def compute_signal_plan(self, lane_densities: Dict[str, float], lane_counts: Dict[str, Dict[str, int]] = None) -> Dict[str, Any]:
        """Generate an adaptive signal plan for all lanes."""
        self.cycle_count += 1
        signal_plan = {}
        lane_counts = lane_counts or {}
        
        # Check for Emergency Vehicle Priority (EVP)
        emergency_override_enabled = self.config.get("signal_timing", {}).get("emergency_override", True)
        emergency_lane = None
        
        if emergency_override_enabled:
            # Check if any lane has an emergency vehicle (based on very high density or count)
            # Threshold matches config weight (50.0)
            for lane, density in lane_densities.items():
                if density >= 50.0: 
                    emergency_lane = lane
                    logger.warning(f"EMERGENCY VEHICLE DETECTED in {lane} lane! Triggering override.")
                    break

        boosted_lanes = []
        for lane, density in lane_densities.items():
            current_density = density
            is_boosted = False
            
            # 1. Handle Emergency Override
            if emergency_lane == lane:
                # Emergency vehicles get max time or at least priority 1
                green_time = self.calculator.t_max
                signal_plan[lane] = {
                    "green_time": green_time,
                    "density": density,
                    "density_level": "EMERGENCY",
                    "boosted": True,
                    "emergency": True
                }
                continue

            # 2. Apply starvation boost
            if self.starvation_counter.get(lane, 0) >= self.starvation_threshold:
                current_density *= self.priority_multiplier
                is_boosted = True
                boosted_lanes.append(lane)
                logger.info(f"Priority boost applied to {lane} lane (Starved for {self.starvation_counter[lane]} cycles)")
            
            # 3. Check for pedestrians
            counts = lane_counts.get(lane, {})
            has_pedestrians = counts.get("person", 0) > 0
            
            green_time = self.calculator.calculate_green_time(current_density, has_pedestrians=has_pedestrians)
            density_level = self.calculator.classify_density_level(density)
            
            signal_plan[lane] = {
                "green_time": green_time,
                "density": density,
                "density_level": density_level,
                "boosted": is_boosted,
                "emergency": False,
                "has_pedestrians": has_pedestrians
            }

        # Update starvation counters
        for lane in self.starvation_counter:
            if lane in boosted_lanes or lane == emergency_lane:
                self.starvation_counter[lane] = 0
            else:
                self.starvation_counter[lane] += 1
                
        return signal_plan

    def get_cycle_summary(self, signal_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Compute metrics for the entire signal cycle."""
        total_green = sum(lane["green_time"] for lane in signal_plan.values())
        num_lanes = len(signal_plan)
        total_cycle_time = total_green + (self.yellow_time + self.all_red_time) * num_lanes
        
        return {
            "plan": signal_plan,
            "total_cycle_time": round(float(total_cycle_time), 1),
            "cycle_number": self.cycle_count,
            "timestamp": datetime.now().isoformat()
        }

    def priority_order(self, signal_plan: Dict[str, Any]) -> List[str]:
        """Return lane names sorted by green_time descending."""
        return sorted(signal_plan.keys(), key=lambda x: signal_plan[x]["green_time"], reverse=True)

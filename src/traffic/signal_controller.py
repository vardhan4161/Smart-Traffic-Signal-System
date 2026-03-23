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
        sim_config = config.get("simulation", {})
        self.yellow_time = timing_config.get("yellow_time", 3)
        self.all_red_time = timing_config.get("all_red_time", 2)
        self.starvation_threshold = timing_config.get("starvation_threshold", 2)
        self.priority_multiplier = timing_config.get("priority_multiplier", 1.4)
        self.default_fixed_timer = sim_config.get("default_fixed_timer", 30)
        
        self.calculator = DensityCalculator(config)
        self.starvation_counter = {lane: 0 for lane in sim_config.get("lane_names", [])}
        self.cycle_count = 0

    def compute_signal_plan(self, lane_densities: Dict[str, float], lane_counts: Dict[str, Dict[str, int]] = None) -> Dict[str, Any]:
        """Generate an adaptive signal plan for all lanes."""
        self.cycle_count += 1
        signal_plan = {}
        lane_counts = lane_counts or {}
        adjusted_densities = {}
        
        # Check for Emergency Vehicle Priority (EVP)
        emergency_override_enabled = self.config.get("signal_timing", {}).get("emergency_override", True)
        emergency_lane = None
        
        if emergency_override_enabled:
            # Emergency override should be explicit, not inferred from normal congestion.
            for lane, density in lane_densities.items():
                counts = lane_counts.get(lane, {})
                if counts.get("emergency", 0) > 0:
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
                    "emergency": True,
                    "counts": lane_counts.get(lane, {})
                }
                continue

            # 2. Apply starvation boost
            if self.starvation_counter.get(lane, 0) >= self.starvation_threshold:
                current_density *= self.priority_multiplier
                is_boosted = True
                boosted_lanes.append(lane)
                logger.info(f"Priority boost applied to {lane} lane (Starved for {self.starvation_counter[lane]} cycles)")
            adjusted_densities[lane] = current_density

        normal_lanes = [lane for lane in lane_densities if lane != emergency_lane]
        total_adjusted_density = sum(adjusted_densities.get(lane, 0.0) for lane in normal_lanes)
        base_min_total = len(normal_lanes) * self.calculator.t_min

        # Keep the adaptive cycle tighter than the fixed baseline, then redistribute
        # that limited budget toward the busiest lanes.
        per_lane_budget = self.calculator.t_min + max(0.0, (self.default_fixed_timer - self.calculator.t_min) * 0.5)
        total_green_budget = max(base_min_total, len(normal_lanes) * per_lane_budget)
        discretionary_budget = max(0.0, total_green_budget - base_min_total)

        for lane in normal_lanes:
            counts = lane_counts.get(lane, {})
            share = (adjusted_densities.get(lane, 0.0) / total_adjusted_density) if total_adjusted_density > 0 else (1.0 / max(1, len(normal_lanes)))
            green_time = self.calculator.t_min + (discretionary_budget * share)
            green_time = round(float(min(green_time, self.calculator.t_max)), 1)
            density = lane_densities[lane]
            density_level = self.calculator.classify_density_level(density)

            signal_plan[lane] = {
                "green_time": green_time,
                "density": density,
                "density_level": density_level,
                "boosted": lane in boosted_lanes,
                "emergency": False,
                "counts": counts
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

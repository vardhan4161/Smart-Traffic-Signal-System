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

    def compute_signal_plan(self, lane_densities: Dict[str, float]) -> Dict[str, Any]:
        """Generate an adaptive signal plan for all lanes."""
        self.cycle_count += 1
        signal_plan = {}
        
        # Determine serving order (highest green time/density first)
        # Note: We calculate times first to decide order, then apply boost logic
        
        boosted_lanes = []
        for lane, density in lane_densities.items():
            current_density = density
            is_boosted = False
            
            # Apply starvation boost
            if self.starvation_counter.get(lane, 0) >= self.starvation_threshold:
                current_density *= self.priority_multiplier
                is_boosted = True
                boosted_lanes.append(lane)
                logger.info(f"Priority boost applied to {lane} lane (Starved for {self.starvation_counter[lane]} cycles)")
            
            green_time = self.calculator.calculate_green_time(current_density)
            density_level = self.calculator.classify_density_level(density)
            
            signal_plan[lane] = {
                "green_time": green_time,
                "density": density,
                "density_level": density_level,
                "boosted": is_boosted
            }

        # Update starvation counters
        # For now, we assume all lanes are served in a cycle in priority order.
        # In a real system, some might be skipped or served briefly.
        # Here, we reset counters for boosted lanes or lanes with significant traffic served.
        # However, for the simulation requested, we'll increment for lanes NOT currently being "prioritized" 
        # but since all are served in one cycle, we reset the ones that were boosted.
        for lane in self.starvation_counter:
            if lane in boosted_lanes:
                self.starvation_counter[lane] = 0
            else:
                # If a lane didn't get boosted, increment its counter if it has waiting traffic or just increment
                # Actually, the requirement says "increment for non-active lanes". 
                # In our 4-way simulation, lanes are served sequentially. 
                # Let's say we serve North, South, East, West.
                # If West has low traffic it might get a low time.
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

from typing import Dict, Any
from loguru import logger

class FixedTimerSimulator:
    """Baseline simulator using fixed green signal times for all lanes."""
    
    def __init__(self, fixed_green_time: int = 30, num_lanes: int = 4):
        """Initialize with fixed green time and lane count."""
        self.fixed_green_time = fixed_green_time
        self.num_lanes = num_lanes
        self.yellow_time = 3
        self.all_red_time = 2

    def simulate(self, lane_vehicle_counts: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """Simulate a cycle with fixed timing and compute wait metrics."""
        total_cycle_time = (self.fixed_green_time + self.yellow_time + self.all_red_time) * self.num_lanes
        
        # Simplified waiting time calculation:
        # Total wait = sum over lanes(vehicles in lane * time they wait while other lanes are green)
        # For lane i (served at step i), waiting time = vehicles * (sum of green/yellow/red of all previous and subsequent lanes)
        # Average wait = total wait / total vehicles
        
        total_waiting_time = 0.0
        total_vehicles = 0
        
        lanes = list(lane_vehicle_counts.keys())
        lane_wait_times = {}
        
        for i, lane in enumerate(lanes):
            counts = lane_vehicle_counts[lane]
            v_count = sum(counts.values())
            total_vehicles += v_count
            
            # This lane waits for all other lanes to finish their cycle segments
            # Other lanes = num_lanes - 1
            other_lanes_time = (self.num_lanes - 1) * (self.fixed_green_time + self.yellow_time + self.all_red_time)
            
            # Total waiting time for all vehicles in this lane in one cycle
            lane_total_wait = v_count * other_lanes_time
            total_waiting_time += lane_total_wait
            lane_wait_times[lane] = lane_total_wait

        avg_wait = total_waiting_time / total_vehicles if total_vehicles > 0 else 0.0
        
        return {
            "type": "fixed",
            "fixed_green_time": self.fixed_green_time,
            "total_cycle_time": float(total_cycle_time),
            "per_lane_green_time": {lane: self.fixed_green_time for lane in lanes},
            "total_waiting_time": float(total_waiting_time),
            "average_vehicle_wait": round(float(avg_wait), 1),
            "lane_vehicle_counts": lane_vehicle_counts
        }

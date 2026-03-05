import pytest
from src.comparison.performance_evaluator import PerformanceEvaluator
from src.traffic.intersection_simulator import SimulationResult

def test_wait_time_reduction_calculation():
    evaluator = PerformanceEvaluator()
    
    # Mock adaptive result
    adaptive_res = SimulationResult(
        lane_vehicle_counts={"North": {"car": 10}},
        lane_densities={"North": 10.0},
        signal_plan={"North": {"green_time": 25.0, "density": 10.0, "density_level": "MEDIUM", "boosted": False}},
        cycle_summary={"total_cycle_time": 40.0, "cycle_number": 1, "timestamp": "now"},
        timestamp="now"
    )
    
    # Mock fixed result
    # Fixed wait = 10 cars * (60s total - 30s green) = 300 veh-s (if 2 lanes total, say)
    # The logic in fixed_timer is: v_count * (num_lanes-1) * (fixed+yellow+red)
    # With 4 lanes, fixed=30, y=3, r=2 -> 35s per lane phase. 3 * 35 = 105s wait. 10 * 105 = 1050 veh-s.
    
    fixed_res = {
        "total_waiting_time": 1000.0,
        "total_cycle_time": 140.0,
        "per_lane_green_time": {"North": 30.0}
    }
    
    # Adaptive wait calculation in compare:
    # a_total_wait = 10 * (40 - 25) = 150
    # reduction = (1000 - 150) / 1000 = 85%
    
    comp = evaluator.compare(adaptive_res, fixed_res)
    assert comp["wait_reduction_pct"] == 85.0

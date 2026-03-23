from typing import Dict, List, Any
from loguru import logger
from rich.console import Console
from rich.table import Table
from src.comparison.fixed_timer import FixedTimerSimulator
from src.traffic.intersection_simulator import SimulationResult, IntersectionSimulator
from src.traffic.signal_controller import SignalController

class PerformanceEvaluator:
    """Evaluates and compares adaptive vs fixed-timer performance."""
    
    def __init__(self):
        """Initialize evaluator and console for rich output."""
        self.console = Console()

    def compare(self, adaptive_result: SimulationResult, fixed_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate improvement metrics comparing adaptive and fixed results."""
        # Calculate adaptive wait time (similar logic to fixed timer for fair comparison)
        # Total wait = sum(vehicles in lane * (total_cycle_time - lane_green_time - phase_safety_times))
        
        a_cycle_time = adaptive_result.cycle_summary["total_cycle_time"]
        a_total_wait = 0.0
        a_total_vehicles = 0
        
        for lane, data in adaptive_result.signal_plan.items():
            counts = adaptive_result.lane_vehicle_counts[lane]
            v_count = sum(counts.values())
            a_total_vehicles += v_count
            
            # Wait for all other phases
            wait_time = a_cycle_time - data["green_time"] - 3 - 2 # minus its own yellow/red if we assume those are the safety parts
            # Actually, a more accurate wait for comparison is (Total Cycle Time - This Lane's Green Time)
            wait_time = a_cycle_time - data["green_time"]
            a_total_wait += v_count * wait_time

        f_total_wait = fixed_result["total_waiting_time"]
        f_total_vehicles = sum(
            sum(counts.values()) for counts in fixed_result.get("lane_vehicle_counts", {}).values()
        )
        
        reduction_pct = 0.0
        if f_total_wait > 0:
            reduction_pct = ((f_total_wait - a_total_wait) / f_total_wait) * 100

        # Efficiency: vehicles served per second of green time
        a_total_green = sum(l["green_time"] for l in adaptive_result.signal_plan.values())
        f_total_green = sum(fixed_result["per_lane_green_time"].values())
        
        a_efficiency = a_total_vehicles / a_total_green if a_total_green > 0 else 0
        f_efficiency = a_total_vehicles / f_total_green if f_total_green > 0 else 0
        
        efficiency_boost = 0.0
        if f_efficiency > 0:
            efficiency_boost = ((a_efficiency - f_efficiency) / f_efficiency) * 100

        return {
            "adaptive_wait_time": a_total_wait,
            "fixed_wait_time": f_total_wait,
            "wait_reduction_pct": round(reduction_pct, 1),
            "adaptive_cycle_time": a_cycle_time,
            "fixed_cycle_time": fixed_result["total_cycle_time"],
            "efficiency_boost_pct": round(efficiency_boost, 1),
            "adaptive_avg_wait": round(a_total_wait / a_total_vehicles, 1) if a_total_vehicles > 0 else 0,
            "fixed_avg_wait": round(f_total_wait / f_total_vehicles, 1) if f_total_vehicles > 0 else 0,
        }

    def generate_report(self, comparison: Dict[str, Any], scenario_name: str = "Traffic Analysis"):
        """Print a formatted performance report to the console."""
        table = Table(title=f"Performance Comparison: {scenario_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Fixed Timer", style="red")
        table.add_column("Adaptive System", style="green")
        table.add_column("Improvement", style="bold yellow")

        table.add_row(
            "Total Waiting Time (veh-s)",
            f"{comparison['fixed_wait_time']:.1f}",
            f"{comparison['adaptive_wait_time']:.1f}",
            f"{comparison['wait_reduction_pct']}%"
        )
        table.add_row(
            "Average Wait per Vehicle",
            f"{comparison.get('fixed_avg_wait', 0):.1f}s",
            f"{comparison['adaptive_avg_wait']}s",
            "Significant" if comparison['wait_reduction_pct'] > 20 else "Moderate"
        )
        cycle_delta_pct = 0.0
        if comparison["fixed_cycle_time"] > 0:
            cycle_delta_pct = ((comparison["fixed_cycle_time"] - comparison["adaptive_cycle_time"]) / comparison["fixed_cycle_time"]) * 100
        cycle_label = f"{abs(cycle_delta_pct):.1f}% faster" if cycle_delta_pct >= 0 else f"{abs(cycle_delta_pct):.1f}% slower"
        table.add_row(
            "Cycle Duration",
            f"{comparison['fixed_cycle_time']}s",
            f"{comparison['adaptive_cycle_time']}s",
            cycle_label
        )
        
        self.console.print("\n")
        self.console.print(table)

    def run_three_scenarios(self, detector, controller) -> List[Dict[str, Any]]:
        """Run benchmark on three predefined traffic scenarios."""
        scenarios = [
            {
                "name": "Low Density",
                "counts": {
                    "North": {"car": 4, "motorcycle": 2},
                    "South": {"car": 3},
                    "East": {"car": 5, "motorcycle": 1},
                    "West": {"car": 4}
                }
            },
            {
                "name": "Medium Density",
                "counts": {
                    "North": {"car": 12, "bus": 2},
                    "South": {"car": 8, "motorcycle": 4},
                    "East": {"car": 15, "truck": 1},
                    "West": {"car": 10}
                }
            },
            {
                "name": "High Density (Congested)",
                "counts": {
                    "North": {"car": 25, "bus": 4, "truck": 2},
                    "South": {"car": 30, "bus": 2},
                    "East": {"car": 35, "motorcycle": 10},
                    "West": {"car": 28, "truck": 1}
                }
            }
        ]
        
        fixed_sim = FixedTimerSimulator(fixed_green_time=30)
        results = []
        
        for sc in scenarios:
            fresh_controller = SignalController(controller.config)
            simulator = IntersectionSimulator(controller.config, detector, fresh_controller)
            adaptive_res = simulator.run_from_counts(sc["counts"])
            fixed_res = fixed_sim.simulate(sc["counts"])
            comp = self.compare(adaptive_res, fixed_res)
            comp["scenario_name"] = sc["name"]
            comp["adaptive_result"] = adaptive_res
            comp["fixed_result"] = fixed_res
            results.append(comp)
            self.generate_report(comp, sc["name"])
            
        return results

import os
import json
import pandas as pd
from loguru import logger
from src.traffic.intersection_simulator import IntersectionSimulator
from src.traffic.signal_controller import SignalController
from src.comparison.performance_evaluator import PerformanceEvaluator
from src.visualization.results_plotter import ResultsPlotter
from src.logging.session_logger import SessionLogger
from rich.console import Console

console = Console()

def run_demo():
    """Runs a full simulation demo with synthetic data."""
    console.print("[bold green]🚀 Starting Smart Traffic Optimization Demo[/]\n")
    
    config = {
        "detection": {"target_classes": {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}},
        "vehicle_weights": {"car": 1.0, "motorcycle": 0.5, "bus": 2.5, "truck": 3.0, "unknown": 1.0},
        "signal_timing": {
            "t_min": 10, "t_max": 90, "k_factor": 1.5,
            "yellow_time": 3, "all_red_time": 2,
            "starvation_threshold": 2, "priority_multiplier": 1.4
        },
        "simulation": {"lane_names": ["North", "South", "East", "West"], "default_fixed_timer": 30},
        "logging": {"output_dir": "outputs/logs", "csv_filename": "demo_session.csv", "log_level": "INFO"},
        "visualization": {"charts_dir": "outputs/charts", "annotated_dir": "outputs/annotated"}
    }
    
    # Initialize components
    # We use None for detector in run_from_counts
    simulator = IntersectionSimulator(config, detector=None)
    evaluator = PerformanceEvaluator()
    plotter = ResultsPlotter(config)
    logger_inst = SessionLogger(config)
    
    # Define mixed traffic scenario
    lane_counts = {
        "North": {"car": 12, "bus": 3, "motorcycle": 2},
        "South": {"car": 5, "truck": 1},
        "East": {"car": 20, "motorcycle": 5},
        "West": {"car": 3}
    }
    
    # 1. Run Adaptive Simulation
    adaptive_res = simulator.run_from_counts(lane_counts)
    
    # 2. Run Fixed Comparison
    from src.comparison.fixed_timer import FixedTimerSimulator
    fixed_sim = FixedTimerSimulator(fixed_green_time=30)
    fixed_res = fixed_sim.simulate(lane_counts)
    
    # 3. Evaluate Performance
    comparison = evaluator.compare(adaptive_res, fixed_res)
    evaluator.generate_report(comparison, "Demo Scenario")
    
    # 4. Log Results
    logger_inst.log_cycle(adaptive_res.cycle_summary)
    logger_inst.save_csv()
    
    # 5. Plot Results
    plotter.plot_signal_timing_comparison(adaptive_res.signal_plan, 30, "outputs/charts/demo_comparison.png")
    
    console.print("\n[bold green]✅ Demo Complete![/] Check outputs/ directory for results.")

if __name__ == "__main__":
    run_demo()

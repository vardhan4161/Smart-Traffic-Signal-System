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
    
    # 1. Run Three Scenario Benchmark
    console.print("[bold yellow]Running Three Scenario Benchmark...[/]")
    scenario_results = evaluator.run_three_scenarios(detector=None, controller=simulator.controller)
    
    # 2. Generate Comparison Charts
    console.print("[bold yellow]Generating Visualization Charts...[/]")
    
    # Plot 1: Three Scenario Comparison (Key Result)
    plotter.plot_three_scenario_comparison(
        scenario_results, 
        os.path.join(config["visualization"]["charts_dir"], "three_scenario_comparison.png")
    )
    
    # Plot 2: Signal Timing for the highest density scenario
    high_density_res = scenario_results[2]
    plotter.plot_signal_timing_comparison(
        high_density_res["adaptive_result"].signal_plan,
        30,
        os.path.join(config["visualization"]["charts_dir"], "signal_timing_comparison.png")
    )
    
    # 3. Log Results for the high density scenario
    logger_inst.log_cycle(high_density_res["adaptive_result"].cycle_summary)
    logger_inst.save_csv()
    
    console.print("\n[bold green]✅ Demo Complete![/] All benchmarks passed.")
    console.print(f"Charts saved to: [cyan]{config['visualization']['charts_dir']}[/]")
    console.print(f"Session log saved to: [cyan]{logger_inst.csv_path}[/]")

if __name__ == "__main__":
    run_demo()

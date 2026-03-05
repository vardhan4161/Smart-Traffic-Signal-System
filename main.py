import click
import yaml
import os
from loguru import logger
from rich.console import Console
from rich.table import Table
from src.detector.vehicle_detector import VehicleDetector
from src.traffic.intersection_simulator import IntersectionSimulator
from src.comparison.performance_evaluator import PerformanceEvaluator
from src.logging.session_logger import SessionLogger
from src.dashboard.gui_dashboard import GUIDashboard

console = Console()

def load_config():
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)

@click.group()
def cli():
    """Smart Traffic Signal Optimization System CLI."""
    pass

@cli.command()
@click.option("--video", required=True, help="Path to video file")
def detect(video):
    """Run vehicle detection on a video file."""
    config = load_config()
    detector = VehicleDetector(config)
    console.print(f"[bold cyan]Running detection on:[/] {video}")
    detections = detector.detect_video(video)
    console.print(f"[bold green]Complete![/] Total frames processed: {len(detections)}")

@cli.command()
@click.option("--mode", type=click.Choice(["counts", "video"]), default="counts")
def simulate(mode):
    """Run intersection simulation."""
    config = load_config()
    simulator = IntersectionSimulator(config)
    logger_inst = SessionLogger(config)
    
    if mode == "counts":
        # Sample data
        counts = {
            "North": {"car": 12, "bus": 2},
            "South": {"car": 8},
            "East": {"car": 20, "motorcycle": 5},
            "West": {"car": 4}
        }
        result = simulator.run_from_counts(counts)
        
        # Display table
        table = Table(title="Simulation Result - Adaptive Signal Plan")
        table.add_column("Lane")
        table.add_column("Density")
        table.add_column("Level")
        table.add_column("Green Time")
        
        for lane, data in result.signal_plan.items():
            table.add_row(lane, str(data["density"]), data["density_level"], f"{data['green_time']}s")
            
        console.print(table)
        logger_inst.log_cycle(result.cycle_summary)
        logger_inst.save_csv()

@cli.command()
def compare():
    """Benchmark adaptive system against fixed-timer across scenarios."""
    config = load_config()
    detector = VehicleDetector(config)
    evaluator = PerformanceEvaluator()
    # Mocking controller for quick check
    from src.traffic.signal_controller import SignalController
    controller = SignalController(config)
    
    console.print("[bold yellow]Running Benchmarks...[/]")
    evaluator.run_three_scenarios(detector, controller)

@cli.command()
def dashboard():
    """Launch the GUI monitor."""
    gui = GUIDashboard()
    gui.run()

@cli.command()
def report():
    """Generate summary report from latest log."""
    config = load_config()
    logger_inst = SessionLogger(config)
    stats = logger_inst.get_summary_stats()
    
    if stats:
        console.print("[bold green]Session Summary Stats:[/]")
        for k, v in stats.items():
            console.print(f"{k.replace('_', ' ').title()}: [bold cyan]{v}[/]")
    else:
        console.print("[red]No session logs found.[/]")

if __name__ == "__main__":
    cli()

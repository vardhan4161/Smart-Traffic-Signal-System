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
    """Run vehicle detection with tracking on a video file."""
    config = load_config()
    from src.detector.vehicle_detector import VehicleDetector
    from src.traffic.density_calculator import DensityCalculator
    
    detector = VehicleDetector(config)
    calc = DensityCalculator(config)
    
    console.print(f"[bold cyan]🔍 Analyzing Traffic Video:[/] {os.path.basename(video)}")
    
    with console.status("[bold yellow]Inference + Tracking...[/]") as status:
        results = detector.process_video(video)
        
    # Display Results Table
    table = Table(title=f"Detection Report — {os.path.basename(video)}")
    table.add_column("Vehicle", style="cyan")
    table.add_column("Count", justify="center", style="green")
    table.add_column("Density Weight", justify="right", style="magenta")
    
    counts = results["counts"]
    weights = config.get("vehicle_weights", {})
    total_density = 0.0
    
    emojis = {"car": "🚗", "motorcycle": "🏍️", "bus": "🚌", "truck": "🚛"}
    
    for cls in ["car", "motorcycle", "bus", "truck"]:
        count = counts.get(cls, 0)
        weight = weights.get(cls, 1.0)
        d_val = count * weight
        total_density += d_val
        table.add_row(f"{emojis.get(cls, '🚙')} {cls.capitalize()}s", str(count), f"{d_val:.1f}")
        
    table.add_section()
    level = calc.classify_density_level(total_density)
    table.add_row("[bold]TOTAL[/]", f"[bold]{results['total']}[/]", f"[bold]{total_density:.1f} ({level})[/]")
    
    console.print(table)
    
    # Timing info
    green_time = calc.calculate_green_time(total_density)
    fixed_time = config.get("simulation", {}).get("default_fixed_timer", 30)
    diff = green_time - fixed_time
    diff_str = f"+{diff:.1f}s" if diff >= 0 else f"{diff:.1f}s"
    
    console.print(f"[bold yellow]Assigned Green Time:[/] {green_time}s  (vs Fixed {fixed_time}s: {diff_str})")

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

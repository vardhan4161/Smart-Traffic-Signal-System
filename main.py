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
@click.option("--north", default=None, help="Video path for North lane")
@click.option("--south", default=None, help="Video path for South lane")
@click.option("--east",  default=None, help="Video path for East lane")
@click.option("--west",  default=None, help="Video path for West lane")
def simulate(mode, north, south, east, west):
    """Run intersection simulation."""
    config = load_config()
    simulator = IntersectionSimulator(config)
    logger_inst = SessionLogger(config)
    
    if mode == "video":
        lane_videos = {
            "North": north, "South": south, "East": east, "West": west
        }
        missing = [k for k, v in lane_videos.items() if not v]
        if missing:
            console.print(f"[red]Missing video paths for lanes: {', '.join(missing)}[/]")
            console.print("[yellow]Usage:[/] py main.py simulate --mode video --north north.mp4 --south south.mp4 --east east.mp4 --west west.mp4")
            return
        
        from src.detector.vehicle_detector import VehicleDetector
        from src.traffic.density_calculator import DensityCalculator
        from src.traffic.signal_controller import SignalController
        
        detector = VehicleDetector(config)
        calc = DensityCalculator(config)
        controller = SignalController(config)
        
        lane_counts = {}
        
        for lane_name, video_path in lane_videos.items():
            video_path = os.path.abspath(video_path)
            if not os.path.exists(video_path):
                console.print(f"[red]⚠ File not found: {video_path}[/]")
                continue
            console.print(f"[cyan]Analyzing {lane_name}: {os.path.basename(video_path)}...[/]")
            with console.status(f"[yellow]Processing {lane_name}...[/]"):
                detector.reset_tracking()
                result = detector.process_video(video_path)
            lane_counts[lane_name] = result["counts"]
            density = calc.calculate_weighted_density(result["counts"])
            green_time = calc.calculate_green_time(density)
            console.print(f"  [green]✅ {lane_name}:[/] {result['total']} vehicles | density={density:.1f} | green={green_time:.0f}s")
        
        if not lane_counts:
            console.print("[red]No lanes were successfully processed.[/]")
            return
        
        # Run adaptive signal plan
        sim_result = simulator.run_from_counts(lane_counts)
        
        # Print summary table
        table = Table(title="🚦 4-Lane Intersection Signal Plan (Adaptive vs Fixed)")
        table.add_column("Lane", style="cyan")
        table.add_column("Vehicles", justify="center")
        table.add_column("Density", justify="center")
        table.add_column("Level", justify="center")
        table.add_column("Adaptive Green", justify="center", style="green")
        table.add_column("Fixed (30s)", justify="center", style="red")
        table.add_column("Improvement", justify="center", style="yellow")
        
        fixed_time = config.get("simulation", {}).get("default_fixed_timer", 30)
        weights = config.get("vehicle_weights", {})
        
        for lane, plan in sim_result.signal_plan.items():
            counts = lane_counts.get(lane, {})
            total_v = sum(counts.values())
            density = plan["density"]
            level = plan["density_level"]
            green = plan["green_time"]
            diff = green - fixed_time
            diff_str = f"+{diff:.1f}s" if diff >= 0 else f"{diff:.1f}s"
            table.add_row(lane, str(total_v), f"{density:.1f}", level, f"{green}s", f"{fixed_time}s", diff_str)
        
        console.print(table)
        logger_inst.log_cycle(sim_result.cycle_summary)
        logger_inst.save_csv()
        console.print("[bold green]✅ Session log saved to outputs/logs/session_log.csv[/]")

    else:
        # Hardcoded sample counts demo
        counts = {
            "North": {"car": 12, "bus": 2},
            "South": {"car": 8},
            "East": {"car": 20, "motorcycle": 5},
            "West": {"car": 4}
        }
        result = simulator.run_from_counts(counts)
        
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

import yaml
from rich.console import Console
from rich.table import Table

from src.comparison.fixed_timer import FixedTimerSimulator
from src.comparison.performance_evaluator import PerformanceEvaluator
from src.traffic.intersection_simulator import IntersectionSimulator
from src.traffic.signal_controller import SignalController


console = Console()


def load_config():
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    counts = {
        "North": {"car": 40, "bus": 5},
        "South": {"car": 2},
        "East": {"car": 1},
        "West": {"car": 2},
    }

    controller = SignalController(config)
    simulator = IntersectionSimulator(config, detector=None, controller=controller)
    adaptive = simulator.run_from_counts(counts)
    fixed = FixedTimerSimulator(30).simulate(counts)
    comparison = PerformanceEvaluator().compare(adaptive, fixed)

    console.print("[bold cyan]Peak-Lane Jury Scenario[/]")
    console.print(
        f"Adaptive wait reduction: [bold green]{comparison['wait_reduction_pct']}%[/] | "
        f"Adaptive cycle: [bold]{comparison['adaptive_cycle_time']}s[/] vs Fixed [bold]{comparison['fixed_cycle_time']}s[/]"
    )

    table = Table(title="Lane Allocation")
    table.add_column("Lane", style="cyan")
    table.add_column("Counts", style="white")
    table.add_column("Density", justify="right", style="magenta")
    table.add_column("Green Time", justify="right", style="green")

    for lane, plan in adaptive.signal_plan.items():
        table.add_row(
            lane,
            str(adaptive.lane_vehicle_counts[lane]),
            f"{plan['density']:.1f}",
            f"{plan['green_time']}s",
        )

    console.print(table)


if __name__ == "__main__":
    main()

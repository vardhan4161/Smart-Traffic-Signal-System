import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import Dict, List, Any
from loguru import logger

class ResultsPlotter:
    """Generates charts and visualizations for traffic performance analysis."""
    
    def __init__(self, config: dict):
        """Set visualization styles."""
        self.charts_dir = config.get("visualization", {}).get("charts_dir", "outputs/charts")
        os.makedirs(self.charts_dir, exist_ok=True)
        sns.set_theme(style="darkgrid")
        self.palette = sns.color_palette("husl", 8)

    def plot_signal_timing_comparison(self, adaptive_plan: Dict[str, Any], fixed_time: int, save_path: str):
        """Generate a bar chart comparing adaptive green times against fixed-timer baseline."""
        try:
            lanes = list(adaptive_plan.keys())
            adaptive_times = [data["green_time"] for data in adaptive_plan.values()]
            fixed_times = [fixed_time] * len(lanes)
            
            df = pd.DataFrame({
                "Lane": lanes * 2,
                "Green Time (s)": adaptive_times + fixed_times,
                "Type": ["Adaptive"] * len(lanes) + ["Fixed (30s)"] * len(lanes)
            })
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x="Lane", y="Green Time (s)", hue="Type")
            plt.title("Adaptive vs Fixed-Timer Signal Allocation", fontsize=14)
            plt.ylabel("Seconds")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Signal timing comparison chart saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to plot signal comparison: {e}")

    def plot_density_vs_green_time(self, session_log_df: pd.DataFrame, save_path: str):
        """Generate a scatter plot of weighted density versus assigned green time."""
        try:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=session_log_df, x="weighted_density", y="assigned_green_time", 
                            hue="density_level", style="density_level", s=100)
            sns.regplot(data=session_log_df, x="weighted_density", y="assigned_green_time", 
                        scatter=False, color="gray", line_kws={"linestyle": "--"})
            
            plt.title("Traffic Density vs Assigned Green Time", fontsize=14)
            plt.xlabel("Weighted Density Score")
            plt.ylabel("Green Time (s)")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            logger.error(f"Failed to plot density vs green time: {e}")

    def plot_vehicle_composition(self, vehicle_counts: Dict[str, int], lane_name: str, save_path: str):
        """Generate a pie chart of vehicle type breakdown for a lane."""
        try:
            # Filter zero counts
            data = {k: v for k, v in vehicle_counts.items() if v > 0}
            if not data: return
            
            plt.figure(figsize=(8, 8))
            plt.pie(data.values(), labels=data.keys(), autopct='%1.1f%%', 
                    colors=sns.color_palette("pastel"), startangle=140)
            plt.title(f"Vehicle Composition: {lane_name} Lane", fontsize=14)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            logger.error(f"Failed to plot vehicle composition: {e}")

    def plot_three_scenario_comparison(self, scenario_results: List[Dict[str, Any]], save_path: str):
        """Generate a 3-panel comparison of wait time reduction across scenarios."""
        try:
            scenarios = [r["scenario_name"] for r in scenario_results]
            fixed_waits = [r["fixed_wait_time"] for r in scenario_results]
            adaptive_waits = [r["adaptive_wait_time"] for r in scenario_results]
            reductions = [r["wait_reduction_pct"] for r in scenario_results]
            
            fig, ax1 = plt.subplots(figsize=(12, 7))
            
            x = range(len(scenarios))
            width = 0.35
            
            ax1.bar([p - width/2 for p in x], fixed_waits, width, label='Fixed Wait', color='salmon')
            ax1.bar([p + width/2 for p in x], adaptive_waits, width, label='Adaptive Wait', color='skyblue')
            
            ax1.set_xlabel('Scenario')
            ax1.set_ylabel('Total Waiting Time (veh-s)')
            ax1.set_title('Efficiency Gains Across Traffic Scenarios', fontsize=16)
            ax1.set_xticks(x)
            ax1.set_xticklabels(scenarios)
            ax1.legend(loc='upper left')
            
            # Add reduction percentages as text
            for i, rect in enumerate(ax1.patches[len(scenarios):]): # adaptive bars
                height = rect.get_height()
                ax1.annotate(f'-{reductions[i]}%', 
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontweight='bold', color='green')

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Performance report chart saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to plot scenario comparison: {e}")

    def generate_all_charts(self, results: Dict[str, Any], output_dir: str) -> List[str]:
        """Sequence call for all plotting functions."""
        paths = []
        # Not fully implemented here as it depends on exact results structure
        # but provided as a stub for the pipeline.
        return paths

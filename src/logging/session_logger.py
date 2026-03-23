import pandas as pd
import os
import json
from loguru import logger
from datetime import datetime
from typing import Dict, Any

class SessionLogger:
    """Handles logging of simulation cycles and data persistence."""
    
    def __init__(self, config: dict):
        """Initialize loggers and data structure."""
        log_config = config.get("logging", {})
        self.output_dir = log_config.get("output_dir", "outputs/logs")
        self.filename = log_config.get("csv_filename", "session_log.csv")
        self.csv_path = os.path.join(self.output_dir, self.filename)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup loguru
        logger.add(os.path.join(self.output_dir, "session.log"), rotation="10 MB", level=log_config.get("log_level", "INFO"))
        
        # Initialize empty DataFrame
        self.columns = [
            "timestamp", "cycle_number", "lane_name", "vehicle_counts_json",
            "weighted_density", "density_level", "assigned_green_time",
            "was_boosted", "total_cycle_time"
        ]
        self.df = pd.DataFrame(columns=self.columns)

    def log_cycle(self, cycle_summary: Dict[str, Any]):
        """Record data from a single simulation cycle."""
        try:
            timestamp = cycle_summary.get("timestamp", datetime.now().isoformat())
            cycle_num = cycle_summary.get("cycle_number", 0)
            total_cycle_time = cycle_summary.get("total_cycle_time", 0.0)
            plan = cycle_summary.get("plan", {})
            
            new_rows = []
            for lane, data in plan.items():
                row = {
                    "timestamp": timestamp,
                    "cycle_number": cycle_num,
                    "lane_name": lane,
                    "vehicle_counts_json": json.dumps(data.get("counts", {})),
                    "weighted_density": data.get("density", 0.0),
                    "density_level": data.get("density_level", "UNKNOWN"),
                    "assigned_green_time": data.get("green_time", 0.0),
                    "was_boosted": data.get("boosted", False),
                    "total_cycle_time": total_cycle_time
                }
                new_rows.append(row)
            
            self.df = pd.concat([self.df, pd.DataFrame(new_rows)], ignore_index=True)
            logger.info(f"Logged cycle {cycle_num} with {len(plan)} lanes.")
        except Exception as e:
            logger.error(f"Failed to log cycle: {e}")

    def save_csv(self, path: str = None):
        """Save the accumulated log data to a CSV file."""
        try:
            target_path = path or self.csv_path
            self.df.to_csv(target_path, index=False)
            logger.info(f"Session data saved to {target_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute basic statistics from the logged data."""
        if self.df.empty:
            return {}
        
        try:
            stats = {
                "total_cycles": int(self.df["cycle_number"].nunique()),
                "avg_green_time": round(float(self.df["assigned_green_time"].mean()), 2),
                "busiest_lane": self.df.groupby("lane_name")["weighted_density"].mean().idxmax(),
                "most_common_density": self.df["density_level"].mode()[0] if not self.df["density_level"].mode().empty else "N/A"
            }
            return stats
        except Exception as e:
            logger.error(f"Error computing summary stats: {e}")
            return {}

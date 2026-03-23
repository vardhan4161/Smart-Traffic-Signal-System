import json
from pathlib import Path

from src.logging.session_logger import SessionLogger


def test_log_cycle_persists_lane_counts():
    tmp_path = Path(".tmp/test_session_logger")
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {
        "logging": {
            "output_dir": str(tmp_path),
            "csv_filename": "session_log.csv",
            "log_level": "INFO",
        }
    }
    logger_inst = SessionLogger(config)
    cycle_summary = {
        "timestamp": "2026-03-22T12:00:00",
        "cycle_number": 1,
        "total_cycle_time": 55.0,
        "plan": {
            "North": {
                "counts": {"car": 4, "bus": 1},
                "density": 6.5,
                "density_level": "MEDIUM",
                "green_time": 29.8,
                "boosted": False,
            }
        },
    }

    logger_inst.log_cycle(cycle_summary)

    assert len(logger_inst.df) == 1
    assert json.loads(logger_inst.df.iloc[0]["vehicle_counts_json"]) == {"car": 4, "bus": 1}

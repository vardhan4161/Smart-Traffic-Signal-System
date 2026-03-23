from src.traffic.intersection_simulator import IntersectionSimulator


class StubDetector:
    def __init__(self, results):
        self.results = results

    def process_video(self, path):
        return self.results[path]


def test_run_from_videos_uses_process_video_results():
    config = {
        "simulation": {"lane_names": ["North", "South"]},
        "signal_timing": {"t_min": 10, "t_max": 90, "k_factor": 1.5},
        "vehicle_weights": {"car": 1.0, "bus": 2.5, "unknown": 1.0},
    }
    detector = StubDetector(
        {
            "north.mp4": {"counts": {"car": 3, "bus": 1}},
            "south.mp4": {"counts": {"car": 2}},
        }
    )

    sim = IntersectionSimulator(config, detector=detector)
    result = sim.run_from_videos({"North": "north.mp4", "South": "south.mp4"})

    assert result.lane_vehicle_counts["North"] == {"car": 3, "bus": 1}
    assert result.lane_densities["North"] == 5.5
    assert result.signal_plan["North"]["counts"] == {"car": 3, "bus": 1}


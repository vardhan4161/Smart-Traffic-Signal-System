import pytest
from src.traffic.signal_controller import SignalController

@pytest.fixture
def config():
    return {
        "simulation": {"lane_names": ["North", "South", "East", "West"]},
        "signal_timing": {
            "t_min": 10, "t_max": 90, "k_factor": 1.5,
            "yellow_time": 3, "all_red_time": 2,
            "starvation_threshold": 2, "priority_multiplier": 1.4
        },
        "vehicle_weights": {"car": 1.0, "unknown": 1.0}
    }

def test_signal_plan_returns_all_lanes(config):
    controller = SignalController(config)
    densities = {"North": 10.0, "South": 5.0, "East": 20.0, "West": 2.0}
    plan = controller.compute_signal_plan(densities)
    assert len(plan) == 4
    assert set(plan.keys()) == set(config["simulation"]["lane_names"])

def test_starvation_counter_increments(config):
    controller = SignalController(config)
    densities = {"North": 10.0, "South": 5.0, "East": 20.0, "West": 2.0}
    controller.compute_signal_plan(densities)
    assert controller.starvation_counter["North"] == 1
    controller.compute_signal_plan(densities)
    assert controller.starvation_counter["North"] == 2

def test_starvation_boost_applied_after_threshold(config):
    controller = SignalController(config)
    densities = {"North": 10.0, "South": 10.0, "East": 10.0, "West": 10.0}
    
    # Run 2 cycles to reach threshold=2
    controller.compute_signal_plan(densities)
    controller.compute_signal_plan(densities)
    
    # 3rd cycle should apply boost
    plan = controller.compute_signal_plan(densities)
    assert plan["North"]["boosted"] is True
    # Counter should reset
    assert controller.starvation_counter["North"] == 0

def test_priority_order_descending(config):
    controller = SignalController(config)
    plan = {
        "North": {"green_time": 20.0},
        "South": {"green_time": 40.0},
        "East": {"green_time": 10.0},
        "West": {"green_time": 60.0}
    }
    order = controller.priority_order(plan)
    assert order == ["West", "South", "North", "East"]

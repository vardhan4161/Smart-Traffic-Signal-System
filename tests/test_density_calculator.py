import pytest
from src.traffic.density_calculator import DensityCalculator

@pytest.fixture
def config():
    return {
        "vehicle_weights": {"car": 1.0, "motorcycle": 0.5, "bus": 2.5, "truck": 3.0, "unknown": 1.0},
        "signal_timing": {"t_min": 10, "t_max": 90, "k_factor": 1.5}
    }

def test_weighted_density_cars_only(config):
    calc = DensityCalculator(config)
    counts = {"car": 10}
    assert calc.calculate_weighted_density(counts) == 10.0

def test_weighted_density_mixed_vehicles(config):
    calc = DensityCalculator(config)
    counts = {"car": 10, "bus": 2, "motorcycle": 4} # 10*1 + 2*2.5 + 4*0.5 = 10 + 5 + 2 = 17
    assert calc.calculate_weighted_density(counts) == 17.0

def test_green_time_minimum_enforced(config):
    calc = DensityCalculator(config)
    assert calc.calculate_green_time(0) == 10.0
    assert calc.calculate_green_time(-5) == 10.0

def test_green_time_maximum_enforced(config):
    calc = DensityCalculator(config)
    assert calc.calculate_green_time(100) == 90.0

def test_density_classification_all_levels(config):
    calc = DensityCalculator(config)
    assert calc.classify_density_level(3) == "LOW"
    assert calc.classify_density_level(8) == "MEDIUM"
    assert calc.classify_density_level(20) == "HIGH"
    assert calc.classify_density_level(40) == "CRITICAL"

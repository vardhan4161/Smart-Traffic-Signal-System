from loguru import logger

class DensityCalculator:
    """Calculates weighted traffic density and assigned green signal time."""
    
    def __init__(self, config: dict):
        """Load weights and signal timing parameters from config."""
        self.weights = config.get("vehicle_weights", {})
        timing_config = config.get("signal_timing", {})
        self.t_min = timing_config.get("t_min", 10)
        self.t_max = timing_config.get("t_max", 90)
        self.k_factor = timing_config.get("k_factor", 1.5)

    def calculate_weighted_density(self, vehicle_counts: dict) -> float:
        """Calculate total weighted density score for a lane."""
        density = 0.0
        for v_type, count in vehicle_counts.items():
            weight = self.weights.get(v_type, self.weights.get("unknown", 1.0))
            density += count * weight
        return density

    def calculate_green_time(self, density: float) -> float:
        """Assign green signal time using the intelligent formula."""
        # Formula: Tg = clamp(T_min + density * k_factor, T_min, T_max)
        tg = self.t_min + (density * self.k_factor)
        tg = max(self.t_min, min(tg, self.t_max))
        return round(float(tg), 1)

    def classify_density_level(self, density: float) -> str:
        """Classify numerical density into categorical labels."""
        if density < 5:
            return "LOW"
        elif 5 <= density < 15:
            return "MEDIUM"
        elif 15 <= density < 30:
            return "HIGH"
        else:
            return "CRITICAL"

"""Emission estimation helpers for the mixed fleet environment."""


class EmissionModel:
    """Rudimentary emission calculator for fuel vehicles."""

    def __init__(self, co2_per_unit_fuel: float):
        self.co2_per_unit_fuel = co2_per_unit_fuel

    def estimate(self, fuel_consumed: float) -> float:
        """Return estimated CO2 emission for the consumed fuel."""
        return max(0.0, fuel_consumed) * self.co2_per_unit_fuel


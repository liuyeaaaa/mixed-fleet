"""Energy consumption and recharge helpers for the mixed fleet environment."""

import numpy as np


class EnergyModel:
    """Lightweight utilities to model electric/fuel consumption and charging."""

    def __init__(self, elec_rate: float, fuel_rate: float):
        # per-unit-distance consumption
        self.elec_rate = elec_rate
        self.fuel_rate = fuel_rate

    def consume_electric(self, soc: float, distance: float) -> float:
        """Return remaining state-of-charge after traveling the given distance."""
        consumed = distance * self.elec_rate
        return max(0.0, soc - consumed)

    def consume_fuel(self, fuel: float, distance: float) -> float:
        """Return remaining fuel after traveling the given distance."""
        consumed = distance * self.fuel_rate
        return max(0.0, fuel - consumed)

    @staticmethod
    def recharge(current: float, amount: float, cap: float) -> float:
        """Top up with the given amount but do not exceed capacity."""
        return float(np.clip(current + amount, 0.0, cap))


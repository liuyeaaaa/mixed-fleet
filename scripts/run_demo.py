"""Quick smoke test for MixedFleetEnv.

Usage:
    python scripts/run_demo.py
"""

import sys
from pathlib import Path
import numpy as np

# Ensure the env modules are importable
ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "envs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ENV_DIR))

from mixed_fleet_env import MixedFleetEnv  # noqa: E402


def main():
    np.random.seed(0)

    config = {
        "num_zones": 16,
        "grid_shape": (16, 16),
        "fleet": {"AET": 1, "HET": 1, "HGT": 1},
        "initial_soc": 10.0,
        "initial_fuel": 8.0,
        "soc_cap": 12.0,
        "fuel_cap": 10.0,
        "lambda_matrix": np.full((16, 16), 0.2),
        "elec_rate": 0.5,
        "fuel_rate": 0.3,
        "charge_stations": [0],
        "fuel_stations": [3],
        "max_time_steps": 5,
        "fare_per_trip": 5.0,
        "elec_price": 1.0,
        "fuel_price": 1.0,
        "co2_price": 0.5,
        "co2_per_unit_fuel": 2.0,
    }

    env = MixedFleetEnv(config)
    state = env.reset(config)
    print("Initial state:")
    print({k: v for k, v in state.items() if k != "grid_demand"})
    print("Initial demand matrix:\n", env.demand_matrix)

    # One step of all-WAIT actions to exercise state update and reward path.
    actions = {vid: 0 for vid in range(len(state["vehicle_pos"]))}
    next_state, reward, done, info = env.step(actions)

    print("\nAfter one step (all WAIT):")
    print({k: v for k, v in next_state.items() if k != "grid_demand"})
    print("Demand matrix:\n", env.demand_matrix)
    print("Reward:", reward)
    print("Done:", done, "Info:", info)


if __name__ == "__main__":
    main()

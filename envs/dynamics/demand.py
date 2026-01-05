"""Demand generation helpers for the mixed fleet environment."""

import numpy as np


class PoissonDemand:
    """Generate origin-destination demand using a Poisson process."""

    def __init__(self, lambda_matrix: np.ndarray):
        self.lambda_matrix = np.asarray(lambda_matrix)

    def sample(self) -> np.ndarray:
        """Return a matrix of sampled trip counts for the next step."""
        return np.random.poisson(self.lambda_matrix)


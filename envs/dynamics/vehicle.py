"""Vehicle movement helpers for the mixed fleet environment."""


class VehicleKinematics:
    """Simple grid-based movement logic."""

    @staticmethod
    def manhattan_distance(src: int, dst: int, grid_shape) -> int:
        """Return Manhattan distance between two zone ids on an HxW grid."""
        h, w = grid_shape
        src_r, src_c = divmod(src, w)
        dst_r, dst_c = divmod(dst, w)
        return abs(src_r - dst_r) + abs(src_c - dst_c)

    @staticmethod
    def move(zone: int, target: int) -> int:
        """Deterministically move to the target zone (placeholder for routing)."""
        return target


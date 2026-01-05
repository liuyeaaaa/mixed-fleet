# models/action/action_mask.py
import numpy as np
from models.action.atomic_action import AtomicAction

class ActionMask:
    """
    根据当前状态，为单辆车生成动作合法性掩码
    """

    def __init__(self, num_zones, grid_shape, charge_stations, fuel_stations):
        self.num_zones = num_zones
        self.H, self.W = grid_shape
        self.charge_stations = set(charge_stations)
        self.fuel_stations = set(fuel_stations)
        # 额外的约束可在运行时注入，例如高需求区域优先
        self.allowed_zones = None
        self.demand_threshold = None

        self.atomic = AtomicAction(num_zones)

    def get_mask(
        self,
        vehicle_type,      # "AET", "HET", "HGT"
        vehicle_zone,      # 当前所在 zone id
        vehicle_energy,    # SOC or fuel
        vehicle_status,    # 0: idle, 1: busy
        demand_matrix=None # 可选的需求矩阵，用于限制 GOTO
    ):
        """
        返回 shape = [action_size] 的 0/1 mask
        """
        mask = np.zeros(self.atomic.action_size, dtype=np.int8)

        # ---------- 1. WAIT ----------
        # 忙碌车辆不能 WAIT（比如正在送客）
        if vehicle_status == 0:
            mask[self.atomic.WAIT] = 1

        # ---------- 2. GOTO_ZONE ----------
        # 能量必须 > 0
        if vehicle_energy > 0 and vehicle_status == 0:
            allowed = range(self.num_zones) if self.allowed_zones is None else self.allowed_zones
            # 若提供了需求矩阵且设置阈值，则只允许需求高于阈值的区域
            if demand_matrix is not None and self.demand_threshold is not None:
                demand_flat = demand_matrix.flatten()
                allowed = [z for z in allowed if demand_flat[z] >= self.demand_threshold]

            for zone in allowed:
                action_id = self.atomic.GOTO_START + zone
                mask[action_id] = 1

        # ---------- 3. CHARGE ----------
        if vehicle_type in ["AET", "HET"]:
            # 必须在充电站，且空闲
            if (vehicle_zone in self.charge_stations) and vehicle_status == 0:
                mask[self.atomic.CHARGE] = 1

        # ---------- 4. REFUEL ----------
        if vehicle_type == "HGT":
            if (vehicle_zone in self.fuel_stations) and vehicle_status == 0:
                mask[self.atomic.REFUEL] = 1

        return mask

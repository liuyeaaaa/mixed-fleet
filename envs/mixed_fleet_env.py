# envs/mixed_fleet_env.py
import numpy as np
from base_env import BaseEnv
from models.action.atomic_action import AtomicAction
from envs.dynamics.vehicle import VehicleKinematics
from envs.dynamics.energy import EnergyModel
from envs.dynamics.demand import PoissonDemand
from envs.dynamics.emission import EmissionModel

class MixedFleetEnv(BaseEnv):
    def __init__(self, config):
        """
        初始化混合车队仿真环境

        config: dict
          包含城市参数、出租车车队组成、需求分布等
        """
        self.config = config
        self.reset(config)

    def reset(self, config):
        """重置环境至初始状态"""
        self.time_step = 0

        # 城市区域数量
        self.num_zones = config["num_zones"]

        # 车队组成，三类车辆数量
        self.fleet_AET = config["fleet"]["AET"]
        self.fleet_HET = config["fleet"]["HET"]
        self.fleet_HGT = config["fleet"]["HGT"]

        # 初始化剩余 SOC / 油量
        self.soc_AET = np.ones(self.fleet_AET) * config["initial_soc"]
        self.fuel_HGT = np.ones(self.fleet_HGT) * config["initial_fuel"]
        self.vehicle_status = np.zeros(self.fleet_AET + self.fleet_HET + self.fleet_HGT, dtype=int)

        # 初始化车辆位置（随机均匀分布）
        self.vehicle_zones = np.random.randint(0, self.num_zones,
                                               size=(self.fleet_AET + self.fleet_HET + self.fleet_HGT))

        # 初始化需求矩阵
        self.demand_model = PoissonDemand(config["lambda_matrix"])
        self.demand_matrix = np.zeros((self.num_zones, self.num_zones))

        # 动力学模块
        self.vehicle_model = VehicleKinematics()
        self.energy_model = EnergyModel(
            elec_rate=config["elec_rate"],
            fuel_rate=config["fuel_rate"],
        )
        self.emission_model = EmissionModel(
            co2_per_unit_fuel=config.get("co2_per_unit_fuel", 1.0)
        )

        # 车辆能力上限
        self.soc_cap = config["soc_cap"]
        self.fuel_cap = config["fuel_cap"]

        self.done = False

        # 返回初始状态
        return self._get_state()

    def step(self, actions):
        """
        执行一步

        actions: dict
          包含各 agent 的动作（调度 / 订单匹配 / 充电 / 路径选择）
        """
        self.time_step += 1
        atomic = AtomicAction(self.num_zones)

        for veh_id, action_id in actions.items():
            action_type, target = atomic.decode(action_id)

            if action_type == "WAIT":
                pass

            elif action_type == "GOTO":
                self.vehicle_zones[veh_id] = target

            elif action_type == "CHARGE":
                self._send_to_charge(veh_id)

            elif action_type == "REFUEL":
                self._send_to_refuel(veh_id)

        # ===== 1) 解析动作 =====
        # 这里假设 actions 已经按车辆类型分组，例如：
        # actions["AET_dispatch"], actions["HET_dispatch"], actions["HGT_dispatch"]

        # 下面是伪代码，需要你具体填
        # for veh_idx, action in enumerate(actions["AET_dispatch"]):
        #     self._apply_action_AET(veh_idx, action)

        # ===== 2) 更新环境动态 =====
        self._update_vehicle_states()
        self._update_demand()

        # ===== 3) 获取下一个状态 =====
        next_state = self._get_state()

        # ===== 4) 奖励计算 =====
        reward = self._compute_reward()

        # ===== 5) 检查是否结束 =====
        self.done = (self.time_step >= self.config["max_time_steps"])

        info = {}

        return next_state, reward, self.done, info

    def render(self):
        # 你可以把车辆分布、当前 SOC 显示出来
        pass

    def _get_state(self):
        """
        返回当前状态：
        - vehicle_pos: 所有车辆位置 [num_vehicles]
        - vehicle_energy: 所有车辆剩余能量 [num_vehicles]
        - vehicle_status: 车辆状态（是否空闲）
        - grid_demand: 当前格网需求矩阵 [H, W]
        - charge_grid: 是否有充电站 [H, W]
        - fuel_grid: 是否有加油站 [H, W]
        - time_step: 当前时间步
        """
        state = {}

        # 地图格网结构
        H, W = self.config["grid_shape"]

        # 车辆总数
        num_vehicles = len(self.vehicle_zones)

        # 1) 车辆位置（格子索引）
        state["vehicle_pos"] = self.vehicle_zones.copy()

        # 2) 车辆能量
        # AET 在前面, 然后 HET, 最后 HGT
        state["vehicle_energy"] = np.concatenate([
            self.soc_AET,
            np.zeros(self.fleet_HET),  # HET 不需要能量
            self.fuel_HGT
        ])

        # 3) 车辆状态（0 = 空闲, 1 = 忙）
        state["vehicle_status"] = self.vehicle_status.copy()

        # 4) 当前需求(历史数据驱动)
        state["grid_demand"] = self.demand_matrix.reshape(H, W)

        # 5) 充电站网格（1 表示该格有充电站，否则 0）
        charge_grid = np.zeros((H, W), dtype=int)
        for cell in self.config["charge_stations"]:
            charge_grid[cell] = 1
        state["charge_grid"] = charge_grid

        # 6) 加油站网格（同上）
        fuel_grid = np.zeros((H, W), dtype=int)
        for cell in self.config["fuel_stations"]:
            fuel_grid[cell] = 1
        state["fuel_grid"] = fuel_grid

        # 7) 当前时间步
        state["time_step"] = self.time_step

        return state

    def _update_vehicle_states(self):
        """更新每辆车的状态（例如位置变化 / SOC 变化 / 充电状态）"""
        # 这里简单假设：如果车辆处于忙碌状态，维持忙碌直到外部逻辑重置；空闲车辆能量随距离衰减
        # 能量消耗基于上一步的移动距离（示例用 1 替代实际距离）
        distance = 1.0
        self.soc_AET = np.array([
            self.energy_model.consume_electric(soc, distance)
            for soc in self.soc_AET
        ])
        self.fuel_HGT = np.array([
            self.energy_model.consume_fuel(fuel, distance)
            for fuel in self.fuel_HGT
        ])

    def _update_demand(self):
        """更新需求矩阵"""
        self.demand_matrix = self.demand_model.sample()

    def _compute_reward(self):
        """计算 reward"""
        # 这个 reward 是全局 reward（平台 + 司机）
        # 简化示例：收入 - 能耗成本 - 排放成本
        revenue = self.demand_matrix.sum() * self.config.get("fare_per_trip", 1.0)
        elec_cost = (self.config.get("elec_price", 1.0) *
                     (self.config["initial_soc"] * self.fleet_AET - self.soc_AET.sum()))
        fuel_used = self.config["initial_fuel"] * self.fleet_HGT - self.fuel_HGT.sum()
        fuel_cost = self.config.get("fuel_price", 1.0) * fuel_used
        emission_cost = self.config.get("co2_price", 0.0) * self.emission_model.estimate(fuel_used)
        return revenue - elec_cost - fuel_cost - emission_cost

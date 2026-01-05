# envs/base_env.py
from abc import ABC, abstractmethod

class BaseEnv(ABC):
    """抽象环境基类，定义统一接口"""

    @abstractmethod
    def reset(self, config):
        """重置环境，返回初始状态"""
        pass

    @abstractmethod
    def step(self, actions):
        """执行动作，返回 (next_state, reward, done, info)"""
        pass

    @abstractmethod
    def render(self):
        """可视化（可选）"""
        pass

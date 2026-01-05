# models/action/atomic_action.py

class AtomicAction:
    """
    原子动作编码器
    """
    WAIT = 0

    def __init__(self, num_zones):
        self.num_zones = num_zones

        # 动作编号
        self.GOTO_START = 1
        self.GOTO_END = self.GOTO_START + num_zones - 1

        self.CHARGE = self.GOTO_END + 1
        self.REFUEL = self.GOTO_END + 2

        self.action_size = self.GOTO_END + 3

    def decode(self, action_id):
        """
        把 action_id 解码为可执行语义
        """
        if action_id == self.WAIT:
            return ("WAIT", None)

        elif self.GOTO_START <= action_id <= self.GOTO_END:
            zone_id = action_id - self.GOTO_START
            return ("GOTO", zone_id)

        elif action_id == self.CHARGE:
            return ("CHARGE", None)

        elif action_id == self.REFUEL:
            return ("REFUEL", None)

        else:
            raise ValueError(f"Invalid action id: {action_id}")

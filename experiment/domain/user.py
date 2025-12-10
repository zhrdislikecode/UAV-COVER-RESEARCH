import numpy as np

class User:
    def __init__(self, position):
        self.position = np.array(position)
        self.cover_num = 0
        self.direction = None


    def move(self, direction):
        """根据方向更新用户的位置"""
        self.position += direction
        self.direction = direction


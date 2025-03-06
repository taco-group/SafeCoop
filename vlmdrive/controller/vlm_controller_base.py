import numpy as np
from collections import deque
from abc import ABC, abstractmethod

class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class VLMControllerBase(ABC, object):
    def __init__(self, config):
        self.turn_controller = PIDController(
            K_P=config['turn_KP'], 
            K_I=config['turn_KI'], 
            K_D=config['turn_KD'], 
            n=config['turn_n']
        )
        self.speed_controller = PIDController(
            K_P=config['speed_KP'],
            K_I=config['speed_KI'],
            K_D=config['speed_KD'],
            n=config['speed_n'],
        )
        self.collision_buffer = np.array(config['collision_buffer'])
        self.config = config
        self.detect_threshold = config['detect_threshold']
        self.stop_steps = 0
        self.forced_forward_steps = 0

        self.red_light_steps = 0
        self.block_red_light = 0

        self.block_stop_sign_distance = (
            0  # it means in 30m, stop sign will not take effect again
        )
        self.stop_sign_trigger_times = 0
        
    def compute_steering(self, x, y):
        """
        x: float, m, x-axis
        y: float, m, y-axis
        return: float, [-1, 1], steering
        """
        theta = np.arctan2(x, y)
        steering = theta / np.pi
        return steering


    @abstractmethod
    def run_step(self, route_info):
        pass
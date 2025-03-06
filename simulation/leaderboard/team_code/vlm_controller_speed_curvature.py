import numpy as np
from collections import deque
from team_code.render_v2x import render, render_self_car

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

def downsample_waypoints(waypoints, precision=0.2):
    """
    waypoints: [float lits], 10 * 2, m
    """
    downsampled_waypoints = []
    downsampled_waypoints.append(np.array([0, 0]))
    last_waypoint = np.array([0.0, 0.0])
    for i in range(10):
        now_waypoint = waypoints[i]
        dis = np.linalg.norm(now_waypoint - last_waypoint)
        if dis > precision:
            interval = int(dis / precision)
            move_vector = (now_waypoint - last_waypoint) / (interval + 1)
            for j in range(interval):
                downsampled_waypoints.append(last_waypoint + move_vector * (j + 1))
        downsampled_waypoints.append(now_waypoint)
        last_waypoint = now_waypoint
    return downsampled_waypoints

def collision_detections(map1, map2, threshold=0.04):
    """
    map1: rendered surround vehicles
    map2: self-car
    """
    assert map1.shape == map2.shape
    overlap_map = (map1 > 0.01) & (map2 > 0.01)
    ratio = float(np.sum(overlap_map)) / np.sum(map2 > 0)
    ratio2 = float(np.sum(overlap_map)) / np.sum(map1 > 0)
    if ratio < threshold:
        return True
    else:
        return False

def get_max_safe_distance(meta_data, downsampled_waypoints, t, collision_buffer, threshold):
    surround_map = render(meta_data.reshape(20, 20, 7), t=t)[0][:100, 40:140]
    if np.sum(surround_map) < 1:
        return np.linalg.norm(downsampled_waypoints[-3])
    # need to render self-car map
    hero_bounding_box = np.array([2.45, 1.0]) + collision_buffer
    safe_distance = 0.0
    for i in range(len(downsampled_waypoints) - 2):
        aim = (downsampled_waypoints[i + 1] + downsampled_waypoints[i + 2]) / 2.0
        loc = downsampled_waypoints[i]
        ori = aim - loc
        self_car_map = render_self_car(loc=loc, ori=ori, box=hero_bounding_box)[
            :100, 40:140
        ]
        if collision_detections(surround_map, self_car_map, threshold) is False:
            break
        safe_distance = max(safe_distance, np.linalg.norm(loc))
    return safe_distance

class VLMControllerSpeedCurvature(object):
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


    def run_step(
        self, route_info
    ):
        """
        Currently, we generate the desired speed according to predicted waypoints only!
        In the next step, we need to consider the GLOBAL speed to finish the route in time.

        route_info: {
            'speed': float, m/s, current speed,
            'target_speed': [float list], 10,
            'curvature': [float list], 10,
            'dt': float,
            'target':
        }
        """
        speed = route_info['speed']
        target_speed = np.array(route_info['target_speed'])[0]
        curvature = np.array(route_info['curvature'])[0]
        
        angle = curvature / 180 * 5 # 4 is a hard-coded value for inhensing the steering angle.
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        print("steer:", steer)

        brake = False
        # get desired speed according to the future waypoints
        delta = np.clip(target_speed - speed, 0.0, self.config['clip_delta'])
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config['max_throttle'])

        if speed > target_speed * self.config['brake_ratio']:
            brake = True

        meta_info_1 = "speed: {:.2f}, target_speed: {:.2f}, angle: {:.2f}, [{}]".format(
            speed,
            target_speed,
            angle,
            ", ".join(f"{val:.2f}" for val in self.turn_controller._window)
        )
        
        meta_info_2 = "stop_steps:N/A"
        meta_info = {
            1: meta_info_1,
            2: meta_info_2,
        }

        return steer, throttle, brake, meta_info

from vlmdrive import VLMDRIVE_REGISTRY
from vlmdrive.controller.vlm_controller_base import VLMControllerBase
import numpy as np

@VLMDRIVE_REGISTRY.register
class VLMControllerWaypoint(VLMControllerBase):
    
    def compute_steering(self, x, y):
        """
        x: float, m, x-axis
        y: float, m, y-axis
        return: float, [-1, 1], steering
        """
        theta = np.arctan2(x, y)
        steering = theta / np.pi
        return steering


    def angle_to_steering(self, angle, thres=0.8):
        
        if angle < thres and angle > -thres:
            
            x = angle
            alpha = 15 
            gamma = 0.25
            beta = 15 
            lambda_ = 30

            # scaling functions
            exp_scale = thres * np.sign(x) * (1 - np.exp(-alpha * np.abs(x)))
            power_scale = thres * np.sign(x) * (np.abs(x) ** gamma)
            tanh_scale = thres * np.tanh(beta * x)
            logit_scale = thres * (2 / (1 + np.exp(-lambda_ * x)) - 1)
            
            return exp_scale
        else:
            return angle

    
    def run_step(self, route_info):
        """
        Currently, we generate the desired speed according to predicted waypoints only!
        In the next step, we need to consider the GLOBAL speed to finish the route in time.

        route_info: {
            'speed': float, m/s, current speed,
            'waypoints': [float list], K * 2, m,
            'target':
        }
        """
        speed = route_info['speed']
        waypoints = np.array(route_info['waypoints'])
        angle = self.compute_steering(waypoints[0][0], waypoints[0][1] + 1e-6)
        steer = self.angle_to_steering(angle)
        steer = self.turn_controller.step(steer)
        steer = np.clip(steer, -1.0, 1.0)

        brake = False
        # get desired speed according to the future waypoints
        target_speed = np.linalg.norm(waypoints[0], ord=2, axis=0)
        
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

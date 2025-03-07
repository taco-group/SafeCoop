from vlmdrive import VLMDRIVE_REGISTRY
from vlmdrive.controller.vlm_controller_base import VLMControllerBase
import numpy as np

@VLMDRIVE_REGISTRY.register
class VLMControllerControl(VLMControllerBase):
    def run_step(self, route_info):
        """
        Currently, we generate the desired speed according to predicted waypoints only!
        In the next step, we need to consider the GLOBAL speed to finish the route in time.

        route_info: {
            'speed': float, m/s, current speed,
            'steering': [float list], K,
            'throttle': [float list], K,
            'brake': [boolean list], K,
            'target':
        }
        """
        speed = route_info['speed']
        steering = np.array(route_info['steering'])[0]
        throttle = np.array(route_info['throttle'])[0]
        brake = np.array(route_info['brake'])[0]
        
        steer = self.turn_controller.step(steering)
        steer = np.clip(steer, -1.0, 1.0)
        print("steer:", steer)
        throttle = np.clip(throttle, 0.0, self.config['max_throttle'])
        
        meta_info_1 = "speed: {:.2f}, [{}]".format(
            speed,
            ", ".join(f"{val:.2f}" for val in self.turn_controller._window)
        )
        
        meta_info_2 = "stop_steps:N/A"
        meta_info = {
            1: meta_info_1,
            2: meta_info_2,
        }

        return steer, throttle, brake, meta_info

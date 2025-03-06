from vlmdrive import VLMDRIVE_REGISTRY
from vlmdrive.controller.vlm_controller_base import VLMControllerBase
import numpy as np

@VLMDRIVE_REGISTRY.register
class VLMControllerWaypoint(VLMControllerBase):
    def run_step(self, route_info):
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

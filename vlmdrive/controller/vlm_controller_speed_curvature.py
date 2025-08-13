from vlmdrive import VLMDRIVE_REGISTRY
from vlmdrive.controller.vlm_controller_base import VLMControllerBase
import numpy as np

# @VLMDRIVE_REGISTRY.register
# class VLMControllerSpeedCurvature(VLMControllerBase):
#     def run_step(self, route_info, buffer_idx=0):
#         """
#         Currently, we generate the desired speed according to predicted waypoints only!
#         In the next step, we need to consider the GLOBAL speed to finish the route in time.

#         route_info: {
#             'speed': float, m/s, current speed,
#             'target_speed': [float list], 10,
#             'curvature': [float list], 10,
#             'dt': float,
#             'target':
#         }
#         """
#         speed = route_info['speed']
#         target_speed = np.array(route_info['target_speed'])[buffer_idx]
#         curvature = np.array(route_info['curvature'])[buffer_idx]
        
#         steer = curvature / 180 * 3 # 3 is a hard-coded value for enhensing the steering angle.
#         steer = self.turn_controller.step(steer)
#         steer = np.clip(steer, -1.0, 1.0)
#         print("steer:", steer)

#         brake = False
#         # get desired speed according to the future waypoints
#         delta = np.clip(target_speed - speed, 0.0, self.config['clip_delta'])
#         throttle = self.speed_controller.step(delta)
#         throttle = np.clip(throttle, 0.0, self.config['max_throttle'])

#         if speed > target_speed * self.config['brake_ratio']:
#             brake = True

#         # meta_info_1 = "speed: {:.2f}, target_speed: {:.2f}, angle: {:.2f}, [{}]".format(
#         #     speed,
#         #     target_speed,
#         #     curvature,
#         #     ", ".join(f"{val:.2f}" for val in self.turn_controller._window)
#         # )
        
#         meta_info_1 = ""
        
#         meta_info_2 = "stop_steps:N/A"
#         meta_info = {
#             1: meta_info_1,
#             2: meta_info_2,
#         }

#         return steer, throttle, brake, meta_info


class PIDSpeedController:
    def __init__(self, kp=0.8, ki=0.1, kd=0.05, dt=0.05, i_clip=2.0):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.i_term = 0.0
        self.prev_error = None
        self.i_clip = i_clip  # anti-windup limit

    def step(self, error):
        """Given speed error (m/s), return desired accel (m/s^2)."""
        # Proportional
        p = self.kp * error
        # Integral
        self.i_term += error * self.dt
        self.i_term = max(min(self.i_term, self.i_clip), -self.i_clip)
        i = self.ki * self.i_term
        # Derivative
        if self.prev_error is None:
            d = 0.0
        else:
            d = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        return p + i + d
    
    

@VLMDRIVE_REGISTRY.register
class VLMControllerSpeedCurvature(VLMControllerBase):
    
    def __init__(self, config):
        
        super().__init__(config)
        
        # Units: speed m/s, curvature 1/m, dt seconds
        self.config.setdefault('wheelbase_m', 2.875)         # sedan-ish wheelbase
        self.config.setdefault('delta_max_deg', 70.0)        # CARLA front wheel max steer angle
        self.config.setdefault('understeer_gain', 1e-3)      # Ku for high-speed understeer compensation
        self.config.setdefault('mu', 0.9)                    # tire-road friction coeff
        self.config.setdefault('g', 9.81)
        self.config.setdefault('a_throttle_max', 3.0)        # achievable forward accel (m/s^2)
        self.config.setdefault('a_brake_max', 6.0)           # achievable braking decel (m/s^2)
        self.config.setdefault('max_throttle', 0.8)          # cap to avoid wheelspin
        self.config.setdefault('brake_ratio', 1.05)          # extra safety to trigger brake when overspeeding
        self.config.setdefault('dt', 0.05)                   # fallback if route_info['dt'] missing

        self.speed_controller = PIDSpeedController(
            kp=self.config.get('speed_KP', 0.8),
            ki=self.config.get('speed_KI', 0.1),
            kd=self.config.get('speed_KD', 0.05),
            dt=self.config.get('dt', 0.05),
            i_clip=self.config.get('speed_i_clip', 2.0)  # anti-windup limit
        )
    
    
    def run_step(self, route_info, buffer_idx=0):
        """
        Map (target speed, curvature) to (steer, throttle, brake) at each tick.

        route_info: {
            'speed': float,           # current speed [m/s]
            'target_speed': list,     # length >= buffer_idx+1, [m/s]
            'curvature': list,        # length >= buffer_idx+1, [1/m], signed: left>0, right<0
            'dt': float (optional),   # seconds
            'target': ... (ignored here)
        }
        Returns:
            steer in [-1,1], throttle in [0,1], brake_bool, meta_info (dict)
        """
        # ------- Read inputs -------
        v      = float(route_info['speed'])
        v_ref  = float(np.array(route_info['target_speed'])[buffer_idx])
        kappa  = float(np.array(route_info['curvature'])[buffer_idx])  # expected unit: 1/m
        dt     = float(route_info.get('dt', self.config.get('dt', 0.05)))

        # ------- Vehicle / control parameters -------
        L                 = float(self.config.get('wheelbase_m', 2.875))
        delta_max_deg     = float(self.config.get('delta_max_deg', 70.0))
        Ku                = float(self.config.get('understeer_gain', 1e-3))
        mu                = float(self.config.get('mu', 0.9))
        g                 = float(self.config.get('g', 9.81))
        a_throttle_max    = float(self.config.get('a_throttle_max', 3.0))
        a_brake_max       = float(self.config.get('a_brake_max', 6.0))
        max_throttle_cap  = float(self.config.get('max_throttle', 0.8))
        brake_ratio       = float(self.config.get('brake_ratio', 1.05))

        delta_max = np.deg2rad(delta_max_deg)

        # ======================================================
        # 1) Curvature -> steering (bicycle model + mild Ku term)
        #    δ ≈ atan(L * κ) [+ Ku * v^2 * κ], then normalize by δ_max
        # ======================================================
        # Protect against NaN / extreme κ
        if not np.isfinite(kappa):
            kappa = 0.0

        delta_cmd = np.arctan(L * kappa) + Ku * (v**2) * kappa
        # Clamp to physical steering limit
        delta_cmd = float(np.clip(delta_cmd, -delta_max, +delta_max))

        # Convert to CARLA steer in [-1, 1] by normalizing to δ_max
        steer = delta_cmd / delta_max

        # Optional smoothing via your turn_controller (if it's a low-pass / PID on steer)
        if hasattr(self, 'turn_controller') and self.turn_controller is not None:
            steer = float(self.turn_controller.step(steer))

        steer = float(np.clip(steer, -1.0, 1.0))

        # ======================================================
        # 2) Speed control -> desired longitudinal acceleration
        #    a_cmd = PID(v_ref - v), then apply friction-circle constraint
        # ======================================================
        e_v = v_ref - v  # allow negative (need braking)
        # Expectation: speed_controller.step returns desired accel [m/s^2]
        a_cmd = float(self.speed_controller.step(e_v))

        # Lateral acceleration from curvature at current speed
        a_y = (v**2) * abs(kappa)  # [m/s^2]

        # Friction circle: limit available longitudinal acceleration
        # If lateral exceeds mu*g, we must brake
        if a_y >= mu * g:
            # lateral is saturated; force braking at a_brake_max
            a_cmd = -a_brake_max
        else:
            a_x_max = float(np.sqrt((mu * g)**2 - a_y**2))  # available long. accel
            a_cmd = float(np.clip(a_cmd, -a_brake_max, a_x_max))

        # ======================================================
        # 3) Accel -> throttle/brake
        #    Use linear maps to [0,1], capped, mutually exclusive
        # ======================================================
        if a_cmd >= 0.0:
            throttle = float(np.clip(a_cmd / a_throttle_max, 0.0, 1.0))
            brake_val = 0.0
        else:
            throttle = 0.0
            brake_val = float(np.clip(-a_cmd / a_brake_max, 0.0, 1.0))

        # Cap throttle (to be gentle)
        throttle = float(np.clip(throttle, 0.0, max_throttle_cap))

        # Your interface returns a boolean brake; keep it
        # (If you later switch to CARLA VehicleControl, pass brake=brake_val instead.)
        brake_bool = (brake_val > 0.05) or (v > v_ref * brake_ratio)

        # ======================================================
        # Meta info for debugging/plots
        # ======================================================
        meta_info = {
            1: f"v={v:.2f} m/s | v_ref={v_ref:.2f} | kappa={kappa:.4f} 1/m | steer={steer:.3f}",
            2: f"a_cmd={a_cmd:.2f} m/s^2 | a_y={a_y:.2f} | thr={throttle:.2f} | brk_val={brake_val:.2f}"
        }

        return steer, throttle, brake_bool, meta_info
import numpy as np
from collections import deque
from abc import ABC, abstractmethod

# class PIDController(object):
#     def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
#         self._K_P = K_P
#         self._K_I = K_I
#         self._K_D = K_D

#         self._window = deque([0 for _ in range(n)], maxlen=n)
#         self._max = 0.0
#         self._min = 0.0

#     def step(self, error):
#         self._window.append(error)
#         self._max = max(self._max, abs(error))
#         self._min = -abs(self._max)

#         if len(self._window) >= 2:
#             integral = np.mean(self._window)
#             derivative = self._window[-1] - self._window[-2]
#         else:
#             integral = 0.0
#             derivative = 0.0

#         return self._K_P * error + self._K_I * integral + self._K_D * derivative

class PIDController(object):
    """Enhanced PID controller with anti-windup, dynamic limits, and window-based methods"""
    
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, window_size=20, dt=1.0):
        self._K_P = K_P        # Proportional gain
        self._K_I = K_I        # Integral gain
        self._K_D = K_D        # Derivative gain
        self._window_size = window_size
        self._error_history = deque(maxlen=window_size)  # Error history
        self._time_history = deque(maxlen=window_size)   # Time history
        self._dt = dt          # Default time interval
        self._last_error = 0.0 # Last recorded error
        self._last_time = 0.0  # Last recorded time
        self._integral = 0.0   # Integral term
        self.reset()
    
    def reset(self):
        """Reset the controller state"""
        self._error_history.clear()
        self._time_history.clear()
        self._max_error = 0.0
        self._last_error = 0.0
        self._last_time = 0.0
        self._integral = 0.0
    
    def step(self, error, current_time=None):
        """Compute and return the control output for a given error"""
        # Compute time interval
        if current_time is not None:
            if self._last_time != 0.0:
                dt = current_time - self._last_time
            else:
                dt = self._dt
            self._last_time = current_time
        else:
            dt = self._dt
        
        # Update error and time history
        self._error_history.append(error)
        self._time_history.append(dt)
        
        # Update maximum error value (with a decay factor)
        decay_factor = 0.999
        self._max_error = max(self._max_error * decay_factor, abs(error))
        
        # Compute proportional term
        p_term = self._K_P * error
        
        # Compute integral term
        if self._K_I != 0.0:
            # Update integral value
            self._integral += error * dt
            
            # Dynamically limit the integral term to prevent windup
            error_ratio = abs(error) / (self._max_error if self._max_error != 0.0 else 1.0)
            integral_limit = self._max_error * (0.5 + 0.5 * error_ratio)
            self._integral = max(-integral_limit, min(integral_limit, self._integral))
            
            i_term = self._K_I * self._integral
        else:
            i_term = 0.0
        
        # Compute derivative term
        if self._K_D != 0.0 and len(self._error_history) >= 2:
            # Smooth the derivative calculation to reduce noise impact
            alpha = 0.7  # Smoothing coefficient
            current_derivative = (error - self._last_error) / dt
            
            if len(self._error_history) >= 3:
                prev_dt = self._time_history[-2] if self._time_history[-2] != 0.0 else dt
                prev_derivative = (self._error_history[-2] - self._error_history[-3]) / prev_dt
                smooth_derivative = alpha * current_derivative + (1.0 - alpha) * prev_derivative
            else:
                smooth_derivative = current_derivative
            
            d_term = self._K_D * smooth_derivative
        else:
            d_term = 0.0
        
        # Update last recorded error
        self._last_error = error
        
        # Compute and return control output
        control_output = p_term + i_term + d_term
        
        return control_output



# class KalmanPIDController(object):
#     """PID controller based on Kalman filtering"""
    
#     def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, dt=1.0,
#                  process_variance=1e-4, measurement_variance=1e-2):
#         # PID parameters
#         self._K_P = K_P
#         self._K_I = K_I
#         self._K_D = K_D
#         self._dt = dt
        
#         # Kalman filter state
#         # State vector x = [error, error_rate]
#         self._x = np.zeros(2)  # State estimate [error, error rate]
#         self._P = np.eye(2)    # State covariance matrix
        
#         # State transition matrix
#         self._F = np.array([[1, dt],
#                            [0, 1]])
        
#         # Measurement matrix (we only measure error)
#         self._H = np.array([[1, 0]])
        
#         # Process noise covariance
#         self._Q = np.array([[dt**4/4, dt**3/2],
#                            [dt**3/2, dt**2]]) * process_variance
        
#         # Measurement noise covariance
#         self._R = np.array([[measurement_variance]])
        
#         # PID controller state
#         self._integral = 0.0
#         self._max_error = 0.0
#         self._last_time = 0.0
    
#     def reset(self):
#         """Reset the controller state"""
#         self._x = np.zeros(2)
#         self._P = np.eye(2)
#         self._integral = 0.0
#         self._max_error = 0.0
#         self._last_time = 0.0
    
#     def _update_kalman(self, measured_error, dt):
#         """Update Kalman filter state"""
#         # Update state transition matrix and process noise covariance
#         self._F[0, 1] = dt
#         self._Q[0, 0] = dt**4/4
#         self._Q[0, 1] = dt**3/2
#         self._Q[1, 0] = dt**3/2
#         self._Q[1, 1] = dt**2
        
#         # Prediction step
#         x_pred = self._F @ self._x
#         P_pred = self._F @ self._P @ self._F.T + self._Q
        
#         # Update step
#         z = np.array([measured_error])
#         y = z - self._H @ x_pred  # Residual (measurement innovation)
#         S = self._H @ P_pred @ self._H.T + self._R  # Residual covariance
#         K = P_pred @ self._H.T @ np.linalg.inv(S)  # Kalman gain
        
#         self._x = x_pred + K @ y
#         self._P = (np.eye(2) - K @ self._H) @ P_pred
        
#         return self._x
    
#     def step(self, measured_error, current_time=None):
#         """Compute and return the control output"""
#         # Compute time interval
#         if current_time is not None:
#             if self._last_time != 0.0:
#                 dt = current_time - self._last_time
#             else:
#                 dt = self._dt
#             self._last_time = current_time
#         else:
#             dt = self._dt
        
#         # Estimate state using the Kalman filter
#         filtered_state = self._update_kalman(measured_error, dt)
#         filtered_error = filtered_state[0]  # Filtered error
#         error_derivative = filtered_state[1]  # Error rate
        
#         # Update maximum error value
#         decay_factor = 0.995
#         self._max_error = max(self._max_error * decay_factor, abs(filtered_error))
        
#         # Compute proportional term
#         p_term = self._K_P * filtered_error
        
#         # Compute integral term
#         if self._K_I != 0.0:
#             self._integral += filtered_error * dt
            
#             # Dynamically limit integral to prevent windup
#             integral_limit = 1.5 * self._max_error
#             self._integral = max(-integral_limit, min(integral_limit, self._integral))
            
#             i_term = self._K_I * self._integral
#         else:
#             i_term = 0.0
        
#         # Compute derivative term (using Kalman-estimated error rate)
#         d_term = self._K_D * error_derivative
        
#         # Return control output
#         return p_term + i_term + d_term



class VLMControllerBase(ABC, object):
    def __init__(self, config):
        self.turn_controller = PIDController(
            K_P=config['turn_KP'], 
            K_I=config['turn_KI'], 
            K_D=config['turn_KD'], 
            window_size=config['turn_n']
        )
        self.speed_controller = PIDController(
            K_P=config['speed_KP'],
            K_I=config['speed_KI'],
            K_D=config['speed_KD'],
            window_size=config['speed_n'],
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
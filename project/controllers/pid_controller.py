import warnings
import numpy as np
from .p_controller import P_Controller

# --- Suppress warnings for cleaner output ---
warnings.filterwarnings("ignore", category=UserWarning)


class PID_Controller(P_Controller):
    """A standard PID controller for speed tracking."""
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.ki = config["pid_params"]["pid_scalar_ki"]
        self.kd = config["pid_params"]["pid_scalar_kd"]
        self._integral = 0
        self._previous_error = 0
        self.dt = config["dt"]

    def get_action(self, current_speed, target_speed, current_rpm, current_gear, future_slopes=None):
        speed_error = target_speed - current_speed
        self._integral += speed_error * self.dt
        self._integral = np.clip(self._integral, -5000, 5000)
        derivative = (speed_error - self._previous_error) / self.dt
        self._previous_error = speed_error
        
        torque_cmd_nm = (self.kp * speed_error) + (self.ki * self._integral) + (self.kd * derivative)
        torque_cmd = np.clip(torque_cmd_nm, -self.max_pos_tq, self.max_pos_tq) / self.max_pos_tq
        
        _, gear_cmd, _ = super().get_action(current_speed, target_speed, current_rpm, current_gear, future_slopes)
        brake_cmd = 0.0
        return torque_cmd, gear_cmd, brake_cmd

    def reset(self):
        self._integral = 0
        self._previous_error = 0

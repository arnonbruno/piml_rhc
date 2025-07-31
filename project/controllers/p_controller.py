import numpy as np
import warnings

# --- Suppress warnings for cleaner output ---
warnings.filterwarnings("ignore", category=UserWarning)

############################################################
# 1. CONTROLLER DEFINITIONS
############################################################

class P_Controller:
    """A simple Proportional controller for speed tracking."""
    def __init__(self, config, **kwargs):
        self.kp = config["pid_params"]["pid_scalar_kp"]
        self.max_pos_tq = config["vehicle_physical_params"]["max_pos_tq"]
        self.gear_ratios = config["vehicle_physical_params"]["gear_ratios"]
        self.rpm_min_shift = 1000
        self.rpm_max_shift = 2200

    def get_action(self, current_speed, target_speed, current_rpm, current_gear, future_slopes=None):
        speed_error = target_speed - current_speed
        torque_cmd_nm = self.kp * speed_error
        torque_cmd = np.clip(torque_cmd_nm, -self.max_pos_tq, self.max_pos_tq) / self.max_pos_tq
        
        if current_rpm < self.rpm_min_shift and current_gear > 0:
            gear_cmd = current_gear - 1
        elif current_rpm > self.rpm_max_shift and current_gear < (len(self.gear_ratios) - 1):
            gear_cmd = current_gear + 1
        else:
            gear_cmd = current_gear
            
        brake_cmd = 0.0
        return torque_cmd, gear_cmd, brake_cmd
# memoization_controller.py
import numpy as np
import time
import hashlib
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Assuming utils has necessary functions; import if available
try:
    from utils import max_torque_at_rpm_fast, calc_energy_efficiency_torch
    import torch
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    # Fallback stubs if utils not available
    def max_torque_at_rpm_fast(rpm):
        return torch.full_like(rpm, 2600.0)  # Simplified fallback

    def calc_energy_efficiency_torch(rpm, torque, engine_params):
        return torch.full_like(rpm, 0.35)  # Average efficiency fallback

class MemoizationController:
    """
    A memoization-based controller using a precomputed lookup table (LUT) for actions.
    The LUT is built offline over a discretized state space (speed, speed error, avg slope)
    and queried online with interpolation. Uses per-track vehicle params for accuracy.
    """
    def __init__(self, config, track_data, args=None, **kwargs):
        self.config = config
        self.track_data = track_data
        self.dt = config.get("dt", 1.0)
        self.max_pos_tq = config["vehicle_physical_params"]["max_pos_tq"]
        self.gear_ratios = np.array(config["vehicle_physical_params"]["gear_ratios"])
        self.final_drive_ratio = config.get("final_drive_ratio", 3.42)
        self.rpm_min = config["vehicle_physical_params"].get("rpm_min", 600.0)
        self.rpm_max = config.get("rpm_max_sim", 3500.0)
        self.engine_params = config.get("engine_params", {})
        self.device = torch.device("cpu")  # CPU for simplicity; can switch to GPU if needed

        # Pull actual per-track vehicle params
        vparams = track_data['vehicle_params']
        self.wheel_radius = vparams['wheel_radius']
        self.mass = vparams['mass']
        self.rho = vparams['rho']
        self.cd = vparams['Cd']
        self.area = vparams['A']
        self.crr = vparams['Crr']

        # Discretization for LUT (adjust for granularity vs. compute)
        self.speed_bins = np.arange(0, 30.1, 0.1)  # 0-30 m/s in 1 m/s steps
        self.error_bins = np.arange(-5, 5.1, 0.1)  # Speed error -5 to 5 m/s
        self.slope_bins = np.arange(-0.1, 0.11, 0.01)  # Avg slope rad -0.1 to 0.1

        # LUT shape: (speed, error, slope) -> (torque_norm, gear_idx, brake)
        self.lut_shape = (len(self.speed_bins), len(self.error_bins), len(self.slope_bins))
        self.lut = np.zeros(self.lut_shape + (3,))  # [..., 0]=torque, [1]=gear, [2]=brake

        # Caching per unique vehicle config
        cache_dir = Path(config["paths"].get("cache_dir", "cache/dp_oracle")) / "memo_lut"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / self._get_lut_hash()

        # Build or load LUT
        if self.cache_path.exists():
            print("Memoization: Loading cached LUT...")
            self.lut = np.load(self.cache_path)
        else:
            print("Memoization: Building LUT from scratch...")
            self._build_lut()
            np.save(self.cache_path, self.lut)

        # State for reset
        self.current_gear = 2  # Starting gear, as in other controllers

    def _get_lut_hash(self):
        """Hash for LUT caching based on bins and per-vehicle params (rounded for stability)."""
        params_str = (
            f"{self.speed_bins}{self.error_bins}{self.slope_bins}"
            f"{self.max_pos_tq}{self.gear_ratios}{self.final_drive_ratio}"
            f"{self.rpm_min}{self.rpm_max}{self.engine_params}"
            f"{self.wheel_radius:.3f}{self.mass:.0f}{self.rho:.2f}{self.cd:.2f}{self.area:.1f}{self.crr:.4f}"  # Rounded to group similar
        )
        return hashlib.sha256(params_str.encode()).hexdigest() + ".npy"

    def _build_lut(self):
        """Populate the LUT by optimizing actions for each grid point."""
        start_time = time.time()
        for i, speed in enumerate(self.speed_bins):
            for j, error in enumerate(self.error_bins):
                for k, avg_slope in enumerate(self.slope_bins):
                    # Simulate "optimal" action: simple heuristic optimization
                    # (e.g., torque to minimize energy + RMSE, gear to keep RPM sweet spot)
                    target_speed = speed + error
                    rpm = (speed / self.wheel_radius) * self.gear_ratios * self.final_drive_ratio * (60 / (2 * np.pi))
                    valid_gears = np.where((rpm >= self.rpm_min) & (rpm <= self.rpm_max))[0]
                    if len(valid_gears) == 0:
                        # Fallback to closest gear
                        gear = np.argmin(np.abs(rpm - (self.rpm_min + self.rpm_max) / 2))
                    else:
                        # Choose gear closest to sweet spot (e.g., 1700 RPM)
                        sweet_rpm_diff = np.abs(rpm[valid_gears] - 1700)
                        gear = valid_gears[np.argmin(sweet_rpm_diff)]

                    # Torque: Proportional to error, clipped
                    torque_norm = np.clip(error * 0.5, -1.0, 1.0)  # Simple P-like

                    # Brake: If overspeed and downhill
                    brake = 1.0 if (error < -1.0 and avg_slope < -0.05) else 0.0

                    # Refine with energy cost (if utils available)
                    if UTILS_AVAILABLE:
                        engine_rpm = torch.tensor([rpm[gear]])
                        torque_nm = torch.tensor([torque_norm * self.max_pos_tq])
                        eff = calc_energy_efficiency_torch(engine_rpm, torque_nm, self.engine_params).item()
                        # Penalize low eff by adjusting torque slightly
                        if eff < 0.2:
                            torque_norm *= 1.1  # Boost if inefficient
                            torque_norm = np.clip(torque_norm, -1.0, 1.0)

                    self.lut[i, j, k] = [torque_norm, gear, brake]

        print(f"Memoization: LUT built in {time.time() - start_time:.2f} seconds. Shape: {self.lut_shape}")

    def _query_lut(self, current_speed, speed_error, avg_slope):
        """Nearest-neighbor + linear interpolation query."""
        i = np.argmin(np.abs(self.speed_bins - current_speed))
        j = np.argmin(np.abs(self.error_bins - speed_error))
        k = np.argmin(np.abs(self.slope_bins - avg_slope))

        # Simple nearest
        action = self.lut[i, j, k]

        # Linear interp for speed (example; extend to others if needed)
        if i < len(self.speed_bins) - 1:
            denom = self.speed_bins[i+1] - self.speed_bins[i]
            if denom != 0:
                alpha = (current_speed - self.speed_bins[i]) / denom
                action = (1 - alpha) * action + alpha * self.lut[i+1, j, k]
            # No else needed; alpha=0 if denom=0 (though uniform prevents)

        return action

    def get_action(self, current_speed, target_speed, current_rpm, current_gear, future_slopes=None):
        if future_slopes is None:
            future_slopes = [0.0] * 5  # Default short horizon

        speed_error = target_speed - current_speed
        avg_slope = np.mean(future_slopes)  # Avg over horizon

        action = self._query_lut(current_speed, speed_error, avg_slope)
        torque_cmd = float(action[0])
        gear_cmd = int(np.round(action[1]))  # Discretize gear
        brake_cmd = float(action[2])

        # Add hysteresis to prevent gear chatter
        if abs(gear_cmd - self.current_gear) < 2:
            gear_cmd = self.current_gear

        # Update internal gear
        self.current_gear = gear_cmd

        return torque_cmd, gear_cmd, brake_cmd

    def reset(self):
        """Reset internal state."""
        self.current_gear = 2
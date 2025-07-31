# controllers/dp_controller.py
import numpy as np
import time
import hashlib
from pathlib import Path

class DPOracle:
    """
    A from-scratch, robust offline Dynamic Programming solver to find the
    near-optimal energy trajectory. This version incorporates the final fixes
    for start-gear selection and absolute torque storage.
    """
    def __init__(self, config, track_data):
        # --- Config & Parameters ---
        self.config = config
        self.track_data = track_data
        dp_params = config.get("dp_oracle_params", {})
        phys_params = config["vehicle_physical_params"]
        vparams = track_data['vehicle_params']

        # --- DP Discretization ---
        self.v_min_mps = dp_params.get("v_min_mps", 2.0)
        self.v_max_mps = dp_params.get("v_max_mps", 30.0)
        self.v_step_mps = dp_params.get("v_step_mps", 0.5)
        self.torque_steps = dp_params.get("torque_steps", 21)
        self.v_nodes = np.arange(self.v_min_mps, self.v_max_mps + self.v_step_mps, self.v_step_mps)
        self.torque_nodes = np.linspace(-1.0, 1.0, self.torque_steps)
        self.g_nodes = np.arange(config["num_gears"])

        # --- Cost Weights ---
        self.w_energy = dp_params.get("w_energy", 1.0)
        self.w_speed = dp_params.get("w_speed", 30.0)
        self.w_shift = dp_params.get("w_shift", 5.0)
        self.w_terminal = dp_params.get("w_terminal", 100.0)

        # --- Vehicle Physics ---
        self.mass = vparams['mass']
        self.wheel_radius = vparams['wheel_radius']
        self.rho = vparams['rho']
        self.cd = vparams['Cd']
        self.area = vparams['A']
        self.crr = vparams['Crr']
        self.dt = config.get("dt", 1.0)
        self.max_pos_tq = phys_params["max_pos_tq"]
        self.gear_ratios = np.array(phys_params["gear_ratios"])
        self.final_drive_ratio = config["final_drive_ratio"]
        self.rpm_min = phys_params["rpm_min"]
        self.rpm_max = phys_params["rpm_max"]
        self.idle_fuel_kw = 5.0

        # --- Grid Dimensions ---
        self.K = len(track_data['slopes'])
        self.N_v = len(self.v_nodes)
        self.N_g = len(self.g_nodes)

        # --- Caching ---
        cache_dir = Path(config["paths"].get("cache_dir", "cache/dp_oracle"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / self._get_track_hash()

    def _get_track_hash(self):
        """Creates a unique hash for the track and parameters for caching."""
        params_str = (
            f"{self.v_min_mps}{self.v_max_mps}{self.v_step_mps}{self.torque_steps}"
            f"{self.w_energy}{self.w_speed}{self.w_shift}{self.w_terminal}"
        )
        data_str = (
            str(self.track_data['slopes'].tolist()) +
            str(self.track_data['vehicle_params']) +
            str(self.track_data['mission_target_v_profile'].tolist())
        )
        full_str = params_str + data_str
        return hashlib.sha256(full_str.encode()).hexdigest() + ".npz"

    def solve(self):
        if self.cache_path.exists():
            print(f"DP: Loading cached solution from {self.cache_path}")
            data = np.load(self.cache_path, allow_pickle=True)
            return {"trajectory": data['trajectory'].tolist(), "solve_time_s": data['solve_time_s'].item()}

        start_time = time.time()
        print("DP: Solving from scratch...")

        J = np.full((self.K, self.N_v, self.N_g), np.inf, dtype=np.float64)
        policy_torque = np.zeros((self.K, self.N_v, self.N_g), dtype=np.float32)
        policy_gear_next = np.zeros((self.K, self.N_v, self.N_g), dtype=np.int8)

        v_target_final = self.track_data['mission_target_v_profile'][-1]
        J[-1, :, :] = self.w_terminal * ((self.v_nodes - v_target_final)[:, np.newaxis] ** 2)

        for k in range(self.K - 2, -1, -1):
            v_target_next = self.track_data['mission_target_v_profile'][k + 1]
            slope_rad = self.track_data['slopes'][k]

            for v_idx, v_k in enumerate(self.v_nodes):
                for g_idx, gear_ratio_k in enumerate(self.gear_ratios):
                    total_gear_ratio = gear_ratio_k * self.final_drive_ratio
                    engine_rpm = (v_k / self.wheel_radius) * total_gear_ratio * (30 / np.pi)
                    
                    if not (self.rpm_min <= engine_rpm <= self.rpm_max):
                        continue
                    
                    # Assuming max_torque_at_rpm_fast is available and works as before
                    # If not, a simplified model would be needed. Here we assume it's part of the environment.
                    try:
                        from utils import max_torque_at_rpm_fast
                        import torch
                        max_tq_at_rpm = max_torque_at_rpm_fast(torch.tensor([engine_rpm])).item()
                    except (ImportError, NameError):
                        max_tq_at_rpm = self.max_pos_tq # Fallback if util is not available

                    total_resistance = 0.5 * self.rho * self.cd * self.area * (v_k ** 2) + \
                                       self.mass * 9.81 * self.crr * np.cos(slope_rad) + \
                                       self.mass * 9.81 * np.sin(slope_rad)

                    torque_nm_vec = np.clip(self.torque_nodes * self.max_pos_tq, -self.max_pos_tq, max_tq_at_rpm)
                    propulsive_force_vec = (torque_nm_vec * total_gear_ratio) / self.wheel_radius
                    accel_vec = (propulsive_force_vec - total_resistance) / self.mass
                    v_next_vec = v_k + accel_vec * self.dt
                    
                    valid_mask = (v_next_vec >= self.v_min_mps) & (v_next_vec <= self.v_max_mps)
                    v_next_indices = np.argmin(np.abs(self.v_nodes[:, np.newaxis] - v_next_vec), axis=0)
                    
                    power_watts_vec = propulsive_force_vec * v_k
                    fuel_power_vec = np.where(power_watts_vec > 0, power_watts_vec / 0.35, self.idle_fuel_kw * 1000)
                    fuel_cost_vec = self.w_energy * (fuel_power_vec * self.dt / 3.6e6)
                    speed_error_cost_vec = self.w_speed * ((v_next_vec - v_target_next)**2)
                    stage_cost_vec = fuel_cost_vec + speed_error_cost_vec
                    
                    future_costs_mat = J[k + 1, v_next_indices, :]
                    shift_penalties = np.array([self.w_shift if g_idx != g_next_idx else 0 for g_next_idx in self.g_nodes])
                    future_costs_with_shift = future_costs_mat + shift_penalties[np.newaxis, :]
                    
                    min_future_costs = np.min(future_costs_with_shift, axis=1)
                    best_g_next_indices = np.argmin(future_costs_with_shift, axis=1)
                    
                    total_costs = stage_cost_vec + min_future_costs
                    total_costs[~valid_mask] = np.inf
                    
                    if np.all(np.isinf(total_costs)): continue
                    
                    best_torque_idx = np.argmin(total_costs)
                    
                    J[k, v_idx, g_idx] = total_costs[best_torque_idx]
                    policy_torque[k, v_idx, g_idx] = torque_nm_vec[best_torque_idx] # FIX 1: Store absolute Nm
                    policy_gear_next[k, v_idx, g_idx] = best_g_next_indices[best_torque_idx]

        solution = self.reconstruct_trajectory(J, policy_torque, policy_gear_next, time.time() - start_time)
        np.savez_compressed(self.cache_path, trajectory=np.array(solution["trajectory"], dtype=object), solve_time_s=np.array(solution['solve_time_s']))
        return solution

    def reconstruct_trajectory(self, J, policy_torque, policy_gear_next, solve_time):
        optimal_trajectory = []
        
        # --- FIX 2: Choose FEASIBLE initial gear automatically ---
        current_v = self.track_data['initial_speed']
        v_idx = np.argmin(np.abs(self.v_nodes - current_v))
        finite_J_costs = J[0, v_idx, :]
        if not np.any(np.isfinite(finite_J_costs)):
            raise RuntimeError(f"DP-Oracle: No feasible initial state found for speed {current_v:.2f} m/s.")
        current_g_idx = int(np.nanargmin(finite_J_costs))
        # --- END FIX ---
        
        for k in range(self.K - 1):
            v_idx = np.argmin(np.abs(self.v_nodes - current_v))
            v_idx = np.clip(v_idx, 0, self.N_v - 1)
            g_idx = np.clip(current_g_idx, 0, self.N_g - 1)

            torque_nm = policy_torque[k, v_idx, g_idx]
            g_next_idx = int(policy_gear_next[k, v_idx, g_idx])
            
            # The controller returns normalized torque for the simulation framework
            torque_norm = torque_nm / self.max_pos_tq
            optimal_trajectory.append({"torque_cmd": float(torque_norm), "gear_cmd": int(current_g_idx), "brake_cmd": 0.0})
            
            total_gear_ratio = self.gear_ratios[g_idx] * self.final_drive_ratio
            slope_rad = self.track_data['slopes'][k]
            total_resistance = 0.5 * self.rho * self.cd * self.area * (current_v**2) + \
                               self.mass * 9.81 * self.crr * np.cos(slope_rad) + \
                               self.mass * 9.81 * np.sin(slope_rad)
            
            propulsive_force = (torque_nm * total_gear_ratio) / self.wheel_radius
            accel = (propulsive_force - total_resistance) / self.mass
            current_v += accel * self.dt
            current_v = np.clip(current_v, self.v_min_mps, self.v_max_mps)
            current_g_idx = g_next_idx
        
        print(f"\n--- DP Solution Summary ---")
        print(f"  Solve time: {solve_time:.2f} seconds")
        torque_cmds = [cmd['torque_cmd'] for cmd in optimal_trajectory]
        if torque_cmds:
            zero_torque_count = sum(1 for t in torque_cmds if abs(t * self.max_pos_tq) < 1.0)
            print(f"  Zero torque commands: {zero_torque_count}/{len(torque_cmds)} ({100*zero_torque_count/len(torque_cmds):.1f}%)")
            avg_abs_torque_nm = np.mean(np.abs([t * self.max_pos_tq for t in torque_cmds]))
            print(f"  Avg torque magnitude: {avg_abs_torque_nm:.1f} Nm")

        return {"trajectory": optimal_trajectory, "solve_time_s": solve_time}


class DP_Controller:
    """
    A controller that loads a pre-computed optimal trajectory from the DPOracle
    and plays it back during the simulation.
    """
    def __init__(self, config, track_data, **kwargs):
        self.config = config
        self.track_data = track_data
        self.current_step = 0
        oracle = DPOracle(config, track_data)
        solution = oracle.solve()
        self.trajectory = solution["trajectory"]
        self.solve_time = solution["solve_time_s"]

    def get_action(self, current_speed, target_speed, current_rpm, current_gear, future_slopes=None):
        if self.current_step < len(self.trajectory):
            action = self.trajectory[self.current_step]
            self.current_step += 1
            return action["torque_cmd"], action["gear_cmd"], action["brake_cmd"]
        else:
            return 0.0, current_gear, 1.0

    def reset(self):
        self.current_step = 0
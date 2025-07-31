# MPC_Controller_HeuristicGear.py

import os
import warnings

# --- Core Scientific Libraries ---
import torch



class MPC_Controller_HeuristicGear:
    """
    A robust MPC controller using realistic physics and the L-BFGS optimizer.
    This provides a fair and challenging benchmark for the PIML model.
    """
    def __init__(self, config, device, track_data, args, **kwargs):
        self.config = config
        self.device = device
        
        # MPC parameters
        mpc_params = config.get("mpc_params", {})
        self.horizon = args.mpc_horizon_dev if args.mpc_horizon_dev else config["future_horizon_steps"]
        self.optim_steps = args.mpc_optim_steps_dev if args.mpc_optim_steps_dev else mpc_params.get("optim_steps", 20) # L-BFGS needs fewer steps than Adam
        
        # Vehicle & Physics parameters, converted to tensors on the correct device
        phys_params = config["vehicle_physical_params"]
        vparams_dict = track_data['vehicle_params']
        
        self.num_gears = config["num_gears"]
        self.gear_ratios = torch.tensor(phys_params["gear_ratios"], device=device, dtype=torch.float32)
        self.final_drive_ratio = float(config.get("final_drive_ratio", 3.42))
        self.max_pos_tq = float(phys_params["max_pos_tq"])
        self.dt = float(config["dt"])
        
        self.wheel_radius = torch.tensor(vparams_dict['wheel_radius'], device=device, dtype=torch.float32)
        self.mass = torch.tensor(vparams_dict['mass'], device=device, dtype=torch.float32)
        self.rho = torch.tensor(vparams_dict.get('rho', 1.225), device=device, dtype=torch.float32)
        self.cd = torch.tensor(vparams_dict.get('Cd', 0.6), device=device, dtype=torch.float32)
        self.area = torch.tensor(vparams_dict.get('A', 10.0), device=device, dtype=torch.float32)
        self.crr = torch.tensor(vparams_dict.get('Crr', 0.007), device=device, dtype=torch.float32)
        
        self.rpm_min = float(phys_params.get("rpm_min", 600.0))
        self.rpm_max = float(phys_params.get("rpm_max", 2500.0))
        self.engine_params = config.get("engine_params", {})

        # Cost function weights, tuned for realistic trade-offs
        self.w_energy = 1.0
        self.w_speed_rmse = 15.0
        self.w_rpm_hinge = 30.0
        self.w_torque_smooth = 0.5
        self.w_brake_effort = 2.0

        # State tracking
        self.pid = os.getpid()
        self.prev_torque = torch.tensor(0.0, device=device)

    def _get_gear_candidates(self, current_speed_mps):
        """Smarter gear selection based on plausible RPM range at the CURRENT speed."""
        if current_speed_mps < 1.0:
            return [0, 1]

        wheel_omega_rad_s = current_speed_mps / self.wheel_radius
        engine_omega_rad_s = wheel_omega_rad_s * self.gear_ratios * self.final_drive_ratio
        rpms_for_all_gears = engine_omega_rad_s * (60.0 / (2 * torch.pi))

        rpm_lower_bound, rpm_upper_bound = 800.0, 2800.0
        valid_indices = torch.where(
            (rpms_for_all_gears >= rpm_lower_bound) & (rpms_for_all_gears <= rpm_upper_bound)
        )[0]
        
        if len(valid_indices) == 0:
            closest_idx = torch.argmin(torch.abs(rpms_for_all_gears - (rpm_lower_bound + rpm_upper_bound) / 2))
            return [closest_idx.item()]
            
        return valid_indices.tolist()

    def _simulate_horizon_for_mpc(self, initial_speed, torque_actions, brake_actions, gear_idx, future_slopes_tensor, target_speed_tensor):
        """A self-contained, realistic physics simulation for the MPC's optimization loop."""
        from utils import max_torque_at_rpm_fast, calc_energy_efficiency_torch

        current_speed = initial_speed.clone()
        total_energy_kwh = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        total_speed_error_sq = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        total_rpm_hinge_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        full_gear_ratio = self.gear_ratios[gear_idx] * self.final_drive_ratio

        for step in range(self.horizon):
            slope_rad = future_slopes_tensor[step]
            
            air_drag = 0.5 * self.rho * self.cd * self.area * (current_speed ** 2)
            rolling_res = self.mass * 9.81 * self.crr * torch.cos(slope_rad)
            slope_force = self.mass * 9.81 * torch.sin(slope_rad)
            
            wheel_omega = current_speed / self.wheel_radius
            engine_omega = wheel_omega * full_gear_ratio
            engine_rpm = engine_omega * (60.0 / (2 * torch.pi))
            
            operational_rpm = torch.clamp(engine_rpm, self.rpm_min, self.rpm_max)
            max_tq_at_rpm = max_torque_at_rpm_fast(operational_rpm.unsqueeze(0)).squeeze(0)
            
            brake_force = torch.relu(brake_actions[step]) * 150000.0
            applied_torque_norm = torch.where(brake_actions[step] > 0.01, torch.tensor(0.0, device=self.device), torque_actions[step])
            applied_torque_nm = torch.clamp(applied_torque_norm * self.max_pos_tq, max=max_tq_at_rpm)
            
            propulsive_force = applied_torque_nm * full_gear_ratio / self.wheel_radius
            net_force = propulsive_force - air_drag - rolling_res - slope_force - brake_force
            accel = net_force / self.mass
            current_speed = torch.maximum(current_speed + accel * self.dt, torch.tensor(0.0, device=self.device))
            
            efficiency = calc_energy_efficiency_torch(operational_rpm.unsqueeze(0), applied_torque_nm.unsqueeze(0), self.engine_params).squeeze(0)
            input_power_watts = (applied_torque_nm * engine_omega) / torch.clamp(efficiency, min=0.1)
            step_energy_joules = torch.relu(input_power_watts) * self.dt
            total_energy_kwh += step_energy_joules / 3.6e6
            
            total_speed_error_sq += (current_speed - target_speed_tensor[step]) ** 2
            
            rpm_sweet_spot_min, rpm_sweet_spot_max = 1000.0, 2000.0
            total_rpm_hinge_loss += torch.relu(rpm_sweet_spot_min - engine_rpm) + torch.relu(engine_rpm - rpm_sweet_spot_max)

        torque_smoothness_loss = torch.mean(torch.diff(torque_actions, prepend=self.prev_torque.unsqueeze(0)) ** 2)

        final_cost = (
            self.w_energy * total_energy_kwh +
            self.w_speed_rmse * torch.sqrt(total_speed_error_sq / self.horizon) +
            self.w_rpm_hinge * (total_rpm_hinge_loss / self.horizon) / 1000.0 +
            self.w_torque_smooth * torque_smoothness_loss +
            self.w_brake_effort * torch.mean(torch.relu(brake_actions))
        )
        return final_cost

    def _optimize_actions(self, gear_idx, current_speed, future_slopes_tensor, target_speed_tensor):
        """
        ### PATCH ###
        Optimization routine using the L-BFGS optimizer with a standard closure pattern.
        """
        torque_params = torch.zeros(self.horizon, device=self.device, requires_grad=True)
        brake_params = torch.full((self.horizon,), -5.0, device=self.device, requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [torque_params, brake_params],
            lr=1.0,
            max_iter=self.optim_steps,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            torque_actions = torch.tanh(torque_params)
            brake_actions = torch.sigmoid(brake_params)
            
            cost = self._simulate_horizon_for_mpc(
                current_speed, torque_actions, brake_actions, gear_idx, future_slopes_tensor, target_speed_tensor
            )
            
            if torch.isnan(cost) or torch.isinf(cost):
                print(f"[MPC Worker {self.pid}] Warning: NaN/Inf cost in gear {gear_idx}. Returning high cost.")
                return torch.tensor(float('inf'), device=self.device)

            cost.backward()
            return cost
        
        optimizer.step(closure)

        # Final evaluation with the optimized parameters
        final_torque = torch.tanh(torque_params).detach()
        final_brake = torch.sigmoid(brake_params).detach()
        final_cost = self._simulate_horizon_for_mpc(
            current_speed, final_torque, final_brake, gear_idx, future_slopes_tensor, target_speed_tensor
        )
        
        return final_cost.item(), final_torque, final_brake

    def get_action(self, current_speed, target_speed, current_rpm, current_gear, future_slopes):
        """Main control loop: select best gear and get corresponding actions."""
        future_slopes_tensor = torch.tensor(future_slopes, device=self.device, dtype=torch.float32)
        target_speed_tensor = torch.full((self.horizon,), target_speed, device=self.device, dtype=torch.float32)

        gear_candidates = self._get_gear_candidates(current_speed)
        
        best_cost = float('inf')
        best_gear = current_gear
        best_torque_action = 0.0
        best_brake_action = 0.0

        for gear_idx in gear_candidates:
            cost, torque_seq, brake_seq = self._optimize_actions(
                gear_idx, torch.tensor(current_speed, device=self.device, dtype=torch.float32), future_slopes_tensor, target_speed_tensor
            )
            
            #print(f"[MPC Worker {self.pid}] Gear {gear_idx}: cost={cost:.4f}", flush=True)

            if cost < best_cost and torque_seq is not None:
                best_cost = cost
                best_gear = gear_idx
                best_torque_action = torque_seq[0].item()
                best_brake_action = brake_seq[0].item()
        
        self.prev_torque = torch.tensor(best_torque_action, device=self.device)

        #print(f"[MPC Success] Worker {self.pid}: Chosen Gear={best_gear}, Cost={best_cost:.4f}, Torque={best_torque_action:.3f}, Brake={best_brake_action:.3f}", flush=True)
        return best_torque_action, best_gear, best_brake_action

    def reset(self):
        """Reset controller state."""
        self.prev_torque = torch.tensor(0.0, device=self.device)
        #print(f"[MPC Worker {self.pid}] Controller state reset.")

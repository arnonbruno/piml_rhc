# utils.py
# --- COMPLETE AND FINAL PATCHED VERSION ---
# This version includes all expert-recommended patches AND restores all
# necessary data generation helper functions to fix the `NameError`.

import multiprocessing
import torch
import numpy as np
import random
import math
import os
import time
import json
from functools import partial

# Performance optimizations
torch.backends.cudnn.benchmark = True

##################################
### Utility Functions
##################################

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device(os.getenv("CUDA_DEVICE", "cuda"))
    return torch.device("cpu")

############################################################
# Optimized Normalization and Stats Functions
############################################################
def compute_minmax_stats(windows_X_raw):
    if not windows_X_raw:
        raise ValueError("Cannot compute stats on an empty list of windows.")
    all_data = np.concatenate(windows_X_raw, axis=0)
    feature_mins = all_data.min(axis=0)
    feature_maxs = all_data.max(axis=0)
    eps = 1e-6
    range_mask = (feature_maxs - feature_mins) < eps
    feature_mins[range_mask] -= 0.5
    feature_maxs[range_mask] += 0.5
    zero_range_mask = feature_mins == feature_maxs
    feature_maxs[zero_range_mask] = feature_mins[zero_range_mask] + eps
    return feature_mins, feature_maxs

def apply_minmax_normalization(windows_X_raw, feature_mins, feature_maxs):
    denominator = feature_maxs - feature_mins
    denominator[denominator < 1e-6] = 1e-6
    windows_X_norm = []
    for arr in windows_X_raw:
        arr_norm = (arr - feature_mins) / denominator
        arr_norm = np.clip(arr_norm, 0.0, 1.0)
        windows_X_norm.append(arr_norm)
    return windows_X_norm

############################################################
### Engine Efficiency and Torque Curve (Configurable)
############################################################

def calc_energy_efficiency_torch(rpm, torque, engine_params):
    max_torque_engine = engine_params.get("max_pos_tq", 2600.0)
    peak_efficiency_value = engine_params.get("peak_efficiency_value", 0.45)
    min_operational_efficiency = engine_params.get("min_operational_efficiency", 0.15)
    absolute_min_floor_eff = engine_params.get("absolute_min_floor_eff", 0.10)
    efficient_rpm_center = engine_params.get("efficient_rpm_center", 1200.0)
    efficient_torque_center = engine_params.get("efficient_torque_center", 2200.0)
    rpm_stddev_sweet_spot = engine_params.get("rpm_stddev_sweet_spot", 250.0)
    torque_stddev_sweet_spot = engine_params.get("torque_stddev_sweet_spot", 600.0)
    rpm_term_ss = (rpm - efficient_rpm_center) / (rpm_stddev_sweet_spot + 1e-8)
    tq_term_ss = (torque - efficient_torque_center) / (torque_stddev_sweet_spot + 1e-8)
    eff_sweet_spot_gaussian = torch.exp(-0.5 * (rpm_term_ss**2 + tq_term_ss**2))
    realistic_eff = min_operational_efficiency + (peak_efficiency_value - min_operational_efficiency) * eff_sweet_spot_gaussian
    very_low_load_eff_value = engine_params.get("very_low_load_eff_value", 0.15)
    low_load_center_frac = engine_params.get("low_load_torque_center_fraction", 0.05)
    low_load_steepness = engine_params.get("low_load_steepness_factor", 0.02)
    eff_at_low_load_factor = torch.sigmoid((torque - (low_load_center_frac * max_torque_engine)) * low_load_steepness)
    current_calculated_eff = very_low_load_eff_value + (realistic_eff - very_low_load_eff_value) * eff_at_low_load_factor
    lugging_rpm_thresh = engine_params.get("lugging_rpm_threshold", 950.0)
    lugging_torque_frac_thresh = engine_params.get("lugging_torque_threshold_fraction", 0.40)
    lugging_torque_abs_thresh = lugging_torque_frac_thresh * max_torque_engine
    target_eff_in_lugging = engine_params.get("lugging_target_efficiency", 0.18)
    is_lugging = (rpm < lugging_rpm_thresh) & (torque > lugging_torque_abs_thresh) & (torque > 0)
    final_eff = torch.where(is_lugging, torch.full_like(rpm, target_eff_in_lugging), current_calculated_eff)
    final_eff = torch.maximum(final_eff, torch.tensor(absolute_min_floor_eff, device=rpm.device, dtype=rpm.dtype))
    return final_eff

@torch.jit.script
def max_torque_at_rpm_fast(rpm: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(rpm)
    mask1 = (rpm >= 0) & (rpm < 600)
    out = torch.where(mask1, 1000.0 * (rpm / 600.0), out)
    mask2 = (rpm >= 600) & (rpm < 1000)
    out = torch.where(mask2, 1000.0 + 1600.0 * (rpm - 600.0) / 400.0, out)
    mask3 = (rpm >= 1000) & (rpm <= 1460)
    out = torch.where(mask3, torch.full_like(rpm, 2600.0), out)
    mask4 = (rpm > 1460) & (rpm <= 1800)
    out = torch.where(mask4, 2600.0 - 300.0 * (rpm - 1460.0) / 340.0, out)
    mask5 = (rpm > 1800) & (rpm <= 2100)
    out = torch.where(mask5, 2300.0 - 300.0 * (rpm - 1800.0) / 300.0, out)
    mask6 = (rpm > 2100) & (rpm <= 2200)
    out = torch.where(mask6, 2000.0 - 400.0 * (rpm - 2100.0) / 100.0, out)
    mask7 = (rpm > 2200) & (rpm <= 2500)
    out = torch.where(mask7, 1600.0 - 800.0 * (rpm - 2200.0) / 300.0, out)
    mask8 = (rpm > 2500) & (rpm <= 3000)
    out = torch.where(mask8, 800.0 - 400.0 * (rpm - 2500.0) / 500.0, out)
    mask9 = (rpm > 3000)
    out = torch.where(mask9, torch.full_like(rpm, 100.0), out)
    return torch.relu(out)

def max_torque_at_rpm(rpm_tensor_or_float):
    is_tensor = torch.is_tensor(rpm_tensor_or_float)
    if not is_tensor:
        rpm = torch.tensor([rpm_tensor_or_float], dtype=torch.float32)
    else:
        rpm = rpm_tensor_or_float.float()
    result = max_torque_at_rpm_fast(rpm)
    return result if is_tensor else result.item()

############################################################
### Data Generation Functions (Restored)
############################################################

def simulate_one_step_ref_controller(
    current_speed_mps, current_gear_idx, target_torque_nm, slope_rad,
    vehicle_params_dict, gear_ratios_np, gear_neg_limits_np, max_pos_tq_engine,
    dt=1.0, substeps=5, rpm_min_heuristic=800.0, rpm_max_heuristic=2200.0,
    rpm_min_op=600.0, rpm_max_op=3500.0, max_speed_mps=30.0, final_drive_ratio=3.42
):
    dt_sub = dt / substeps
    speed_mps = float(current_speed_mps)
    wheel_radius = vehicle_params_dict['wheel_radius']
    mass = vehicle_params_dict['mass']
    rho = vehicle_params_dict['rho']
    Cd = vehicle_params_dict['Cd']
    A_val = vehicle_params_dict['A']
    Crr = vehicle_params_dict['Crr']
    if speed_mps < 0.1:
        rpm_before_shift = 0
    else:
        rpm_before_shift = speed_mps / (wheel_radius + 1e-9) * gear_ratios_np[current_gear_idx] * final_drive_ratio * (60.0 / (2 * math.pi))
    if rpm_before_shift < rpm_min_heuristic and current_gear_idx > 0:
        current_gear_idx -= 1
    elif rpm_before_shift > rpm_max_heuristic and current_gear_idx < len(gear_ratios_np) - 1:
        current_gear_idx += 1
    current_gear_ratio = gear_ratios_np[current_gear_idx] * final_drive_ratio
    neg_torque_limit_gear = gear_neg_limits_np[current_gear_idx]
    for _ in range(substeps):
        slope_force = mass * 9.81 * math.sin(slope_rad)
        air_drag = 0.5 * rho * Cd * A_val * (speed_mps**2)
        rolling_res = Crr * mass * 9.81 * math.cos(slope_rad)
        if speed_mps < 0.1:
            engine_rpm = 0
        else:
            wheel_omega_rad_s = speed_mps / (wheel_radius + 1e-9)
            engine_omega_rad_s = wheel_omega_rad_s * current_gear_ratio
            engine_rpm = engine_omega_rad_s * (60.0 / (2 * math.pi))
        clamped_engine_rpm = np.clip(engine_rpm, rpm_min_op, rpm_max_op)
        max_engine_torque_at_rpm_val = max_torque_at_rpm(clamped_engine_rpm)
        final_engine_tq = np.clip(target_torque_nm, neg_torque_limit_gear, max_engine_torque_at_rpm_val)
        wheel_torque_propulsive = final_engine_tq * current_gear_ratio
        total_resistive_force = slope_force + air_drag + rolling_res
        propulsive_force_at_wheels = wheel_torque_propulsive / (wheel_radius + 1e-9)
        net_force = propulsive_force_at_wheels - total_resistive_force
        accel = net_force / mass
        speed_mps += accel * dt_sub
        speed_mps = np.clip(speed_mps, 0.0, max_speed_mps)
    if speed_mps < 0.1:
        final_actual_rpm = 0
    else:
        final_actual_rpm = speed_mps / (wheel_radius + 1e-9) * current_gear_ratio * (60.0 / (2 * math.pi))
    return speed_mps, current_gear_idx, final_actual_rpm

def generate_smooth_slopes(
    total_length_m=1000, segment_length_mean=80, segment_length_std=10,
    base_slope_distribution="mostly_flat", slope_change_std=0.3, max_slope_deg=5.0
):
    segments = []
    length_so_far = 0.0
    if base_slope_distribution == "flat":
        current_slope_deg = np.random.normal(0.0, 0.05)
    elif base_slope_distribution == "mostly_flat":
        choice = np.random.choice(["flat_seg", "mild_up_seg", "mild_down_seg"], p=[0.7, 0.15, 0.15])
        if choice == "flat_seg": base_slope_deg = np.random.normal(0.0, 0.5)
        elif choice == "mild_up_seg": base_slope_deg = np.random.normal(1.5, 0.5)
        else: base_slope_deg = np.random.normal(-1.5, 0.5)
        current_slope_deg = np.clip(base_slope_deg, -max_slope_deg, max_slope_deg)
    else: # mixed
        current_slope_deg = np.random.uniform(-max_slope_deg, max_slope_deg)
    while length_so_far < total_length_m:
        seg_len = max(10, int(np.random.normal(segment_length_mean, segment_length_std)))
        slope_rad = math.radians(current_slope_deg)
        segments.append((seg_len, slope_rad))
        length_so_far += seg_len
        if base_slope_distribution == "flat": delta = np.random.normal(0.0, 0.005)
        else: delta = np.random.normal(0.0, slope_change_std)
        current_slope_deg += delta
        current_slope_deg = np.clip(current_slope_deg, -max_slope_deg, max_slope_deg)
    return segments

def _generate_single_track_data(sample_index, track_length, config_data):
    # --- START: DEFINITIVE FIX FOR MULTIPROCESSING RANDOMNESS ---
    # This block creates a unique seed for each worker process to ensure
    # true randomness across all parallel track generation calls.
    base_seed = config_data.get("seed", 42)
    process_id = os.getpid()
    current_time_ns = time.time_ns()
    
    # The final seed is a robust combination of the base seed, the process ID,
    # the unique sample index for the worker, and the current nanosecond time.
    unique_seed = (base_seed + process_id + sample_index + current_time_ns) % (2**32)
    np.random.seed(unique_seed)
    random.seed(unique_seed)
    # --- END: DEFINITIVE FIX ---

    dt = config_data["dt"]
    mass_range = config_data.get("mass_range_training", (20000, 70000))
    wheel_radius_range = config_data.get("wheel_radius_range_training", (0.45, 0.55))
    frontal_area_range = config_data.get("frontal_area_range_training", (9.0, 11.0))
    gen_params = config_data.get("data_generation_params", {})
    segment_length_mean = gen_params.get("segment_length_mean", 80)
    segment_length_std = gen_params.get("segment_length_std", 10)
    base_slope_distribution = gen_params.get("base_slope_distribution", "mostly_flat")
    slope_change_std_gen = gen_params.get("slope_change_std", 0.1)
    max_slope_deg_gen = gen_params.get("max_slope_deg", 2.0)
    init_speed_range = gen_params.get("init_speed_range", (3.0, 15.0))
    stop_go_prob = gen_params.get("stop_go_prob", 0.0)
    stop_duration_range_s = gen_params.get("stop_duration_range_s", (2, 5))
    forced_downhill_prob = gen_params.get("forced_downhill_prob", 0.0)
    forced_downhill_deg_range = gen_params.get("forced_downhill_deg_range", (-5.0, -2.0))
    segs = generate_smooth_slopes(
        total_length_m=track_length, segment_length_mean=segment_length_mean, segment_length_std=segment_length_std,
        base_slope_distribution=base_slope_distribution, slope_change_std=slope_change_std_gen, max_slope_deg=max_slope_deg_gen)
    if np.random.rand() < forced_downhill_prob and segs:
        idx_to_change = np.random.randint(0, len(segs))
        forced_deg = np.random.uniform(*forced_downhill_deg_range)
        segs[idx_to_change] = (segs[idx_to_change][0], math.radians(forced_deg))
    track_initial_speed = np.random.uniform(init_speed_range[0], init_speed_range[1])
    vparams_current = {
        'wheel_radius': np.random.uniform(wheel_radius_range[0], wheel_radius_range[1]),
        'mass': np.random.uniform(mass_range[0], mass_range[1]), 'rho': np.random.uniform(1.15, 1.25),
        'Cd': np.random.uniform(0.55, 0.75), 'A': np.random.uniform(frontal_area_range[0], frontal_area_range[1]),
        'Crr': np.random.uniform(0.0060, 0.0070)}
    slope_list_for_track = []
    current_track_len = 0
    for seg_len_m, slope_rad_val in segs:
        num_steps_in_segment = int(seg_len_m)
        if current_track_len + num_steps_in_segment > track_length:
            num_steps_in_segment = track_length - current_track_len
        slope_list_for_track.extend([slope_rad_val] * num_steps_in_segment)
        current_track_len += num_steps_in_segment
        if current_track_len >= track_length: break
    track_slopes_np = np.array(slope_list_for_track, dtype=np.float32)
    if len(track_slopes_np) > track_length:
        track_slopes_np = track_slopes_np[:track_length]
    elif len(track_slopes_np) < track_length:
        pad_val = track_slopes_np[-1] if len(track_slopes_np) > 0 else 0.0
        track_slopes_np = np.pad(track_slopes_np, (0, track_length - len(track_slopes_np)), 'constant', constant_values=pad_val)
    
    mission_target_speed = compute_target_speed(track_slopes_np, vparams_current, config_data)
    
    current_ref_speed_mps = float(track_initial_speed)
    phys_params = config_data.get("vehicle_physical_params", {})
    gear_ratios_np = np.array(phys_params.get("gear_ratios", []))
    gear_neg_limits_np = np.array(phys_params.get("gear_neg_limits", []))
    num_gears = len(gear_ratios_np)
    current_ref_gear_idx = min(max(0, config_data.get("ref_start_gear_idx", 2)), num_gears - 1) if num_gears > 0 else 0
    track_ref_speeds_profile = np.zeros(track_length, dtype=np.float32)
    heuristic_target_speed = mission_target_speed * config_data.get("ref_speed_target_fraction", 1.0)
    max_pos_tq_engine = phys_params.get("max_pos_tq", 2600.0)
    stop_start_time = -1
    stop_end_time = -1
    if np.random.rand() < stop_go_prob and track_length > 100:
        stop_duration = np.random.randint(stop_duration_range_s[0], stop_duration_range_s[1] + 1)
        start_range = (track_length // 4, track_length * 3 // 4 - stop_duration)
        if start_range[0] < start_range[1]:
            stop_start_time = np.random.randint(start_range[0], start_range[1])
            stop_end_time = stop_start_time + stop_duration
    for t in range(track_length):
        track_ref_speeds_profile[t] = current_ref_speed_mps
        is_stopping = stop_start_time <= t < stop_end_time
        is_post_stop_reacceleration = t == stop_end_time
        if is_stopping:
            target_torque_nm = config_data.get("ref_torque_braking_fraction", -0.4) * max_pos_tq_engine
        elif is_post_stop_reacceleration:
            current_ref_speed_mps = 0.1; current_ref_gear_idx = 0
            target_torque_nm = config_data.get("ref_torque_accel_fraction", 0.7) * max_pos_tq_engine
        else:
            if current_ref_speed_mps < 0.75 * heuristic_target_speed:
                target_torque_nm = config_data.get("ref_torque_accel_fraction", 0.7) * max_pos_tq_engine
            elif current_ref_speed_mps > 1.1 * heuristic_target_speed:
                target_torque_nm = config_data.get("ref_torque_decel_fraction", -0.1) * max_pos_tq_engine
            else:
                target_torque_nm = config_data.get("ref_torque_cruise_fraction", 0.05) * max_pos_tq_engine
        current_ref_speed_mps, current_ref_gear_idx, _ = simulate_one_step_ref_controller(
            current_speed_mps=current_ref_speed_mps, current_gear_idx=current_ref_gear_idx,
            target_torque_nm=target_torque_nm, slope_rad=track_slopes_np[t],
            vehicle_params_dict=vparams_current, gear_ratios_np=gear_ratios_np,
            gear_neg_limits_np=gear_neg_limits_np, max_pos_tq_engine=max_pos_tq_engine,
            dt=dt, substeps=config_data.get("substeps_ref_sim", 1),
            rpm_min_heuristic=config_data.get("ref_rpm_min_shift", 1000.0),
            rpm_max_heuristic=config_data.get("ref_rpm_max_shift", 2200.0),
            rpm_min_op=phys_params.get("rpm_min", 600.0),
            rpm_max_op=config_data.get("rpm_max_op_ref_sim", phys_params.get("rpm_max", 3500.0)),
            max_speed_mps=config_data.get("max_speed", 30.0),
            final_drive_ratio=phys_params.get("final_drive_ratio", 3.42))
        if is_stopping: current_ref_speed_mps = max(0.0, current_ref_speed_mps)
    return (track_slopes_np, track_ref_speeds_profile, vparams_current, track_initial_speed, mission_target_speed)


def generate_sequences_consistent(n_samples, track_length=1000, config_data=None):
    if config_data is None: raise ValueError("`config_data` must be provided.")
    phys_params = config_data.get("vehicle_physical_params", {})
    if not phys_params.get("gear_ratios") or not phys_params.get("gear_neg_limits"):
        raise ValueError("Gear information is missing or empty in the provided config.")
    num_cpus = os.cpu_count() or 1
    num_cpus_generation = min(config_data.get("num_track_generation_workers", 8), num_cpus)
    print(f"    (Generating tracks using {num_cpus_generation} CPU cores)", flush=True)
    worker_func_track_gen = partial(_generate_single_track_data, track_length=track_length, config_data=config_data)
    with multiprocessing.Pool(processes=num_cpus_generation) as pool:
        sample_indices = list(range(n_samples))
        parallel_results = pool.map(worker_func_track_gen, sample_indices)
    all_tracks_slopes_list = [res[0] for res in parallel_results]
    all_tracks_reference_speeds_list = [res[1] for res in parallel_results]
    vehicle_params_list = [res[2] for res in parallel_results]
    track_initial_speeds_arr = np.array([res[3] for res in parallel_results], dtype=np.float32)
    mission_target_speeds_list = [res[4] for res in parallel_results]
    return (all_tracks_slopes_list, all_tracks_reference_speeds_list,
            vehicle_params_list, track_initial_speeds_arr, mission_target_speeds_list)

def compute_target_speed(track_slopes, vparams, config_data):
    target_speed_logic_cfg = config_data.get("target_speed_logic", {})
    avg_slope_rad = np.mean(track_slopes) if len(track_slopes) > 0 else 0.0
    mass = vparams['mass']
    base_target_speed = target_speed_logic_cfg.get("base_speed_mps", 20.0)
    slope_factor = target_speed_logic_cfg.get("slope_factor", 50.0)
    mass_effect_factor = target_speed_logic_cfg.get("mass_effect_factor", 3.0)
    mass_ref_min = target_speed_logic_cfg.get("mass_ref_min", 20000.0)
    mass_ref_max = target_speed_logic_cfg.get("mass_ref_max", 70000.0)
    min_target_speed = target_speed_logic_cfg.get("min_target_speed_mps", 8.0)
    max_target_speed = target_speed_logic_cfg.get("max_target_speed_mps", 25.0)
    target_speed_mps = base_target_speed - avg_slope_rad * slope_factor
    mass_normalized_effect = ((mass - mass_ref_min) / (mass_ref_max - mass_ref_min + 1e-6)) * mass_effect_factor
    target_speed_mps -= mass_normalized_effect
    target_speed_mps = np.clip(target_speed_mps, min_target_speed, max_target_speed)
    return float(target_speed_mps)

def process_single_track_for_windows(track_data_tuple_extended, config, generate_aux_data=True):
    track_slopes, track_reference_speeds, vparams, _track_start_speed, mission_target_speed = track_data_tuple_extended
    past_context_steps = config["past_context_steps"]
    future_horizon_steps = config["future_horizon_steps"]
    stride = config.get("window_stride", 30)
    agent_operational_mode = config.get("agent_operational_mode", "scalar_target")
    track_windows_X = []
    track_windows_Y = [] if generate_aux_data else None
    track_theoretical_energies = [] if generate_aux_data else None
    track_len = len(track_slopes)
    if len(track_reference_speeds) != track_len:
        return [], [], []
    param_array = np.array([vparams['wheel_radius'], vparams['mass'], vparams['rho'], vparams['Cd'], vparams['A'], vparams['Crr']], dtype=np.float32)
    total_window_len = past_context_steps + future_horizon_steps
    if track_len < total_window_len:
        return [], [], []
    max_idx = track_len - total_window_len
    start_idx = 0
    while start_idx <= max_idx:
        future_start = start_idx + past_context_steps
        future_end = future_start + future_horizon_steps
        window_slopes_segment = track_slopes[start_idx:start_idx + total_window_len]
        time_len_window = len(window_slopes_segment)
        window_feature7_speeds = np.zeros(total_window_len, dtype=np.float32)
        window_feature7_speeds[:past_context_steps] = track_reference_speeds[start_idx:future_start]
        if agent_operational_mode == "scalar_target":
            window_feature7_speeds[past_context_steps:] = mission_target_speed
        else:
            window_feature7_speeds[past_context_steps:] = track_reference_speeds[future_start:future_end]
        current_speed_col = window_feature7_speeds.reshape(-1, 1)
        window_slopes_2d = window_slopes_segment.reshape(-1, 1)
        repeated_params = np.tile(param_array, (time_len_window, 1))
        mission_target_speed_col = np.full((time_len_window, 1), mission_target_speed, dtype=np.float32)
        window_x = np.concatenate([window_slopes_2d, repeated_params, current_speed_col, mission_target_speed_col], axis=1)
        track_windows_X.append(window_x)
        if generate_aux_data:
            dummy_y = np.zeros((future_horizon_steps, 3), dtype=np.float32)
            dummy_y[:, 2] = mission_target_speed
            track_windows_Y.append(dummy_y)
        start_idx += stride
    return track_windows_X, track_windows_Y, track_theoretical_energies

############################################################
### Physics Simulation Functions
############################################################

def simulate_one_step_soft(
    engine_torque, brake_signal, gear_probs, gear_ratios, gear_neg_limits, max_pos_tq,
    current_speed, wheel_radius, vehicle_mass,
    ext_slope_force, ext_air_drag, ext_rolling_res,
    engine_efficiency_func, device, config
):
    
    phys_params = config.get("vehicle_physical_params", {})
    final_drive_ratio = phys_params.get("final_drive_ratio", 3.42)
    rpm_min = phys_params.get("rpm_min", 600.0)
    rpm_max = config.get("rpm_max_sim", phys_params.get("rpm_max", 3500.0))
    max_speed = config.get("max_speed", 30.0)
    dt = config["dt"]
    substeps = config["substeps"]
    max_brake_force = phys_params.get("max_friction_brake_force", 150000.0)
    engine_params = config.get("engine_params", {})
    B = current_speed.shape[0]
    dt_sub = dt / substeps
    if gear_ratios.ndim == 1: gear_ratios_b = gear_ratios.view(1, -1).expand(B, -1)
    else: gear_ratios_b = gear_ratios
    if gear_neg_limits.ndim == 1: gear_neg_limits_b = gear_neg_limits.view(1, -1).expand(B, -1)
    else: gear_neg_limits_b = gear_neg_limits
    if ext_slope_force.ndim == 0 and B > 1: ext_slope_force = ext_slope_force.expand(B)
    if ext_air_drag.ndim == 0 and B > 1: ext_air_drag = ext_air_drag.expand(B)
    if ext_rolling_res.ndim == 0 and B > 1: ext_rolling_res = ext_rolling_res.expand(B)

    # === EXPERT FIX: STRAIGHT-THROUGH ESTIMATOR (STE) FOR HONEST PHYSICS ===
    ratio_soft = (gear_probs * gear_ratios_b).sum(dim=-1) * final_drive_ratio
    gear_idx = gear_probs.argmax(dim=-1)
    ratio_hard = gear_ratios[gear_idx] * final_drive_ratio
    ratio_for_physics = ratio_soft + (ratio_hard - ratio_soft).detach()
    
    neg_limit_soft = (gear_probs * gear_neg_limits_b).sum(dim=-1)
    speeds_sim = current_speed.clone()
    total_engine_energy_kwh = torch.zeros_like(speeds_sim, device=device)
    total_friction_brake_kwh = torch.zeros_like(speeds_sim, device=device)
    rpm_lower_violation_sum_dt = torch.zeros_like(speeds_sim, device=device)
    rpm_upper_violation_sum_dt = torch.zeros_like(speeds_sim, device=device)
    engine_torque_scaled_cmd = engine_torque.squeeze(-1) * max_pos_tq
    brake_signal_cmd = brake_signal.squeeze(-1)
    raw_rpm_last_substep = torch.zeros_like(speeds_sim, device=device)
    final_engine_tq_last_substep = torch.zeros_like(speeds_sim, device=device)
    
    speeds_sim = torch.clamp(speeds_sim, min=1e-6)
    
    for i_substep in range(substeps):
        wheel_omega_rad_s = speeds_sim / (wheel_radius + 1e-8)
        raw_rpm = wheel_omega_rad_s * ratio_for_physics * (60.0 / (2 * math.pi))
        
        rpm_viol_lower_substep = torch.relu(rpm_min - raw_rpm)
        rpm_viol_upper_substep = torch.relu(raw_rpm - rpm_max)
        rpm_lower_violation_sum_dt += rpm_viol_lower_substep
        rpm_upper_violation_sum_dt += rpm_viol_upper_substep
        engine_rpm_op = torch.clamp(raw_rpm, min=rpm_min, max=rpm_max)
        max_engine_torque_at_this_rpm = max_torque_at_rpm_fast(engine_rpm_op)
        final_engine_tq_applied = torch.clamp(engine_torque_scaled_cmd, min=neg_limit_soft, max=max_engine_torque_at_this_rpm)
        engine_omega_op_rad_s = engine_rpm_op * (2 * math.pi / 60.0)
        
        input_power_crank_watts = final_engine_tq_applied * engine_omega_op_rad_s

        if engine_efficiency_func is not None:
            current_eff = engine_efficiency_func(engine_rpm_op, final_engine_tq_applied, engine_params)
            safe_current_eff = torch.clamp(current_eff, min=0.1)

            # --- START: FINAL, ROBUST ENERGY LOGIC ---

            # 1. Get parameters and define constants.
            idle_fuel_power_kw = engine_params.get("idle_fuel_power_kw", 5.0)
            brake_penalty_fraction = 0.15 # Pumping loss penalty

            # 2. Create a device- and dtype-aware tensor for idle power.
            #    This prevents potential device/type mismatch errors during training.
            idle_power_watts = torch.tensor(
                idle_fuel_power_kw * 1000.0,
                device=input_power_crank_watts.device,
                dtype=input_power_crank_watts.dtype
            )

            # 3. Create boolean masks for the three distinct engine states.
            propulsion_mask = input_power_crank_watts > 0
            braking_mask = input_power_crank_watts < 0

            # 4. Calculate fuel power for each state.
            fuel_input_power_watts = torch.zeros_like(input_power_crank_watts)

            # Propulsion: Standard BSFC-based calculation.
            fuel_input_power_watts[propulsion_mask] = (
                input_power_crank_watts[propulsion_mask] / safe_current_eff[propulsion_mask]
            )
            # Engine Braking: Cost is idle fuel + pumping losses.
            fuel_input_power_watts[braking_mask] = (
                idle_power_watts
                + brake_penalty_fraction * torch.abs(input_power_crank_watts[braking_mask])
            )
            # Idling: Cost is idle fuel.
            fuel_input_power_watts[~(propulsion_mask | braking_mask)] = idle_power_watts
            # --- END: CORRECTED ENERGY LOGIC ---
        else:
            fuel_input_power_watts = torch.where(input_power_crank_watts > 0, input_power_crank_watts / 0.35, torch.zeros_like(input_power_crank_watts))
        engine_energy_joules_substep = fuel_input_power_watts * dt_sub
        applied_brake_force = brake_signal_cmd * max_brake_force
        friction_brake_power_watts = applied_brake_force * speeds_sim
        friction_brake_joules_substep = torch.nan_to_num(friction_brake_power_watts * dt_sub, nan=0.0, posinf=0.0, neginf=0.0)
        
        effective_wheel_torque = final_engine_tq_applied * ratio_for_physics
        
        propulsive_force_at_wheels = effective_wheel_torque / (wheel_radius + 1e-8)
        net_force_on_vehicle = propulsive_force_at_wheels - ext_slope_force - ext_air_drag - ext_rolling_res - applied_brake_force
        accel = net_force_on_vehicle / vehicle_mass
        speeds_sim = torch.clamp(speeds_sim + accel * dt_sub, 0.0, max_speed)
        total_engine_energy_kwh += engine_energy_joules_substep / 3.6e6
        total_friction_brake_kwh += friction_brake_joules_substep / 3.6e6
        if i_substep == substeps - 1:
            raw_rpm_last_substep = raw_rpm.clone()
            final_engine_tq_last_substep = final_engine_tq_applied.clone()
            
    logged_ratio = ratio_hard
            
    return (speeds_sim, total_engine_energy_kwh, total_friction_brake_kwh,
            {"rpm_lower": rpm_lower_violation_sum_dt, "rpm_upper": rpm_upper_violation_sum_dt},
            logged_ratio, neg_limit_soft, raw_rpm_last_substep, final_engine_tq_last_substep)


def simulate_horizon_soft(
    batch_torque, batch_brake_signal, batch_gear_probs, gear_ratios, gear_neg_limits, max_pos_tq,
    initial_speed, slopes, vehicle_params,
    engine_efficiency_func, device, config,
    p_scheduled_sampling=0.0, 
    reference_speeds=None, 
    track_gear_indices=True,
    **kwargs 
):
    B, horizon, _ = batch_torque.shape
    vp_wheel_radius, vp_mass, vp_rho, vp_Cd, vp_A, vp_Crr = vehicle_params.T
    current_speeds_sim = initial_speed.clone().to(device)
    if current_speeds_sim.ndim == 0: current_speeds_sim = current_speeds_sim.unsqueeze(0)
    if current_speeds_sim.shape[0] != B and B > 1: current_speeds_sim = current_speeds_sim.expand(B)

    sim_speeds_over_horizon = torch.zeros(B, horizon, device=device)
    sim_energies_kwh_step = torch.zeros(B, horizon, device=device)
    sim_friction_brake_kwh_step = torch.zeros(B, horizon, device=device)
    sim_rpm_lower_violations_sum_horizon = torch.zeros(B, device=device)
    sim_rpm_upper_violations_sum_horizon = torch.zeros(B, device=device)
    sim_gear_indices_horizon = torch.zeros(B, horizon, dtype=torch.long, device=device) if track_gear_indices else None
    sim_raw_rpms_last_substep_horizon = torch.zeros(B, horizon, device=device)
    sim_ratio_soft_horizon = torch.zeros(B, horizon, device=device)
    sim_final_engine_tq_horizon = torch.zeros(B, horizon, device=device)

    for t in range(horizon):
        if p_scheduled_sampling > 0 and reference_speeds is not None:
            use_student_forcing = torch.rand(B, device=device) < p_scheduled_sampling
            teacher_forced_speed = reference_speeds[:, t]
            current_speeds_sim = torch.where(use_student_forcing, current_speeds_sim, teacher_forced_speed)

        torque_cmd_t = batch_torque[:, t, :]
        brake_signal_t = batch_brake_signal[:, t, :]
        gear_probs_t = batch_gear_probs[:, t, :]
        slope_rad_t = slopes[:, t]
        current_ext_slope_force = vp_mass * 9.81 * torch.sin(slope_rad_t)
        current_ext_air_drag = 0.5 * vp_rho * vp_Cd * vp_A * (current_speeds_sim**2)
        current_ext_rolling_res = vp_mass * 9.81 * vp_Crr * torch.cos(slope_rad_t)

        (next_speeds_sim, eng_energy_kwh_dt, friction_brake_kwh_dt,
         rpm_violations_dt_dict, ratio_dt, _neg_limit_soft_dt,
         raw_rpm_last_substep_dt, final_engine_tq_dt) = simulate_one_step_soft(
            engine_torque=torque_cmd_t, brake_signal=brake_signal_t, gear_probs=gear_probs_t,
            gear_ratios=gear_ratios, gear_neg_limits=gear_neg_limits, max_pos_tq=max_pos_tq,
            current_speed=current_speeds_sim, wheel_radius=vp_wheel_radius, vehicle_mass=vp_mass,
            ext_slope_force=current_ext_slope_force, ext_air_drag=current_ext_air_drag,
            ext_rolling_res=current_ext_rolling_res, engine_efficiency_func=engine_efficiency_func,
            device=device, config=config)
        
        current_speeds_sim = next_speeds_sim
        sim_speeds_over_horizon[:, t] = current_speeds_sim
        sim_energies_kwh_step[:, t] = eng_energy_kwh_dt
        sim_friction_brake_kwh_step[:, t] = friction_brake_kwh_dt
        sim_rpm_lower_violations_sum_horizon += rpm_violations_dt_dict['rpm_lower']
        sim_rpm_upper_violations_sum_horizon += rpm_violations_dt_dict['rpm_upper']
        sim_raw_rpms_last_substep_horizon[:, t] = raw_rpm_last_substep_dt
        sim_ratio_soft_horizon[:, t] = ratio_dt 
        sim_final_engine_tq_horizon[:, t] = final_engine_tq_dt
        if track_gear_indices and sim_gear_indices_horizon is not None:
            sim_gear_indices_horizon[:, t] = torch.argmax(gear_probs_t, dim=-1)

    return (sim_speeds_over_horizon, sim_energies_kwh_step, sim_friction_brake_kwh_step,
            sim_rpm_lower_violations_sum_horizon, sim_rpm_upper_violations_sum_horizon,
            sim_gear_indices_horizon, sim_raw_rpms_last_substep_horizon,
            sim_ratio_soft_horizon, sim_final_engine_tq_horizon)

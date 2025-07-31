# main.py
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import time
import numpy as np
from pathlib import Path
import copy
import torch.nn.functional as F
import math
from collections import deque
import random
import multiprocessing
from functools import partial

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

# setting random seed for all libraries
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
# setting pytorch random seed
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from models import HorizonTransformer
from utils import (
    set_seeds, get_device,
    compute_minmax_stats, apply_minmax_normalization,
    generate_sequences_consistent,
    process_single_track_for_windows,
    calc_energy_efficiency_torch,
    simulate_horizon_soft,
)

class HorizonDataset(Dataset):
    def __init__(self, windows_X, device="cpu"):
        self.windows_X = windows_X
        self.device = device
    def __len__(self):
        return len(self.windows_X)
    def __getitem__(self, idx):
        return torch.tensor(self.windows_X[idx], dtype=torch.float32)

def get_tau(current_epoch, total_epochs, tau_start=5.0, tau_end=0.5, decay_rate=0.975):
    tau = tau_start * (decay_rate ** current_epoch)
    return max(tau, tau_end)

def get_scheduled_sampling_p(epoch_in_stage, total_epochs_in_stage, stage_idx, num_stages):
    return 0.0

class DwaBalancer:
    def __init__(self, task_list, device, temperature=2.0, window_size=3):
        self.task_list = task_list
        self.T = temperature
        self.window_size = window_size
        self.device = device
        self.loss_history = {task: deque(maxlen=window_size) for task in self.task_list}
        self.weights = {task: 1.0 for task in self.task_list}
        print(f"✅ DWA Balancer initialized for tasks: {task_list}")

    def update_losses(self, epoch_component_losses):
        for task in self.task_list:
            if task in epoch_component_losses and epoch_component_losses[task] > 0:
                self.loss_history[task].append(epoch_component_losses[task])

    def compute_weights(self, max_weight_cap=5.0):
        if len(next(iter(self.loss_history.values()), [])) < self.window_size:
            return self.weights

        avg_losses_prev = {task: np.mean(list(self.loss_history[task])[:-1]) for task in self.task_list if len(self.loss_history[task]) > 1}
        avg_losses_curr = {task: np.mean(list(self.loss_history[task])) for task in self.task_list if len(self.loss_history[task]) > 0}

        ratios = {}
        for task in self.task_list:
            if task in avg_losses_prev and task in avg_losses_curr and avg_losses_prev[task] > 1e-8:
                raw_ratio = avg_losses_curr[task] / avg_losses_prev[task]
                ratios[task] = torch.clamp(torch.tensor(raw_ratio, device=self.device), 0.3, 3.0)
            else:
                ratios[task] = torch.tensor(1.0, device=self.device)

        exp_ratios = {task: torch.exp(r / self.T) for task, r in ratios.items()}
        sum_exp_ratios = sum(exp_ratios.values()) + 1e-8

        for task in self.task_list:
            weight = len(self.task_list) * exp_ratios[task] / sum_exp_ratios
            self.weights[task] = min(weight.item(), max_weight_cap)
        return self.weights

def run_diagnostic_callback(model, diagnostic_batch, config, feature_mins_tensor, feature_maxs_tensor, gear_ratios_tensor, device, epoch_info_str):
    """
    Runs the model on a single, fixed truck from a validation batch to provide
    qualitative insight into the model's behavior at a point in training.
    """
    print("-" * 80)
    print(f"🕵️  Running Diagnostic Callback for: {epoch_info_str}")
    print("-" * 80)

    model.eval()
    with torch.no_grad():
        batch_x = diagnostic_batch.to(device)
        truck_x = batch_x[0]

        vp_norm = truck_x[0, 1:7]
        vp_denorm_factors = (feature_maxs_tensor[1:7] - feature_mins_tensor[1:7]).clamp(min=1e-6)
        vp_denorm = vp_norm * vp_denorm_factors + feature_mins_tensor[1:7]
        truck_mass = vp_denorm[1].item()
        frontal_area = vp_denorm[4].item()

        slope_norm = truck_x[:, 0]
        slope_denorm_factor = (feature_maxs_tensor[0] - feature_mins_tensor[0]).clamp(min=1e-6)
        slope_rad = slope_norm * slope_denorm_factor + feature_mins_tensor[0]
        slope_deg = torch.rad2deg(slope_rad)

        speed_denorm_factor = (feature_maxs_tensor[7] - feature_mins_tensor[7]).clamp(min=1e-6)
        initial_speed_mps = (truck_x[config["past_context_steps"] - 1, 7] * speed_denorm_factor + feature_mins_tensor[7]).item()
        target_speed_mps = (truck_x[config["past_context_steps"], 7] * speed_denorm_factor + feature_mins_tensor[7]).item()

        print(f"  Track Info (Single Truck): Mass={truck_mass:.0f} kg, Area={frontal_area:.1f} m^2, Start Speed={initial_speed_mps:.1f} m/s, Target Speed={target_speed_mps:.1f} m/s")
        print(f"                           Slope Profile (deg): Min={slope_deg.min():.2f}, Avg={slope_deg.mean():.2f}, Max={slope_deg.max():.2f}")

        raw_torque, gear_probs_soft, gear_probs_hard, brake_signal = model(batch_x, tau=0.1)

        sim_outputs = simulate_horizon_soft(
            batch_torque=raw_torque, batch_brake_signal=brake_signal,
            batch_gear_probs=gear_probs_hard,
            gear_ratios=gear_ratios_tensor,
            gear_neg_limits=model.gear_neg_limits, max_pos_tq=model.max_pos_tq,
            initial_speed=batch_x[:, config["past_context_steps"] - 1, 7] * speed_denorm_factor + feature_mins_tensor[7],
            slopes=batch_x[:, config["past_context_steps"]:, 0] * slope_denorm_factor + feature_mins_tensor[0],
            vehicle_params=batch_x[:, 0, 1:7] * vp_denorm_factors + feature_mins_tensor[1:7],
            engine_efficiency_func=calc_energy_efficiency_torch,
            device=device, config=config, track_gear_indices=True
        )

        sim_speeds = sim_outputs[0][0]
        sim_energy_step = sim_outputs[1][0]
        sim_gears = sim_outputs[5][0]
        sim_rpms = sim_outputs[6][0]
        sim_torques = sim_outputs[8][0]

        ref_speeds_horizon = truck_x[config["past_context_steps"]:, 7] * speed_denorm_factor + feature_mins_tensor[7]
        speed_error_rmse = torch.sqrt(torch.mean((sim_speeds - ref_speeds_horizon)**2)).item()

        rpm_max_cfg = config.get("rpm_max_sim", 3500.0)
        rpm_min_cfg = config.get("vehicle_physical_params", {}).get("rpm_min", 600.0)
        violation_steps_count = torch.sum((sim_rpms > rpm_max_cfg) | (sim_rpms < rpm_min_cfg)).item()

        total_energy = sim_energy_step.sum().item()
        num_gear_shifts = torch.sum(sim_gears[:-1] != sim_gears[1:]).item()
        gear_sequence_str = " ".join(map(str, sim_gears.tolist()))

        print("\n  Model Behavior (Single Truck):")
        print(f"    ⚙️ Gear Sequence: {gear_sequence_str}")
        print(f"    - Shifts: {num_gear_shifts}")
        print(f"    📈 RPM (sim):     Min={sim_rpms.min():.0f}, Avg={sim_rpms.mean():.0f}, Max={sim_rpms.max():.0f} | Violations: {violation_steps_count} steps")
        print(f"    💪 Torque (sim):  Min={sim_torques.min():.0f}, Avg={sim_torques.mean():.0f}, Max={sim_torques.max():.0f} Nm")
        print(f"    🚗 Speed Error (RMSE vs Ref): {speed_error_rmse:.3f} m/s")
        print(f"    ⚡️ Total Energy Consumed (horizon): {total_energy:.4f} kWh")
        print("-" * 80 + "\n")


def loss_function_horizon_configurable(
    batch_X, model, raw_torque_model_output,
    gear_probs_model_output_soft, gear_probs_model_output_hard,
    brake_signal_model_output,
    config, feature_mins_tensor, feature_maxs_tensor, gear_ratios_tensor,
    tau=1.0, hard=False, mode="research", p_scheduled_sampling=0.0,
    ablation_name="l-base"
):
    device = batch_X.device
    B, _, _ = batch_X.shape
    past_context_steps = config["past_context_steps"]
    future_horizon_steps = config["future_horizon_steps"]
    loss_cfg = config["loss_function_params"]
    piml_params = config.get("piml_specific_params", {})
    use_amp = config.get("use_amp", True)
    enable_efficiency_func = (mode == "research")

    slope_norm_horizon = batch_X[:, past_context_steps:past_context_steps + future_horizon_steps, 0]
    slope_denorm_factor = (feature_maxs_tensor[0] - feature_mins_tensor[0]).clamp(min=1e-6)
    slope_horizon = slope_norm_horizon * slope_denorm_factor + feature_mins_tensor[0]
    vehicle_params_norm = batch_X[:, 0, 1:7]
    vp_denorm_factors = (feature_maxs_tensor[1:7] - feature_mins_tensor[1:7]).clamp(min=1e-6)
    vehicle_params_denorm = vehicle_params_norm * vp_denorm_factors + feature_mins_tensor[1:7]

    if past_context_steps > 0:
        initial_speed_norm = batch_X[:, past_context_steps - 1, 7]
    else:
        initial_speed_norm = batch_X[:, 0, 7]
    speed_denorm_factor = (feature_maxs_tensor[7] - feature_mins_tensor[7]).clamp(min=1e-6)
    initial_speed_for_sim = initial_speed_norm * speed_denorm_factor + feature_mins_tensor[7]
    dyn_ref_speed_norm_h = batch_X[:, past_context_steps:past_context_steps + future_horizon_steps, 7]
    dyn_ref_speed_horizon = dyn_ref_speed_norm_h * speed_denorm_factor + feature_mins_tensor[7]

    efficiency_func = calc_energy_efficiency_torch if enable_efficiency_func else None

    with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float32):
        (all_sim_speeds, step_sim_energy_kwh, step_friction_brake_E,
         rpm_low_viol, rpm_high_viol, _, all_raw_rpm, _, all_final_eng_tq) = simulate_horizon_soft(
            batch_torque=raw_torque_model_output, batch_brake_signal=brake_signal_model_output,
            batch_gear_probs=gear_probs_model_output_hard,
            gear_ratios=gear_ratios_tensor,
            gear_neg_limits=model.gear_neg_limits, max_pos_tq=model.max_pos_tq,
            initial_speed=initial_speed_for_sim, slopes=slope_horizon,
            vehicle_params=vehicle_params_denorm, engine_efficiency_func=efficiency_func,
            device=device, config=config,
            p_scheduled_sampling=p_scheduled_sampling, reference_speeds=dyn_ref_speed_horizon,
            track_gear_indices=(mode == "research")
        )

    _, actual_horizon_len = all_sim_speeds.shape

    is_descent = slope_horizon.mean(dim=1) < (-0.5 * math.pi / 180.0)
    v0 = initial_speed_for_sim
    v_final_target = dyn_ref_speed_horizon[:, -1]
    is_speed_above_ref = v0 > v_final_target
    need_brake = is_descent | is_speed_above_ref
    excess_neg_torque = torch.relu(-(raw_torque_model_output) - 0.3)
    bad_neg_torque = excess_neg_torque * (~need_brake).float()
    neg_torque_pen_loss = bad_neg_torque.mean() * 2.0

    if actual_horizon_len > 0:
        tgt_speed = dyn_ref_speed_horizon[:, :actual_horizon_len]
        n_terminal = loss_cfg.get("speed_loss_n_terminal", 10)
        non_terminal_len = max(0, actual_horizon_len - n_terminal)
        speed_loss_per_B = torch.zeros(B, device=device)
        if non_terminal_len > 0:
            err_non_term = all_sim_speeds[:, :non_terminal_len] - tgt_speed[:, :non_terminal_len]
            speed_w = torch.linspace(loss_cfg.get("speed_loss_weight_start", 0.2), loss_cfg.get("speed_loss_weight_end", 1.0), steps=non_terminal_len, device=device).unsqueeze(0)
            weighted_mse = ((err_non_term * speed_w) ** 2).mean(dim=1)
            speed_loss_per_B += weighted_mse
        if n_terminal > 0 and actual_horizon_len > non_terminal_len:
            err_term = all_sim_speeds[:, non_terminal_len:] - tgt_speed[:, non_terminal_len:]
            delta = loss_cfg.get("speed_loss_delta_terminal", 0.5)
            corridor = F.relu(err_term.abs() - delta)
            term_pen = (corridor ** 2).mean(dim=1)
            speed_loss_per_B += term_pen * loss_cfg.get("speed_loss_weight_end", 1.0)
        speed_loss = speed_loss_per_B.mean()
    else:
        speed_loss = torch.tensor(0.0, device=device)

    energy_loss = step_sim_energy_kwh.sum(dim=1).abs().mean()
    is_stalled = (all_sim_speeds < config.get("stall_threshold_speed", 0.5)).any(dim=1).float()
    stall_loss = is_stalled.mean() if B > 0 else torch.tensor(0.0, device=device)

    if B > 0:
        rpm_min_cfg = config.get("vehicle_physical_params", {}).get("rpm_min", 600.0)
        rpm_max_cfg = config.get("vehicle_physical_params", {}).get("rpm_max", 3500.0)
        rpm_deviation = torch.zeros_like(all_raw_rpm)
        rpm_deviation = torch.where(all_raw_rpm < rpm_min_cfg, rpm_min_cfg - all_raw_rpm, rpm_deviation)
        rpm_deviation = torch.where(all_raw_rpm > rpm_max_cfg, all_raw_rpm - rpm_max_cfg, rpm_deviation)
        rpm_loss_normalized = rpm_deviation / rpm_max_cfg
        rpm_loss = (rpm_loss_normalized ** 2).mean()
        rpm_loss = torch.clamp(rpm_loss, max=50.0)
    else:
        rpm_loss = torch.tensor(0.0, device=device)

    lug_rpm_thr = piml_params.get("lugging_rpm_threshold", 950.0)
    lug_tq_frac = piml_params.get("lugging_torque_threshold_fraction", 0.40)
    max_tq = config.get("vehicle_physical_params",{}).get("max_pos_tq", 2600.0)
    lug_tq_thr = lug_tq_frac * max_tq
    is_lugging = (all_raw_rpm < lug_rpm_thr) & (all_final_eng_tq > lug_tq_thr)
    lug_pen = torch.relu(lug_rpm_thr - all_raw_rpm) * all_final_eng_tq.relu() * is_lugging
    max_lugging_penalty = config.get("loss_function_params", {}).get("max_lugging_penalty", 500.0)
    clamped_lug_pen = torch.clamp(lug_pen, max=max_lugging_penalty)
    lugging_loss = torch.nan_to_num(clamped_lug_pen.mean())

    if gear_probs_model_output_hard.shape[1] > 1:
        shift_pen = (gear_probs_model_output_hard[:,1:,:] - gear_probs_model_output_hard[:,:-1,:]).abs().mean()
    else:
        shift_pen = torch.tensor(0.0, device=device)

    entropy_loss = -(gear_probs_model_output_soft * torch.log(gear_probs_model_output_soft + 1e-9)).sum(dim=-1).mean()

    if piml_params.get("use_gear_plausibility_penalty", False):
        speed_breaks = torch.tensor(piml_params.get("gear_plausibility_speed_breaks", []), device=device)
        sel_gear_idx = gear_probs_model_output_hard.argmax(dim=-1).float()
        max_gear = torch.searchsorted(speed_breaks, all_sim_speeds)
        plaus_viol = torch.relu(sel_gear_idx - max_gear.float())
        gear_plausibility_loss = plaus_viol.mean()
    else:
        gear_plausibility_loss = torch.tensor(0.0, device=device)

    if mode == "research":
        if actual_horizon_len > 0 and enable_efficiency_func:
            eng_params = config.get("engine_params", {})
            inst_eff = efficiency_func(all_raw_rpm, all_final_eng_tq, eng_params)
            eff_target = piml_params.get("target_efficiency_threshold", 0.30)
            eff_short = torch.relu(eff_target - inst_eff) * (all_final_eng_tq > 0.0)
            efficiency_loss = (eff_short**2).mean()
        else:
            efficiency_loss = torch.tensor(0.0, device=device)
        braking_loss = step_friction_brake_E.sum(dim=1).mean() if B > 0 else torch.tensor(0.0, device=device)
        brake_effort_loss = brake_signal_model_output.mean()
    else:
        efficiency_loss = torch.tensor(0.0, device=device)
        braking_loss = torch.tensor(0.0, device=device)
        brake_effort_loss = torch.tensor(0.0, device=device)

    speed_loss *= loss_cfg.get("speed_weight", 0.07)
    energy_loss *= loss_cfg.get("energy_weight", 0.15)

    loss_components = {
        "speed": speed_loss, "energy": energy_loss, "rpm": rpm_loss,
        "gear_shift": shift_pen, "lugging": lugging_loss,
        "neg_torque_pen": neg_torque_pen_loss, "stall": stall_loss,
        "gear_entropy": entropy_loss, "gear_plausibility": gear_plausibility_loss,
        "efficiency": efficiency_loss, "braking": braking_loss,
        "brake_effort": brake_effort_loss
    }

    return loss_components, all_sim_speeds

def plot_loss_components(history, plots_dir_path, training_mode, time_str):
    """
    Generates a plot showing the evolution of each individual validation loss
    component over the course of training.
    """
    plt.figure(figsize=(18, 10))
    all_tasks = history['val_comps'].keys()
    epochs = range(len(history['val_total']))
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_tasks)))
    color_map = {task: color for task, color in zip(all_tasks, colors)}

    for task in all_tasks:
        if task in history['val_comps'] and len(history['val_comps'][task]) > 0:
            plt.plot(epochs, history['val_comps'][task], label=f'Val {task}', color=color_map[task])

    plt.title(f"Validation Loss Components - {training_mode.title()} Mode", fontsize=16)
    plt.xlabel("Global Epoch", fontsize=12)
    plt.ylabel("Component Loss Value (Raw, Unscaled)", fontsize=12)
    plt.yscale('log')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plot_path = plots_dir_path / f"loss_components_{training_mode}_{time_str}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Loss components plot saved to {plot_path}")

def main():
    training_start_time = time.time()

    print("Loading configurable PIML configuration...", flush=True)
    config_path = "config/config.json"
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}", flush=True)
        return

    # ==============================================================================
    # START: ABLATION CONFIGURATION
    # ==============================================================================
    ablation_name = "arc-large"
    print(f"🔬 RUNNING ABLATION: {ablation_name.upper()}")

    # --- Configure Loss Terms ---
    piml_params_main = config.get("piml_specific_params", {})
    base_learnable_tasks = ["speed", "energy", "rpm", "gear_shift"]
    if config.get("training_mode", "research") == "research":
        research_tasks = ["efficiency", "braking", "brake_effort"]
        tasks_with_s_params = base_learnable_tasks + research_tasks
    else:
        tasks_with_s_params = base_learnable_tasks
    if piml_params_main.get("use_gear_plausibility_penalty", False):
        tasks_with_s_params.append("gear_plausibility")
    
    if ablation_name == 'l-se':
        tasks_with_s_params = ["speed", "energy"]
    elif ablation_name == 'l-se-rpm':
        tasks_with_s_params = ["speed", "energy", "rpm"]
    elif ablation_name == 'l-se-shift':
        tasks_with_s_params = ["speed", "energy", "gear_shift"]
    elif ablation_name == 'l-sers':
        tasks_with_s_params = ["speed", "energy", "rpm", "gear_shift"]

    # --- Configure Architecture ---
    d_model_abl, nhead_abl = config["d_model"], config["nhead"]
    enc_layers_abl, dec_layers_abl = config["num_encoder_layers"], config["num_decoder_layers"]
    
    if ablation_name == "arc-small":
        print("   -> Using SMALLER architecture: 2 encoder, 1 decoder, d_model=128")
        d_model_abl, nhead_abl, enc_layers_abl, dec_layers_abl = 128, 4, 2, 1
    elif ablation_name == "arc-large":
        print("   -> Using LARGER architecture: 6 encoder, 4 decoder, d_model=384")
        d_model_abl, nhead_abl, enc_layers_abl, dec_layers_abl = 384, 8, 6, 4
    # ==============================================================================
    # END: ABLATION CONFIGURATION
    # ==============================================================================

    training_mode = config.get("training_mode", "research")
    print(f"🎯 Training Mode: {training_mode.upper()}", flush=True)
    base_batch_for_lr = config.get("base_batch_for_lr", config["batch_size"])
    accum_steps = config.get("accum_steps", 1)
    effective_batch = config["batch_size"] * accum_steps
    scaled_lr = config["lr"] * (effective_batch / base_batch_for_lr)
    print(f"📏 Scaled LR = {scaled_lr:.6f}  (effective batch {effective_batch})", flush=True)
    set_seeds(config["seed"])
    device = get_device()
    print(f"✅ Running on device: {device}", flush=True)
    paths = config.get("paths", {})
    project_base_path = Path(paths.get("base_project_dir", "."))
    config_dir_path = project_base_path / paths.get("config_dir", "config")
    checkpoints_dir_path = project_base_path / paths.get("checkpoints_dir", "checkpoints")
    plots_dir_path = project_base_path / paths.get("plots_dir", "plots")
    for path in [config_dir_path, checkpoints_dir_path, plots_dir_path]:
        path.mkdir(parents=True, exist_ok=True)
    
    curriculum_stages = config.get("curriculum_stages", [])
    if not curriculum_stages:
        base_config_stage = copy.deepcopy(config)
        base_config_stage.update({"stage_name": "Single Stage", "epochs_in_stage": config["epochs"]})
        curriculum_stages = [base_config_stage]
    
    # --- Handle Curriculum Ablations ---
    if ablation_name == 'curr-no':
        print("   -> Running NO CURRICULUM ablation. Using final stage settings for all epochs.")
        total_epochs = sum(s.get("epochs_in_stage", config["epochs"]) for s in curriculum_stages)
        final_stage_params = curriculum_stages[-1]
        final_stage_params['epochs_in_stage'] = total_epochs
        curriculum_stages = [final_stage_params]
    elif ablation_name == 'curr-short':
        print("   -> Running SHORT CURRICULUM ablation. Removing 'Hill Start BOOTCAMP' stage.")
        curriculum_stages = [s for s in curriculum_stages if "Hill Start BOOTCAMP" not in s["stage_name"]]


    stats_samples = config.get("n_samples_for_stats", 2500)
    print(f"Generating normalization stats ({stats_samples} samples)...", flush=True)
    all_slopes_stats, all_ref_speeds_stats, vehicle_params_stats, start_speeds_stats, mission_targets_stats = generate_sequences_consistent(
        n_samples=stats_samples, track_length=config["track_length"], config_data=config)
    print("Creating windows for stats...", flush=True)
    all_track_data_stats = list(zip(all_slopes_stats, all_ref_speeds_stats, vehicle_params_stats, start_speeds_stats, mission_targets_stats))
    with multiprocessing.Pool(processes=config.get("num_track_generation_workers", 8)) as pool:
        worker_func_stats = partial(process_single_track_for_windows, config=config, generate_aux_data=False)
        results_stats = pool.map(worker_func_stats, all_track_data_stats)
    windows_X_raw_stats = [item for res in results_stats for item in res[0]]
    if not windows_X_raw_stats:
        print("ERROR: No windows created for stats.", flush=True); return
    print("Computing and saving feature statistics...", flush=True)
    feature_mins_np, feature_maxs_np = compute_minmax_stats(windows_X_raw_stats)
    torch.save(torch.tensor(feature_mins_np, dtype=torch.float32), config_dir_path / paths.get("feature_mins_filename", "feature_mins.pt"))
    torch.save(torch.tensor(feature_maxs_np, dtype=torch.float32), config_dir_path / paths.get("feature_maxs_filename", "feature_maxs.pt"))
    feature_mins_tensor = torch.tensor(feature_mins_np, dtype=torch.float32, device=device)
    feature_maxs_tensor = torch.tensor(feature_maxs_np, dtype=torch.float32, device=device)

    horizon_model = HorizonTransformer(
        num_gears=config["num_gears"], input_dim=len(feature_mins_np), 
        d_model=d_model_abl, nhead=nhead_abl,
        num_encoder_layers=enc_layers_abl, num_decoder_layers=dec_layers_abl,
        dropout=config["dropout"], future_horizon=config["future_horizon_steps"],
        past_context=config["past_context_steps"], model_architecture=config.get("model_architecture", "parallel")
    ).to(device)

    if hasattr(torch, 'compile') and config.get("use_torch_compile", True):
        try:
            horizon_model = torch.compile(horizon_model)
            print("Model compiled successfully.", flush=True)
        except Exception as e:
            print(f"torch.compile() failed: {e}. Proceeding without it.", flush=True)

    print(f"   - Using learnable task set: {tasks_with_s_params}")

    s_params = torch.nn.Parameter(torch.zeros(len(tasks_with_s_params), device=device))
    dwa_params = config.get("dwa_params", {})
    dwa_balancer = DwaBalancer(task_list=tasks_with_s_params, device=device, temperature=dwa_params.get("temperature", 2.0))

    optimizer = optim.AdamW([
        {'params': horizon_model.parameters(), 'lr': scaled_lr, 'weight_decay': 1e-4},
        {'params': [s_params], 'lr': scaled_lr * 2.0, 'weight_decay': 0.0}
    ])

    phys_params_main = config.get("vehicle_physical_params", {})
    gear_ratios_tensor_main = torch.tensor(phys_params_main.get("gear_ratios", []), dtype=torch.float32, device=device)
    scaler = torch.amp.GradScaler(enabled=config.get("use_amp", True))
    best_val_loss_overall = float('inf')
    global_epoch_count = 0
    ss_transition_epoch = None
    ss_freeze_duration = 3
    
    all_tasks_for_logging = [
        "speed", "energy", "rpm", "efficiency", "gear_shift", "braking", 
        "brake_effort", "gear_plausibility", "stall", "gear_entropy", "neg_torque_pen"
    ]
    history = {
        "train_total": [], "val_total": [],
        "train_comps": {t: [] for t in all_tasks_for_logging},
        "val_comps": {t: [] for t in all_tasks_for_logging},
        "dwa_weights": {t: [] for t in tasks_with_s_params}}

    past_sim_speeds_buffer = None

    print(f"🚀 Starting {training_mode} mode training with {len(curriculum_stages)} curriculum stages", flush=True)

    for stage_idx, stage_params in enumerate(curriculum_stages):
        stage_name = stage_params.get("stage_name", f"Stage {stage_idx + 1}")
        epochs_in_stage = stage_params.get("epochs_in_stage", config["epochs"])
        print(f"\n{'='*40}\n🚀 Starting: {stage_name}\n{'='*40}", flush=True)

        current_stage_config = copy.deepcopy(config)
        stage_data_overrides = stage_params.get("data_gen_params_override", {})
        if stage_data_overrides:
            current_stage_config["data_generation_params"].update(stage_data_overrides)

        stage_loss_overrides = stage_params.get("loss_function_params_override", {})
        if stage_loss_overrides:
            print(f"   -> Overriding loss function params for this stage.")
            current_stage_config["loss_function_params"].update(stage_loss_overrides)

        stage_piml_overrides = stage_params.get("piml_specific_params_override", {})
        if stage_piml_overrides:
            print(f"   -> Overriding PIML-specific params for this stage.")
            current_stage_config["piml_specific_params"].update(stage_piml_overrides)

        if "mass_range_training" in stage_params:
            print(f"   -> Overriding mass_range_training for this stage to: {stage_params['mass_range_training']}")
            current_stage_config["mass_range_training"] = stage_params["mass_range_training"]

        lr_override = stage_params.get("lr_override")

        n_samples_stage = stage_params.get("n_samples_per_stage", config["n_samples"])
        print(f"Generating data for {stage_name} ({n_samples_stage} samples)...", flush=True)
        all_slopes, all_ref_speeds, vehicle_params, start_speeds, mission_targets = generate_sequences_consistent(
            n_samples=n_samples_stage,
            track_length=current_stage_config["data_generation_params"].get("track_length", config["track_length"]),
            config_data=current_stage_config)
        all_track_data = list(zip(all_slopes, all_ref_speeds, vehicle_params, start_speeds, mission_targets))
        with multiprocessing.Pool(processes=config.get("num_track_generation_workers", 8)) as pool:
            worker_func = partial(process_single_track_for_windows, config=current_stage_config, generate_aux_data=False)
            results_stage = pool.map(worker_func, all_track_data)
        windows_X_raw = [item for res in results_stage for item in res[0]]
        if not windows_X_raw: print(f"ERROR: No windows for {stage_name}. Skipping.", flush=True); continue
        print(f"Created {len(windows_X_raw)} windows for {stage_name}.", flush=True)
        windows_X_norm = apply_minmax_normalization(windows_X_raw, feature_mins_np, feature_maxs_np)
        train_X, val_X = train_test_split(windows_X_norm, test_size=0.2, random_state=config["seed"])

        diagnostic_X = val_X[:config.get("batch_size", 256)]
        diagnostic_dataset = HorizonDataset(diagnostic_X)
        diagnostic_loader = DataLoader(diagnostic_dataset, batch_size=config["batch_size"], shuffle=False)
        persistent_diagnostic_batch = next(iter(diagnostic_loader))
        print(f"   -> Created a fixed diagnostic set with {len(diagnostic_X)} windows.")

        train_dataset = HorizonDataset(train_X)
        val_dataset = HorizonDataset(val_X)
        loader_params = {"batch_size": config["batch_size"], "num_workers": config.get("training_params", {}).get("num_dataloader_workers", 4), "pin_memory": True}
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_params)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_params)
        
        # --- Scheduler Selection ---
        if ablation_name == "lr-onecycle-off":
            print("   -> Using simpler CosineAnnealingLR scheduler.")
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs_in_stage * len(train_loader))
        else: # Default to OneCycleLR
            target_lr_for_stage = lr_override if lr_override is not None else scaled_lr
            one_cycle_params = config.get("one_cycle_lr_params", {})
            max_lr_for_cycle = target_lr_for_stage * one_cycle_params.get("max_lr_override_factor", 1.0)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lr_for_cycle,
                epochs=epochs_in_stage,
                steps_per_epoch=len(train_loader),
                pct_start=one_cycle_params.get("pct_start", 0.3),
                anneal_strategy=one_cycle_params.get("anneal_strategy", "cos"),
                div_factor=one_cycle_params.get("div_factor", 25.0),
                final_div_factor=one_cycle_params.get("final_div_factor", 1e4)
            )

        stage_best_val_loss = float('inf')
        patience_counter = 0
        max_patience_epochs = config.get("max_patience", 15)
        total_epochs_for_calc = sum(s.get("epochs_in_stage", config["epochs"]) for s in curriculum_stages)
        training_params = config.get("training_params", {})
        hard_gumbel_start_epoch = int(training_params.get("hard_gumbel_start_epoch_fraction", 0.8) * total_epochs_for_calc)
        s_params_warmup_epochs = training_params.get("s_params_warmup_epochs", 3)

        for epoch_in_stage in range(epochs_in_stage):
            current_global_epoch = global_epoch_count
            tau_params = config.get("training_params", {})
            tau = get_tau(current_global_epoch, total_epochs_for_calc, tau_params.get("tau_start", 5.0), tau_params.get("tau_end", 0.5), tau_params.get("tau_decay_rate", 0.98))
            use_hard_gumbel_training = (current_global_epoch >= hard_gumbel_start_epoch)
            is_initial_warmup = current_global_epoch < s_params_warmup_epochs

            p_sched_samp = get_scheduled_sampling_p(
                epoch_in_stage=epoch_in_stage,
                total_epochs_in_stage=epochs_in_stage,
                stage_idx=stage_idx,
                num_stages=len(curriculum_stages)
            )

            is_ss_transition_freeze = False
            if ss_transition_epoch is None and p_sched_samp > 0:
                ss_transition_epoch = current_global_epoch
                print(f"   -> 🔒 Scheduled Sampling activated! Freezing s_params for {ss_freeze_duration} epochs.")

            if ss_transition_epoch is not None:
                epochs_since_ss_start = current_global_epoch - ss_transition_epoch
                is_ss_transition_freeze = epochs_since_ss_start < ss_freeze_duration
                if epochs_since_ss_start == ss_freeze_duration:
                    print(f"   -> 🔓 Unfreezing s_params after SS transition period.")

            should_freeze_s_params = is_initial_warmup or is_ss_transition_freeze
            s_params.requires_grad_(not should_freeze_s_params)

            # --- DWA Ablation Logic ---
            dwa_weights = dwa_balancer.compute_weights(max_weight_cap=dwa_params.get("max_weight", 3.0))
            if ablation_name == "dwa-off":
                dwa_weights = {task: 1.0 for task in tasks_with_s_params}


            if is_initial_warmup:
                warmup_status = "WARM-UP (s_params frozen)"
            elif is_ss_transition_freeze:
                epochs_left = ss_freeze_duration - (current_global_epoch - ss_transition_epoch)
                warmup_status = f"SS-TRANSITION (s_params frozen, {epochs_left} left)"
            else:
                warmup_status = "s_params active"
            print(f"Stage '{stage_name}' - Epoch {epoch_in_stage+1}/{epochs_in_stage} (Global {current_global_epoch+1}) - Tau: {tau:.4f} - Hard G-S: {use_hard_gumbel_training} - SS p={p_sched_samp:.3f} - {warmup_status}", flush=True)
            horizon_model.train()
            epoch_train_loss = 0.0
            num_train_batches = 0
            epoch_train_comps = {task: 0.0 for task in all_tasks_for_logging}

            for batch_x in train_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                B, L, F = batch_x.shape
                past_context_steps = config["past_context_steps"]
                num_train_batches += 1

                batch_x_for_model = batch_x.clone()
                if p_sched_samp > 0 and past_sim_speeds_buffer is not None and past_sim_speeds_buffer.shape[0] == B:
                    ground_truth_past_speeds = batch_x_for_model[:, :past_context_steps, 7]
                    mask = (torch.rand(B, 1, device=device) < p_sched_samp).expand_as(ground_truth_past_speeds)

                    speed_denorm_factor = (feature_maxs_tensor[7] - feature_mins_tensor[7]).clamp(min=1e-6)
                    sim_speeds_normalized_buffer = (past_sim_speeds_buffer - feature_mins_tensor[7]) / speed_denorm_factor

                    corrupted_past_speeds = torch.where(mask, sim_speeds_normalized_buffer[:, :past_context_steps], ground_truth_past_speeds)
                    batch_x_for_model[:, :past_context_steps, 7] = corrupted_past_speeds.detach()

                with torch.amp.autocast('cuda', enabled=config.get("use_amp", True)):
                    raw_torque, gear_probs_soft, gear_probs_hard, brake_signal = horizon_model(batch_x_for_model, tau=tau)

                    loss_dict, all_sim_speeds = loss_function_horizon_configurable(
                        batch_X=batch_x, model=horizon_model,
                        raw_torque_model_output=raw_torque, gear_probs_model_output_soft=gear_probs_soft, gear_probs_model_output_hard=gear_probs_hard,
                        brake_signal_model_output=brake_signal, config=current_stage_config,
                        feature_mins_tensor=feature_mins_tensor, feature_maxs_tensor=feature_maxs_tensor,
                        gear_ratios_tensor=gear_ratios_tensor_main, tau=tau,
                        hard=use_hard_gumbel_training, mode=training_mode,
                        p_scheduled_sampling=p_sched_samp,
                        ablation_name=ablation_name)

                    past_sim_speeds_buffer = all_sim_speeds.detach()

                    for task_name, loss_val in loss_dict.items():
                        if task_name in epoch_train_comps: epoch_train_comps[task_name] += loss_val.item()

                    total_loss = torch.tensor(0.0, device=device)
                    loss_cfg = current_stage_config["loss_function_params"]

                    for i, task_name in enumerate(tasks_with_s_params):
                        raw_loss_component = loss_dict.get(task_name, torch.tensor(0.0, device=device))
                        raw_loss_component = torch.nan_to_num(raw_loss_component)
                        stable_loss_component = raw_loss_component + 1e-8

                        if not torch.isfinite(raw_loss_component):
                             print(f"Warning: Non-finite raw loss for task '{task_name}' detected. Skipping.")
                             continue

                        if task_name == "rpm":
                            scaled_loss = stable_loss_component * loss_cfg.get("gamma_rpm", 1.0)
                        else:
                            scaled_loss = stable_loss_component

                        uncertainty_loss = 0.5 * (scaled_loss * torch.exp(-s_params[i]) + s_params[i])
                        final_task_loss = dwa_weights[task_name] * uncertainty_loss
                        total_loss += final_task_loss

                    if not ablation_name.startswith('l-'):
                        stall_loss_component = loss_dict.get('stall', 0.0) * loss_cfg.get("stall_penalty_weight", 10.0)
                        total_loss += stall_loss_component
                        entropy_loss_component = loss_dict.get('gear_entropy', 0.0) * loss_cfg.get("gear_entropy_weight", 0.02)
                        total_loss += entropy_loss_component
                        neg_torque_loss_component = loss_dict.get('neg_torque_pen', 0.0)
                        total_loss += neg_torque_loss_component

                if not torch.isfinite(total_loss):
                    print("ERROR: Total loss became non-finite. Skipping optimizer step.")
                    continue

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(total_loss).backward()
                
                # --- CORRECTED OPTIMIZER/SCALER/SCHEDULER PATTERN ---
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(horizon_model.parameters(), config.get("gradient_clip_value", 1.0))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                # --- END OF FIX ---

                with torch.no_grad():
                    piml_params = current_stage_config.get("piml_specific_params", {})
                    log_sigma2_min = piml_params.get("dwa_log_sigma2_min", -2.0)
                    log_sigma2_max = piml_params.get("dwa_log_sigma2_max", 2.0)
                    s_params.clamp_(log_sigma2_min, log_sigma2_max)
                epoch_train_loss += total_loss.item()

            avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0.0
            history["train_total"].append(avg_train_loss)
            for task_name in all_tasks_for_logging:
                if task_name in epoch_train_comps:
                    history["train_comps"][task_name].append(epoch_train_comps[task_name] / num_train_batches if num_train_batches > 0 else 0.0)

            horizon_model.eval()
            epoch_val_loss = 0.0
            num_val_batches = 0
            epoch_val_comps = {task: 0.0 for task in all_tasks_for_logging}
            with torch.no_grad():
                for batch_x_val in val_loader:
                    batch_x_val = batch_x_val.to(device, non_blocking=True)
                    num_val_batches += 1
                    raw_torque_val, gear_probs_soft_val, gear_probs_hard_val, brake_signal_val = horizon_model(batch_x_val, tau=config.get("training_params",{}).get("tau_end", 0.5))

                    loss_dict_val, _ = loss_function_horizon_configurable(
                        batch_X=batch_x_val, model=horizon_model,
                        raw_torque_model_output=raw_torque_val,
                        gear_probs_model_output_soft=gear_probs_soft_val,
                        gear_probs_model_output_hard=gear_probs_hard_val,
                        brake_signal_model_output=brake_signal_val,
                        config=current_stage_config,
                        feature_mins_tensor=feature_mins_tensor,
                        feature_maxs_tensor=feature_maxs_tensor,
                        gear_ratios_tensor=gear_ratios_tensor_main,
                        tau=config.get("training_params",{}).get("tau_end", 0.5),
                        hard=True,
                        mode=training_mode,
                        p_scheduled_sampling=0.0,
                        ablation_name=ablation_name
                    )
                    for task_name, loss_val in loss_dict_val.items():
                        if task_name in epoch_val_comps: epoch_val_comps[task_name] += loss_val.item()

                    total_val_loss = torch.tensor(0.0, device=device)
                    loss_cfg = current_stage_config["loss_function_params"]

                    for i, task_name in enumerate(tasks_with_s_params):
                        current_loss_component_val = loss_dict_val.get(task_name, torch.tensor(0.0, device=device))
                        current_loss_component_val = torch.nan_to_num(current_loss_component_val)
                        stable_loss_component_val = current_loss_component_val + 1e-8

                        if not torch.isfinite(current_loss_component_val): continue

                        if task_name == "rpm":
                            scaled_loss = stable_loss_component_val * loss_cfg.get("gamma_rpm", 1.0)
                        else:
                            scaled_loss = stable_loss_component_val

                        uncertainty_loss = 0.5 * (scaled_loss * torch.exp(-s_params[i]) + s_params[i])
                        final_task_loss = dwa_weights[task_name] * uncertainty_loss
                        total_val_loss += final_task_loss
                    
                    if not ablation_name.startswith('l-'):
                        stall_loss_val = loss_dict_val.get('stall', 0.0) * loss_cfg.get("stall_penalty_weight", 10.0)
                        total_val_loss += stall_loss_val
                        entropy_loss_val = loss_dict_val.get('gear_entropy', 0.0) * loss_cfg.get("gear_entropy_weight", 0.02)
                        total_val_loss += entropy_loss_val
                        neg_torque_loss_val = loss_dict_val.get('neg_torque_pen', 0.0)
                        total_val_loss += neg_torque_loss_val

                    if torch.isfinite(total_val_loss): epoch_val_loss += total_val_loss.item()

            avg_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else float('inf')

            history["val_total"].append(avg_val_loss)

            avg_val_comps = {}
            for task_name in all_tasks_for_logging:
                if task_name in epoch_val_comps:
                    avg_val_comp = epoch_val_comps[task_name] / num_val_batches if num_val_batches > 0 else 0.0
                    history["val_comps"][task_name].append(avg_val_comp)
                    avg_val_comps[task_name] = avg_val_comp
            dwa_balancer.update_losses(avg_val_comps)
            print(f"  [Epoch {current_global_epoch+1}] Train Total Loss: {avg_train_loss:.4f}, Val Total Loss: {avg_val_loss:.4f}")
            val_comps_str = ", ".join([f"{t}={history['val_comps'][t][-1]:.4f}" for t in all_tasks_for_logging if t in history['val_comps'] and history['val_comps'][t]])
            print(f"    Val (Raw, Unscaled): {val_comps_str}")
            s_params_cpu = s_params.data.cpu().numpy()
            dwa_weights_cpu = {t: w for t, w in dwa_weights.items()}
            print(f"    s_params (log σ^2): " + ", ".join([f"{t}={s:.4f}" for t, s in zip(tasks_with_s_params, s_params_cpu)]))
            print(f"    DWA weights:       " + ", ".join([f"{t}={w:.4f}" for t, w in dwa_weights_cpu.items()]))
            print(f"    Current LR:        {optimizer.param_groups[0]['lr']:.6f}")

            diagnostic_frequency = config.get("diagnostic_frequency", 5)
            is_last_epoch = (epoch_in_stage == epochs_in_stage - 1)

            if (epoch_in_stage + 1) % diagnostic_frequency == 0 or is_last_epoch:
                epoch_str = f"Stage '{stage_name}', Epoch {epoch_in_stage+1}"
                run_diagnostic_callback(
                    model=horizon_model,
                    diagnostic_batch=persistent_diagnostic_batch,
                    config=current_stage_config,
                    feature_mins_tensor=feature_mins_tensor,
                    feature_maxs_tensor=feature_maxs_tensor,
                    gear_ratios_tensor=gear_ratios_tensor_main,
                    device=device,
                    epoch_info_str=epoch_str
                )

            model_suffix = f"_{ablation_name}" if ablation_name != "l-base" else ""
            if not is_initial_warmup and avg_val_loss < best_val_loss_overall:
                best_val_loss_overall = avg_val_loss
                torch.save(horizon_model.state_dict(), checkpoints_dir_path / f"best_model_overall{model_suffix}.pth")
                print(f"    🏆 New OVERALL best validation loss: {best_val_loss_overall:.4f}. Overall best model saved.")
            if not is_initial_warmup and avg_val_loss < stage_best_val_loss:
                stage_best_val_loss = avg_val_loss
                safe_stage_name = "".join(c for c in stage_name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
                torch.save(horizon_model.state_dict(), checkpoints_dir_path / f"best_model_stage_{stage_idx+1}_{safe_stage_name}{model_suffix}.pth")
                print(f"    🏅 New STAGE '{stage_name}' best validation loss: {stage_best_val_loss:.4f}. Stage best model saved.")
                patience_counter = 0
            else:
                if not is_initial_warmup:
                    patience_counter += 1
                    if patience_counter >= max_patience_epochs:
                        print(f"  🛑 Early stopping STAGE after {patience_counter} epochs without improvement.")
                        break
            global_epoch_count += 1

    print(f"\n🏁 {training_mode.title()} training completed!", flush=True)

    time_str = time.strftime('%Y%m%d_%H%M%S')
    plt.figure(figsize=(15, 5))
    plt.plot(history['train_total'], label='Train Loss')
    plt.plot(history['val_total'], label='Validation Loss')
    plt.title(f"Training Progress - {training_mode.title()} Mode")
    plt.xlabel("Global Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plot_path = plots_dir_path / f"training_{training_mode}_{time_str}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📉 Training plot saved to {plot_path}")

    plot_loss_components(history, plots_dir_path, training_mode, time_str)

    final_model_path = checkpoints_dir_path / f"final_model_{training_mode}_{time_str}.pth"
    torch.save(horizon_model.state_dict(), final_model_path)
    print(f"💾 Final model saved to {final_model_path}")

    training_end_time = time.time()
    total_training_duration_s = training_end_time - training_start_time
    total_training_duration_h = total_training_duration_s / 3600
    print(f"\n\n==================================================")
    print(f"TOTAL WALL-CLOCK TRAINING TIME: {total_training_duration_s:.2f} seconds ({total_training_duration_h:.2f} hours)")
    print(f"==================================================")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

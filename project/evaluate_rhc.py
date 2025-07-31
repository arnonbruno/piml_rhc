# evaluate_rhc.py (Final Fairness Patch Applied)
import os
import json
import argparse
import time
from pathlib import Path
import copy
import warnings
import hashlib
import torch.multiprocessing as mp
import platform

# --- Core Scientific Libraries ---
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

# --- Plotting Libraries ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Computational Metrics Libraries ---
import psutil
from fvcore.nn import FlopCountAnalysis

# --- Statistical Analysis Libraries ---
from scipy.stats import shapiro, friedmanchisquare
import scikit_posthocs as sp
from analysis import run_statistical_analysis, plot_aggregate_results
import pingouin as pg
from statsmodels.stats.multitest import multipletests

# --- Import project-specific modules ---
from models import HorizonTransformer
from utils import (
    set_seeds,
    generate_sequences_consistent,
    simulate_one_step_soft,
    calc_energy_efficiency_torch
)
from controllers.p_controller import P_Controller
from controllers.pid_controller import PID_Controller
from controllers.mpc_controller import MPC_Controller_HeuristicGear
from controllers.piml_rhc_controller import PIML_RHC_Controller
from controllers.dp_controller import DP_Controller
from controllers.memoization_controller import MemoizationController

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid")

#setting all seeds to 42
set_seeds(42)


# ==============================================================================
# 1. SIMULATION CORE
# ==============================================================================

def run_simulation(controller, track_data, config, device, args, feature_mins=None, feature_maxs=None):
    if hasattr(controller, 'reset'):
        controller.reset()

    track_len = len(track_data['slopes'])
    past_context = config["past_context_steps"]
    horizon = getattr(controller, 'horizon', config["future_horizon_steps"])
    num_features = len(feature_mins) if feature_mins is not None else 9

    phys_params = config["vehicle_physical_params"]
    vparams_dict = track_data['vehicle_params']
    gear_ratios_tensor = torch.tensor(phys_params["gear_ratios"], device=device, dtype=torch.float32)
    gear_neg_limits_tensor = torch.tensor(phys_params["gear_neg_limits"], device=device, dtype=torch.float32)
    max_pos_tq = float(phys_params["max_pos_tq"])
    vehicle_mass_tensor = torch.tensor([vparams_dict['mass']], device=device, dtype=torch.float32)
    wheel_radius_tensor = torch.tensor([vparams_dict['wheel_radius']], device=device, dtype=torch.float32)
    rpm_min_val, rpm_max_val = phys_params["rpm_min"], config.get("rpm_max_sim", 3500.0)
    param_array = np.array([vparams_dict['wheel_radius'], vparams_dict['mass'], vparams_dict['rho'], vparams_dict['Cd'], vparams_dict['A'], vparams_dict['Crr']])

    current_speed = track_data['initial_speed']
    current_rpm = 0.0
    current_gear = config.get("ref_start_gear_idx", 2)
    
    history_window = np.zeros((past_context, num_features))
    for i in range(past_context):
        history_window[i, 0] = track_data['slopes'][0]
        history_window[i, 1:7] = param_array
        history_window[i, 7] = current_speed
        history_window[i, 8] = track_data['mission_target']

    results = []
    process = psutil.Process(os.getpid())
    CPU_POWER_W, GPU_POWER_W = 65, 350

    for t in range(track_len):
        # By default, the target is the time-varying profile
        target_speed_step = track_data['mission_target_v_profile'][t]
        ## for testing purposes, use mission target as target speed
        #target_speed_step = track_data['mission_target']
        
        # FINAL FAIRNESS FIX: If the controller is PIML-RHC, its target for scoring
        # must be the scalar mission target it was trained to follow.
        #if isinstance(controller, PIML_RHC_Controller):
        #    target_speed_step = track_data['mission_target']

        future_slopes = np.array([track_data['slopes'][min(t + i, track_len - 1)] for i in range(horizon)])

        start_time = time.perf_counter()
        if isinstance(controller, PIML_RHC_Controller):
            future_references = np.zeros((horizon, num_features))
            ref_speeds_horizon = [track_data['mission_target_v_profile'][min(t + i, track_len - 1)] for i in range(horizon)]
            for i in range(horizon):
                future_references[i, 0] = future_slopes[i]
                future_references[i, 1:7] = param_array
                #future_references[i, 7] = current_speed
                future_references[i, 7] = ref_speeds_horizon[i]
                future_references[i, 8] = track_data['mission_target']
            torque_cmd, gear_cmd, brake_cmd = controller.get_action(history_window, future_references)
        else:
            # Other controllers use the time-varying profile as their target
            torque_cmd, gear_cmd, brake_cmd = controller.get_action(current_speed, track_data['mission_target_v_profile'][t], current_rpm, current_gear, future_slopes)
        end_time = time.perf_counter()
        
        wall_time_s = end_time - start_time
        peak_mem_bytes = process.memory_info().rss
        compute_joules = wall_time_s * CPU_POWER_W

        current_gear = int(np.round(gear_cmd))
        torque_tensor = torch.tensor([[torque_cmd]], device=device)
        brake_tensor = torch.tensor([[brake_cmd]], device=device)
        gear_probs_tensor = F.one_hot(torch.tensor([current_gear]), num_classes=len(gear_ratios_tensor)).float().to(device)
        speed_tensor = torch.tensor([current_speed], device=device)
        slope_rad_t = torch.tensor(track_data['slopes'][t], device=device)

        with torch.no_grad():
            (next_speeds_sim, eng_energy_kwh_dt, _, _, _, _, raw_rpm_last_substep_dt, final_engine_tq_dt) = simulate_one_step_soft(
                engine_torque=torque_tensor, brake_signal=brake_tensor, gear_probs=gear_probs_tensor,
                gear_ratios=gear_ratios_tensor, gear_neg_limits=gear_neg_limits_tensor, max_pos_tq=max_pos_tq,
                current_speed=speed_tensor, wheel_radius=wheel_radius_tensor, vehicle_mass=vehicle_mass_tensor,
                ext_slope_force=(vehicle_mass_tensor * 9.81 * torch.sin(slope_rad_t)),
                ext_air_drag=(0.5 * vparams_dict['rho'] * vparams_dict['Cd'] * vparams_dict['A'] * (speed_tensor**2)),
                ext_rolling_res=(vehicle_mass_tensor * 9.81 * vparams_dict['Crr'] * torch.cos(slope_rad_t)),
                engine_efficiency_func=calc_energy_efficiency_torch,
                device=device, config=config)

        next_speed, next_rpm = next_speeds_sim.item(), raw_rpm_last_substep_dt.item()
        applied_torque, energy_kwh = final_engine_tq_dt.item(), eng_energy_kwh_dt.item()
        
        results.append({
            'time': t, 'speed': current_speed, 'target_speed': target_speed_step, 'rpm': current_rpm,
            'gear': current_gear, 'applied_torque': applied_torque, 'energy_kwh': energy_kwh,
            'wall_time_ms': wall_time_s * 1000, 'peak_memory_mb': peak_mem_bytes / (1024 * 1024),
            'compute_joules': compute_joules
        })

        current_speed, current_rpm = next_speed, next_rpm
        if isinstance(controller, PIML_RHC_Controller):
            new_history_row = np.zeros(num_features)
            new_history_row[0] = track_data['slopes'][t]
            new_history_row[1:7] = param_array
            new_history_row[7] = current_speed
            new_history_row[8] = track_data['mission_target']
            history_window = np.roll(history_window, -1, axis=0)
            history_window[-1, :] = new_history_row

    return pd.DataFrame(results)

# ==============================================================================
# 2. EVALUATION WORKER & MAIN LOOP
# ==============================================================================

def evaluate_track(task_args):
    track_idx, scenario_key, track_data, config, args, feature_mins, feature_maxs = task_args
    device = torch.device("cpu")
    torch.set_num_threads(4)
    
    print(f"[Worker {os.getpid()}] Processing Track #{track_idx} ({scenario_key})...", flush=True)

    controllers = {
        "P-Control": P_Controller(config, args=args),
        "PID": PID_Controller(config, args=args),
        "MPC-Heuristic": MPC_Controller_HeuristicGear(config, device, track_data, args=args),
        "Memoization": MemoizationController(config, track_data, args=args),
        "PIML-RHC": PIML_RHC_Controller(args.model_path, config, feature_mins, feature_maxs, device, args=args),
        "DP-Oracle": DP_Controller(config, track_data)
    }

    track_results = []
    for name, controller in controllers.items():
        try:
            results_df = run_simulation(controller, track_data, config, device, args, feature_mins, feature_maxs)
            
            wall_time = controller.solve_time * 1000 if hasattr(controller, 'solve_time') else results_df['wall_time_ms'].mean()
            
            sim_compute_joules = results_df['compute_joules'].sum()
            if name == "DP-Oracle":
                planning_joules = controller.solve_time * 65
                total_compute_kj = (sim_compute_joules + planning_joules) / 1000
            else:
                total_compute_kj = sim_compute_joules / 1000

            summary = {
                "Track Index": track_idx, "Scenario": scenario_key, "Controller": name,
                "Total Energy (kWh)": results_df['energy_kwh'].sum(),
                "Speed RMSE (m/s)": np.sqrt(np.mean((results_df['speed'] - results_df['target_speed'])**2)),
                "Torque Smoothness": results_df['applied_torque'].diff().abs().mean(),
                "Total Gear Shifts": results_df['gear'].diff().ne(0).sum(),
                "Planning Time (ms)": wall_time,
                "Peak Memory (MB)": results_df['peak_memory_mb'].mean(),
                "Compute Energy (kJ)": total_compute_kj
            }
            track_results.append(summary)
        except Exception as e:
            print(f"  ERROR: Controller '{name}' failed on track {track_idx}. Error: {e}")
            import traceback
            traceback.print_exc()
            
    return track_results

# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================

def main(args):
    if platform.system() != "Windows" and mp.get_start_method(allow_none=True) != "forkserver":
        mp.set_start_method("forkserver", force=True)

    with open(args.config_path, "r") as f:
        config = json.load(f)

    plots_dir = Path(config["paths"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(config["seed"])

    config_dir = Path(config["paths"]["config_dir"])
    feature_mins = torch.load(config_dir / config["paths"]["feature_mins_filename"]).numpy()
    feature_maxs = torch.load(config_dir / config["paths"]["feature_maxs_filename"]).numpy()

    all_tasks = []
    track_counter = 0
    scenarios_to_test = {
        "Stage1_Gentle": "Stage 1: Gentle Introduction",
        "Stage2_StopGoHills": "Stage 2: Stop, Go, and Hills",
        "Stage3_Complex": "Stage 3: Full Complexity"
    }

    for scenario_key, stage_name in scenarios_to_test.items():
        print(f"\nGenerating data for Scenario: {stage_name}...")
        stage_config = next((s for s in config["curriculum_stages"] if s["stage_name"] == stage_name), {})
        eval_stage_config = copy.deepcopy(config)
        eval_stage_config["data_generation_params"].update(stage_config.get("data_gen_params_override", {}))
        
        track_len_eval = args.track_len or config["evaluation_rhc_params"].get("rhc_eval_track_len", 100)
        
        slopes, ref_speeds, vparams, starts, missions = generate_sequences_consistent(
            n_samples=args.num_tracks, track_length=track_len_eval, config_data=eval_stage_config
        )
        
        for s, ref, v, st, m in zip(slopes, ref_speeds, vparams, starts, missions):
            track_data = {'slopes': s, 'vehicle_params': v, 'initial_speed': st, 
                          'mission_target': m, 'mission_target_v_profile': ref}
            all_tasks.append((track_counter, scenario_key, track_data, config, args, feature_mins, feature_maxs))
            track_counter += 1

    print(f"\n--- Starting Evaluation on {len(all_tasks)} tracks using {args.workers} workers ---")
    all_results = []
    with mp.Pool(processes=args.workers) as pool:
        for result_chunk in tqdm(pool.imap_unordered(evaluate_track, all_tasks), total=len(all_tasks)):
            all_results.extend(result_chunk)

    print("\n--- Evaluation Complete ---")
    if not all_results:
        print("No results were generated. Exiting.")
        return

    summary_df = pd.DataFrame(all_results)
    
    long_form_path = plots_dir / "evaluation_results_long_form.csv"
    summary_df.to_csv(long_form_path, index=False)
    print(f"\n💾 Long-form results saved to {long_form_path}")

    display_cols = ['Total Energy (kWh)', 'Speed RMSE (m/s)', 'Torque Smoothness', 
                    'Total Gear Shifts', 'Planning Time (ms)', 'Peak Memory (MB)', 'Compute Energy (kJ)']
    
    summary_df.rename(columns={"Wall Time (ms)": "Planning Time (ms)"}, inplace=True, errors='ignore')
    
    summary_grouped = summary_df.groupby(['Scenario', 'Controller'])[display_cols]
    
    summary_mean = summary_grouped.mean()
    summary_std = summary_grouped.std().fillna(0)
    
    summary_display = summary_mean.round(2).astype(str) + " ± " + summary_std.round(2).astype(str)
    
    print("\n--- Aggregate Results Summary (Mean ± Std Dev) ---\n")
    print(summary_display.to_string())
    
    csv_path = plots_dir / "evaluation_summary_mean_std.csv"
    summary_display.to_csv(csv_path)
    print(f"\nDetailed summary table saved to {csv_path}")
    
    print("\n--- Aggregate Results Summary (Median ± MAD) ---\n")
    summary_median = summary_grouped.median()
    # MAD (Median Absolute Deviation) requires a custom aggregation function
    from scipy.stats import median_abs_deviation
    summary_mad = summary_grouped.agg(lambda x: median_abs_deviation(x, nan_policy='omit'))
    
    summary_display_median = summary_median.round(2).astype(str) + " ± " + summary_mad.round(2).astype(str)
    print(summary_display_median.to_string())
    
    median_csv_path = plots_dir / "evaluation_summary_median_mad.csv"
    summary_display_median.to_csv(median_csv_path)
    print(f"\nDetailed median summary table saved to {median_csv_path}")

    plot_aggregate_results(summary_df, config)
    if args.run_stats:
        run_statistical_analysis(summary_df, config)

    print("\n✅ Evaluation script finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust and Fair RHC Evaluation Suite.")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model_overall.pth", help="Path to the trained PIML-RHC model.")
    parser.add_argument("--config_path", type=str, default="config/config.json", help="Path to the configuration JSON file.")
    parser.add_argument("--num_tracks", type=int, default=10, help="Number of test tracks per scenario.")
    parser.add_argument("--track_len", type=int, default=None, help="Override track length for this run.")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2), help="Number of parallel processes for evaluation.")
    parser.add_argument("--run_stats", action="store_true", help="Run the full statistical analysis suite on the results.")
    
    parser.add_argument("--mpc_horizon_dev", type=int, default=None, help="DEV: Override MPC horizon for quick tests.")
    parser.add_argument("--mpc_optim_steps_dev", type=int, default=None, help="DEV: Override MPC optimization steps for quick tests.")
    
    args = parser.parse_args()
    main(args)

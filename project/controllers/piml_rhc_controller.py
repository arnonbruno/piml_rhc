import torch
import numpy as np
from models import HorizonTransformer
from utils import apply_minmax_normalization
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class PIML_RHC_Controller:
    """Wraps the trained HorizonTransformer model for Receding Horizon Control."""
    def __init__(self, model_path, config, feature_mins, feature_maxs, device, **kwargs):
        self.config = config
        self.device = device
        self.feature_mins = feature_mins
        self.feature_maxs = feature_maxs
        self.past_context = config["past_context_steps"]
        self.horizon = config["future_horizon_steps"]
        
        self.model = HorizonTransformer(
            num_gears=config["num_gears"], input_dim=len(feature_mins), d_model=config["d_model"],
            nhead=config["nhead"], num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            dropout=0.0, future_horizon=self.horizon,
            past_context=self.past_context
        ).to(device)

        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_state_dict)
        self.model.eval()

    @torch.no_grad()
    def get_action(self, history_window, future_references):
        model_input_raw = np.vstack([history_window, future_references])
        model_input_norm = apply_minmax_normalization([model_input_raw], self.feature_mins, self.feature_maxs)[0]
        input_tensor = torch.tensor(model_input_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        raw_torque, _, gear_probs_hard, brake_signal = self.model(input_tensor, tau=0.1)
        
        # FIX: Clip outputs to their valid physical ranges for robustness
        torque_action = float(np.clip(raw_torque[0, 0, 0].item(), -1.0, 1.0))
        gear_action = torch.argmax(gear_probs_hard[0, 0, :]).item()
        brake_action = float(np.clip(brake_signal[0, 0, 0].item(), 0.0, 1.0))
        
        return torque_action, gear_action, brake_action

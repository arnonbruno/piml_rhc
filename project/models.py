import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# loading model config
# This path should be correct for your environment
try:
    with open('/home/bruno/Documents/light-of-aman/Thesis/Python/project/config/config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("Warning: Could not load config.json in models.py. Using default values.")
    config = {}


############################################################
### 1) Gumbel-Softmax
############################################################
def gumbel_softmax(logits: torch.Tensor,
                          tau: float = 1.0,
                          hard: bool = False,
                          eps: float = 1e-8) -> torch.Tensor:
    """
    Identical interface to the original but:
    • All math is done in float32 to prevent Inf/NaN in half precision.
    • Logits are clamped to a sensible range before exp/softmax.
    • Returns a tensor in the *original* dtype so it plugs in transparently.
    """
    dtype_orig = logits.dtype
    logits_fp32 = logits.float()
    logits_fp32 = torch.clamp(logits_fp32, -40.0, 40.0)

    gumbel_noise = -torch.log(-torch.log(
        torch.rand_like(logits_fp32) + eps) + eps)

    y = F.softmax((logits_fp32 + gumbel_noise) / max(tau, 1e-4), dim=-1)

    if hard:
        index = y.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y).scatter_(-1, index, 1.0)
        y = (y_hard - y).detach() + y

    return y.to(dtype_orig)

############################################################
### 2) Positional Encoding
############################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].squeeze(1)
        return self.dropout(x)

############################################################
### 3) HorizonTransformer
############################################################
class HorizonTransformer(nn.Module):
    def __init__(
        self,
        num_gears=15,
        input_dim=9,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        future_horizon=60,
        past_context=10,
        model_architecture="parallel"
    ):
        super().__init__()
        self.num_gears = num_gears
        self.future_horizon = future_horizon
        self.past_context = past_context
        self.d_model = d_model
        self.model_architecture = model_architecture

        if self.model_architecture != "parallel":
            raise NotImplementedError(f"Model architecture '{self.model_architecture}' is not supported. Use 'parallel'.")

        self.input_embed = nn.Sequential(nn.Linear(input_dim, d_model), nn.ReLU())
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.fc_torque = nn.Linear(d_model, 1)
        self.fc_gear_logits = nn.Linear(d_model, num_gears)
        self.fc_brake = nn.Linear(d_model, 1)

        self._init_weights()

        vehicle_phys_params = config.get("vehicle_physical_params", {})
        self.gear_neg_limits = nn.Parameter(
            torch.tensor(vehicle_phys_params.get("gear_neg_limits", []), dtype=torch.float32),
            requires_grad=False
        )
        self.max_pos_tq = vehicle_phys_params.get("max_pos_tq", 2600.0)

        if len(self.gear_neg_limits) == 0:
            raise ValueError("`gear_neg_limits` not found or empty in config's `vehicle_physical_params`.")

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc_torque.weight)
        nn.init.zeros_(self.fc_torque.bias)
        with torch.no_grad(): self.fc_torque.bias += 0.2
            
        nn.init.xavier_uniform_(self.fc_gear_logits.weight)
        nn.init.zeros_(self.fc_gear_logits.bias)

        nn.init.xavier_uniform_(self.fc_brake.weight)
        nn.init.zeros_(self.fc_brake.bias)
        with torch.no_grad(): self.fc_brake.bias -= 5.0

    def _forward_parallel(self, x, tau):
        x = x.to(self.fc_torque.weight.device)
        B, total_len, _ = x.shape

        x_embed = self.input_embed(x)
        x_pe = self.pos_encoding(x_embed)
        enc_out = self.transformer_encoder(x_pe)

        start_idx = self.past_context
        future_out = enc_out[:, start_idx:, :]

        raw_torque = 0.95 * torch.tanh(self.fc_torque(future_out))

        gear_logits = self.fc_gear_logits(future_out)
        gear_logits_2d = gear_logits.view(-1, self.num_gears)

        # Calculate BOTH soft and hard versions
        gear_probs_soft_2d = gumbel_softmax(gear_logits_2d, tau=tau, hard=False)
        gear_probs_hard_2d = gumbel_softmax(gear_logits_2d, tau=tau, hard=True)

        # Reshape both
        gear_probs_soft = gear_probs_soft_2d.view(B, self.future_horizon, self.num_gears)
        gear_probs_hard = gear_probs_hard_2d.view(B, self.future_horizon, self.num_gears)

        brake_signal = torch.sigmoid(self.fc_brake(future_out))

        # Return the torque, SOFT probs, HARD probs, and brake signal
        return raw_torque, gear_probs_soft, gear_probs_hard, brake_signal

    def forward(self, x, tau=1.0):
        """
        Main forward pass for the parallel architecture.
        Now returns both soft and hard gear probabilities.
        """
        # All internal math is done in float32 for stability.
        with torch.amp.autocast('cuda', enabled=False):
            # Unpack the four return values from our updated helper method
            raw_torque, gear_probs_soft, gear_probs_hard, brake_signal = \
                self._forward_parallel(x.float(), tau)

        # Return all four tensors, cast to the original input dtype for AMP compatibility
        return (
            raw_torque.to(x.dtype),
            gear_probs_soft.to(x.dtype),
            gear_probs_hard.to(x.dtype),
            brake_signal.to(x.dtype)
        )
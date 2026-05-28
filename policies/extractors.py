"""
Deep sequence feature extractors for PPO (V3 / Phase 4).

The environment emits a *flattened* observation: ``window_size * n_features``
technical-feature values followed by ``N_ACCOUNT_FEATURES`` account-state
scalars. The default ``MlpPolicy`` treats this as an unordered vector, throwing
away the temporal structure.

These custom :class:`BaseFeaturesExtractor` subclasses instead **reshape the
flat window back into a ``(time, features)`` sequence** and process it with a
recurrent (LSTM), attention (Transformer) or convolutional (CNN) encoder before
concatenating the account state. They plug straight into Stable-Baselines3 via
``policy_kwargs`` — no change to the environment or data pipeline is required.

Because the encoders operate purely on the observation window (which contains
only causal, past-and-present features), they introduce no look-ahead.
"""
from __future__ import annotations

import math
from typing import Tuple

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env.gold_trading_env import N_ACCOUNT_FEATURES


class _SequenceExtractor(BaseFeaturesExtractor):
    """Base class that splits the flat observation into (sequence, account).

    Subclasses build ``self.encoder`` (consuming ``(B, T, n_features)`` and
    producing ``(B, encoder_out)``) and ``self.head`` (mapping
    ``encoder_out + n_account -> features_dim``).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        window_size: int,
        features_dim: int = 128,
        n_account_features: int = N_ACCOUNT_FEATURES,
    ) -> None:
        super().__init__(observation_space, features_dim)
        total = int(observation_space.shape[0])
        self.window_size = int(window_size)
        self.n_account = int(n_account_features)
        seq_len_total = total - self.n_account
        if seq_len_total <= 0 or seq_len_total % self.window_size != 0:
            raise ValueError(
                f"Observation dim {total} incompatible with window_size "
                f"{window_size} and {self.n_account} account features."
            )
        self.n_features = seq_len_total // self.window_size

    def _split(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = obs.shape[0]
        cut = self.window_size * self.n_features
        seq = obs[:, :cut].view(batch, self.window_size, self.n_features)
        account = obs[:, cut:]
        return seq, account


class LSTMExtractor(_SequenceExtractor):
    """LSTM over the time axis; uses the final hidden state (PPO + LSTM)."""

    def __init__(
        self,
        observation_space: gym.Space,
        window_size: int,
        features_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        n_account_features: int = N_ACCOUNT_FEATURES,
    ) -> None:
        super().__init__(observation_space, window_size, features_dim, n_account_features)
        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + self.n_account, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        seq, account = self._split(observations)
        out, _ = self.lstm(seq)           # (B, T, hidden)
        last = out[:, -1, :]              # final timestep summary
        return self.head(torch.cat([last, account], dim=1))


class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for the Transformer."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerExtractor(_SequenceExtractor):
    """Transformer encoder with temporal attention pooling."""

    def __init__(
        self,
        observation_space: gym.Space,
        window_size: int,
        features_dim: int = 128,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        n_account_features: int = N_ACCOUNT_FEATURES,
    ) -> None:
        super().__init__(observation_space, window_size, features_dim, n_account_features)
        self.input_proj = nn.Linear(self.n_features, d_model)
        self.pos_encoder = _PositionalEncoding(d_model, max_len=max(self.window_size, 64))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Learned attention pooling over time (a single query attends to all steps).
        self.attn_pool = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(d_model + self.n_account, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        seq, account = self._split(observations)
        x = self.input_proj(seq)
        x = self.pos_encoder(x)
        z = self.encoder(x)                                  # (B, T, d_model)
        weights = torch.softmax(self.attn_pool(z), dim=1)    # (B, T, 1)
        pooled = (z * weights).sum(dim=1)                    # (B, d_model)
        return self.head(torch.cat([pooled, account], dim=1))


class CNNExtractor(_SequenceExtractor):
    """1-D CNN over the time axis for local chart-pattern extraction."""

    def __init__(
        self,
        observation_space: gym.Space,
        window_size: int,
        features_dim: int = 128,
        channels: Tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        n_account_features: int = N_ACCOUNT_FEATURES,
    ) -> None:
        super().__init__(observation_space, window_size, features_dim, n_account_features)
        layers = []
        in_ch = self.n_features
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            ]
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool1d(1))  # global temporal pooling
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(in_ch + self.n_account, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        seq, account = self._split(observations)
        # Conv1d expects (B, channels=n_features, length=time).
        x = seq.transpose(1, 2)
        z = self.conv(x).squeeze(-1)                 # (B, channels[-1])
        return self.head(torch.cat([z, account], dim=1))


class CNNLSTMExtractor(_SequenceExtractor):
    """Hybrid: a 1-D CNN front-end (local patterns) feeding an LSTM (temporal).

    Convolution first extracts local chart-shape features at each time step, then
    the LSTM models how those evolve over the window. More expressive than either
    alone — and therefore more prone to overfitting on low-SNR data, so judge it
    by walk-forward / grid robustness, not a single backtest. The convolution
    keeps the time length (padding), so the LSTM still sees ``window_size`` steps.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        window_size: int,
        features_dim: int = 128,
        channels: Tuple[int, ...] = (32,),
        kernel_size: int = 3,
        hidden_size: int = 128,
        num_layers: int = 1,
        n_account_features: int = N_ACCOUNT_FEATURES,
    ) -> None:
        super().__init__(observation_space, window_size, features_dim, n_account_features)
        layers = []
        in_ch = self.n_features
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)          # keeps time length
        self.lstm = nn.LSTM(in_ch, hidden_size, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + self.n_account, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        seq, account = self._split(observations)     # (B, T, F)
        x = seq.transpose(1, 2)                       # (B, F, T)
        x = self.conv(x).transpose(1, 2)              # (B, T, C)
        out, _ = self.lstm(x)                         # (B, T, H)
        last = out[:, -1, :]
        return self.head(torch.cat([last, account], dim=1))

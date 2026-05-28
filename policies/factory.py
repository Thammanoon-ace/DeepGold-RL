"""
Policy / feature-extractor factory (V3 / Phase 4).

Translates ``TrainingConfig.policy_arch`` (+ its architecture params) into the
``policy_kwargs`` dict that Stable-Baselines3 expects, so the training scripts
can switch between MLP, LSTM, Transformer and CNN encoders purely via config.
"""
from __future__ import annotations

from typing import Any, Dict

from config.config import TrainingConfig
from env.gold_trading_env import N_ACCOUNT_FEATURES
from policies.extractors import (
    CNNExtractor,
    CNNLSTMExtractor,
    LSTMExtractor,
    TransformerExtractor,
)

SUPPORTED_ARCHS = ("mlp", "lstm", "transformer", "cnn", "cnn_lstm")


def build_policy_kwargs(cfg: TrainingConfig, window_size: int) -> Dict[str, Any]:
    """Return the ``policy_kwargs`` for the configured architecture.

    For non-MLP architectures a custom feature extractor is selected; the
    ``net_arch`` heads after the extractor are kept small since the extractor
    already produces a rich representation.
    """
    arch = (cfg.policy_arch or "mlp").lower()
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unknown policy_arch {arch!r}; choose from {SUPPORTED_ARCHS}.")

    if arch == "mlp":
        return dict(net_arch=cfg.net_arch)

    common = dict(window_size=window_size, n_account_features=N_ACCOUNT_FEATURES,
                  features_dim=cfg.features_dim)
    head_arch = [64]  # small post-extractor MLP for the policy/value heads

    if arch == "lstm":
        return dict(
            features_extractor_class=LSTMExtractor,
            features_extractor_kwargs={**common, "hidden_size": cfg.lstm_hidden,
                                       "num_layers": cfg.lstm_layers},
            net_arch=head_arch,
        )
    if arch == "transformer":
        return dict(
            features_extractor_class=TransformerExtractor,
            features_extractor_kwargs={**common, "d_model": cfg.transformer_d_model,
                                       "nhead": cfg.transformer_nhead,
                                       "num_layers": cfg.transformer_layers},
            net_arch=head_arch,
        )
    if arch == "cnn_lstm":
        return dict(
            features_extractor_class=CNNLSTMExtractor,
            features_extractor_kwargs={**common, "channels": tuple(cfg.cnn_channels),
                                       "kernel_size": cfg.cnn_kernel,
                                       "hidden_size": cfg.lstm_hidden,
                                       "num_layers": cfg.lstm_layers},
            net_arch=head_arch,
        )
    # cnn
    return dict(
        features_extractor_class=CNNExtractor,
        features_extractor_kwargs={**common, "channels": tuple(cfg.cnn_channels),
                                   "kernel_size": cfg.cnn_kernel},
        net_arch=head_arch,
    )

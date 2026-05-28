"""
Centralized configuration for the DeepGold RL trading framework.

All tunable parameters live here as typed dataclasses so that the rest of the
code base never hard-codes magic numbers.  Values can be overridden at runtime
from a YAML file (see ``config/config.yaml``) via :meth:`Config.from_yaml`.

Design notes
------------
* Keeping configuration separate from logic (requirement #10) makes the
  framework reproducible: a single YAML file fully describes an experiment.
* Train/test dates implement walk-forward validation (requirement #14): the
  agent only ever *sees* pre-2025 data during training, while 2025 is held out
  strictly for evaluation.  This is enforced downstream by the data loader and
  the normalizer (which is fit on training data only).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

try:
    import yaml  # PyYAML
except ImportError:  # pragma: no cover - yaml is a declared dependency
    yaml = None


# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
# Project root = parent directory of this ``config`` package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Bars per trading year by timeframe (24h sessions, ~252 trading days). Used to
# annualize Sharpe/Sortino correctly when evaluating on different timeframes.
BARS_PER_YEAR = {"M5": 288 * 252, "M15": 96 * 252, "H1": 24 * 252, "H4": 6 * 252}


@dataclass
class PathsConfig:
    """Absolute paths to the standard project folders."""

    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    models: Path = PROJECT_ROOT / "models"
    logs: Path = PROJECT_ROOT / "logs"
    notebooks: Path = PROJECT_ROOT / "notebooks"

    def ensure(self) -> None:
        """Create every folder if it does not already exist."""
        for p in (self.data, self.models, self.logs, self.notebooks):
            Path(p).mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """How raw market data is loaded, cleaned and split."""

    symbol: str = "XAUUSD"
    timeframe: str = "M5"               # one of: M5, M15, H1
    csv_filename: str = "XAUUSD_M5.csv"  # file expected inside data/
    datetime_column: str = "time"
    # Column names as exported by MetaTrader 5 (lower-cased on load).
    ohlcv_columns: List[str] = field(
        default_factory=lambda: ["open", "high", "low", "close", "tick_volume"]
    )
    # Missing-value strategy: 'ffill' | 'drop' | 'interpolate'.
    missing_value_strategy: str = "ffill"

    # ---- Walk-forward split (no future leakage) ------------------------- #
    # Everything strictly BEFORE ``test_start`` is training data; everything
    # from ``test_start`` (inclusive) up to ``test_end`` is the held-out set.
    train_start: Optional[str] = None          # None => earliest available
    test_start: str = "2025-01-01"
    test_end: Optional[str] = None             # None => latest available


@dataclass
class FeatureConfig:
    """Technical-indicator parameters (requirement #2)."""

    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ema_fast: int = 12
    ema_slow: int = 26
    atr_period: int = 14
    volatility_window: int = 20

    # ---- V3 / Phase 4 additions ---------------------------------------- #
    # Higher timeframes whose indicators are merged (causally) onto the base
    # timeframe, e.g. ["H1"] or ["H1", "H4"]. Empty = single-timeframe.
    multi_timeframe: List[str] = field(default_factory=list)
    # Lookback for the volatility-regime baseline (0 disables the feature).
    regime_window: int = 100
    # V3.5 / 5B: append RegimeDetector signals (regime_trend, regime_vol) to the
    # observation so the agent can condition behaviour on the market regime
    # (e.g. ride strong trends, be cautious when ranging). Causal.
    use_regime_features: bool = False

    # ---- V3 / Phase 4B: Indicator Expansion System --------------------- #
    # Opt-in extra feature GROUPS (see utils/indicators.FEATURE_GROUPS):
    # 'trend' | 'momentum' | 'volatility' | 'candle' | 'structure' | 'volume'.
    # Empty keeps only the core feature set. Add incrementally and validate each
    # group with walk-forward (scripts/feature_ab.py) before keeping it.
    feature_groups: List[str] = field(default_factory=list)
    # If >0, drop engineered features whose abs pairwise correlation exceeds this
    # (fit on training data only) to avoid redundant/correlated indicators.
    correlation_threshold: float = 0.0

    # Parameters for the Phase 4B indicators.
    sma_period: int = 50
    vwap_window: int = 50
    roc_period: int = 10
    stoch_rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    hist_vol_window: int = 50
    structure_window: int = 50
    slope_window: int = 20
    volume_window: int = 20

    # Columns fed to the agent (filled automatically by FeatureEngineer).
    feature_columns: List[str] = field(default_factory=list)


@dataclass
class EnvConfig:
    """Trading-environment economics and risk controls (requirements #3/#13)."""

    # Account.
    initial_balance: float = 10_000.0
    # Observation lookback window (number of bars the agent sees).
    window_size: int = 32

    # Instrument economics (XAUUSD defaults; broker dependent).
    contract_size: float = 100.0        # ounces per 1.0 lot
    leverage: float = 30.0              # 1:30 retail leverage
    spread: float = 0.20                # quoted in price units (USD/oz)
    slippage: float = 0.05              # extra adverse price on fill
    commission_per_lot: float = 7.0     # round-turn commission per lot (USD)

    # Risk management (hard safety limits).
    max_position_lots: float = 1.0      # absolute cap on position size
    risk_fraction: float = 0.02         # fraction of balance risked per trade
    stop_loss_pct: float = 0.01         # 1% adverse move closes the trade
    take_profit_pct: float = 0.02       # 2% favourable move closes the trade
    atr_period: int = 14                # ATR period the env computes for vol-targeted sizing
    max_drawdown_pct: float = 0.40      # episode terminates beyond this DD

    # Reward mode: 'absolute' = scaled equity change (default);
    # 'excess' = equity change MINUS a buy-and-hold benchmark's change (reward
    #   accrues only for beating passive holding);
    # 'dsr' = Differential Sharpe Ratio of the *excess* (over buy-and-hold)
    #   returns — an online reward that maximises the INFORMATION RATIO vs
    #   buy-and-hold (the risk-adjusted dimension BH still wins). Targets Sharpe,
    #   not raw return, and inherently penalises volatile equity.
    reward_mode: str = "absolute"
    dsr_eta: float = 0.01               # EMA rate for the differential Sharpe

    # Reward shaping weights (requirement #4).  Kept small and interpretable
    # to avoid reward hacking (requirement #14).
    reward_scaling: float = 1.0e-3      # scales raw equity delta into ~[-1,1]
    drawdown_penalty_weight: float = 0.5
    overtrading_penalty: float = 1.0e-4  # per executed trade
    holding_penalty: float = 0.0         # optional cost for idle capital

    # Minimum number of bars between two new entries (anti-overtrading).
    min_bars_between_trades: int = 0

    # ---- V3.5 / Phase 5C: trade-frequency control (defaults = neutral) -- #
    # Minimum bars a position must be held before a *signal* Close is allowed
    # (stop-loss / take-profit always fire regardless). 0 = no minimum.
    min_hold_bars: int = 0
    # Cap on new entries per episode (0 = unlimited). Hard brake on churn.
    max_trades_per_episode: int = 0
    # Make the overtrading penalty grow with trade count: penalty for the k-th
    # entry = overtrading_penalty * (1 + trade_penalty_growth * n_trades).
    # 0 = constant penalty (unchanged behaviour).
    trade_penalty_growth: float = 0.0

    # ---- V3.5 reward/sizing redesign (default off = unchanged behaviour) ---- #
    # Volatility-targeted sizing: size each position so the stop-loss risk is a
    # fixed fraction of balance based on ATR (constant $-risk across calm/volatile
    # regimes) instead of a fixed % of price. The env computes ATR from its own
    # OHLC (causal). SL distance = vol_target_risk_atr * ATR; TP keeps the same
    # reward:risk ratio (take_profit_pct / stop_loss_pct).
    volatility_target_sizing: bool = False
    vol_target_risk_atr: float = 1.5


@dataclass
class TrainingConfig:
    """PPO hyper-parameters and runtime options (requirements #5/#12)."""

    total_timesteps: int = 1_000_000
    n_envs: int = 4                     # vectorized environments
    seed: int = 42
    device: str = "auto"                # 'auto' picks CUDA when available

    # Algorithm: 'ppo' (on-policy, default) or 'dqn' (off-policy, experimental).
    # SAC is intentionally not offered — it is continuous-action only, while our
    # action space is Discrete(4); DQN is the discrete off-policy counterpart.
    algo: str = "ppo"

    # PPO hyper-parameters (Stable-Baselines3 defaults, lightly tuned).
    policy: str = "MlpPolicy"
    learning_rate: float = 3.0e-4
    n_steps: int = 2048                 # rollout length per env
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01              # encourage exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: List[int] = field(default_factory=lambda: [128, 128])

    # ---- Deep sequence architecture (V3 / Phase 4) --------------------- #
    # policy_arch: 'mlp' (default) | 'lstm' | 'transformer' | 'cnn'.
    # Non-MLP archs use a custom feature extractor (see policies/) that treats
    # the observation window as a (time x features) sequence. GPU is worthwhile
    # for these; the tiny MLP is usually faster on CPU.
    policy_arch: str = "mlp"
    features_dim: int = 128             # extractor output size (non-MLP archs)
    lstm_hidden: int = 128
    lstm_layers: int = 1
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_layers: int = 2
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64])
    cnn_kernel: int = 3

    # ---- DQN-specific hyper-parameters (used only when algo='dqn') ------ #
    dqn_buffer_size: int = 100_000
    dqn_learning_starts: int = 10_000
    dqn_target_update_interval: int = 2_000
    dqn_train_freq: int = 4
    dqn_exploration_fraction: float = 0.2
    dqn_exploration_final_eps: float = 0.05

    # Checkpointing / logging.
    checkpoint_freq: int = 50_000       # steps between checkpoints (per env)
    eval_freq: int = 50_000             # steps between evaluations
    model_name: str = "ppo_gold"
    tensorboard_subdir: str = "tensorboard"


@dataclass
class BacktestConfig:
    """Backtest / evaluation options (requirement #6)."""

    deterministic: bool = True          # greedy policy during evaluation
    risk_free_rate: float = 0.0         # annualized, for Sharpe
    # Bars per year used to annualize Sharpe (M5 ~= 288 bars/day * 252 days).
    bars_per_year: int = 288 * 252
    save_plots: bool = True


@dataclass
class LiveConfig:
    """MetaTrader 5 live-trading parameters (requirement #9).

    Credentials should come from environment variables, never source control.
    """

    enabled: bool = False               # master safety switch (off by default)
    symbol: str = "XAUUSD"
    timeframe: str = "M5"
    magic_number: int = 20250526        # identifies bot orders
    poll_seconds: int = 30
    # Live risk guards (independent of the environment limits).
    max_open_positions: int = 1
    max_lots: float = 0.10
    dry_run: bool = True                # log intended orders, do not send them


@dataclass
class Config:
    """Top-level container aggregating every sub-config."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Build a Config, overriding defaults with values from a YAML file.

        Unknown keys are ignored so old config files stay forward-compatible.
        """
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configs.")
        cfg = cls()
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        section_map = {
            "data": cfg.data,
            "features": cfg.features,
            "env": cfg.env,
            "training": cfg.training,
            "backtest": cfg.backtest,
            "live": cfg.live,
        }
        for section, obj in section_map.items():
            for key, value in (raw.get(section) or {}).items():
                if hasattr(obj, key):
                    setattr(obj, key, value)
        return cfg

    def to_dict(self) -> dict:
        """Return a JSON/YAML-serializable representation (paths as strings)."""
        d = {
            "data": asdict(self.data),
            "features": asdict(self.features),
            "env": asdict(self.env),
            "training": asdict(self.training),
            "backtest": asdict(self.backtest),
            "live": asdict(self.live),
        }
        return d


# A ready-to-use default instance for quick scripts and notebooks.
default_config = Config()

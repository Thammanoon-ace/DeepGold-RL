"""Smoke + baseline test for ExecutionGoldEnv.

Checks three things:

1. **Wiring** — obs shape matches the declared space, episode lengths are
   bounded, ``terminated``/``truncated`` fire correctly.
2. **TWAP shadow correctness** — when the agent acts at exactly TWAP rate
   (action=2 every step) its shortfall should equal the shadow's, so
   ``bps_savings_vs_twap`` ≈ 0. This is the env's internal sanity check.
3. **Tension** — running fixed strategies over many episodes, the *distribution*
   of shortfalls should differ (front-load takes more impact, back-load takes
   more drift) so there's signal to learn. If they're identical the env
   isn't modelling impact properly.

Run with: ``python scripts/_test_execution_env.py``
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config.config import EnvConfig
from env.execution_env import ExecutionGoldEnv


def make_env(seed: int = 0) -> ExecutionGoldEnv:
    rng = np.random.default_rng(seed)
    n_bars, n_features = 5000, 4
    # Random-walk close so episodes see meaningful drift.
    close = 2000.0 + np.cumsum(rng.normal(0, 4, n_bars))
    high = close + np.abs(rng.normal(0, 3, n_bars))
    low = close - np.abs(rng.normal(0, 3, n_bars))
    prices = np.stack([high, low, close], axis=1)
    feats = rng.normal(0, 1, (n_bars, n_features)).astype(np.float32)
    cfg = EnvConfig(window_size=16, spread=0.20, slippage=0.05)
    return ExecutionGoldEnv(feats, prices, cfg, deadline_range=(32, 64),
                            target_lots_range=(0.5, 2.0),
                            fixed_cost_bps=1.0, impact_bps_per_lot=10.0,
                            random_start=True, seed=seed)


def run_strategy(env: ExecutionGoldEnv, action_picker, n_episodes: int,
                 reset_seed_base: int = 1000):
    """Run one fixed strategy across N episodes; return shortfall stats.

    ``action_picker(env, step_idx)`` returns an int action each step.
    """
    sf, twap_sf, savings = [], [], []
    for ep in range(n_episodes):
        env.reset(seed=reset_seed_base + ep)
        t, done, info = 0, False, {}
        while not done:
            a = action_picker(env, t)
            _, _, term, trunc, info = env.step(a)
            done = term or trunc
            t += 1
        sf.append(info["shortfall_bps"])
        twap_sf.append(info["twap_shortfall_bps"])
        savings.append(info["bps_savings_vs_twap"])
    return np.array(sf), np.array(twap_sf), np.array(savings)


def main() -> None:
    # ---- 1. Wiring smoke ------------------------------------------------ #
    env = make_env(seed=0)
    obs, _ = env.reset(seed=42)
    expected = env.observation_space.shape[0]
    assert obs.shape == (expected,), f"obs shape {obs.shape} != {(expected,)}"
    assert env.action_space.n == 4
    print(f"[1] wiring  : obs_dim={expected} actions={env.action_space.n}  OK")

    # ---- 2. Full random episode runs to completion --------------------- #
    np.random.seed(0)
    info = {}
    for _ in range(200):
        info_step = env.step(int(np.random.randint(4)))[-1]
        if info_step:                       # terminal step carries the final dict
            info = info_step
            break
    assert info, "random episode never terminated within 200 steps"
    assert "bps_savings_vs_twap" in info
    print(f"[2] random ep: terminates, info keys: {sorted(info.keys())[:4]}...  OK")

    # ---- 3. TWAP-as-agent sanity (shadow correctness) ------------------ #
    sf, twap_sf, savings = run_strategy(env, lambda env, t: 2, n_episodes=100)
    print(f"[3] agent==TWAP rate (action=2 every step):")
    print(f"      agent shortfall  median {np.median(sf):+.3f} bps")
    print(f"      TWAP  shortfall  median {np.median(twap_sf):+.3f} bps")
    print(f"      bps_savings      median {np.median(savings):+.4f} bps  (expect ~0)")
    assert abs(np.median(savings)) < 0.5, (
        f"agent at TWAP rate should match shadow to <0.5 bps median, got "
        f"{np.median(savings):+.3f}"
    )
    print(f"      shadow matches agent at action=2  OK")

    # ---- 4. Strategy comparison (must produce different distributions) -- #
    print("\n[4] strategy comparison (100 episodes each, same seeds):")
    print(f"{'strategy':<14} | {'median sf':>11} | {'mean sf':>9} | "
          f"{'p10..p90':>16} | {'beats TWAP':>11}")
    print("-" * 75)
    strategies = {
        "pause-then-end": lambda env, t: 0,                         # action=0 always → forced finish
        "back-load":      lambda env, t: 0 if env.remaining_bars > 1 else 3,
        "front-load":     lambda env, t: 3 if env.remaining_lots > 0.1 else 0,
        "TWAP (action 2)": lambda env, t: 2,
        "random":         lambda env, t: int(np.random.randint(4)),
    }
    out = {}
    for name, picker in strategies.items():
        np.random.seed(7)
        sf, _, sav = run_strategy(make_env(seed=0), picker, n_episodes=100)
        out[name] = (sf, sav)
        beats = float(np.mean(sav > 0)) * 100.0
        print(f"{name:<14} | {np.median(sf):>+8.2f} bps | "
              f"{np.mean(sf):>+6.2f} | "
              f"[{np.percentile(sf, 10):>+5.1f}, {np.percentile(sf, 90):>+5.1f}] | "
              f"{beats:>9.1f} %")

    # ---- 5. Sanity asserts on the comparison --------------------------- #
    # The cleanest single-slice impact test: ``pause-then-end`` waits, then
    # the forced-finish dumps the entire remaining lot in one slice, paying
    # full impact_bps_per_lot × target_lots. That MUST be much worse than
    # TWAP (which spreads the lot across ~deadline bars).
    pause_median = np.median(out["pause-then-end"][0])
    twap_median = np.median(out["TWAP (action 2)"][0])
    assert pause_median > twap_median + 5.0, (
        f"forced single-slice finish ({pause_median:.2f} bps) should be ≥5 bps "
        f"worse than TWAP ({twap_median:.2f}); the impact term may be too weak"
    )
    # Strategies should produce distinct distributions (not bug-collapsed).
    assert not np.allclose(out["front-load"][0], out["TWAP (action 2)"][0]), \
        "front-load and TWAP shortfall arrays should differ"
    print(f"\nimpact tension confirmed: forced-finish - TWAP = "
          f"{pause_median - twap_median:+.1f} bps median")
    print("EXEC-ENV-OK")


if __name__ == "__main__":
    main()

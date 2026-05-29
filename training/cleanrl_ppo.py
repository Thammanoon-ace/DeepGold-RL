"""
CleanRL-style PPO — fully torch-native, GPU-resident training loop.

Pairs with :class:`env.torch_vec_env.TorchVecGoldEnv`: the rollout, advantage
computation and PPO update all run on the GPU in torch, so with a GPU env the
device is the primary compute engine (no per-step CPU<->GPU transfer, no SB3).

The policy (:class:`ActorCritic`) exposes an SB3-compatible ``predict(obs)`` so a
trained agent can still be evaluated with the scalar env + ``run_episode`` and
the standard metric/robustness tooling — keeping evaluation identical to every
other experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from env.gold_trading_env import N_ACCOUNT_FEATURES


def _orthogonal(layer: nn.Module, std: float = np.sqrt(2), bias: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Actor-critic mirroring the SB3 ``cnn`` policy (policies.extractors).

    The flat obs is split into a ``(window, n_features)`` block + account-state
    scalars. For ``arch="cnn"`` the block goes through the same pipeline SB3 uses
    — ``CNNExtractor`` (conv(32,64)+ReLU -> global avg pool -> Linear+ReLU to
    ``features_dim``) followed by the ``net_arch`` ``[128,128]`` Tanh
    ``mlp_extractor`` — before the orthogonally-initialised policy/value heads.
    This matching is deliberate: the original CleanRL net (no ``net_arch`` MLP,
    no conv ortho-init) under-performed SB3 on the trend folds, so the GPU engine
    is being brought to architectural parity (see docs/memory gpu-ppo-vs-sb3).
    """

    def __init__(self, obs_dim: int, window: int, n_features: int,
                 n_actions: int = 4, arch: str = "cnn",
                 cnn_channels: Tuple[int, ...] = (32, 64), features_dim: int = 128,
                 net_arch: Tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        self.window, self.n_features = window, n_features
        self.n_account = N_ACCOUNT_FEATURES
        self.arch = arch

        # --- Feature extractor (mirrors policies.extractors) --------------- #
        if arch == "cnn":
            # SB3 CNNExtractor: conv(32,64)+ReLU -> global avg pool -> Linear+ReLU
            layers, in_ch = [], n_features
            for out_ch in cnn_channels:
                layers += [_orthogonal(nn.Conv1d(in_ch, out_ch, 3, padding=1)), nn.ReLU()]
                in_ch = out_ch
            layers.append(nn.AdaptiveAvgPool1d(1))
            self.encoder = nn.Sequential(*layers)
            self.proj = nn.Sequential(
                _orthogonal(nn.Linear(in_ch + self.n_account, features_dim)), nn.ReLU())
            mlp_in = features_dim
        else:  # mlp: SB3 MlpPolicy feeds the flat obs straight to the mlp_extractor
            self.encoder = None
            self.proj = None
            mlp_in = obs_dim

        # --- SB3 mlp_extractor (net_arch, Tanh), shared by pi & vf --------- #
        mlp, last = [], mlp_in
        for h in net_arch:
            mlp += [_orthogonal(nn.Linear(last, h)), nn.Tanh()]
            last = h
        self.mlp = nn.Sequential(*mlp)
        self.actor = _orthogonal(nn.Linear(last, n_actions), std=0.01)
        self.critic = _orthogonal(nn.Linear(last, 1), std=1.0)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        if self.arch == "cnn":
            cut = self.window * self.n_features
            seq, acct = obs[:, :cut], obs[:, cut:]
            x = seq.view(-1, self.window, self.n_features).transpose(1, 2)
            enc = self.encoder(x).squeeze(-1)
            feat = self.proj(torch.cat([enc, acct], dim=1))
        else:
            feat = obs
        return self.mlp(feat)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self._features(obs)).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None):
        feat = self._features(obs)
        logits = self.actor(feat)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(feat).squeeze(-1)

    # --- SB3-compatible inference (for scalar-env evaluation) ---------- #
    @torch.no_grad()
    def _logits(self, observation) -> torch.Tensor:
        p = next(self.parameters())
        dev, p_dtype = p.device, p.dtype
        # Accept numpy arrays (scalar-env path) AND torch tensors (Tier 2.1
        # GPU-resident eval path) — for the latter we skip the host round-trip.
        if isinstance(observation, torch.Tensor):
            t = observation if observation.ndim == 2 else observation.unsqueeze(0)
            t = t.to(device=dev, dtype=p_dtype)
        else:
            obs = np.asarray(observation, dtype=np.float32)
            t = torch.as_tensor(obs[None] if obs.ndim == 1 else obs,
                                dtype=p_dtype, device=dev)
        return self.actor(self._features(t))

    @torch.no_grad()
    def predict(self, observation, deterministic: bool = True):
        single = (observation.ndim == 1 if isinstance(observation, torch.Tensor)
                  else np.asarray(observation).ndim == 1)
        logits = self._logits(observation)
        a = (logits.argmax(-1) if deterministic
             else torch.distributions.Categorical(logits=logits).sample())
        a = a.cpu().numpy()
        return (int(a[0]) if single else a), None

    @torch.no_grad()
    def action_probs(self, observation) -> np.ndarray:
        """Mean-poolable categorical action probabilities, shape (B, A).

        Lets :class:`policies.ensemble.EnsemblePolicy` average GPU-trained
        agents the same way it averages SB3 agents (duck-typed there).
        """
        return torch.softmax(self._logits(observation), dim=-1).cpu().numpy()

    @torch.no_grad()
    def action_probs_torch(self, observation: torch.Tensor) -> torch.Tensor:
        """GPU-resident counterpart of :meth:`action_probs` (Tier 2.1).

        Accepts a torch tensor on any device, returns the (B, A) softmax tensor
        on the model's device — no numpy round-trip. The GPU-vec eval loop and
        :meth:`policies.ensemble.EnsemblePolicy.predict_torch` use this to keep
        the whole inference path on the GPU.
        """
        return torch.softmax(self._logits(observation), dim=-1)

    @torch.no_grad()
    def predict_torch(self, observation: torch.Tensor) -> torch.Tensor:
        """GPU-resident argmax counterpart of :meth:`predict` (Tier 2.1).

        Duck-types with :meth:`policies.ensemble.EnsemblePolicy.predict_torch`
        so ``backtest.run_episode_torch_vec`` can call either uniformly.
        """
        single = observation.ndim == 1
        if single:
            observation = observation.unsqueeze(0)
        return self._logits(observation).argmax(dim=-1).to(torch.long)


@dataclass
class PPOConfig:
    total_timesteps: int = 300_000
    n_steps: int = 256
    n_epochs: int = 10            # SB3 default; more passes => policy actually moves
    minibatch_size: int = 2048    # SB3 grid used 512; small minibatches => many
                                  # gradient steps/rollout so the policy actually
                                  # learns to trade instead of collapsing to Hold.
                                  # (overrides num_minibatches when set)
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3.0e-4
    target_kl: Optional[float] = None   # SB3 default: no early stop (avoids
                                        # the policy freezing near its init)
    normalize_reward: bool = True       # match SB3 VecNormalize(norm_reward=True):
                                        # scale the small excess/dsr rewards so the
                                        # signal is large enough to learn from
    compile: bool = False               # Tier 2.4: torch.compile the ActorCritic
                                        # in 'reduce-overhead' mode (CUDA graphs).
                                        # Off by default until the bench confirms a
                                        # net speedup on this codebase.


class _RewardNormalizer:
    """SB3-VecNormalize-style reward scaling: divide by the running std of the
    discounted return. Amplifies the tiny per-step excess/DSR rewards so PPO
    optimises them instead of drifting to a constant (Hold) action."""

    def __init__(self, num_envs: int, gamma: float, dev, dtype) -> None:
        self.gamma = gamma
        self.ret = torch.zeros(num_envs, device=dev, dtype=dtype)
        self.mean = torch.zeros((), device=dev, dtype=dtype)
        self.var = torch.ones((), device=dev, dtype=dtype)
        self.count = torch.tensor(1e-4, device=dev, dtype=dtype)

    def __call__(self, reward: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        self.ret = self.ret * self.gamma + reward
        b_mean, b_var, b_count = self.ret.mean(), self.ret.var(unbiased=False), self.ret.numel()
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean = self.mean + delta * b_count / tot
        m2 = self.var * self.count + b_var * b_count + delta * delta * self.count * b_count / tot
        self.var = m2 / tot
        self.count = tot
        out = torch.clamp(reward / torch.sqrt(self.var + 1e-8), -10.0, 10.0)
        self.ret = self.ret * (1.0 - done)   # reset discounted return on episode end
        return out


def train_cleanrl_ppo(env, arch: str = "cnn", ppo: Optional[PPOConfig] = None,
                      seed: int = 0, log_every: int = 0) -> Tuple[ActorCritic, List[float]]:
    """Train an ActorCritic on a TorchVecGoldEnv with a GPU-native PPO loop."""
    ppo = ppo or PPOConfig()
    dev = env.device
    torch.manual_seed(seed)
    ac = ActorCritic(env.obs_dim, env.window, env.n_features, arch=arch).to(dev)
    if ppo.compile:
        # torch.compile returns a wrapper that forwards attr access to the
        # underlying module, so opt/state_dict/predict_torch still work.
        ac = torch.compile(ac, mode="reduce-overhead")
    opt = torch.optim.Adam(ac.parameters(), lr=ppo.learning_rate, eps=1e-5)

    N, T = env.num_envs, ppo.n_steps
    obs = torch.zeros((T, N, env.obs_dim), device=dev)
    actions = torch.zeros((T, N), dtype=torch.long, device=dev)
    logprobs = torch.zeros((T, N), device=dev)
    rewards = torch.zeros((T, N), device=dev)
    dones = torch.zeros((T, N), device=dev)
    values = torch.zeros((T, N), device=dev)

    next_obs = env.reset()
    next_done = torch.zeros(N, device=dev)
    batch_size = N * T
    mb_size = (min(ppo.minibatch_size, batch_size) if ppo.minibatch_size
               else max(batch_size // ppo.num_minibatches, 1))
    updates = max(ppo.total_timesteps // batch_size, 1)
    ep_returns: List[float] = []
    rnorm = _RewardNormalizer(N, ppo.gamma, dev, env.dtype) if ppo.normalize_reward else None

    for update in range(updates):
        for t in range(T):
            obs[t], dones[t] = next_obs, next_done
            with torch.no_grad():
                a, lp, _, v = ac.get_action_and_value(next_obs)
            actions[t], logprobs[t], values[t] = a, lp, v
            next_obs, r, d = env.step(a)
            next_done = d.to(env.dtype)
            rewards[t] = rnorm(r, next_done) if rnorm is not None else r
            if env.last_ep_returns.numel():
                ep_returns.extend(env.last_ep_returns.detach().cpu().tolist())

        # GAE.
        with torch.no_grad():
            next_value = ac.get_value(next_obs)
            adv = torch.zeros_like(rewards)
            lastgae = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nonterminal = 1.0 - (next_done if t == T - 1 else dones[t + 1])
                nextval = next_value if t == T - 1 else values[t + 1]
                delta = rewards[t] + ppo.gamma * nextval * nonterminal - values[t]
                adv[t] = lastgae = delta + ppo.gamma * ppo.gae_lambda * nonterminal * lastgae
            returns = adv + values

        b_obs = obs.reshape(-1, env.obs_dim)
        b_act = actions.reshape(-1)
        b_lp = logprobs.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = values.reshape(-1)
        idx = np.arange(batch_size)
        for _ in range(ppo.n_epochs):
            np.random.shuffle(idx)
            stop = False
            for start in range(0, batch_size, mb_size):
                mb = idx[start:start + mb_size]
                _, newlp, ent, newval = ac.get_action_and_value(b_obs[mb], b_act[mb])
                ratio = (newlp - b_lp[mb]).exp()
                a_mb = b_adv[mb]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = torch.maximum(-a_mb * ratio,
                                   -a_mb * torch.clamp(ratio, 1 - ppo.clip_range, 1 + ppo.clip_range)).mean()
                v_loss = 0.5 * ((newval - b_ret[mb]) ** 2).mean()
                loss = pg - ppo.ent_coef * ent.mean() + ppo.vf_coef * v_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), ppo.max_grad_norm)
                opt.step()
                if ppo.target_kl is not None:
                    with torch.no_grad():
                        if ((b_lp[mb] - newlp).mean().abs()) > ppo.target_kl:
                            stop = True; break
            if stop:
                break
        if log_every and update % log_every == 0 and ep_returns:
            recent = np.mean(ep_returns[-50:])
            print(f"  update {update}/{updates} | ep_return(last50) {recent:+.2f}%")

    return ac, ep_returns

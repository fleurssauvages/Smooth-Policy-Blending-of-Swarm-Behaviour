# ppo_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 3, hidden_dim: int = 128):
        super().__init__()

        self.actor_body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor):
        raise NotImplementedError

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_policy_dist(self, obs: torch.Tensor) -> Normal:
        x = self.actor_body(obs)
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd).expand_as(mean)
        return Normal(mean, std)

    def sample_action(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            raw_action: sampled unconstrained action
            blend_weights: softmax(raw_action)
            log_prob: log prob under Normal over raw_action
            value: critic value
        """
        dist = self.get_policy_dist(obs)
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        blend_weights = torch.softmax(raw_action, dim=-1)
        value = self.get_value(obs)
        return raw_action, blend_weights, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, raw_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.get_policy_dist(obs)
        log_prob = dist.log_prob(raw_actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.get_value(obs)
        return log_prob, entropy, value


@dataclass
class PPOConfig:
    obs_dim: int
    act_dim: int = 3
    hidden_dim: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    minibatch_size: int = 256
    device: str = "cpu"


class PPOAgent:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.net = ActorCritic(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

    def act(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            raw_action, blend_weights, log_prob, value = self.net.sample_action(obs_t)
        return (
            raw_action.squeeze(0).cpu().numpy(),
            blend_weights.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def value(self, obs: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value = self.net.get_value(obs_t)
        return float(value.item())

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_nonterminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_nonterminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.cfg.gamma * next_value * next_nonterminal - values[t]
            lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values
        return advantages, returns

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        raw_actions = torch.as_tensor(batch["raw_actions"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.shape[0]
        idxs = np.arange(n)

        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
        }
        n_updates = 0

        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(idxs)

            for start in range(0, n, self.cfg.minibatch_size):
                mb_idx = idxs[start:start + self.cfg.minibatch_size]

                mb_obs = obs[mb_idx]
                mb_actions = raw_actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                new_log_probs, entropy, values = self.net.evaluate_actions(mb_obs, mb_actions)
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio,
                    1.0 - self.cfg.clip_coef,
                    1.0 + self.cfg.clip_coef,
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = 0.5 * ((values - mb_returns) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.cfg.vf_coef * value_loss
                    - self.cfg.ent_coef * entropy_loss
                )

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.optim.step()

                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_loss.item())
                stats["approx_kl"] += float(approx_kl)
                n_updates += 1

        if n_updates > 0:
            for k in stats:
                stats[k] /= n_updates

        return stats

    def save(self, path: str):
        torch.save(
            {
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "config": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["model_state_dict"])
        self.optim.load_state_dict(ckpt["optimizer_state_dict"])
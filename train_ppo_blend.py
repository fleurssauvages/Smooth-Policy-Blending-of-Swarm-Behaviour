# train_ppo_blend.py
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

from RL.fish_wrapper import FishBlendEnv
from RL.ppo_agent import PPOAgent, PPOConfig


def main():
    total_updates = 100
    rollout_steps = 2048
    save_every = 25
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    env = FishBlendEnv(seed=0, do_animation=False, visit_grid_res=24)
    obs_dim = 7

    agent = PPOAgent(
        PPOConfig(
            obs_dim=obs_dim,
            act_dim=6,   # was 3
            hidden_dim=128,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            update_epochs=10,
            minibatch_size=256,
            device="cpu",
        )
    )

    obs = env.reset(seed=0)

    for update in range(1, total_updates + 1):
        storage = defaultdict(list)
        ep_returns = []
        ep_lengths = []
        ep_return = 0.0
        ep_len = 0

        for step in range(rollout_steps):
            raw_action, _, log_prob, value = agent.act(obs)

            next_obs, reward, done, info = env.step(raw_action)

            storage["obs"].append(obs.copy())
            storage["raw_actions"].append(raw_action.copy())
            storage["log_probs"].append(log_prob)
            storage["values"].append(value)
            storage["rewards"].append(reward)
            storage["dones"].append(float(done))

            ep_return += reward
            ep_len += 1
            obs = next_obs

            if done:
                ep_returns.append(ep_return)
                ep_lengths.append(ep_len)
                ep_return = 0.0
                ep_len = 0
                obs = env.reset()

        last_value = agent.value(obs)

        batch = {
            "obs": np.asarray(storage["obs"], dtype=np.float32),
            "raw_actions": np.asarray(storage["raw_actions"], dtype=np.float32),
            "log_probs": np.asarray(storage["log_probs"], dtype=np.float32),
            "values": np.asarray(storage["values"], dtype=np.float32),
            "rewards": np.asarray(storage["rewards"], dtype=np.float32),
            "dones": np.asarray(storage["dones"], dtype=np.float32),
        }

        advantages, returns = agent.compute_gae(
            rewards=batch["rewards"],
            values=batch["values"],
            dones=batch["dones"],
            last_value=last_value,
        )
        batch["advantages"] = advantages
        batch["returns"] = returns

        stats = agent.update(batch)

        mean_return = float(np.mean(ep_returns)) if ep_returns else float("nan")
        mean_len = float(np.mean(ep_lengths)) if ep_lengths else float("nan")

        print(
            f"[update {update:04d}] "
            f"episodes={len(ep_returns):3d} "
            f"mean_return={mean_return:8.3f} "
            f"mean_len={mean_len:6.1f} "
            f"policy_loss={stats['policy_loss']:.4f} "
            f"value_loss={stats['value_loss']:.4f} "
            f"entropy={stats['entropy']:.4f} "
            f"approx_kl={stats['approx_kl']:.6f}"
        )

        if update % save_every == 0:
            ckpt_path = os.path.join(save_dir, f"ppo_blend_update_{update:04d}.pt")
            agent.save(ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(save_dir, "ppo_blend_final.pt")
    agent.save(final_path)
    print(f"Saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
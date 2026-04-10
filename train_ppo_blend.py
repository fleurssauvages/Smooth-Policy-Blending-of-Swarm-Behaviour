from __future__ import annotations

import os
import multiprocessing as mp
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from RL.fish_wrapper import FishBlendEnv
from RL.ppo_agent import PPOAgent, PPOConfig


# ---------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------
def _env_worker(remote, layout_name: str, seed: int, visit_grid_res: int):
    env = FishBlendEnv(
        seed=seed,
        do_animation=False,
        visit_grid_res=visit_grid_res,
        layout_name=layout_name,
    )

    try:
        while True:
            cmd, data = remote.recv()

            if cmd == "reset":
                obs = env.reset(seed=data)
                remote.send(obs)

            elif cmd == "step":
                obs, reward, done, info = env.step(data)
                if done:
                    next_obs = env.reset()
                    remote.send((obs, reward, done, info, next_obs))
                else:
                    remote.send((obs, reward, done, info, obs))

            elif cmd == "close":
                remote.close()
                break

            else:
                raise ValueError(f"Unknown command: {cmd}")

    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------
# Parallel env manager
# ---------------------------------------------------------------------
class ParallelFishEnv:
    def __init__(
        self,
        num_envs: int,
        layout_name: str,
        base_seed: int,
        visit_grid_res: int = 24,
    ):
        self.num_envs = num_envs
        self.remotes = []
        self.processes = []

        for i in range(num_envs):
            parent_remote, child_remote = mp.Pipe()
            proc = mp.Process(
                target=_env_worker,
                args=(child_remote, layout_name, base_seed + i, visit_grid_res),
                daemon=True,
            )
            proc.start()
            child_remote.close()

            self.remotes.append(parent_remote)
            self.processes.append(proc)

    def reset(self, seeds: list[int] | None = None) -> np.ndarray:
        if seeds is None:
            seeds = [None] * self.num_envs

        for remote, seed in zip(self.remotes, seeds):
            remote.send(("reset", seed))

        obs = [remote.recv() for remote in self.remotes]
        return np.asarray(obs, dtype=np.float32)

    def step(self, actions: np.ndarray):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))

        results = [remote.recv() for remote in self.remotes]

        obs, rewards, dones, infos, next_obs = zip(*results)

        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            list(infos),
            np.asarray(next_obs, dtype=np.float32),
        )

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass

        for proc in self.processes:
            proc.join(timeout=1.0)


# ---------------------------------------------------------------------
# Training per stage
# ---------------------------------------------------------------------
def run_stage(
    agent: PPOAgent,
    stage_name: str,
    total_updates: int,
    rollout_steps: int,
    num_envs: int,
    save_every: int,
    save_dir: str,
    success_threshold: float,
    length_threshold: float,
    success_window: int,
    seed: int = 0,
):
    print("\n" + "=" * 80)
    print(f"Starting stage: {stage_name}")
    print("=" * 80)

    envs = ParallelFishEnv(
        num_envs=num_envs,
        layout_name=stage_name,
        base_seed=seed * 10000,
        visit_grid_res=24,
    )

    obs = envs.reset(seeds=[seed * 10000 + i for i in range(num_envs)])

    recent_success_rates = deque(maxlen=success_window)
    recent_mean_lengths = deque(maxlen=success_window)

    best_success_window = -np.inf
    best_length_window = np.inf

    ep_returns = np.zeros(num_envs, dtype=np.float32)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)

    try:
        for update in range(1, total_updates + 1):
            storage = defaultdict(list)

            finished_returns = []
            finished_lengths = []
            finished_successes = []

            for _ in range(rollout_steps):
                raw_actions = []
                log_probs = []
                values = []

                for env_idx in range(num_envs):
                    raw_action, _, log_prob, value = agent.act(obs[env_idx])
                    raw_actions.append(raw_action)
                    log_probs.append(log_prob)
                    values.append(value)

                raw_actions = np.asarray(raw_actions, dtype=np.float32)
                log_probs = np.asarray(log_probs, dtype=np.float32)
                values = np.asarray(values, dtype=np.float32)

                curr_obs, rewards, dones, infos, next_obs = envs.step(raw_actions)

                storage["obs"].append(obs.copy())
                storage["raw_actions"].append(raw_actions.copy())
                storage["log_probs"].append(log_probs.copy())
                storage["values"].append(values.copy())
                storage["rewards"].append(rewards.copy())
                storage["dones"].append(dones.copy())

                ep_returns += rewards
                ep_lengths += 1

                for i in range(num_envs):
                    if dones[i]:
                        finished_returns.append(float(ep_returns[i]))
                        finished_lengths.append(int(ep_lengths[i]))
                        finished_successes.append(float(infos[i].get("success", False)))
                        ep_returns[i] = 0.0
                        ep_lengths[i] = 0

                obs = next_obs

            last_values = np.asarray([agent.value(obs_i) for obs_i in obs], dtype=np.float32)

            batch = {
                "obs": np.asarray(storage["obs"], dtype=np.float32),            # (T, N, obs_dim)
                "raw_actions": np.asarray(storage["raw_actions"], dtype=np.float32),  # (T, N, act_dim)
                "log_probs": np.asarray(storage["log_probs"], dtype=np.float32),      # (T, N)
                "values": np.asarray(storage["values"], dtype=np.float32),            # (T, N)
                "rewards": np.asarray(storage["rewards"], dtype=np.float32),          # (T, N)
                "dones": np.asarray(storage["dones"], dtype=np.float32),              # (T, N)
            }

            advantages = np.zeros_like(batch["rewards"], dtype=np.float32)
            returns = np.zeros_like(batch["rewards"], dtype=np.float32)

            for env_idx in range(num_envs):
                adv_i, ret_i = agent.compute_gae(
                    rewards=batch["rewards"][:, env_idx],
                    values=batch["values"][:, env_idx],
                    dones=batch["dones"][:, env_idx],
                    last_value=last_values[env_idx],
                )
                advantages[:, env_idx] = adv_i
                returns[:, env_idx] = ret_i

            # Flatten (T, N, ...) -> (T*N, ...)
            batch_flat = {
                "obs": batch["obs"].reshape(-1, batch["obs"].shape[-1]),
                "raw_actions": batch["raw_actions"].reshape(-1, batch["raw_actions"].shape[-1]),
                "log_probs": batch["log_probs"].reshape(-1),
                "values": batch["values"].reshape(-1),
                "rewards": batch["rewards"].reshape(-1),
                "dones": batch["dones"].reshape(-1),
                "advantages": advantages.reshape(-1),
                "returns": returns.reshape(-1),
            }

            stats = agent.update(batch_flat)

            mean_return = float(np.mean(finished_returns)) if finished_returns else float("nan")
            mean_len = float(np.mean(finished_lengths)) if finished_lengths else float("nan")
            success_rate = float(np.mean(finished_successes)) if finished_successes else 0.0

            recent_success_rates.append(success_rate)
            recent_mean_lengths.append(mean_len)

            success_rate_window = float(np.mean(recent_success_rates))
            length_window = float(np.mean(recent_mean_lengths))

            best_success_window = max(best_success_window, success_rate_window)
            best_length_window = min(best_length_window, length_window)

            print(
                f"[{stage_name:>8} | update {update:04d}] "
                f"episodes={len(finished_returns):3d} "
                f"mean_return={mean_return:8.3f} "
                f"mean_len={mean_len:6.1f} "
                f"success_rate={success_rate:6.3f} "
                f"success_win={success_rate_window:6.3f} "
                f"len_win={length_window:6.1f} "
                f"policy_loss={stats['policy_loss']:.4f} "
                f"value_loss={stats['value_loss']:.4f} "
                f"entropy={stats['entropy']:.4f} "
                f"approx_kl={stats['approx_kl']:.6f}"
            )

            if update % save_every == 0:
                ckpt_path = os.path.join(save_dir, f"ppo_{stage_name}_update_{update:04d}.pt")
                agent.save(ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            if (
                len(recent_success_rates) == success_window
                and len(recent_mean_lengths) == success_window
                and success_rate_window >= success_threshold
                and length_window <= length_threshold
            ):
                print(
                    f"\nEarly stopping stage '{stage_name}' "
                    f"because success_win={success_rate_window:.3f} >= {success_threshold:.3f} "
                    f"and len_win={length_window:.1f} <= {length_threshold:.1f}\n"
                )
                break

    finally:
        envs.close()

    final_stage_path = os.path.join(save_dir, f"ppo_{stage_name}_final.pt")
    agent.save(final_stage_path)
    print(f"Saved final stage checkpoint: {final_stage_path}")
    print(
        f"Best rolling stats for '{stage_name}': "
        f"best_success_win={best_success_window:.3f}, "
        f"best_len_win={best_length_window:.1f}"
    )

    return agent


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    mp.set_start_method("spawn", force=True)

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    obs_dim = 7
    act_dim = 3

    stages = [
        ("wall_2", 20, 0.90, 105.0),
        ("wall_4", 50, 0.85, 180.0),
        ("wall_8", 50, 0.75, 200.0),
        ("wall_12", 100, 0.65, 300.0),
        ("wall_16", 100, 0.65, 350.0),
    ]

    num_envs = 8
    rollout_steps = 256   # 8 * 256 = 2048 samples per update
    save_every = 25
    success_window = 5

    agent = PPOAgent(
        PPOConfig(
            obs_dim=obs_dim,
            act_dim=act_dim,
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

    for i, (stage_name, total_updates, success_threshold, length_threshold) in enumerate(stages):
        agent = run_stage(
            agent=agent,
            stage_name=stage_name,
            total_updates=total_updates,
            rollout_steps=rollout_steps,
            num_envs=num_envs,
            save_every=save_every,
            save_dir=save_dir,
            success_threshold=success_threshold,
            length_threshold=length_threshold,
            success_window=success_window,
            seed=i,
        )

    final_path = os.path.join(save_dir, "ppo_blend_final.pt")
    agent.save(final_path)

    print("\n" + "=" * 80)
    print(f"Saved final curriculum checkpoint: {final_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
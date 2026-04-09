from __future__ import annotations
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from RL.fish_wrapper import FishBlendEnv
from RL.ppo_agent import PPOAgent, PPOConfig


def main():
    checkpoint_path = "checkpoints/ppo_blend_final.pt"
    max_eval_steps = 1200
    sleep_s = 0.01   # animation pacing

    # Same obs/action dimensions as training
    agent = PPOAgent(
        PPOConfig(
            obs_dim=7,
            act_dim=3,
            hidden_dim=128,
            device="cpu",   # set "cuda" if you trained on GPU and want that here too
        )
    )
    agent.load(checkpoint_path)
    agent.net.eval()

    # Animated env
    env = FishBlendEnv(
        seed=123,
        do_animation=True,
        visit_grid_res=24,
    )

    obs = env.reset(seed=123)

    # Access the underlying animated figure from FishGoalEnv
    fig = env.env.fig if hasattr(env.env, "fig") else None

    weight_text = None
    info_text = None
    if fig is not None:
        weight_text = fig.text(
            0.02, 0.95, "",
            fontsize=11,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        info_text = fig.text(
            0.02, 0.84, "",
            fontsize=10,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    episode_return = 0.0

    for step in range(max_eval_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            # deterministic evaluation:
            # use actor mean -> softmax
            dist = agent.net.get_policy_dist(obs_t)
            raw_action = dist.mean
            blend_weights = torch.softmax(raw_action, dim=-1)

        weights = blend_weights.squeeze(0).cpu().numpy()
        obs, reward, done, info = env.step(weights)
        episode_return += reward

        w_group, w_goal, w_explore = weights.tolist()

        # Print to terminal
        print(
            f"step={step:04d} | "
            f"group={w_group:.3f} "
            f"goal={w_goal:.3f} "
            f"explore={w_explore:.3f} | "
            f"goal_dist={info['goal_distance']:.3f} | "
            f"obs%={100.0*info['observed_percentage']:.2f}"
        )

        # Overlay weights on animation
        if weight_text is not None:
            weight_text.set_text(
                "Blend weights\n"
                f"Grouped:     {w_group:.3f}\n"
                f"Goal:        {w_goal:.3f}\n"
                f"Exploration: {w_explore:.3f}"
            )

        if info_text is not None:
            info_text.set_text(
                "State\n"
                f"Goal distance: {info['goal_distance']:.3f}\n"
                f"Goal progress: {info['goal_progress']:.3f}\n"
                f"Observed %:    {100.0*info['observed_percentage']:.2f}\n"
                f"Cohesion:      {info['cohesion']:.3f}\n"
                f"Dispersion:    {info['dispersion']:.3f}\n"
                f"Alignment:     {info['alignment']:.3f}\n"
                f"Blocked frac:  {info['blocked_fraction']:.3f}\n"
                f"Alive frac:    {info['alive_fraction']:.3f}\n"
                f"Return:        {episode_return:.3f}"
            )

        if fig is not None:
            plt.pause(0.001)

            if not plt.fignum_exists(fig.number):
                print("Figure closed by user.")
                break

        time.sleep(sleep_s)

    if fig is not None:
        plt.show()


if __name__ == "__main__":
    main()
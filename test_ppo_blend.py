from __future__ import annotations
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from RL.fish_wrapper import FishBlendEnv
from RL.ppo_agent import PPOAgent, PPOConfig


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def main():
    checkpoint_path = "checkpoints/ppo_blend_final.pt"
    max_eval_steps = 1200
    sleep_s = 0.01

    agent = PPOAgent(
        PPOConfig(
            obs_dim=7,
            act_dim=6,   # 3 blend outputs + 3 relative goal outputs
            hidden_dim=128,
            device="cpu",
        )
    )
    agent.load(checkpoint_path)
    agent.net.eval()

    env = FishBlendEnv(
        seed=123,
        do_animation=True,
        visit_grid_res=24,
    )

    obs = env.reset(seed=123)

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
            0.02, 0.78, "",
            fontsize=10,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    episode_return = 0.0

    # History
    step_hist = []
    group_hist = []
    goal_hist = []
    explore_hist = []

    rel_goal_x_hist = []
    rel_goal_y_hist = []
    rel_goal_z_hist = []

    local_goal_x_hist = []
    local_goal_y_hist = []
    local_goal_z_hist = []

    for step in range(max_eval_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            dist = agent.net.get_policy_dist(obs_t)
            raw_action_t = dist.mean

        raw_action = raw_action_t.squeeze(0).cpu().numpy()

        # Decode only for display
        blend_logits = raw_action[:3]
        rel_goal_raw = raw_action[3:]

        blend_weights = softmax_np(blend_logits)
        w_group, w_goal, w_explore = blend_weights.tolist()

        obs, reward, done, info = env.step(raw_action)
        episode_return += reward

        # History
        step_hist.append(step)
        group_hist.append(w_group)
        goal_hist.append(w_goal)
        explore_hist.append(w_explore)

        rel_goal_x_hist.append(rel_goal_raw[0])
        rel_goal_y_hist.append(rel_goal_raw[1])
        rel_goal_z_hist.append(rel_goal_raw[2])

        local_goal_x_hist.append(info.get("local_goal_x", np.nan))
        local_goal_y_hist.append(info.get("local_goal_y", np.nan))
        local_goal_z_hist.append(info.get("local_goal_z", np.nan))

        print(
            f"step={step:04d} | "
            f"group={w_group:.3f} "
            f"goal={w_goal:.3f} "
            f"explore={w_explore:.3f} | "
            f"rel_goal=({rel_goal_raw[0]:.3f}, {rel_goal_raw[1]:.3f}, {rel_goal_raw[2]:.3f}) | "
            f"local_goal=({info.get('local_goal_x', np.nan):.3f}, "
            f"{info.get('local_goal_y', np.nan):.3f}, "
            f"{info.get('local_goal_z', np.nan):.3f}) | "
            f"goal_dist={info['goal_distance']:.3f} | "
            f"obs%={100.0 * info['observed_percentage']:.2f}"
        )

        if weight_text is not None:
            weight_text.set_text(
                "Blend weights\n"
                f"Grouped:     {w_group:.3f}\n"
                f"Goal:        {w_goal:.3f}\n"
                f"Exploration: {w_explore:.3f}\n\n"
                "Relative goal output\n"
                f"dx:          {rel_goal_raw[0]:.3f}\n"
                f"dy:          {rel_goal_raw[1]:.3f}\n"
                f"dz:          {rel_goal_raw[2]:.3f}"
            )

        if info_text is not None:
            info_text.set_text(
                "State\n"
                f"Goal distance: {info['goal_distance']:.3f}\n"
                f"Goal progress: {info['goal_progress']:.3f}\n"
                f"Observed %:    {100.0 * info['observed_percentage']:.2f}\n"
                f"Cohesion:      {info['cohesion']:.3f}\n"
                f"Dispersion:    {info['dispersion']:.3f}\n"
                f"Alignment:     {info['alignment']:.3f}\n"
                f"Blocked frac:  {info['blocked_fraction']:.3f}\n"
                f"Alive frac:    {info['alive_fraction']:.3f}\n"
                f"Local goal x:  {info.get('local_goal_x', np.nan):.3f}\n"
                f"Local goal y:  {info.get('local_goal_y', np.nan):.3f}\n"
                f"Local goal z:  {info.get('local_goal_z', np.nan):.3f}\n"
                f"Return:        {episode_return:.3f}"
            )

        if fig is not None:
            plt.pause(0.001)
            if not plt.fignum_exists(fig.number):
                print("Figure closed by user.")
                break

        if done:
            print("Episode finished.")
            break

        time.sleep(sleep_s)

    # Plot 1: blend weights
    plt.figure(figsize=(10, 5))
    plt.plot(step_hist, group_hist, label="Grouped")
    plt.plot(step_hist, goal_hist, label="Goal")
    plt.plot(step_hist, explore_hist, label="Exploration")
    plt.xlabel("Step")
    plt.ylabel("Weight")
    plt.title("Evolution of blend weights")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: raw relative goal outputs
    plt.figure(figsize=(10, 5))
    plt.plot(step_hist, rel_goal_x_hist, label="rel_goal_x_raw")
    plt.plot(step_hist, rel_goal_y_hist, label="rel_goal_y_raw")
    plt.plot(step_hist, rel_goal_z_hist, label="rel_goal_z_raw")
    plt.xlabel("Step")
    plt.ylabel("Raw network output")
    plt.title("Evolution of relative goal outputs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 3: actual local goal used by env
    plt.figure(figsize=(10, 5))
    plt.plot(step_hist, local_goal_x_hist, label="local_goal_x")
    plt.plot(step_hist, local_goal_y_hist, label="local_goal_y")
    plt.plot(step_hist, local_goal_z_hist, label="local_goal_z")
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.title("Evolution of local intermediate goal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
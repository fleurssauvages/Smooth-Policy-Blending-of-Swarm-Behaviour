import numpy as np
import pickle
from Env.env import FishGoalEnv, make_sphere_mesh
import time

def evaluate_theta(env, theta, seeds):
    """Evaluate one parameter vector across multiple seeds."""
    total_R = 0.0
    for seed in seeds:
        env.reset(seed=seed)
        _, R, _, _, _ = env.step(theta.astype(np.float32))
        total_R += R
    return total_R / len(seeds)


def sample_theta(center, std, low=0.0, high=20.0):
    """Sample one candidate around the current center."""
    theta = center + np.random.randn(center.size) * std
    return np.clip(theta, low, high).astype(np.float32)


def main():
    t0 = time.time()
    # --- Simulation Parameters ---
    boid_count = 500
    max_steps = 300
    dt = 1.0

    # --- Monte Carlo Parameters ---
    iters = 5
    samples_per_iter = 50
    eval_episodes = 2
    elite_frac = 0.2
    seed0 = 0

    # Reward weights
    w_goal, w_time, w_diversity, w_coh, w_dis, w_ali = -3.0, 0.0, 0.0, 10.0, 0.0, 5.0 # Grouped Roam
    w_goal, w_time, w_diversity, w_coh, w_dis, w_ali = 0.0, 0.0, 0.0, 0.0, 10.0, 0.0 # Random Exploration
    w_goal, w_time, w_diversity, w_coh, w_dis, w_ali = 10.0, 0.5, 3.0, 0.0, 0.0, 0.0 # Goal-reaching
    name = "goal"

    # Initial parameter center
    theta0 = np.array([
        1.0,  # sep weight
        1.0,  # ali weight
        1.0,  # coh weight
        1.0,  # boundary
        1.0,  # random
        1.0,  # obstacle
        0.3,  # goal
        1.5,  # sep radius
        4.0,  # ali radius
        5.5,  # coh radius
    ], dtype=np.float32)

    # Sampling std
    exploration_std = np.array([
        0.5, 0.5, 0.5, 0.5, 2.0, 0.1, 0.1,
        0.5, 0.5, 0.5
    ], dtype=np.float32)

    # Environment
    verts, faces = make_sphere_mesh(
        R=3.0,
        seg_theta=24,
        seg_phi=24,
        center=(20.0, 20.0, 20.0)
    )

    goals = np.array([
        [34.0, 20.0, 20.0],  # 0 - initial
        [40.0, 20.0, 20.0],  # 1
        [40.0, 30.0, 20.0],  # 2
        [40.0, 10.0, 20.0],  # 3
    ], dtype=np.float32)

    goal_W = np.array([
        [0.0, 2.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

    env = FishGoalEnv(
        boid_count=boid_count,
        max_steps=max_steps,
        dt=dt,
        verts=verts,
        faces=faces,
        goals=goals,
        goal_W=goal_W,
        w_goal=w_goal,
        w_time=w_time,
        w_div=w_diversity,
        w_coh=w_coh,
        w_dis=w_dis,
        w_ali=w_ali,
    )

    # Search state
    current_center = theta0.copy()
    current_std = exploration_std.copy()

    best_R = -1e18
    best_theta = theta0.copy()

    for it in range(iters):
        candidates = []
        returns = []

        # Include current center explicitly
        center_seeds = [seed0 + 100000 * it + r for r in range(eval_episodes)]
        R_center = evaluate_theta(env, current_center, center_seeds)
        candidates.append(current_center.copy())
        returns.append(R_center)

        if R_center > best_R:
            best_R = R_center
            best_theta = current_center.copy()

        # Monte Carlo sampling
        for k in range(samples_per_iter):
            theta_k = sample_theta(current_center, current_std)

            seeds = [
                seed0 + 100000 * it + 1000 * k + r
                for r in range(eval_episodes)
            ]
            R_k = evaluate_theta(env, theta_k, seeds)

            candidates.append(theta_k)
            returns.append(R_k)

            if R_k > best_R:
                best_R = R_k
                best_theta = theta_k.copy()

        candidates = np.array(candidates, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)

        # Elite selection
        n_elite = max(1, int(elite_frac * len(candidates)))
        elite_idx = np.argsort(returns)[-n_elite:]
        elite_thetas = candidates[elite_idx]

        # Update sampling distribution
        current_center = np.mean(elite_thetas, axis=0)

        # Keep fixed-zero std dimensions at zero
        new_std = np.std(elite_thetas, axis=0)
        current_std = np.where(exploration_std > 0.0, np.maximum(0.05, new_std), 0.0)

        print(
            f"[it {it+1:04d}] "
            f"iter_best={returns.max(): .4f}  "
            f"global_best={best_R: .4f}"
        )

    print("==== DONE ====")
    print("Best return:", best_R)
    print("Best theta:", best_theta)
    t1 = time.time()
    print(f"Total time: {t1 - t0:.2f} seconds")
    
    pickle.dump(
        {"best_theta": best_theta, "best_return": best_R},
        open("save/"+name+".pkl", "wb")
    )

    env = FishGoalEnv(
        boid_count=boid_count,
        max_steps=max_steps,
        dt=dt,
        verts=verts,
        faces=faces,
        goals=goals,
        goal_W=goal_W,
        w_goal=w_goal,
        w_time=w_time,
        w_div=w_diversity,
        doAnimation=True,
    )
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(best_theta)

if __name__ == "__main__":
    main()
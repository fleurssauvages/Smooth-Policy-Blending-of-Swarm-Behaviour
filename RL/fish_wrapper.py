# fish_rl_wrapper.py
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np

from Env.env import FishGoalEnv, make_torus_mesh, make_sphere_mesh, make_parallelepiped_mesh, merge_meshes
from RL.swarm_features import update_visited_grid, extract_features


@dataclass
class RewardConfig:
    w_goal_progress: float = 1.0
    w_goal_far: float = 2.0          # extra goal reward when far
    w_novelty: float = 5.0
    w_dispersion_escape: float = 1.0
    w_blocked: float = 0.2
    w_collision: float = 0.5
    w_backward: float = 1.0
    w_blend_smooth: float = 0.1
    blend_change_free: float = 0.10
    success_bonus: float = 50.0

    progress_ref: float = 0.01
    dispersion_target: float = 0.12
    dispersion_scale: float = 0.08


class FishBlendEnv:
    """
    PPO wrapper around the interactive rollout-style env.

    Important:
    This wrapper assumes that after calling:
        env.init_rollout(...)
        env.update_action(...)
        env.step_rollout()

    the current rollout state is available through attributes like:
        env.boid_pos
        env.boid_vel
        env.alive

    If your actual attribute names differ, only `_get_rollout_arrays()`
    below needs to be adjusted.
    """

    def __init__(
        self,
        seed: int = 0,
        do_animation: bool = False,
        visit_grid_res: int = 24,
        reward_cfg: RewardConfig | None = None,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reward_cfg = reward_cfg or RewardConfig()
        self.visit_grid_res = int(visit_grid_res)
        self.prev_blend_weights = None

        # Same geometry as your interactive script
        t1 = make_torus_mesh(
            R=6.0,
            r=2.0,
            segR=12,
            segr=12,
            center=(20.0, 20.0, 20.0),
        )
        s1 = make_sphere_mesh(
            R=5.0,
            center=(28.0, 20.0, 20.0),
            seg_phi=8,
            seg_theta=8,
        )
        s2 = make_sphere_mesh(
            R=5.0,
            center=(12.0, 20.0, 20.0),
            seg_phi=8,
            seg_theta=8,
        )
        verts, faces = merge_meshes([t1, s1, s2])

        walls = []

        # # Outer corridor-like constraints
        # walls.append(make_parallelepiped_mesh(size=(28.0, 2.0, 15.0), center=(20.0, 8.0, 20.0)))
        # walls.append(make_parallelepiped_mesh(size=(28.0, 2.0, 15.0), center=(20.0, 32.0, 20.0)))

        # # Internal maze walls
        # walls.append(make_parallelepiped_mesh(size=(2.0, 14.0, 15.0), center=(14.0, 14.0, 20.0)))
        # walls.append(make_parallelepiped_mesh(size=(2.0, 14.0, 15.0), center=(26.0, 26.0, 20.0)))

        # # walls.append(make_parallelepiped_mesh(size=(10.0, 2.0, 15.0), center=(20.0, 14.0, 20.0)))
        # walls.append(make_parallelepiped_mesh(size=(10.0, 2.0, 15.0), center=(20.0, 26.0, 20.0)))

        # # Central blocker with asymmetric passage
        # walls.append(make_parallelepiped_mesh(size=(2.0, 10.0, 15.0), center=(20.0, 20.0, 20.0)))

        # verts, faces = merge_meshes(walls)

        starts = np.array(
            [
                [5.0, 17.0, 20.0],
                [5.0, 23.0, 20.0],
            ],
            dtype=np.float32,
        )
        goals = np.array([[34.0, 20.0, 20.0]], dtype=np.float32)
        goal_W = np.array([[0.0]], dtype=np.float32)

        self.env = FishGoalEnv(
            boid_count=600,
            max_steps=2000,
            dt=1.0,
            doAnimation=do_animation,
            returnTrajectory=False,
            verts=verts,
            faces=faces,
            starts=starts,
            goals=goals,
            goal_W=goal_W,
            goal_radius=0.5,
        )

        # Expert behavior vectors, as in your interactive script
        self.action_group = pickle.load(open("save/free_roam.pkl", "rb"))["best_theta"].astype(np.float32)
        self.action_goal = pickle.load(open("save/goal_opt.pkl", "rb"))["best_theta"].astype(np.float32)
        self.action_explore = pickle.load(open("save/exploration.pkl", "rb"))["best_theta"].astype(np.float32)

        self.expert_actions = np.stack(
            [self.action_group, self.action_goal, self.action_explore],
            axis=0,
        )  # (3, action_dim)

        self.goal = goals[0].copy()
        self.bound = float(self.env.bound)
        self.max_steps = int(self.env.max_steps)

        self.visited_grid = np.zeros(
            (self.visit_grid_res, self.visit_grid_res, self.visit_grid_res),
            dtype=bool,
        )

        self.prev_goal_distance = None
        self.prev_observed_percentage = None
        self.step_count = 0

    def _get_rollout_arrays(self):
        positions = self.env.boid_pos
        velocities = self.env.boid_vel
        alive = self.env._alive
        return positions, velocities, alive

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is None:
            seed = int(self.rng.integers(0, 2**31 - 1))

        self.env.reset(seed=seed)
        self.env.init_rollout(self.action_group)
        self.prev_blend_weights = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        self.visited_grid.fill(False)
        self.prev_goal_distance = None
        self.prev_observed_percentage = None
        self.step_count = 0

        positions, velocities, alive = self._get_rollout_arrays()
        positions_alive = positions[alive]
        velocities_alive = velocities[alive]

        self.visited_grid = update_visited_grid(
            self.visited_grid,
            positions_alive,
            bound=self.bound,
        )

        obs, aux = extract_features(
            positions=positions_alive,
            velocities=velocities_alive,
            goal=self.goal,
            visited_grid=self.visited_grid,
            bound=self.bound,
            prev_goal_distance=None,
            prev_observed_percentage=None,
        )

        self.prev_goal_distance = aux["goal_distance"]
        self.prev_observed_percentage = aux["observed_percentage"]
        return obs

    def step(self, blend_weights: np.ndarray):
        blend_weights = np.asarray(blend_weights, dtype=np.float32)
        blend_weights = np.clip(blend_weights, 1e-8, None)
        blend_weights /= blend_weights.sum()

        action = (blend_weights[:, None] * self.expert_actions).sum(axis=0).astype(np.float32)
        self.env.update_action(action)
        self.env.step_rollout()
        self.step_count += 1

        positions, velocities, alive = self._get_rollout_arrays()
        positions_alive = positions[alive]
        velocities_alive = velocities[alive]

        self.visited_grid = update_visited_grid(
            self.visited_grid,
            positions_alive,
            bound=self.bound,
        )

        obs, aux = extract_features(
            positions=positions_alive,
            velocities=velocities_alive,
            goal=self.goal,
            visited_grid=self.visited_grid,
            bound=self.bound,
            prev_goal_distance=self.prev_goal_distance,
            prev_observed_percentage=self.prev_observed_percentage,
        )

        reward = self._compute_reward(aux, alive, positions_alive, blend_weights)
        self.prev_blend_weights = blend_weights.copy()
        done, success = self._compute_done_success(positions_alive, alive)

        self.prev_goal_distance = aux["goal_distance"]
        self.prev_observed_percentage = aux["observed_percentage"]

        info = {
            "success": bool(success),
            "goal_distance": float(aux["goal_distance"]),
            "goal_progress": float(aux["goal_progress"]),
            "observed_percentage": float(aux["observed_percentage"]),
            "novelty_gain": float(aux["novelty_gain"]),
            "cohesion": float(aux["cohesion"]),
            "dispersion": float(aux["dispersion"]),
            "alignment": float(aux["alignment"]),
            "blocked_fraction": float(aux["blocked_fraction"]),
            "alive_fraction": float(np.mean(alive.astype(np.float32))),
            "blend_group": float(blend_weights[0]),
            "blend_goal": float(blend_weights[1]),
            "blend_explore": float(blend_weights[2]),
        }
        return obs, reward, done, info

    def _compute_reward(
        self,
        aux: dict,
        alive: np.ndarray,
        positions_alive: np.ndarray,
        blend_weights: np.ndarray | None = None,
    ) -> float:
        cfg = self.reward_cfg

        dead_fraction = 1.0 - float(np.mean(alive.astype(np.float32)))

        goal_progress = float(aux["goal_progress"])
        goal_distance = float(aux["goal_distance"])
        novelty_gain = float(aux["novelty_gain"])
        dispersion = float(aux["dispersion"])
        blocked_fraction = float(aux["blocked_fraction"])
        observed_percentage = float(aux.get("observed_percentage", 0.0))

        # -------------------------------------------------
        # 1) Stagnation estimate: 0 = progressing, 1 = stuck
        # -------------------------------------------------
        progress_ref = getattr(cfg, "progress_ref", 0.01)
        progress_pos = max(goal_progress, 0.0)
        stagnation = 1.0 - min(progress_pos / progress_ref, 1.0)
        stagnation = stagnation ** 2

        # -------------------------------------------------
        # 2) Distance-based goal weighting
        # -------------------------------------------------
        # Normalize distance with workspace diagonal
        diag = np.sqrt(3.0) * float(self.bound)
        dist_norm = np.clip(goal_distance / max(diag, 1e-8), 0.0, 1.0)

        # Far from goal -> larger reward for progress
        goal_distance_boost = 1.0 + cfg.w_goal_far * dist_norm

        # Goal progress dominates when not stuck
        goal_term = (
            cfg.w_goal_progress
            * goal_distance_boost
            * (1.0 - stagnation)
            * progress_pos
        )

        # Mild penalty for moving away from the goal
        backward_penalty = cfg.w_backward * max(-goal_progress, 0.0)

        # -------------------------------------------------
        # 3) Novelty becomes more important when stuck
        # -------------------------------------------------
        novelty_term = (
            cfg.w_novelty
            * stagnation
            * novelty_gain
            * (1.0 - observed_percentage)
        )

        # -------------------------------------------------
        # 4) Moderate dispersion can help escape traps
        # -------------------------------------------------
        disp_error = (dispersion - cfg.dispersion_target) / max(cfg.dispersion_scale, 1e-8)
        dispersion_term = (
            cfg.w_dispersion_escape
            * stagnation
            * np.exp(-(disp_error ** 2))
        )

        # -------------------------------------------------
        # 5) Optional smoothness penalty on blend changes
        # -------------------------------------------------
        smooth_penalty = 0.0
        if blend_weights is not None and getattr(self, "prev_blend_weights", None) is not None:
            delta = blend_weights - self.prev_blend_weights
            delta_norm = float(np.linalg.norm(delta))
            excess = max(0.0, delta_norm - cfg.blend_change_free)
            smooth_penalty = cfg.w_blend_smooth * (excess ** 2)

            # when stuck, allow somewhat faster switching
            smooth_penalty *= (1.0 - 0.7 * stagnation)

        # -------------------------------------------------
        # 6) Total reward
        # -------------------------------------------------
        reward = 0.0
        reward += goal_term
        reward += novelty_term
        reward += dispersion_term
        reward -= backward_penalty
        reward -= cfg.w_blocked * blocked_fraction
        reward -= cfg.w_collision * dead_fraction
        reward -= smooth_penalty

        if self._goal_reached(positions_alive):
            reward += cfg.success_bonus

        return float(reward)

    def _goal_reached(self, positions_alive: np.ndarray) -> bool:
        if positions_alive.shape[0] == 0:
            return False
        centroid = positions_alive.mean(axis=0)
        d = np.linalg.norm(centroid - self.goal)
        goal_radius = float(getattr(self.env, "goal_radius", 2.0))
        return bool(d <= goal_radius)

    def _compute_done_success(self, positions_alive: np.ndarray, alive: np.ndarray):
        success = self._goal_reached(positions_alive)
        timeout = self.step_count >= self.max_steps
        everyone_dead = not np.any(alive)
        done = success or timeout or everyone_dead
        return bool(done), bool(success)
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np

from Env.env import FishGoalEnv, make_parallelepiped_mesh, make_torus_mesh, merge_meshes
from RL.swarm_features import update_visited_grid, extract_features


@dataclass
class RewardConfig:
    w_goal_progress: float = 1.0
    w_novelty: float = 0.5
    w_time: float = 0.1
    success_bonus: float = 50.0
    progress_ref: float = 0.1

    blend_smoothing_alpha: float = 0.0
    w_blend_smooth: float = 1.0      


class FishBlendEnv:
    """
    Blend-only PPO wrapper with 3 experts:
      - group
      - goal
      - exploration

    Action space:
      3 logits -> softmax -> [group_weight, goal_weight, explore_weight]

    Scenarios:
      - no wall
      - wall_2
      - wall_4
      - wall_8
      - wall_12
      - wall_16
    """

    def __init__(
        self,
        seed: int = 0,
        do_animation: bool = False,
        visit_grid_res: int = 24,
        reward_cfg: RewardConfig | None = None,
        layout_name: str = "wall_2",
        save_animation: bool = False,
    ):
        self.seed = seed
        self.layout_name = layout_name
        self.rng = np.random.default_rng(seed)
        self.reward_cfg = reward_cfg or RewardConfig()
        self.visit_grid_res = int(visit_grid_res)

        verts, faces = self._build_layout(self.layout_name)

        starts = np.array(
            [
                [2.0, 20.0, 20.0],
            ],
            dtype=np.float32,
        )
        goals = np.array([[38.0, 20.0, 20.0]], dtype=np.float32)
        goal_W = np.array([[0.0]], dtype=np.float32)

        self.env = FishGoalEnv(
            boid_count=600,
            max_steps=600,
            dt=1.0,
            doAnimation=do_animation,
            saveAnimation=save_animation,
            returnTrajectory=False,
            verts=verts,
            faces=faces,
            starts=starts,
            goals=goals,
            goal_W=goal_W,
            goal_radius=4.0,
        )

        # Group + goal + exploration experts
        self.action_group = pickle.load(open("save/grouped_roam.pkl", "rb"))["best_theta"].astype(np.float32)
        self.action_goal = pickle.load(open("save/goal.pkl", "rb"))["best_theta"].astype(np.float32)
        self.action_explore = pickle.load(open("save/random_roam.pkl", "rb"))["best_theta"].astype(np.float32)

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
        self.prev_weights = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def _build_layout(self, layout_name: str):
        """
        Single square obstacle between start and goal.
        Side lengths increase with scenario difficulty.
        """
        size_map = {
            "no_wall": 0.0,
            "wall_2": 2.0,
            "wall_4": 4.0,
            "wall_8": 8.0,
            "wall_12": 12.0,
            "wall_16": 16.0,
            "test": 16.0,
        }

        if layout_name not in size_map:
            raise ValueError(
                f"Unknown layout_name: {layout_name}. "
                f"Expected one of {list(size_map.keys())}"
            )

        side = size_map[layout_name]

        if side == 0.0:
            return None, None

        wall = make_parallelepiped_mesh(
            size=(1.0, side, side),
            center=(20.0, 20.0, 20.0),
        )

        verts, faces = merge_meshes([wall])

        if layout_name == "test":
            walls = []
            walls.append(make_parallelepiped_mesh(
                size=(1.0, 16.0, 16.0),
                center=(20.0, 25.0, 15.0),
            ))
            walls.append(make_parallelepiped_mesh(
                size=(1.0, 16.0, 16.0),
                center=(10.0, 15.0, 25.0),
            ))
            walls.append(make_parallelepiped_mesh(
                size=(1.0, 16.0, 16.0),
                center=(30.0, 25.0, 20.0),
            ))
            verts, faces = merge_meshes(walls)

        return verts, faces

    def _get_rollout_arrays(self):
        positions = self.env.boid_pos
        velocities = self.env.boid_vel
        alive = self.env._alive
        return positions, velocities, alive

    def _decode_action(self, action3: np.ndarray):
        action3 = np.asarray(action3, dtype=np.float32).reshape(-1)
        if action3.shape[0] != 3:
            raise ValueError(f"Expected action of shape (3,), got {action3.shape}")

        x = action3 - np.max(action3)
        exp_x = np.exp(x)
        weights = exp_x / np.sum(exp_x)

        return weights.astype(np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is None:
            seed = int(self.rng.integers(0, 2**31 - 1))

        self.env.reset(seed=seed)
        self.env.init_rollout(self.action_group)

        self.visited_grid.fill(False)
        self.prev_goal_distance = None
        self.prev_observed_percentage = None
        self.step_count = 0

        # Always use the global goal directly
        self.env.update_goal(
            goals=self.goal[None, :],
            goal_idx=np.zeros(self.env.boid_pos.shape[0], dtype=np.int32),
            goal_gain=1.0,
        )

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
        self.prev_weights = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        return obs.astype(np.float32)

    def _smooth_weights(self, target_weights: np.ndarray) -> np.ndarray:
        alpha = float(self.reward_cfg.blend_smoothing_alpha)
        alpha = np.clip(alpha, 1e-4, 1.0)

        smoothed = (1.0 - alpha) * self.prev_weights + alpha * target_weights
        smoothed = np.clip(smoothed, 1e-8, None)
        smoothed = smoothed / np.sum(smoothed)
        return smoothed.astype(np.float32)
    
    def step(self, action3: np.ndarray):
        target_weights = self._decode_action(action3)
        weights = self._smooth_weights(target_weights)
        blend_delta = weights - self.prev_weights

        # Keep simulator goal fixed to the global goal
        self.env.update_goal(
            goals=self.goal[None, :],
            goal_idx=np.zeros(self.env.boid_pos.shape[0], dtype=np.int32),
            goal_gain=1.0,
        )

        action = (weights[:, None] * self.expert_actions).sum(axis=0).astype(np.float32)
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

        reward = self._compute_reward(aux, positions_alive, blend_delta=blend_delta)
        done, success, frac_goal = self._compute_done_success(positions_alive, alive)

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
            "weight_group": float(weights[0]),
            "weight_goal": float(weights[1]),
            "weight_explore": float(weights[2]),
            "frac_goal": float(frac_goal),
        }
        self.prev_weights = weights.copy()
        return obs.astype(np.float32), reward, done, info

    def _compute_reward(
        self,
        aux: dict,
        positions_alive: np.ndarray,
        blend_delta: np.ndarray | None = None,
    ) -> float:
        cfg = self.reward_cfg

        goal_progress = float(aux["goal_progress"])
        novelty_gain = float(aux["novelty_gain"])
        observed_percentage = float(aux.get("observed_percentage", 0.0))

        progress_pos = max(goal_progress, 0.0)

        # Stagnation / stuck estimate
        stagnation = 1.0 - min(progress_pos / max(cfg.progress_ref, 1e-8), 1.0)
        stagnation = stagnation ** 2

        # Reward forward progress
        goal_term = cfg.w_goal_progress * progress_pos

        # Reward novelty only when stuck
        novelty_term = (
            cfg.w_novelty
            * stagnation
            * novelty_gain
            * (1.0 - observed_percentage)
        )

        # Penalize large changes in blend weights to encourage smoother transitions
        blend_smooth_penalty = 0.0
        if blend_delta is not None:
            blend_smooth_penalty = cfg.w_blend_smooth * float(np.sum(blend_delta ** 2))

        # Penalize time
        time_penalty = cfg.w_time

        reward = 0.0
        reward += goal_term
        reward += novelty_term
        reward -= time_penalty
        reward -= blend_smooth_penalty
        reward += cfg.success_bonus * self._goal_fraction(positions_alive)

        return float(reward)

    def _goal_fraction(self, positions_alive: np.ndarray) -> float:
        if positions_alive.shape[0] == 0:
            return 0.0

        goal_radius = float(getattr(self.env, "goal_radius", 4.0))
        dists = np.linalg.norm(positions_alive - self.goal[None, :], axis=1)
        return float(np.mean(dists <= goal_radius))
    
    def _compute_done_success(self, positions_alive: np.ndarray, alive: np.ndarray):
        frac_goal = self._goal_fraction(positions_alive)
        timeout = self.step_count >= self.max_steps
        everyone_dead = not np.any(alive)

        success = frac_goal > 0.5
        done = success or timeout or everyone_dead
        return bool(done), bool(success), float(frac_goal)
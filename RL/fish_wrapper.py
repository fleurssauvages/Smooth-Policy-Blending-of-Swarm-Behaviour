from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np

from Env.env import FishGoalEnv, make_parallelepiped_mesh, merge_meshes
from RL.swarm_features import update_visited_grid, extract_features


@dataclass
class RewardConfig:
    w_goal_progress: float = 2.0
    w_novelty: float = 3.0
    w_blend_smooth: float = 0.1
    success_bonus: float = 50.0


class FishBlendEnv:
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

        # size of relative intermediate goal around centroid
        self.local_goal_scale = np.array([8.0, 8.0, 4.0], dtype=np.float32)
        self.goal_gain = 1.0

        walls = []
        walls.append(make_parallelepiped_mesh(size=(28.0, 2.0, 15.0), center=(20.0, 8.0, 20.0)))
        walls.append(make_parallelepiped_mesh(size=(28.0, 2.0, 15.0), center=(20.0, 32.0, 20.0)))
        walls.append(make_parallelepiped_mesh(size=(2.0, 14.0, 15.0), center=(14.0, 14.0, 20.0)))
        walls.append(make_parallelepiped_mesh(size=(2.0, 14.0, 15.0), center=(26.0, 26.0, 20.0)))
        walls.append(make_parallelepiped_mesh(size=(10.0, 2.0, 15.0), center=(20.0, 26.0, 20.0)))
        walls.append(make_parallelepiped_mesh(size=(2.0, 10.0, 15.0), center=(20.0, 20.0, 20.0)))
        verts, faces = merge_meshes(walls)

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

        self.action_group = pickle.load(open("save/free_roam.pkl", "rb"))["best_theta"].astype(np.float32)
        self.action_goal = pickle.load(open("save/goal_opt.pkl", "rb"))["best_theta"].astype(np.float32)
        self.action_explore = pickle.load(open("save/exploration.pkl", "rb"))["best_theta"].astype(np.float32)

        self.expert_actions = np.stack(
            [self.action_group, self.action_goal, self.action_explore],
            axis=0,
        )

        self.goal = goals[0].copy()          # global goal used for reward
        self.current_local_goal = self.goal.copy()
        self.prev_local_goal = None

        # smoothness: 0 -> keep previous, 1 -> use new candidate directly
        self.local_goal_alpha = 0.2

        # optional additional hard cap on local goal displacement per step
        self.max_local_goal_step = 1.5

        # enforce strict monotonic decrease to global goal
        self.local_goal_min_improvement = 1e-3

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

    def _clip_point_to_bounds(self, p: np.ndarray) -> np.ndarray:
        return np.clip(p, 0.0, self.bound).astype(np.float32)

    def _smooth_local_goal(self, candidate_goal: np.ndarray) -> np.ndarray:
        candidate_goal = np.asarray(candidate_goal, dtype=np.float32)

        if self.prev_local_goal is None:
            return candidate_goal.copy()

        # Exponential smoothing
        smoothed = (
            (1.0 - self.local_goal_alpha) * self.prev_local_goal
            + self.local_goal_alpha * candidate_goal
        ).astype(np.float32)

        # Optional hard cap on goal displacement per step
        delta = smoothed - self.prev_local_goal
        delta_norm = np.linalg.norm(delta)
        if delta_norm > self.max_local_goal_step and delta_norm > 1e-8:
            smoothed = self.prev_local_goal + delta / delta_norm * self.max_local_goal_step

        return smoothed.astype(np.float32)


    def _project_goal_toward_global(self, goal_candidate: np.ndarray) -> np.ndarray:
        """
        Enforce:
            ||goal_t - global_goal|| <= ||goal_{t-1} - global_goal|| - min_improvement
        """
        goal_candidate = np.asarray(goal_candidate, dtype=np.float32)
        global_goal = self.goal.astype(np.float32)

        if self.prev_local_goal is None:
            return goal_candidate.copy()

        prev_dist = np.linalg.norm(self.prev_local_goal - global_goal)
        cand_vec = goal_candidate - global_goal
        cand_dist = np.linalg.norm(cand_vec)

        target_max_dist = max(0.0, prev_dist - self.local_goal_min_improvement)

        # already valid
        if cand_dist <= target_max_dist:
            return goal_candidate.copy()

        # if candidate exactly at global goal direction is undefined
        if cand_dist < 1e-8:
            return goal_candidate.copy()

        # project onto sphere centered at global goal
        projected = global_goal + cand_vec / cand_dist * target_max_dist
        return projected.astype(np.float32)

    def _decode_action(self, action6: np.ndarray):
        action6 = np.asarray(action6, dtype=np.float32).reshape(-1)
        if action6.shape[0] != 6:
            raise ValueError(f"Expected action of shape (6,), got {action6.shape}")

        blend_logits = action6[:3]
        rel_goal_raw = action6[3:]

        # stable softmax
        x = blend_logits - np.max(blend_logits)
        exp_x = np.exp(x)
        blend_weights = exp_x / np.sum(exp_x)

        rel_goal = np.tanh(rel_goal_raw) * self.local_goal_scale
        return blend_weights.astype(np.float32), rel_goal.astype(np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is None:
            seed = int(self.rng.integers(0, 2**31 - 1))

        self.env.reset(seed=seed)
        self.env.init_rollout(self.action_group)

        self.prev_blend_weights = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.current_local_goal = self.goal.copy()
        self.prev_local_goal = None

        self.visited_grid.fill(False)
        self.prev_goal_distance = None
        self.prev_observed_percentage = None
        self.step_count = 0

        positions, velocities, alive = self._get_rollout_arrays()
        positions_alive = positions[alive]
        velocities_alive = velocities[alive]

        # initialize local goal to current centroid
        if positions_alive.shape[0] > 0:
            centroid = positions_alive.mean(axis=0).astype(np.float32)
            self.current_local_goal = centroid.copy()
            self.prev_local_goal = centroid.copy()

            self.env.update_goal(
                goals=self.current_local_goal[None, :],
                goal_idx=np.zeros(self.env.boid_pos.shape[0], dtype=np.int32),
                goal_gain=self.goal_gain,
            )
            
        self.visited_grid = update_visited_grid(
            self.visited_grid,
            positions_alive,
            bound=self.bound,
        )

        obs, aux = extract_features(
            positions=positions_alive,
            velocities=velocities_alive,
            goal=self.goal,  # keep global goal in observation/reward features
            visited_grid=self.visited_grid,
            bound=self.bound,
            prev_goal_distance=None,
            prev_observed_percentage=None,
        )

        self.prev_goal_distance = aux["goal_distance"]
        self.prev_observed_percentage = aux["observed_percentage"]
        return obs

    def step(self, action6: np.ndarray):
        blend_weights, rel_goal = self._decode_action(action6)

        positions, velocities, alive = self._get_rollout_arrays()
        positions_alive = positions[alive]

        if positions_alive.shape[0] > 0:
            centroid = positions_alive.mean(axis=0).astype(np.float32)
        else:
            centroid = self.goal.copy()

        # candidate from policy output
        candidate_goal = centroid + rel_goal
        candidate_goal = self._clip_point_to_bounds(candidate_goal)

        # 1) smooth it
        local_goal = self._smooth_local_goal(candidate_goal)
        local_goal = self._clip_point_to_bounds(local_goal)

        # 2) force monotonic progress toward global goal
        local_goal = self._project_goal_toward_global(local_goal)
        local_goal = self._clip_point_to_bounds(local_goal)

        self.current_local_goal = local_goal.copy()
        self.prev_local_goal = local_goal.copy()

        self.env.update_goal(
            goals=local_goal[None, :],
            goal_idx=np.zeros(self.env.boid_pos.shape[0], dtype=np.int32),
            goal_gain=self.goal_gain,
        )

        # blend experts as before
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
            goal=self.goal,  # still the global goal
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
            "local_goal_x": float(local_goal[0]),
            "local_goal_y": float(local_goal[1]),
            "local_goal_z": float(local_goal[2]),
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

        goal_progress = float(aux["goal_progress"])
        novelty_gain = float(aux["novelty_gain"])
        observed_percentage = float(aux.get("observed_percentage", 0.0))

        # -------------------------------------------------
        # 1) Stagnation estimate: 0 = progressing, 1 = stuck
        # -------------------------------------------------
        progress_ref = getattr(cfg, "progress_ref", 0.01)
        progress_pos = max(goal_progress, 0.0)

        stagnation = 1.0 - min(progress_pos / progress_ref, 1.0)
        stagnation = stagnation ** 2

        # -------------------------------------------------
        # 2) Goal progress (only positive progress)
        # -------------------------------------------------
        goal_term = cfg.w_goal_progress * progress_pos

        # -------------------------------------------------
        # 3) Novelty (only matters when stuck)
        # -------------------------------------------------
        novelty_term = (
            cfg.w_novelty
            * stagnation
            * novelty_gain
            * (1.0 - observed_percentage)
        )

        # -------------------------------------------------
        # 4) Smoothness penalty on blending
        # -------------------------------------------------
        smooth_penalty = 0.0
        if blend_weights is not None and getattr(self, "prev_blend_weights", None) is not None:
            delta = blend_weights - self.prev_blend_weights
            delta_norm = float(np.linalg.norm(delta))

            # no free threshold anymore → always penalize
            smooth_penalty = cfg.w_blend_smooth * (delta_norm ** 2)

            # allow faster switching when stuck
            smooth_penalty *= (1.0 - 0.7 * stagnation)

        # -------------------------------------------------
        # 5) Total reward
        # -------------------------------------------------
        reward = 0.0
        reward += goal_term
        reward += novelty_term
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
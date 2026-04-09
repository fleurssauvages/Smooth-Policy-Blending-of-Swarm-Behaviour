from __future__ import annotations
import numpy as np

def _safe_mean(x: np.ndarray, axis=None):
    if x.size == 0:
        return 0.0
    return np.mean(x, axis=axis)


def compute_centroid(positions: np.ndarray) -> np.ndarray:
    if positions.size == 0:
        return np.zeros(3, dtype=np.float32)
    return positions.mean(axis=0).astype(np.float32)


def compute_cohesion(positions: np.ndarray) -> float:
    """
    Mean distance to the swarm centroid.
    Lower = tighter group.
    """
    if positions.shape[0] == 0:
        return 0.0
    centroid = compute_centroid(positions)
    d = np.linalg.norm(positions - centroid[None, :], axis=1)
    return float(np.mean(d))


def compute_dispersion(positions: np.ndarray) -> float:
    """
    Standard deviation of distances to centroid.
    Lower = more homogeneous compactness.
    """
    if positions.shape[0] == 0:
        return 0.0
    centroid = compute_centroid(positions)
    d = np.linalg.norm(positions - centroid[None, :], axis=1)
    return float(np.std(d))


def compute_alignment(velocities: np.ndarray, eps: float = 1e-8) -> float:
    """
    Norm of the mean unit velocity.
    Range ~ [0,1].
    1 = everyone moves in same direction.
    """
    if velocities.shape[0] == 0:
        return 0.0
    norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    unit = velocities / np.maximum(norms, eps)
    mean_dir = unit.mean(axis=0)
    return float(np.linalg.norm(mean_dir))


def compute_goal_distance(positions: np.ndarray, goal: np.ndarray) -> float:
    """
    Distance from swarm centroid to current goal.
    """
    if positions.shape[0] == 0:
        return 0.0
    centroid = compute_centroid(positions)
    return float(np.linalg.norm(centroid - goal))


def compute_blocked_fraction(
    velocities: np.ndarray,
    speed_threshold: float = 0.02,
) -> float:
    """
    Fraction of agents with very low speed.
    """
    if velocities.shape[0] == 0:
        return 1.0
    speed = np.linalg.norm(velocities, axis=1)
    return float(np.mean(speed < speed_threshold))


def update_visited_grid(
    visited_grid: np.ndarray,
    positions: np.ndarray,
    bound: float,
) -> np.ndarray:
    """
    Mark visited cells from boid positions.
    visited_grid shape: (R, R, R), dtype=bool
    """
    R = visited_grid.shape[0]
    if positions.shape[0] == 0:
        return visited_grid

    # map [0, bound] -> [0, R-1]
    idx = np.floor((positions / max(bound, 1e-8)) * (R - 1)).astype(np.int32)
    idx = np.clip(idx, 0, R - 1)

    visited_grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return visited_grid


def compute_observed_percentage(visited_grid: np.ndarray) -> float:
    """
    Fraction of workspace cells ever visited.
    """
    return float(np.mean(visited_grid.astype(np.float32)))


def compute_novelty_gain(
    prev_observed_percentage: float,
    current_observed_percentage: float,
) -> float:
    return float(current_observed_percentage - prev_observed_percentage)


def normalize_features(
    cohesion: float,
    dispersion: float,
    alignment: float,
    observed_percentage: float,
    goal_distance: float,
    goal_progress: float,
    blocked_fraction: float,
    bound: float,
) -> np.ndarray:
    """
    Simple normalization to keep PPO stable.
    """
    diag = np.sqrt(3.0) * bound

    cohesion_n = np.clip(cohesion / max(diag, 1e-8), 0.0, 1.0)
    dispersion_n = np.clip(dispersion / max(diag, 1e-8), 0.0, 1.0)
    alignment_n = np.clip(alignment, 0.0, 1.0)
    observed_n = np.clip(observed_percentage, 0.0, 1.0)
    goal_dist_n = np.clip(goal_distance / max(diag, 1e-8), 0.0, 1.0)

    # goal progress can be negative/positive; normalize by bound
    goal_prog_n = np.clip(goal_progress / max(bound, 1e-8), -1.0, 1.0)
    blocked_n = np.clip(blocked_fraction, 0.0, 1.0)

    return np.array(
        [
            cohesion_n,
            dispersion_n,
            alignment_n,
            observed_n,
            goal_dist_n,
            goal_prog_n,
            blocked_n,
        ],
        dtype=np.float32,
    )


def extract_features(
    positions: np.ndarray,
    velocities: np.ndarray,
    goal: np.ndarray,
    visited_grid: np.ndarray,
    bound: float,
    prev_goal_distance: float | None,
    prev_observed_percentage: float | None,
    speed_threshold: float = 0.02,
) -> tuple[np.ndarray, dict]:
    """
    Main feature extractor.

    Returns
    -------
    obs : np.ndarray shape (7,)
    aux : dict
        raw metrics, useful for rewards/logging.
    """
    cohesion = compute_cohesion(positions)
    dispersion = compute_dispersion(positions)
    alignment = compute_alignment(velocities)
    goal_distance = compute_goal_distance(positions, goal)
    blocked_fraction = compute_blocked_fraction(
        velocities, speed_threshold=speed_threshold
    )

    observed_percentage = compute_observed_percentage(visited_grid)

    goal_progress = 0.0 if prev_goal_distance is None else (prev_goal_distance - goal_distance)
    novelty_gain = 0.0 if prev_observed_percentage is None else (
        observed_percentage - prev_observed_percentage
    )

    obs = normalize_features(
        cohesion=cohesion,
        dispersion=dispersion,
        alignment=alignment,
        observed_percentage=observed_percentage,
        goal_distance=goal_distance,
        goal_progress=goal_progress,
        blocked_fraction=blocked_fraction,
        bound=bound,
    )

    aux = {
        "cohesion": float(cohesion),
        "dispersion": float(dispersion),
        "alignment": float(alignment),
        "goal_distance": float(goal_distance),
        "goal_progress": float(goal_progress),
        "blocked_fraction": float(blocked_fraction),
        "observed_percentage": float(observed_percentage),
        "novelty_gain": float(novelty_gain),
    }
    return obs, aux
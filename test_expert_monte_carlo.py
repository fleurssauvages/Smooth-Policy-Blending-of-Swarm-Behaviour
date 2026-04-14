import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from Env.env import FishGoalEnv, make_torus_mesh, make_sphere_mesh, merge_meshes
from controllers.actionblender import ActionPolygonBlender

if __name__ == "__main__":
    t1 = make_torus_mesh(
        R=6.0,
        r=2.0,
        segR=12,
        segr=12,
        center=(20.0, 20.0, 20.0)
    )

    s1 = make_sphere_mesh(
        R=5.0,
        center=(28.0, 20.0, 20.0),
        seg_phi=8,
        seg_theta=8
    )

    s2 = make_sphere_mesh(
        R=5.0,
        center=(12.0, 20.0, 20.0),
        seg_phi=8,
        seg_theta=8
    )

    verts, faces = merge_meshes([t1, s1, s2])

    starts = np.array([
        [5.0, 17.0, 20.0],
        [5.0, 23.0, 20.0]], dtype=np.float32)

    goals = np.array([
        [34.0, 20.0, 20.0],
    ], dtype=np.float32)

    goal_W = np.array([
        [0.0],
    ], dtype=np.float32)

    env = FishGoalEnv(
        boid_count=600,
        max_steps=500,
        dt=1,
        doAnimation=True,
        returnTrajectory=False,
        verts=verts,
        faces=faces,
        starts=starts,
        goals=goals,
        goal_W=goal_W,
    )
    env.reset(seed=0)

    action_goal = pickle.load(open("save/goal.pkl", "rb"))["best_theta"].astype(np.float32)
    action_free_roam = pickle.load(open("save/grouped_roam.pkl", "rb"))["best_theta"].astype(np.float32)
    action_exploratory = pickle.load(open("save/random_roam.pkl", "rb"))["best_theta"].astype(np.float32)

    action_presets = [
        action_free_roam,
        action_goal,
        action_exploratory,
    ]

    labels = [
        "Grouped",
        "Goal",
        "Exploration",
    ]

    for i, a in enumerate(action_presets):
        if a.shape != action_presets[0].shape:
            raise ValueError(f"Action {i} has incompatible shape {a.shape}")

    env.init_rollout(action_presets[0])

    blender = ActionPolygonBlender(
        fig=env.fig,
        actions=action_presets,
        labels=labels,
        on_change=env.update_action,
        ax_rect=(0.0, 0.05, 0.36, 0.36),
    )


    try:
        while True:
            _ = env.step_rollout()

            if not plt.fignum_exists(env.fig.number):
                print("Figure closed -> stopping rollout")
                break

            plt.pause(0.001)

    finally:
        env.reset(seed=0)
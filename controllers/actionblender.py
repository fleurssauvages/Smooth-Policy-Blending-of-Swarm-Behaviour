import numpy as np
from matplotlib.path import Path

class ActionPolygonBlender:
    """
    Draggable polygon-based action blender.

    Each polygon vertex corresponds to one action vector.
    A draggable point inside the polygon defines generalized barycentric
    weights (mean value coordinates), which are used to blend the actions.

    Notes
    -----
    - Works best for convex polygons.
    - By default, vertices are arranged as a regular n-gon.
    - The blended action is:
          action = sum_i w_i * action_i
      with w_i >= 0 and sum_i w_i = 1 for points inside the polygon.
    """

    def __init__(
        self,
        fig,
        actions,
        on_change,
        labels=None,
        ax_rect=(0.72, 0.05, 0.24, 0.24),
        polygon_radius=1.0,
        initial_point=None,
        point_size=80,
    ):
        self.fig = fig
        self.actions = [np.asarray(a, dtype=np.float32) for a in actions]
        self.on_change = on_change
        self.labels = labels
        self.n = len(self.actions)
        self.point_size = point_size

        if self.n < 3:
            raise ValueError("ActionPolygonBlender requires at least 3 actions/vertices.")

        first_shape = self.actions[0].shape
        for i, a in enumerate(self.actions):
            if a.shape != first_shape:
                raise ValueError(
                    f"All actions must have the same shape. "
                    f"Action 0 shape={first_shape}, action {i} shape={a.shape}"
                )

        self.ax = fig.add_axes(ax_rect)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-1.25 * polygon_radius, 1.25 * polygon_radius)
        self.ax.set_ylim(-1.25 * polygon_radius, 1.25 * polygon_radius)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Regular polygon vertices
        angles = np.linspace(0, 2 * np.pi, self.n, endpoint=False) + np.pi / 2
        self.vertices = np.c_[
            polygon_radius * np.cos(angles),
            polygon_radius * np.sin(angles),
        ]

        self.path = Path(self.vertices)

        # Draw polygon
        poly_closed = np.vstack([self.vertices, self.vertices[0]])
        self.ax.plot(poly_closed[:, 0], poly_closed[:, 1], lw=2)

        self.ax.scatter(self.vertices[:, 0], self.vertices[:, 1], s=50)

        if self.labels is not None:
            if len(self.labels) != self.n:
                raise ValueError("labels must have same length as actions")
            for (x, y), label in zip(self.vertices, self.labels):
                self.ax.text(
                    x * 1.10,
                    y * 1.10,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                )

        # Initial draggable point
        if initial_point is None:
            initial_point = np.mean(self.vertices, axis=0)

        initial_point = np.asarray(initial_point, dtype=float)
        if not self.path.contains_point(initial_point):
            initial_point = self._project_to_polygon(initial_point)

        self.point = initial_point
        self.handle = self.ax.scatter(
            [self.point[0]],
            [self.point[1]],
            s=self.point_size,
            zorder=5,
            picker=True,
        )

        self.weight_text = self.ax.text(
            0.02, 0.02, "", transform=self.ax.transAxes,
            ha="left", va="bottom", fontsize=8
        )

        self.dragging = False

        self.cid_press = fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.cid_release = fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.cid_motion = fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

        self._emit_current_action()

    def get_weights(self):
        return self._mean_value_coordinates(self.point, self.vertices)

    def get_action(self):
        w = self.get_weights()
        out = np.zeros_like(self.actions[0], dtype=np.float32)
        for wi, ai in zip(w, self.actions):
            out += wi * ai
        return out.astype(np.float32)

    def _emit_current_action(self):
        weights = self.get_weights()
        action = self.get_action()
        # self._update_label(weights)
        self.on_change(action)

    def _update_label(self, weights):
        txt = "\n".join([f"{i}: {wi:.2f}" for i, wi in enumerate(weights)])
        self.weight_text.set_text(txt)

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        mouse = np.array([event.xdata, event.ydata], dtype=float)
        d = np.linalg.norm(mouse - self.point)
        if d < 0.12:
            self.dragging = True

    def _on_release(self, event):
        self.dragging = False

    def _on_motion(self, event):
        if not self.dragging:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        p = np.array([event.xdata, event.ydata], dtype=float)
        if not self.path.contains_point(p):
            p = self._project_to_polygon(p)

        self.point = p
        self.handle.set_offsets([self.point])
        self._emit_current_action()
        self.fig.canvas.draw_idle()

    def _project_to_polygon(self, p):
        """
        Project a point to the closest point on the polygon boundary.
        """
        best_q = None
        best_d2 = np.inf
        n = len(self.vertices)

        for i in range(n):
            a = self.vertices[i]
            b = self.vertices[(i + 1) % n]
            q = self._project_point_to_segment(p, a, b)
            d2 = np.sum((p - q) ** 2)
            if d2 < best_d2:
                best_d2 = d2
                best_q = q

        return best_q

    @staticmethod
    def _project_point_to_segment(p, a, b):
        ab = b - a
        denom = np.dot(ab, ab)
        if denom <= 1e-12:
            return a.copy()
        t = np.dot(p - a, ab) / denom
        t = np.clip(t, 0.0, 1.0)
        return a + t * ab

    @staticmethod
    def _mean_value_coordinates(p, verts, eps=1e-12):
        """
        Mean value coordinates for a point inside a convex polygon.

        Returns nonnegative weights summing to 1.
        """
        n = len(verts)
        r = verts - p[None, :]
        d = np.linalg.norm(r, axis=1)

        # If point is extremely close to a vertex, snap fully to that vertex
        i_close = np.argmin(d)
        if d[i_close] < 1e-10:
            w = np.zeros(n, dtype=np.float64)
            w[i_close] = 1.0
            return w

        angles = np.zeros(n, dtype=np.float64)
        for i in range(n):
            ri = r[i]
            rj = r[(i + 1) % n]
            cross = ri[0] * rj[1] - ri[1] * rj[0]
            dot = np.dot(ri, rj)
            angles[i] = np.arctan2(cross, dot)

        tan_half = np.tan(angles / 2.0)

        w = np.zeros(n, dtype=np.float64)
        for i in range(n):
            w[i] = (tan_half[i - 1] + tan_half[i]) / max(d[i], eps)

        w_sum = np.sum(w)
        if abs(w_sum) < eps:
            # Fallback: inverse-distance weights
            inv_d = 1.0 / np.maximum(d, eps)
            return inv_d / np.sum(inv_d)

        w /= w_sum

        # Small numerical cleanup
        w = np.maximum(w, 0.0)
        w /= np.sum(w)
        return w

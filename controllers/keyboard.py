import time
import threading
import numpy as np


class Keyboard3D:
    """
    Figure-based keyboard controller.

    Outputs:
      - v_trans : (3,) translation velocity
      - v_rot   : (3,) reserved, always zero for now
      - buttons : (2,) float array for edge detection

    Default mapping:
      W / S         -> +X / -X
      D / A         -> +Y / -Y
      Up / Down     -> +Z / -Z
      Space         -> buttons[0]
      Right shift   -> buttons[1]
    """

    def __init__(
        self,
        fig,
        trans_scale=10.0,
        rate_hz=60.0,
        lowpass=0.0,
    ):
        self.fig = fig
        self.trans_scale = float(trans_scale)
        self.rate_hz = float(rate_hz)
        self.lowpass = float(lowpass)

        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        self._pressed = set()
        self._v_trans = np.zeros(3, dtype=float)
        self._v_rot = np.zeros(3, dtype=float)
        self._buttons = np.zeros(2, dtype=np.float32)

        self._cid_press = None
        self._cid_release = None
        self._cid_close = None

    def start(self):
        if self._running:
            return

        self._cid_press = self.fig.canvas.mpl_connect("key_press_event", self._on_press)
        self._cid_release = self.fig.canvas.mpl_connect("key_release_event", self._on_release)
        self._cid_close = self.fig.canvas.mpl_connect("close_event", self._on_close)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        try:
            if self._cid_press is not None:
                self.fig.canvas.mpl_disconnect(self._cid_press)
            if self._cid_release is not None:
                self.fig.canvas.mpl_disconnect(self._cid_release)
            if self._cid_close is not None:
                self.fig.canvas.mpl_disconnect(self._cid_close)
        except Exception:
            pass

    def read(self):
        with self._lock:
            return self._v_trans.copy(), self._v_rot.copy(), self._buttons.copy()

    def _on_press(self, event):
        if event.key is None:
            return
        with self._lock:
            self._pressed.add(event.key)

    def _on_release(self, event):
        if event.key is None:
            return
        with self._lock:
            self._pressed.discard(event.key)

    def _on_close(self, event):
        self.stop()

    def _loop(self):
        dt = 1.0 / max(self.rate_hz, 1.0)
        alpha = self.lowpass

        while self._running:
            with self._lock:
                pressed = set(self._pressed)

            trans = np.zeros(3, dtype=float)

            # X axis
            if "w" in pressed or "z" in pressed:
                trans[0] += 1.0
            if "s" in pressed:
                trans[0] -= 1.0

            # Y axis (left/right)
            if "d" in pressed:
                trans[1] += 1.0
            if "a" in pressed or "q" in pressed:
                trans[1] -= 1.0

            # Z axis
            if "up" in pressed:
                trans[2] += 1.0
            if "down" in pressed:
                trans[2] -= 1.0

            # normalize diagonal motion
            n = np.linalg.norm(trans)
            if n > 1e-12:
                trans = trans / n

            trans *= self.trans_scale

            buttons = np.zeros(2, dtype=np.float32)
            if " " in pressed or "space" in pressed:
                buttons[0] = 1.0
            if "shift" in pressed or "right shift" in pressed:
                buttons[1] = 1.0

            with self._lock:
                if alpha > 0.0:
                    self._v_trans = (1.0 - alpha) * trans + alpha * self._v_trans
                else:
                    self._v_trans = trans

                self._v_rot[:] = 0.0
                self._buttons = buttons

            time.sleep(dt)
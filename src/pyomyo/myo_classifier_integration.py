from collections import Counter, deque

from .enums import emg_mode
from .pyomyo import Myo


class MyoClassifier(Myo):
    """Adds higher-level pose classification and handling onto Myo."""

    def __init__(self, cls, tty=None, mode=emg_mode.PREPROCESSED, hist_len=25):
        Myo.__init__(self, tty, mode=mode)
        # Add a classifier
        self.cls = cls
        self.hist_len = hist_len
        self.history = deque([0] * self.hist_len, self.hist_len)
        self.history_cnt = Counter(self.history)
        self.add_emg_handler(self.emg_handler)
        self.last_pose = None

        self.pose_handlers = []

    def emg_handler(self, emg, moving):
        y = self.cls.classify(emg)
        self.history_cnt[self.history[0]] -= 1
        self.history_cnt[y] += 1
        self.history.append(y)

        r, n = self.history_cnt.most_common(1)[0]
        if self.last_pose is None or (
            n > self.history_cnt[self.last_pose] + 5 and n > self.hist_len / 2
        ):
            self.on_raw_pose(r)
            self.last_pose = r

    def add_raw_pose_handler(self, h):
        self.pose_handlers.append(h)

    def on_raw_pose(self, pose):
        for h in self.pose_handlers:
            h(pose)

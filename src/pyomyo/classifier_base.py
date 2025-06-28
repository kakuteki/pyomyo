import os
import struct
from collections import Counter, deque

import numpy as np

from .utils import pack

SUBSAMPLE = 3
K = 15


class Classifier(object):
    """A wrapper for nearest-neighbor classifier that stores
    training data in vals0, ..., vals9.dat."""

    def __init__(self, name="Classifier", color=(0, 200, 0)):
        # Add some identifiers to the classifier to identify what model was used in different screenshots
        self.name = name
        self.color = color
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(self.data_dir, exist_ok=True)  # Ensure data directory exists

        for i in range(10):
            with open(os.path.join(self.data_dir, "vals%d.dat" % i), "ab") as f:
                pass
        self.read_data()

    def store_data(self, cls, vals):
        with open(os.path.join(self.data_dir, "vals%d.dat" % cls), "ab") as f:
            f.write(pack("8H", *vals))

        self.train(np.vstack([self.X, vals]), np.hstack([self.Y, [cls]]))

    def read_data(self):
        X = []
        Y = []
        for i in range(10):
            X.append(
                np.fromfile(
                    os.path.join(self.data_dir, "vals%d.dat" % i), dtype=np.uint16
                ).reshape((-1, 8))
            )
            Y.append(i + np.zeros(X[-1].shape[0]))

        self.train(np.vstack(X), np.hstack(Y))

    def delete_data(self):
        for i in range(10):
            with open(os.path.join(self.data_dir, "vals%d.dat" % i), "wb") as f:
                pass
        self.read_data()

    def train(self, X, Y):
        self.X = X
        self.Y = Y
        self.model = None

    def nearest(self, d):
        dists = ((self.X - d) ** 2).sum(1)
        ind = dists.argmin()
        return self.Y[ind]

    def classify(self, d):
        if self.X.shape[0] < K * SUBSAMPLE:
            return 0
        return self.nearest(d)


class Live_Classifier(Classifier):
    """
    General class for all Sklearn classifiers
    Expects something you can call .fit and .predict on
    """

    def __init__(self, classifier, name="Live Classifier", color=(0, 55, 175)):
        self.model = classifier
        Classifier.__init__(self, name=name, color=color)

    def train(self, X, Y):
        self.X = X
        self.Y = Y

        if self.X.shape[0] > 0 and self.Y.shape[0] > 0:
            self.model.fit(self.X, self.Y)

    def classify(self, emg):
        if self.X.shape[0] == 0 or self.model == None:
            # We have no data or model, return 0
            return 0

        x = np.array(emg).reshape(1, -1)
        pred = self.model.predict(x)
        return int(pred[0])

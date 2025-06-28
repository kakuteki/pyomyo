import os
import sys
from collections import deque

import joblib
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from tensorflow.keras.models import load_model

from pyomyo import Myo, emg_mode
from pyomyo.Classifier import EMGHandler


# -----------------------------
# EMG Plot Widget
# -----------------------------
class EMGPlotWidget(pg.PlotWidget):
    def __init__(self, num_channels=8, max_points=100, parent=None):
        super().__init__(parent)
        self.num_channels = num_channels
        self.max_points = max_points
        self.data = np.zeros((num_channels, max_points))
        self.curves = []
        self.setBackground("w")
        self.setYRange(-128, 128)
        self.setXRange(0, max_points)
        for i in range(num_channels):
            pen = pg.mkPen(width=2)
            self.curves.append(self.plot(pen=pen))

    def update_plot(self, emg):
        if len(emg) != self.num_channels:
            return
        self.data[:, :-1] = self.data[:, 1:]
        self.data[:, -1] = emg
        for i, curve in enumerate(self.curves):
            curve.setData(self.data[i])


# -----------------------------
# LSTM Gesture Classifier
# -----------------------------
class LSTMGestureClassifier:
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        encoder_path: str,
        window_size: int = 200,
        step: int = 50,
        threshold: float = 0.7,
    ):
        # パラメータ
        self.window_size = window_size
        self.step = step
        self.threshold = threshold

        # モデル・スケーラー・ラベルエンコーダの読み込み
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)

        # バッファ
        self.emg_buffer = deque(maxlen=self.window_size)
        self.counter = 0

        # 状態
        self.last_prediction = None
        self.last_confidence = 0.0
        self.pred_history = deque(maxlen=5)

    def preprocess_window(self, window: np.ndarray) -> np.ndarray:
        # 入力形状の検証
        if len(window.shape) == 1:  # (8,)
            window = window.reshape(1, -1)  # (1, 8)
        elif len(window.shape) == 2:  # (timesteps, 8)
            pass  # そのまま

        # スケーリング
        scaled = self.scaler.transform(window)

        # モデルの入力形状に合わせてリシェイプ
        # LSTMの場合は (samples, timesteps, features) の形式が必要
        if len(scaled.shape) == 2:
            scaled = scaled.reshape(1, -1, scaled.shape[1])

        return scaled


def classify(self, emg_sample: np.ndarray):
    self.emg_buffer.append(emg_sample)
    self.counter += 1

    # ウィンドウサイズに達していない場合はNoneを返す
    if len(self.emg_buffer) < self.window_size:
        return None

        # stepサイズに応じてサンプリング
        if self.counter % self.step != 0:
            return self.last_prediction

        # ウィンドウデータを取得
        window = np.array(self.emg_buffer[-self.window_size :], dtype=np.float32)

        # 前処理
        X = self.preprocess_window(window)

        # 推論
        proba = self.model.predict(X, verbose=0)[0]
        idx = int(np.argmax(proba))
        conf = float(np.max(proba))

        # 信頼度が閾値未満の場合は前回の予測を保持
        if conf < self.threshold and self.last_prediction is not None:
            return self.last_prediction

        # 予測を更新
        self.last_confidence = conf
        self.pred_history.append(idx)

        # スムージング（移動平均）
        if len(self.pred_history) >= 3:
            idx = int(np.median(self.pred_history))  # 中央値を使用

        self.last_prediction = idx
        return idx


# -----------------------------
# GUI アプリケーション
# -----------------------------
class GestureClassifierApp(QMainWindow):
    def __init__(self, classifier: LSTMGestureClassifier):
        super().__init__()
        self.classifier = classifier
        self.emg_handler = None
        self.init_ui()
        self.init_myo()

    def init_ui(self):
        self.setWindowTitle("LSTM Gesture Classifier")
        self.setGeometry(100, 100, 800, 600)
        w = QWidget()
        self.setCentralWidget(w)
        layout = QVBoxLayout(w)
        self.label_g = QLabel("Gesture: -", self)
        layout.addWidget(self.label_g)
        self.label_c = QLabel("Confidence: -", self)
        layout.addWidget(self.label_c)
        self.plot = EMGPlotWidget(num_channels=8)
        layout.addWidget(self.plot)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(50)

    def init_myo(self):
        self.myo = Myo(None, mode=emg_mode.PREPROCESSED)
        self.emg_handler = EMGHandler(self.myo)
        self.emg_handler.classifier = self.classifier
        self.myo.add_emg_handler(self.emg_handler)
        self.myo.connect()
        self.myo_timer = QTimer(self)
        self.myo_timer.timeout.connect(self.myo_loop)
        self.myo_timer.start(10)

    def myo_loop(self):
        self.myo.run()
        if hasattr(self.emg_handler, "emg"):
            pred = self.classifier.classify(self.emg_handler.emg)
            if pred is not None:
                name = self.classifier.label_to_name(pred)
                conf = self.classifier.last_confidence
                self.emg_handler.last_gesture = name
                self.emg_handler.last_confidence = conf

    def update_ui(self):
        if hasattr(self.emg_handler, "last_gesture"):
            self.label_g.setText(f"Gesture: {self.emg_handler.last_gesture}")
            self.label_c.setText(f"Confidence: {self.emg_handler.last_confidence:.2f}")
            if hasattr(self.emg_handler, "emg"):
                self.plot.update_plot(self.emg_handler.emg)

    def closeEvent(self, event):
        self.timer.stop()
        self.myo_timer.stop()
        self.myo.disconnect()
        event.accept()


# -----------------------------
# エントリポイント
# -----------------------------
def main():
    BASE = os.path.dirname(__file__)
    model_path = os.path.join(BASE, "model.h5")
    scaler_path = os.path.join(BASE, "scaler.pkl")
    encoder_path = os.path.join(BASE, "label_encoder.pkl")

    for p in (model_path, scaler_path, encoder_path):
        if not os.path.exists(p):
            print(f"Error: ファイルが見つかりません: {p}")
            return 1

    app = QApplication(sys.argv)
    clf = LSTMGestureClassifier(
        model_path=model_path,
        scaler_path=scaler_path,
        encoder_path=encoder_path,
        window_size=200,  # 学習時と同じ
        step=50,  # 学習時と同じ
        threshold=0.7,
    )
    win = GestureClassifierApp(clf)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

import multiprocessing
import sys
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pyomyo import Myo, emg_mode

# ----------------------------
# Constants and Configuration
# ----------------------------
SAMPLE_RATE = 200  # Hz
BUFFER_SIZE = 200  # 1 second of data at 200Hz
PLOT_REFRESH_RATE = 30  # Hz


# ----------------------------
# Worker Process
# ----------------------------
def worker(q):
    m = Myo(mode=emg_mode.RAW)
    m.connect()

    def add_to_queue(emg, movement):
        q.put(emg)

    m.add_emg_handler(add_to_queue)

    # Set LEDs to indicate connection
    m.set_leds([128, 0, 0], [128, 0, 0])
    m.vibrate(1)

    print("Myo worker started")
    try:
        while True:
            m.run()
    except KeyboardInterrupt:
        print("Myo worker stopping...")
    except Exception as e:
        print(f"Error in worker: {e}")
    finally:
        m.disconnect()


# ----------------------------
# Main Window
# ----------------------------
class EMGPlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time EMG Viewer (200Hz)")
        self.setGeometry(100, 100, 1200, 800)

        # Data storage
        self.data_buffers = [deque(maxlen=BUFFER_SIZE) for _ in range(8)]
        self.time_buffer = deque(maxlen=BUFFER_SIZE)
        self.sample_times = deque(maxlen=100)
        self.last_update_time = time.time()

        # Initialize UI
        self.init_ui()

        # Start worker process
        self.q = multiprocessing.Queue()
        self.worker_process = multiprocessing.Process(target=worker, args=(self.q,))
        self.worker_process.start()

        # Setup update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(int(1000 / PLOT_REFRESH_RATE))

        # Sample rate calculation
        self.sample_rate = 0
        self.sample_count = 0

    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Info panel
        info_layout = QHBoxLayout()
        self.sample_rate_label = QLabel("Sample Rate: 0.00 Hz")
        self.sample_count_label = QLabel("Samples: 0")
        info_layout.addWidget(self.sample_rate_label)
        info_layout.addWidget(self.sample_count_label)
        info_layout.addStretch()

        # Add plot area
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plots = []
        self.curves = []

        # Create 8 subplots (one for each channel)
        for i in range(8):
            plot = self.plot_widget.addPlot(row=i, col=0)
            plot.setYRange(-128, 128)
            plot.setXRange(0, BUFFER_SIZE / SAMPLE_RATE)
            plot.setLabel("left", f"Ch {i+1}")
            if i < 7:
                plot.hideAxis("bottom")
            else:
                plot.setLabel("bottom", "Time (s)")

            curve = plot.plot(pen=pg.intColor(i, hues=8, maxValue=200))
            self.plots.append(plot)
            self.curves.append(curve)

        # Add widgets to main layout
        layout.addLayout(info_layout)
        layout.addWidget(self.plot_widget)

        # Add control buttons
        control_layout = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self.on_pause_toggled)
        control_layout.addWidget(self.pause_btn)

        # Add channel visibility toggles
        self.visibility_btns = []
        for i in range(8):
            btn = QPushButton(f"Ch {i+1}")
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.toggled.connect(
                lambda checked, idx=i: self.toggle_channel_visibility(idx, checked)
            )
            self.visibility_btns.append(btn)
            control_layout.addWidget(btn)

        layout.addLayout(control_layout)

    def toggle_channel_visibility(self, channel, visible):
        if 0 <= channel < 8:
            self.plots[channel].setVisible(visible)

    def on_pause_toggled(self, checked):
        if checked:
            self.pause_btn.setText("Resume")
        else:
            self.pause_btn.setText("Pause")

    def update_plot(self):
        # Process all available data
        processed_count = 0
        current_time = time.time()

        while not self.q.empty():
            emg_data = self.q.get()
            if not self.pause_btn.isChecked():
                self.time_buffer.append(current_time)
                for i, value in enumerate(emg_data):
                    self.data_buffers[i].append(value)
                processed_count += 1

            # Update sample rate calculation
            self.sample_times.append(time.perf_counter())
            if len(self.sample_times) > 1:
                time_diffs = np.diff(self.sample_times)
                self.sample_rate = 1.0 / np.mean(time_diffs)

        # Update UI
        if not self.pause_btn.isChecked():
            self.sample_count += processed_count
            self.sample_rate_label.setText(f"Sample Rate: {self.sample_rate:.2f} Hz")
            self.sample_count_label.setText(f"Samples: {self.sample_count}")

            # Update plots
            for i in range(8):
                if self.data_buffers[i]:
                    x = np.linspace(
                        0,
                        len(self.data_buffers[i]) / SAMPLE_RATE,
                        len(self.data_buffers[i]),
                    )
                    self.curves[i].setData(x, list(self.data_buffers[i]))

    def closeEvent(self, event):
        # Clean up worker process
        if self.worker_process.is_alive():
            self.worker_process.terminate()
            self.worker_process.join()
        event.accept()


# ----------------------------
# Main Function
# ----------------------------
def main():
    # Enable antialiasing for better looking plots
    pg.setConfigOptions(antialias=True)

    app = QApplication(sys.argv)
    window = EMGPlotWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

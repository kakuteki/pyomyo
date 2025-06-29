import logging
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import butter, filtfilt

from pyomyo import Myo, emg_mode


# -----------------------------
# Logging Configuration
# -----------------------------
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("emg_detector.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# -----------------------------
# Configuration and Constants
# -----------------------------
@dataclass
class Config:
    """Application configuration"""

    # UI Settings
    WINDOW_TITLE: str = "Hand State Detector"
    WINDOW_SIZE: tuple = (800, 400)
    UPDATE_RATE_MS: int = 50  # 20 FPS
    MYO_UPDATE_RATE_MS: int = 10  # 100Hz

    # Signal Processing
    EMG_BUFFER_SIZE: int = 20
    HISTORY_SIZE: int = 5
    SMOOTHING_ALPHA: float = 0.3
    LOWPASS_CUTOFF: float = 5.0
    SAMPLE_RATE: float = 200.0
    FILTER_ORDER: int = 2

    # Threshold Settings
    DEFAULT_THRESHOLD: int = 50
    MIN_THRESHOLD: int = 10
    MAX_THRESHOLD: int = 250
    HYSTERESIS_FACTOR: float = 0.2

    # Calibration Settings (二状態（二点）計測によるレンジ把握 & チャンネル／次元ごとのキャリブレーション)
    CALIBRATION_DURATION_MS: int = 5000
    CALIBRATION_RELAX_PROMPT: str = "手を開いた状態で5秒間保持してください (リラックス)"
    CALIBRATION_CONTRACT_PROMPT: str = (
        "手を強く握った状態で5秒間保持してください (最大収縮)"
    )

    # Online Adaptive Threshold (オンライン適応閾値)
    ADAPTIVE_THRESHOLD_WINDOW_SIZE: int = 100  # 過去のEMG値を保持するウィンドウサイズ
    ADAPTIVE_THRESHOLD_LEARNING_RATE: float = 0.01  # 閾値の適応速度
    ADAPTIVE_THRESHOLD_MIN_DIFF: float = 5.0  # 閾値が適応される最小の変化量

    # Kalman Filter
    KALMAN_PROCESS_VARIANCE: float = 0.05
    KALMAN_MEASUREMENT_VARIANCE: float = 1.0

    # Plot Settings
    PLOT_RANGE_Y: tuple = (-128, 128)
    PLOT_RANGE_X: tuple = (0, 100)
    PLOT_DATA_SIZE: int = 100


class HandState(Enum):
    """Hand state enumeration"""

    OPEN = "開いています"
    CLOSED = "閉じています"


# -----------------------------
# Circular Buffer for Efficient Data Management
# -----------------------------
class CircularBuffer:
    """Efficient circular buffer implementation for plot data"""

    def __init__(self, size: int):
        self.data = np.zeros(size)
        self.size = size
        self.index = 0
        self.full = False

    def append(self, value: float):
        """Add new value to buffer"""
        self.data[self.index] = value
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.full = True

    def get_ordered_data(self) -> np.ndarray:
        """Get data in chronological order"""
        if not self.full:
            return self.data[: self.index]
        return np.concatenate([self.data[self.index :], self.data[: self.index]])


# -----------------------------
# Signal Processing Components
# -----------------------------
class KalmanFilter1D:
    """1D Kalman Filter for EMG signal smoothing"""

    def __init__(
        self, process_variance: float = 1e-5, measurement_variance: float = 0.01
    ):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posterior_estimate = 0.0
        self.posterior_error_estimate = 1.0

    def update(self, measurement: float) -> float:
        """Update filter with new measurement"""
        # Prediction step
        prior_estimate = self.posterior_estimate
        prior_error_estimate = self.posterior_error_estimate + self.process_variance

        # Update step
        kalman_gain = prior_error_estimate / (
            prior_error_estimate + self.measurement_variance
        )
        self.posterior_estimate = prior_estimate + kalman_gain * (
            measurement - prior_estimate
        )
        self.posterior_error_estimate = (1 - kalman_gain) * prior_error_estimate

        return self.posterior_estimate


class AdvancedEMGProcessor:
    """Enhanced EMG signal processing pipeline with improved filtering
    中央値によるロバストベースライン & チャンネル／次元ごとのキャリブレーション対応
    """

    def __init__(self, config: Config):
        self.config = config
        self.kalman_filters = [
            KalmanFilter1D(
                process_variance=config.KALMAN_PROCESS_VARIANCE,
                measurement_variance=config.KALMAN_MEASUREMENT_VARIANCE,
            )
            for _ in range(8)
        ]
        self.emg_buffer = deque(maxlen=config.EMG_BUFFER_SIZE)
        self.exponential_moving_avg = 0.0
        self.lowpass_filter_state = None

        # 中央値によるロバストベースライン & チャンネル／次元ごとのキャリブレーション
        self.channel_baselines = np.zeros(8)  # 各チャンネルのリラックス時のベースライン

        # Initialize Butterworth filter coefficients
        try:
            nyq = 0.5 * self.config.SAMPLE_RATE
            normal_cutoff = self.config.LOWPASS_CUTOFF / nyq
            self.filter_b, self.filter_a = butter(
                self.config.FILTER_ORDER, normal_cutoff, btype="low", analog=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize Butterworth filter: {e}")
            self.filter_b, self.filter_a = None, None

    def process(self, raw_emg: List[float]) -> float:
        """Process raw EMG data with improved filtering pipeline"""
        try:
            # Step 1: Apply Kalman filter to each channel
            filtered_channels = self._apply_kalman_filtering(raw_emg)

            # Step 2: Apply channel baselines (中央値によるロバストベースライン)
            # 各チャンネルの生データからリラックス時のベースラインを差し引く
            normalized_channels = [
                fc - bl for fc, bl in zip(filtered_channels, self.channel_baselines)
            ]

            # Step 3: Calculate RMS of filtered and normalized channels
            emg_rms = self._calculate_rms(normalized_channels)

            # Step 4: Add to moving average buffer
            self.emg_buffer.append(emg_rms)

            # Step 5: Apply lowpass filter on buffered data
            lowpass_output = self._apply_lowpass_filter()

            # Step 6: Apply exponential moving average for final smoothing
            self.exponential_moving_avg = self._apply_exponential_smoothing(
                lowpass_output
            )

            return self.exponential_moving_avg

        except Exception as e:
            logger.exception(f"Error processing EMG data: {e}")
            return 0.0

    def _apply_kalman_filtering(self, raw_emg: List[float]) -> List[float]:
        """Apply Kalman filter to each EMG channel"""
        return [
            self.kalman_filters[i].update(float(value))
            for i, value in enumerate(raw_emg[: len(self.kalman_filters)])
        ]

    def _calculate_rms(self, data: List[float]) -> float:
        """Calculate Root Mean Square"""
        return np.sqrt(np.mean(np.square(data)))

    def _apply_lowpass_filter(self) -> float:
        """Apply Butterworth lowpass filter to buffered data"""
        if len(self.emg_buffer) <= 5 or self.filter_b is None:
            return self.emg_buffer[-1] if self.emg_buffer else 0.0

        try:
            buffer_data = list(self.emg_buffer)
            filtered_data = filtfilt(self.filter_b, self.filter_a, buffer_data)
            last_lowpass_output = filtered_data[-1] if len(filtered_data) > 0 else 0.0
            return last_lowpass_output
        except Exception as e:
            logger.warning(f"Lowpass filter failed, using raw data: {e}")
            return self.emg_buffer[-1] if self.emg_buffer else 0.0

    def _apply_exponential_smoothing(self, current_value: float) -> float:
        """Apply exponential moving average for final smoothing"""
        alpha = self.config.SMOOTHING_ALPHA
        return alpha * current_value + (1 - alpha) * self.exponential_moving_avg

    def set_channel_baselines(self, baselines: np.ndarray):
        """Set the baseline values for each channel"""
        self.channel_baselines = baselines


class HandStateClassifier:
    """Hand state classification with hysteresis and online adaptive threshold"""

    def __init__(self, config: Config):
        self.config = config
        self.threshold = config.DEFAULT_THRESHOLD
        self.current_state = HandState.OPEN
        self.history = deque(maxlen=config.HISTORY_SIZE)

        # オンライン適応閾値
        self.emg_history_for_adaptive_threshold = deque(
            maxlen=config.ADAPTIVE_THRESHOLD_WINDOW_SIZE
        )
        self.relaxed_emg_level = (
            0.0  # キャリブレーションで得られたリラックス時のEMGレベル
        )
        self.contracted_emg_level = (
            0.0  # キャリブレーションで得られた最大収縮時のEMGレベル
        )

    def classify(self, emg_value: float) -> HandState:
        """Classify hand state based on EMG value"""
        # オンライン適応閾値の更新
        self.emg_history_for_adaptive_threshold.append(emg_value)
        self._adapt_threshold()

        upper_threshold = self.threshold * (1 + self.config.HYSTERESIS_FACTOR)
        lower_threshold = self.threshold * (1 - self.config.HYSTERESIS_FACTOR)

        if emg_value > upper_threshold:
            new_state = HandState.CLOSED
        elif emg_value < lower_threshold:
            new_state = HandState.OPEN
        else:
            # Within hysteresis range - maintain current state
            new_state = self.current_state

        # State smoothing (多数決方式)
        self.history.append(new_state)
        if len(self.history) >= self.config.HISTORY_SIZE:
            # 直近の履歴で最も多い状態を現在の状態とする
            open_count = self.history.count(HandState.OPEN)
            closed_count = self.history.count(HandState.CLOSED)
            if open_count > closed_count:
                self.current_state = HandState.OPEN
            elif closed_count > open_count:
                self.current_state = HandState.CLOSED

        return self.current_state

    def set_threshold(self, threshold: int):
        """Set new threshold value"""
        self.threshold = max(
            self.config.MIN_THRESHOLD, min(self.config.MAX_THRESHOLD, threshold)
        )

    def set_calibration_levels(self, relaxed_level: float, contracted_level: float):
        """キャリブレーションで得られたリラックス時と収縮時のEMGレベルを設定"""
        self.relaxed_emg_level = relaxed_level
        self.contracted_emg_level = contracted_level
        # 初期閾値を設定
        if relaxed_level < contracted_level:
            initial_threshold = (relaxed_level + contracted_level) / 2
            self.set_threshold(int(initial_threshold))
        else:
            self.set_threshold(self.config.DEFAULT_THRESHOLD)

    def _adapt_threshold(self):
        """オンライン適応閾値のロジック"""
        if (
            len(self.emg_history_for_adaptive_threshold)
            < self.config.ADAPTIVE_THRESHOLD_WINDOW_SIZE
        ):
            return

        # 過去のEMG値の移動中央値を計算
        current_median_emg = np.median(list(self.emg_history_for_adaptive_threshold))

        # リラックス時と収縮時のレベルが設定されている場合のみ適応
        if self.relaxed_emg_level < self.contracted_emg_level:
            # 現在の中央値がリラックスレベルに近いか、収縮レベルに近いかを判断
            # そして、その中間点に閾値を適応的に調整する
            target_threshold = (self.relaxed_emg_level + self.contracted_emg_level) / 2

            # 現在の閾値と目標閾値の差を計算
            diff = target_threshold - self.threshold

            # 最小変化量を超えている場合のみ閾値を更新
            if abs(diff) > self.config.ADAPTIVE_THRESHOLD_MIN_DIFF:
                # 適応速度に基づいて閾値を微調整
                new_threshold = (
                    self.threshold + diff * self.config.ADAPTIVE_THRESHOLD_LEARNING_RATE
                )
                self.set_threshold(int(new_threshold))
                logger.debug(f"Adaptive threshold updated to {self.threshold}")


class CalibrationManager:
    """Manages calibration process (二状態（二点）計測によるレンジ把握)"""

    def __init__(self, config: Config):
        self.config = config
        self.is_calibrating = False
        self.calibration_phase = 0  # 0: idle, 1: relax, 2: contract
        self.current_phase_samples_emg = []  # RMS値のサンプル
        self.current_phase_samples_raw_emg = (
            []
        )  # 生EMGデータのサンプル (チャンネルごとのベースライン用)
        self.callback: Optional[Callable[[float, float, np.ndarray], None]] = (
            None  # relaxed_level, contracted_level, channel_baselines
        )

        self.relaxed_emg_level = 0.0
        self.contracted_emg_level = 0.0
        self.channel_baselines = np.zeros(8)

    def start_calibration(
        self, phase: int, callback: Callable[[float, float, np.ndarray], None]
    ):
        """Start calibration process for a specific phase"""
        self.is_calibrating = True
        self.calibration_phase = phase
        self.current_phase_samples_emg = []
        self.current_phase_samples_raw_emg = []
        self.callback = callback
        logger.info(f"Calibration started for phase {phase}")

    def add_sample(self, emg_value: float, raw_emg: List[float]):
        """Add calibration sample"""
        if self.is_calibrating:
            self.current_phase_samples_emg.append(emg_value)
            self.current_phase_samples_raw_emg.append(raw_emg)

    def finish_calibration_phase(self) -> bool:
        """Finish current calibration phase and store results"""
        if not self.current_phase_samples_emg:
            logger.warning(
                f"No calibration samples collected for phase {self.calibration_phase}"
            )
            self.is_calibrating = False
            return False

        if self.calibration_phase == 1:  # Relax phase
            self.relaxed_emg_level = np.median(
                self.current_phase_samples_emg
            )  # 中央値によるロバストベースライン
            # 各チャンネルのベースラインを計算 (中央値を使用)
            raw_emg_array = np.array(self.current_phase_samples_raw_emg)
            self.channel_baselines = np.median(raw_emg_array, axis=0)
            logger.info(
                f"Relax calibration completed. Relaxed EMG: {self.relaxed_emg_level:.2f}, Channel Baselines: {self.channel_baselines}"
            )
        elif self.calibration_phase == 2:  # Contract phase
            self.contracted_emg_level = np.median(
                self.current_phase_samples_emg
            )  # 中央値によるロバストベースライン
            logger.info(
                f"Contract calibration completed. Contracted EMG: {self.contracted_emg_level:.2f}"
            )

        self.is_calibrating = False
        return True

    def get_calibration_results(self) -> tuple[float, float, np.ndarray]:
        """Get final calibration results"""
        return self.relaxed_emg_level, self.contracted_emg_level, self.channel_baselines

    def reset_calibration(self):
        """Reset all calibration data"""
        self.relaxed_emg_level = 0.0
        self.contracted_emg_level = 0.0
        self.channel_baselines = np.zeros(8)
        self.is_calibrating = False
        self.calibration_phase = 0
        self.current_phase_samples_emg = []
        self.current_phase_samples_raw_emg = []
        logger.info("Calibration data reset.")


# -----------------------------
# Enhanced UI Components
# -----------------------------
class EnhancedEMGPlotWidget(pg.PlotWidget):
    """Enhanced EMG plot widget with circular buffer"""

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.buffer = CircularBuffer(config.PLOT_DATA_SIZE)
        self._setup_plot()

    def _setup_plot(self):
        """Setup plot appearance and components"""
        self.setBackground("w")
        self.setYRange(*self.config.PLOT_RANGE_Y)
        self.setXRange(*self.config.PLOT_RANGE_X)

        # Data curve
        self.curve = self.plot(pen="b", name="EMG Avg")

        # Threshold line
        self.threshold_line = pg.InfiniteLine(
            angle=0, pen=pg.mkPen("r", width=1, style=Qt.DashLine)
        )
        self.addItem(self.threshold_line)

    def update_plot(self, emg_value: float, state: HandState):
        """Update plot with new EMG value and state (optimized)"""
        # Add new data to circular buffer (no array copying)
        self.buffer.append(emg_value)

        # Update curve data
        plot_data = self.buffer.get_ordered_data()
        self.curve.setData(plot_data)

        # Update curve color based on state
        color = "r" if state == HandState.CLOSED else "g"
        self.curve.setPen(color)

    def update_threshold(self, threshold: int):
        """Update threshold line position"""
        self.threshold_line.setPos(threshold)


class MyoController(QObject):
    """Myo device controller with enhanced error handling"""

    emg_received = pyqtSignal(object)  # raw_emg (List[float])
    connection_error = pyqtSignal(str)

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.myo: Optional[Myo] = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_myo)

    def connect_myo(self) -> bool:
        """Connect to Myo device with enhanced error handling"""
        try:
            self.myo = Myo(None, mode=emg_mode.PREPROCESSED)
            self.myo.add_emg_handler(self._handle_emg)
            self.myo.connect()
            self.timer.start(self.config.MYO_UPDATE_RATE_MS)
            logger.info("Successfully connected to Myo device")
            return True
        except Exception as e:
            error_msg = f"Failed to connect to Myo: {e}"
            logger.error(error_msg)
            self.connection_error.emit(error_msg)
            return False

    def disconnect_myo(self):
        """Disconnect from Myo device"""
        if self.timer.isActive():
            self.timer.stop()
        if self.myo:
            try:
                self.myo.disconnect()
                logger.info("Myo device disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting Myo: {e}")

    def _handle_emg(self, emg_data, moving):
        """Handle EMG data from Myo"""
        try:
            # Convert tuple to list for consistent handling
            emg_list = list(emg_data) if isinstance(emg_data, tuple) else emg_data
            self.emg_received.emit(emg_list)
        except Exception as e:
            logger.exception(f"Error handling EMG data: {e}")

    def _update_myo(self):
        """Update Myo device"""
        try:
            if self.myo:
                self.myo.run()
        except Exception as e:
            logger.exception(f"Error updating Myo: {e}")


# -----------------------------
# Main Application
# -----------------------------
class HandStateDetector(QMainWindow):
    """Main application window with enhanced error handling"""

    def __init__(self):
        super().__init__()
        self.config = Config()

        # Initialize components
        self.emg_processor = AdvancedEMGProcessor(self.config)
        self.classifier = HandStateClassifier(self.config)
        self.calibration_manager = CalibrationManager(self.config)
        self.myo_controller = MyoController(self.config)

        # UI components
        self.state_label: Optional[QLabel] = None
        self.threshold_value: Optional[QLabel] = None
        self.threshold_slider: Optional[QSlider] = None
        self.calibrate_btn: Optional[QPushButton] = None
        self.calibration_status: Optional[QLabel] = None
        self.plot: Optional[EnhancedEMGPlotWidget] = None

        # Timers
        self.ui_timer = QTimer(self)
        self.calibration_timer = QTimer(self)

        self._init_ui()
        self._setup_connections()
        self._start_application()

    def _init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle(self.config.WINDOW_TITLE)
        self.setGeometry(100, 100, *self.config.WINDOW_SIZE)

        # Main widget and layout
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)

        # State display
        self.state_label = QLabel(f"手の状態: {HandState.OPEN.value}")
        self.state_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: green;"
        )
        layout.addWidget(self.state_label)

        # Threshold controls
        self._create_threshold_controls(layout)

        # EMG plot
        self.plot = EnhancedEMGPlotWidget(self.config)
        layout.addWidget(self.plot)

    def _create_threshold_controls(self, parent_layout):
        """Create threshold control widgets"""
        threshold_layout = QVBoxLayout()

        # Threshold value display
        self.threshold_value = QLabel(f"閾値: {self.classifier.threshold}")
        self.threshold_value.setFont(QFont("Arial", 12))
        threshold_layout.addWidget(self.threshold_value)

        # Threshold slider
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(self.config.MIN_THRESHOLD)
        self.threshold_slider.setMaximum(self.config.MAX_THRESHOLD)
        self.threshold_slider.setValue(self.classifier.threshold)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        threshold_layout.addWidget(self.threshold_slider)

        # Calibration button and status
        self.calibrate_btn = QPushButton("キャリブレーション開始")
        self.calibrate_btn.setFont(QFont("Arial", 12))
        threshold_layout.addWidget(self.calibrate_btn)

        self.calibration_status = QLabel("キャリブレーション: 未実施")
        self.calibration_status.setFont(QFont("Arial", 10))
        threshold_layout.addWidget(self.calibration_status)

        parent_layout.addLayout(threshold_layout)

    def _setup_connections(self):
        """Setup signal and slot connections"""
        self.myo_controller.emg_received.connect(self._on_emg_received)
        self.myo_controller.connection_error.connect(self._on_myo_connection_error)
        self.ui_timer.timeout.connect(self._update_ui)
        self.threshold_slider.valueChanged.connect(self._on_threshold_slider_changed)
        self.calibrate_btn.clicked.connect(self._on_calibrate_button_clicked)

    def _start_application(self):
        """Start Myo connection and UI timer"""
        if self.myo_controller.connect_myo():
            self.ui_timer.start(self.config.UPDATE_RATE_MS)

    def _on_emg_received(self, raw_emg: List[float]):
        """Handle raw EMG data from Myo"""
        # キャリブレーション中は生EMGデータもCalibrationManagerに渡す
        if self.calibration_manager.is_calibrating:
            self.calibration_manager.add_sample(
                self.emg_processor.process(raw_emg), raw_emg
            )
        else:
            # 通常処理
            self.emg_processor.process(raw_emg)

    def _update_ui(self):
        """Update UI elements with processed EMG data"""
        current_emg_value = self.emg_processor.exponential_moving_avg
        current_state = self.classifier.classify(current_emg_value)

        self.state_label.setText(f"手の状態: {current_state.value}")
        self.state_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: red;"
            if current_state == HandState.CLOSED
            else "font-size: 24px; font-weight: bold; color: green;"
        )

        self.plot.update_plot(current_emg_value, current_state)
        self.plot.update_threshold(self.classifier.threshold)

    def _on_threshold_slider_changed(self, value: int):
        """Handle threshold slider value change"""
        self.classifier.set_threshold(value)
        self.threshold_value.setText(f"閾値: {self.classifier.threshold}")

    def _on_calibrate_button_clicked(self):
        """Handle calibrate button click (二状態（二点）計測によるレンジ把握)"""
        if not self.calibration_manager.is_calibrating:
            self.calibration_manager.reset_calibration()
            self._start_relax_calibration()
        else:
            QMessageBox.warning(
                self, "キャリブレーション", "キャリブレーションが既に進行中です。"
            )

    def _start_relax_calibration(self):
        """Start the relax phase of calibration"""
        self.calibration_status.setText(
            f"キャリブレーション: {self.config.CALIBRATION_RELAX_PROMPT}"
        )
        self.calibrate_btn.setEnabled(False)
        self.threshold_slider.setEnabled(False)
        self.calibration_manager.start_calibration(
            1, self._on_calibration_complete
        )  # Phase 1: Relax
        self.calibration_timer.singleShot(
            self.config.CALIBRATION_DURATION_MS, self._finish_relax_calibration
        )
        logger.info("Starting relax calibration phase.")

    def _finish_relax_calibration(self):
        """Finish the relax phase and start contract phase"""
        if self.calibration_manager.finish_calibration_phase():
            self._start_contract_calibration()
        else:
            self._reset_calibration_ui()
            QMessageBox.critical(
                self,
                "キャリブレーションエラー",
                "リラックス時のキャリブレーションに失敗しました。サンプルが収集されませんでした。",
            )

    def _start_contract_calibration(self):
        """Start the contract phase of calibration"""
        self.calibration_status.setText(
            f"キャリブレーション: {self.config.CALIBRATION_CONTRACT_PROMPT}"
        )
        self.calibration_manager.start_calibration(
            2, self._on_calibration_complete
        )  # Phase 2: Contract
        self.calibration_timer.singleShot(
            self.config.CALIBRATION_DURATION_MS, self._finish_contract_calibration
        )
        logger.info("Starting contract calibration phase.")

    def _finish_contract_calibration(self):
        """Finish the contract phase and complete calibration"""
        if self.calibration_manager.finish_calibration_phase():
            relaxed_level, contracted_level, channel_baselines = (
                self.calibration_manager.get_calibration_results()
            )
            self.emg_processor.set_channel_baselines(
                channel_baselines
            )  # チャンネルごとのベースラインを設定
            self.classifier.set_calibration_levels(
                relaxed_level, contracted_level
            )  # 分類器にキャリブレーションレベルを設定
            self._on_calibration_complete(
                relaxed_level, contracted_level, channel_baselines
            )
            QMessageBox.information(
                self,
                "キャリブレーション完了",
                "キャリブレーションが完了しました。閾値が自動調整されました。",
            )
        else:
            QMessageBox.critical(
                self,
                "キャリブレーションエラー",
                "最大収縮時のキャリブレーションに失敗しました。サンプルが収集されませんでした。",
            )
        self._reset_calibration_ui()

    def _on_calibration_complete(
        self,
        relaxed_level: float,
        contracted_level: float,
        channel_baselines: np.ndarray,
    ):
        """Callback when calibration is fully complete"""
        # このコールバックは、各フェーズの終了時ではなく、両フェーズ完了後に呼ばれる
        # 閾値の設定はclassifier.set_calibration_levels()で行われるため、ここではUI更新のみ
        self.threshold_value.setText(f"閾値: {self.classifier.threshold}")
        self.threshold_slider.setValue(self.classifier.threshold)
        self.calibration_status.setText("キャリブレーション: 完了")
        logger.info(
            f"Final calibration results - Relaxed: {relaxed_level:.2f}, Contracted: {contracted_level:.2f}, Baselines: {channel_baselines}"
        )
        self._reset_calibration_ui()

    def _reset_calibration_ui(self):
        """Reset calibration UI elements"""
        self.calibrate_btn.setEnabled(True)
        self.threshold_slider.setEnabled(True)
        self.calibration_manager.is_calibrating = False
        self.calibration_manager.calibration_phase = 0

    def _on_myo_connection_error(self, error_msg: str):
        """Handle Myo connection errors"""
        QMessageBox.critical(self, "Myo接続エラー", error_msg)
        self.close()

    def closeEvent(self, event):
        """Handle application close event"""
        logger.info("Application closing...")
        self.myo_controller.disconnect_myo()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    detector = HandStateDetector()
    detector.show()
    sys.exit(app.exec_())

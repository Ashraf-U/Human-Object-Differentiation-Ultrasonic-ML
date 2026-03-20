"""
FIUS Ultrasonic Sensor — Real-Time Classification App (v2)
==========================================================
Classifies: Human vs Chair vs Nothing
Models: 1D-Transformer + 1D-ResNet + 1D-CNN (all raw signal input)

Modes:
  - REAL MODE: Connects to Red Pitaya sensor via UDP
  - SIMULATION MODE: Loads from CSV files for testing without sensor

Setup:
  Place these files in the SAME folder as this script:
    - transformer_model.keras   (from Step 5)
    - resnet_model.keras        (from Step 5)
    - cnn_model.keras           (from Step 3)
  For simulation, also place CSV files in a 'test_data' subfolder.

Install:
  pip install PyQt6 pyqtgraph paramiko numpy pandas tensorflow
"""

import traceback, sys
import numpy as np
import pandas as pd
import os
import time
import socket
import struct

from PyQt6.QtCore import QSize, Qt, QRunnable, pyqtSlot, QThreadPool, QObject, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QMainWindow,
                              QGridLayout, QHBoxLayout, QVBoxLayout, QCheckBox,
                              QLabel, QLineEdit, QFrame, QGroupBox)
from PyQt6.QtGui import QFont

import pyqtgraph as pg
import paramiko


# ==============================================================
#  CONSTANTS
# ==============================================================

N_METADATA_COLS = 17
N_SIGNAL_SAMPLES = 25000
SAMPLING_RATE = 1_953_125
DOWNSAMPLE_FACTOR = 10
DS_LENGTH = N_SIGNAL_SAMPLES // DOWNSAMPLE_FACTOR  # 2500

LABEL_NAMES = ['Human', 'Chair', 'Nothing']
LABEL_COLORS = {
    'Human':   '#e74c3c',
    'Chair':   '#3498db',
    'Nothing': '#2ecc71',
    'No Model': '#95a5a6',
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    'Transformer': os.path.join(SCRIPT_DIR, 'transformer_model.keras'),
    'ResNet':      os.path.join(SCRIPT_DIR, 'resnet_model.keras'),
    'CNN':         os.path.join(SCRIPT_DIR, 'cnn_model.keras'),
}
TEST_DATA_DIR = os.path.join(SCRIPT_DIR, 'test_data')


# ==============================================================
#  SIGNAL PREPROCESSING (same as training)
# ==============================================================

def preprocess_signal(signal_array):
    """
    Convert raw 25,000 signal → normalized 2,500 for model input.
    Identical to training pipeline.
    """
    sig = signal_array.astype(np.float64)

    # Downsample: 25000 → 2500 by averaging every 10 samples
    sig_ds = sig[:DS_LENGTH * DOWNSAMPLE_FACTOR].reshape(DS_LENGTH, DOWNSAMPLE_FACTOR).mean(axis=1)

    # Per-sample normalize: zero mean, unit variance
    mean = sig_ds.mean()
    std = sig_ds.std() + 1e-8
    sig_ds = (sig_ds - mean) / std

    # Reshape for model: (1, 2500, 1)
    return sig_ds.reshape(1, DS_LENGTH, 1).astype(np.float32)


# ==============================================================
#  TRANSFORMER BLOCK (must be defined before loading the model)
# ==============================================================

# Import TensorFlow and Keras at module level for custom layer registration
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    """Single Transformer block with multi-head self-attention."""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config


# ==============================================================
#  MODEL PREDICTOR — All 3 Deep Learning Models
# ==============================================================

class ModelPredictor:
    """Loads and runs Transformer, ResNet, and CNN models."""

    def __init__(self):
        self.models = {}      # name → keras model
        self.loaded = {}      # name → bool

        print("\n" + "=" * 50)
        print("  Loading Deep Learning Models...")
        print("=" * 50)
        print(f"  TensorFlow {tf.__version__} loaded")

        # Load each model (TransformerBlock is now registered above)
        for name, path in MODEL_PATHS.items():
            if os.path.exists(path):
                try:
                    self.models[name] = tf.keras.models.load_model(
                        path, custom_objects={'TransformerBlock': TransformerBlock}
                    )
                    self.loaded[name] = True
                    size_mb = os.path.getsize(path) / 1e6
                    print(f"  ✅ {name:12s} loaded ({size_mb:.1f} MB)")
                except Exception as e:
                    self.loaded[name] = False
                    print(f"  ❌ {name:12s} failed: {e}")
            else:
                self.loaded[name] = False
                print(f"  ⚠️  {name:12s} not found at {path}")

        loaded_count = sum(self.loaded.values())
        print(f"\n  {loaded_count}/{len(MODEL_PATHS)} models loaded")
        print("=" * 50)

    def predict_single(self, model_name, signal_preprocessed):
        """Run one model. Returns (class_name, confidence, time_ms)."""
        if not self.loaded.get(model_name, False):
            return "No Model", 0.0, 0.0

        t0 = time.time()
        proba = self.models[model_name].predict(signal_preprocessed, verbose=0)[0]
        pred_time = (time.time() - t0) * 1000

        pred_class = np.argmax(proba)
        confidence = float(proba[pred_class])
        return LABEL_NAMES[pred_class], confidence, pred_time

    def predict_all(self, signal_array):
        """
        Run all 3 models on a raw signal.
        signal_array: numpy array of shape (25000,)
        Returns dict: {model_name: {class, confidence, time_ms, probabilities}}
        """
        # Preprocess once (shared by all models)
        t0 = time.time()
        sig_input = preprocess_signal(signal_array)
        preprocess_time = (time.time() - t0) * 1000

        results = {}
        for name in ['Transformer', 'ResNet', 'CNN']:
            if self.loaded.get(name, False):
                t0 = time.time()
                proba = self.models[name].predict(sig_input, verbose=0)[0]
                pred_time = (time.time() - t0) * 1000

                pred_class = np.argmax(proba)
                results[name] = {
                    'class': LABEL_NAMES[pred_class],
                    'confidence': float(proba[pred_class]),
                    'time_ms': pred_time,
                    'preprocess_ms': preprocess_time,
                    'probabilities': {
                        'Human': float(proba[0]),
                        'Chair': float(proba[1]),
                        'Nothing': float(proba[2]),
                    }
                }
            else:
                results[name] = {
                    'class': 'No Model',
                    'confidence': 0.0,
                    'time_ms': 0.0,
                    'preprocess_ms': 0.0,
                    'probabilities': {'Human': 0, 'Chair': 0, 'Nothing': 0}
                }

        return results


# ==============================================================
#  SIMULATED SENSOR
# ==============================================================

class SimulatedSensor:
    def __init__(self, test_data_dir=TEST_DATA_DIR):
        self.size_of_raw_adc = N_SIGNAL_SAMPLES
        self.total_data_blocks = 1
        self.sensor_status_message = "SIMULATION MODE — Using test CSV data"
        self.data_loaded = False
        self.sim_data = []
        self.sim_index = 0

        csv_files = {'human': None, 'chair': None, 'nothing': None}

        if os.path.exists(test_data_dir):
            for f in os.listdir(test_data_dir):
                fl = f.lower()
                if fl.endswith('.csv'):
                    if 'human' in fl or 'ashraf' in fl:
                        csv_files['human'] = os.path.join(test_data_dir, f)
                    elif 'chair' in fl:
                        csv_files['chair'] = os.path.join(test_data_dir, f)
                    elif 'nothing' in fl:
                        csv_files['nothing'] = os.path.join(test_data_dir, f)

        for label, path in csv_files.items():
            if path and os.path.exists(path):
                try:
                    print(f"  Loading sim data: {label} from {os.path.basename(path)}")
                    raw = np.genfromtxt(path, delimiter=',', max_rows=50, dtype=np.float32)
                    for i in range(raw.shape[0]):
                        metadata = raw[i, :N_METADATA_COLS]
                        signal = raw[i, N_METADATA_COLS:N_METADATA_COLS + N_SIGNAL_SAMPLES]
                        header_series = pd.Series(metadata, name='header')
                        data_series = pd.Series(signal.astype(np.int16), name='raw_adc')
                        self.sim_data.append((header_series, data_series, label))
                    print(f"    ✅ {raw.shape[0]} samples for {label}")
                except Exception as e:
                    print(f"    ❌ Failed: {e}")

        if self.sim_data:
            np.random.shuffle(self.sim_data)
            self.data_loaded = True
            print(f"  ✅ Simulation ready: {len(self.sim_data)} samples")
        else:
            print(f"  ❌ No data in {test_data_dir}/")

    def get_sensor_status_message(self):
        return self.sensor_status_message

    def get_data_info_from_server(self):
        self.sensor_status_message = "SIMULATION MODE — Ready"
        return time.time() * 1000, list(range(17))

    def get_data_from_server(self, start_time):
        if not self.data_loaded:
            return None, None
        time.sleep(0.3)
        header, data, true_label = self.sim_data[self.sim_index % len(self.sim_data)]
        self.sim_index += 1
        self.sensor_status_message = f"SIMULATION [{self.sim_index}] — True label: {true_label.upper()}"
        return header, data


# ==============================================================
#  RED PITAYA SENSOR (original)
# ==============================================================

class RedPitayaSensor:
    def __init__(self):
        self.size_of_raw_adc = 25000
        self.buffer_size = (self.size_of_raw_adc + 17) * 4
        self.msg_from_client = "-i 1"
        self.hostIP = "169.254.148.148"
        self.data_port = 61231
        self.ssh_port = 22
        self.server_address_port = (self.hostIP, self.data_port)
        self.sensor_status_message = "Waiting to Connect with RedPitaya UDP Server!"
        print(self.sensor_status_message)
        self.udp_client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.header_length = None

    def give_ssh_command(self, command):
        try:
            self.client.connect(self.hostIP, self.ssh_port, "root", "root")
            self.set_sensor_message(f"Connected to Redpitaya {self.hostIP}")
            stdin, stdout, stderr = self.client.exec_command(command)
            output = stdout.read().decode()
            error = stderr.read().decode()
            self.set_sensor_message(f"Output: {output}")
            if error:
                self.set_sensor_message(f"Error: {error}")
            if output:
                return output
        finally:
            self.client.close()
            self.set_sensor_message("Connection closed")

    def set_sensor_message(self, message):
        self.msg_from_client = message

    def get_sensor_status_message(self):
        return self.sensor_status_message

    def send_msg_to_server(self):
        bytes_to_send = str.encode(self.msg_from_client)
        print("Sending message")
        self.udp_client_socket.sendto(bytes_to_send, self.server_address_port)

    def get_data_info_from_server(self):
        self.msg_from_client = "-i 1"
        self.send_msg_to_server()
        packet = self.udp_client_socket.recv(self.buffer_size)
        self.sensor_status_message = f"Sensor Connected at {self.server_address_port}!"
        print(self.sensor_status_message)
        self.header_length = int(struct.unpack('@f', packet[:4])[0])
        self.total_data_blocks = int(struct.unpack('@f', packet[56:60])[0])
        synced_time = int(struct.unpack('@f', packet[20:24])[0])
        header_data = [i[0] for i in struct.iter_unpack('@f', packet[:self.header_length])]
        self.local_time_sync = time.time() * 1000
        self.first_synced_time = synced_time
        return synced_time, header_data

    def get_data_from_server(self, start_time):
        ultrasonic_data = []
        header = []
        for i in range(self.total_data_blocks):
            time.sleep(1 / 1000)
            self.msg_from_client = "-a 1"
            self.send_msg_to_server()
            packet1 = self.udp_client_socket.recv(self.buffer_size)
            if i == 0:
                current_time = time.time() * 1000
                elapsed_time = current_time - self.local_time_sync + start_time
                header = [h[0] for h in struct.iter_unpack('@f', packet1[:self.header_length])]
            current_data_block_number = int(struct.unpack('@f', packet1[60:64])[0])
            if i != current_data_block_number:
                print(f"Error: Expected block{i} but received block{current_data_block_number}")
                break
            redpitaya_acq_time_stamp = int(struct.unpack('@f', packet1[64:68])[0])
            self.sensor_status_message = f"Block {current_data_block_number+1} received at {elapsed_time:.0f}ms"
            for val in struct.iter_unpack('@h', packet1[self.header_length:]):
                ultrasonic_data.append(val[0])
        if len(ultrasonic_data) != self.size_of_raw_adc * self.total_data_blocks:
            return None, None
        header_df = pd.DataFrame(header, columns=['header'])
        raw_df = pd.DataFrame(ultrasonic_data, columns=['raw_adc'])
        return header_df['header'], raw_df['raw_adc']


# ==============================================================
#  WORKER SIGNALS
# ==============================================================

class WorkerSignals(QObject):
    result = pyqtSignal(object)
    prediction = pyqtSignal(object)
    error = pyqtSignal(tuple)
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    saved_signals_count_updated = pyqtSignal(int)
    broken_signals_count_updated = pyqtSignal(int)
    total_signals_count_updated = pyqtSignal(int)


# ==============================================================
#  WORKER THREAD
# ==============================================================

class Worker(QRunnable):
    def __init__(self, func_is_button_checked, rp_sensor, model_predictor=None, *args, **kwargs):
        super().__init__()
        self.func_is_button_checked = func_is_button_checked
        self.rp_sensor = rp_sensor
        self.model_predictor = model_predictor
        self.dataFilePath = None
        self.saving_number_of_signals = None
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.is_running = True
        self.saved_signals_count = 0
        self.total_signals_count = 0
        self.broken_signals_count = 0

    @pyqtSlot()
    def run(self):
        print("Start of thread")
        while self.func_is_button_checked(*self.args, **self.kwargs) and self.is_running:
            try:
                header, data = self.rp_sensor.get_data_from_server(window.start_time)
                self.total_signals_count += 1
                self.signals.total_signals_count_updated.emit(self.total_signals_count)

                if data is None or header is None:
                    print("No valid data received, skipping")
                    self.broken_signals_count += 1
                    self.signals.broken_signals_count_updated.emit(self.broken_signals_count)
                    continue

                # --- ML PREDICTION (all 3 models) ---
                if self.model_predictor is not None:
                    try:
                        signal_np = np.array(data.values, dtype=np.float32)
                        predictions = self.model_predictor.predict_all(signal_np)
                        self.signals.prediction.emit(predictions)
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        traceback.print_exc()

                # --- SAVE DATA ---
                if self.saving_number_of_signals is not None:
                    if self.saved_signals_count < self.saving_number_of_signals:
                        self.save_data(header, data)
                        self.saved_signals_count += 1
                    else:
                        self.signals.saved_signals_count_updated.emit(self.saved_signals_count)
                else:
                    self.saved_signals_count = 0
                    self.total_signals_count = 0
                    self.broken_signals_count = 0

            except:
                traceback.print_exc()
                exctype, value = sys.exc_info()[:2]
                self.signals.error.emit((exctype, value, traceback.format_exc()))
            else:
                self.signals.result.emit(data)
            finally:
                self.signals.finished.emit()

    def save_data(self, header, data):
        combined_df = pd.concat([header, data]).to_frame().transpose()
        file_path = f"{self.dataFilePath}/signal_{self.saving_number_of_signals}.csv"
        if not os.path.exists(self.dataFilePath):
            os.makedirs(self.dataFilePath)
        combined_df.to_csv(file_path, mode='a', index=False, header=False)

    def set_saving_number_of_signals(self, n):
        self.saving_number_of_signals = n

    def set_dataFilePath(self, path):
        self.dataFilePath = path

    def stop(self):
        self.is_running = False


# ==============================================================
#  MODEL RESULT WIDGET (reusable for each model)
# ==============================================================

class ModelResultWidget(QFrame):
    """A compact widget showing one model's prediction result."""

    def __init__(self, model_name, accent_color="#3498db", parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.accent_color = accent_color

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            ModelResultWidget {{
                border: 1px solid #555;
                border-radius: 6px;
                padding: 5px;
                margin: 2px;
            }}
        """)

        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(8, 6, 8, 6)

        # Model name
        self.title = QLabel(model_name)
        self.title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet(f"color: {accent_color};")
        layout.addWidget(self.title)

        # Class prediction (large)
        self.class_label = QLabel("Waiting...")
        self.class_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.class_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.class_label.setStyleSheet("color: #95a5a6; padding: 4px;")
        layout.addWidget(self.class_label)

        # Confidence
        self.conf_label = QLabel("Confidence: —")
        self.conf_label.setFont(QFont("Segoe UI", 9))
        self.conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.conf_label)

        # Probabilities bar
        self.prob_label = QLabel("H: —  C: —  N: —")
        self.prob_label.setFont(QFont("Consolas", 8))
        self.prob_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prob_label.setStyleSheet("color: #888;")
        layout.addWidget(self.prob_label)

        # Timing
        self.time_label = QLabel("—")
        self.time_label.setFont(QFont("Segoe UI", 8))
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet("color: #666;")
        layout.addWidget(self.time_label)

        self.setLayout(layout)

    def update_result(self, result_dict):
        """Update display with prediction result."""
        cls = result_dict['class']
        conf = result_dict['confidence']
        t_ms = result_dict['time_ms']
        probs = result_dict['probabilities']

        # Class name + color
        self.class_label.setText(cls)
        color = LABEL_COLORS.get(cls, '#95a5a6')
        self.class_label.setStyleSheet(f"color: {color}; padding: 4px;")

        # Confidence
        self.conf_label.setText(f"Confidence: {conf*100:.1f}%")

        # Probabilities
        self.prob_label.setText(
            f"H:{probs['Human']*100:5.1f}%  C:{probs['Chair']*100:5.1f}%  N:{probs['Nothing']*100:5.1f}%"
        )

        # Timing
        self.time_label.setText(f"Inference: {t_ms:.1f}ms")


# ==============================================================
#  MAIN WINDOW
# ==============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- Load ML Models ---
        self.model_predictor = ModelPredictor()

        # --- Sensors ---
        self.rp_sensor = RedPitayaSensor()
        self.sim_sensor = SimulatedSensor()
        self.active_sensor = self.rp_sensor
        self.simulation_mode = False

        self.start_time = None
        self.header_info = None
        self.threadpool = QThreadPool()
        self.sensor_status_message = self.rp_sensor.get_sensor_status_message()
        self.app_status_message = "App Started"

        self.button_is_checked = True
        self.realtime_chkbox_checked = False
        self.show_region_to_select = False
        self.raw_adc_data = None
        self.previous_range_selector_region = (100, 1000)

        self.setWindowTitle("FIUS Sensor — Real-Time Object Classification (3 Models)")
        self.setMinimumSize(1300, 750)

        # ==================== GUI LAYOUT ====================
        main_layout = QGridLayout()

        # --- Plot widget ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1a1a2e')
        self.plot_widget.setTitle("Ultrasonic Echo Signal", color='#eee', size='12pt')
        self.plot_widget.setLabel('left', 'Amplitude', color='#aaa')
        self.plot_widget.setLabel('bottom', 'Sample', color='#aaa')
        main_layout.addWidget(self.plot_widget, 0, 0, 1, 2)

        # ==================================================
        # PREDICTION PANEL (RIGHT SIDE) — 3 model results
        # ==================================================
        prediction_group = QGroupBox("  Classification Results")
        prediction_group.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #444;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
        """)
        pred_layout = QVBoxLayout()
        pred_layout.setSpacing(4)

        # Transformer (primary)
        self.transformer_widget = ModelResultWidget("1D-Transformer", "#9b59b6")
        pred_layout.addWidget(self.transformer_widget)

        # ResNet
        self.resnet_widget = ModelResultWidget("1D-ResNet", "#e74c3c")
        pred_layout.addWidget(self.resnet_widget)

        # CNN
        self.cnn_widget = ModelResultWidget("1D-CNN", "#3498db")
        pred_layout.addWidget(self.cnn_widget)

        # Consensus label
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #555;")
        pred_layout.addWidget(sep)

        self.consensus_label = QLabel("Consensus: —")
        self.consensus_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.consensus_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.consensus_label.setStyleSheet("color: #95a5a6; padding: 8px;")
        pred_layout.addWidget(self.consensus_label)

        # Model load status
        status_parts = []
        for name in ['Transformer', 'ResNet', 'CNN']:
            s = "✅" if self.model_predictor.loaded.get(name, False) else "❌"
            status_parts.append(f"{s} {name}")
        self.status_label = QLabel("  ".join(status_parts))
        self.status_label.setFont(QFont("Segoe UI", 8))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #888;")
        pred_layout.addWidget(self.status_label)

        pred_layout.addStretch()
        prediction_group.setLayout(pred_layout)
        prediction_group.setFixedWidth(300)
        main_layout.addWidget(prediction_group, 0, 2, 5, 1)

        # ==================================================
        # CONTROLS
        # ==================================================
        self.controls_layout = QHBoxLayout()
        self.realtime_chkbox = QCheckBox("Realtime")
        self.controls_layout.addWidget(self.realtime_chkbox)
        self.show_region_chkbox = QCheckBox("Region-Select")
        self.controls_layout.addWidget(self.show_region_chkbox)
        self.confirm_region_btn = QPushButton("Confirm")
        self.controls_layout.addWidget(self.confirm_region_btn)
        self.simulation_chkbox = QCheckBox("Simulation Mode (no sensor)")
        self.simulation_chkbox.setStyleSheet("color: #e67e22; font-weight: bold;")
        self.controls_layout.addWidget(self.simulation_chkbox)
        main_layout.addLayout(self.controls_layout, 1, 0, 1, 2)

        # Messages
        self.msg_layout = QHBoxLayout()
        self.server_message_widget = QLabel(self.sensor_status_message)
        self.msg_layout.addWidget(self.server_message_widget)
        self.app_message_widget = QLabel(self.app_status_message)
        self.msg_layout.addWidget(self.app_message_widget)
        self.total_signal_count_message_widget = QLabel("Total: 0")
        self.msg_layout.addWidget(self.total_signal_count_message_widget)
        self.broken_signal_count_message_widget = QLabel("Broken: 0")
        self.msg_layout.addWidget(self.broken_signal_count_message_widget)
        main_layout.addLayout(self.msg_layout, 2, 0, 1, 2)

        # Saving path
        self.path_layout = QHBoxLayout()
        self.path_label = QLabel("Save Path:")
        self.file_path_line_edit = QLineEdit()
        self.path_layout.addWidget(self.path_label)
        self.path_layout.addWidget(self.file_path_line_edit)
        main_layout.addLayout(self.path_layout, 3, 0, 1, 2)

        # Signal count + save
        self.save_layout = QHBoxLayout()
        self.signal_numbers_label = QLabel("Signals to save:")
        self.signal_numbers_line_edit = QLineEdit()
        self.save_data_btn = QPushButton("Save")
        self.save_layout.addWidget(self.signal_numbers_label)
        self.save_layout.addWidget(self.signal_numbers_line_edit)
        self.save_layout.addWidget(self.save_data_btn)
        main_layout.addLayout(self.save_layout, 4, 0, 1, 2)

        # SSH
        self.ssh_layout = QHBoxLayout()
        self.start_sensor_btn = QPushButton("Start Sensor")
        self.stop_sensor_btn = QPushButton("Stop Sensor")
        self.ssh_layout.addWidget(self.start_sensor_btn)
        self.ssh_layout.addWidget(self.stop_sensor_btn)
        main_layout.addLayout(self.ssh_layout, 5, 0, 1, 2)

        # Central widget
        self.widget = QWidget()
        self.widget.setLayout(main_layout)
        self.setCentralWidget(self.widget)

        self.range_selector = pg.LinearRegionItem()

        # Connect signals
        self.show_region_chkbox.stateChanged.connect(self.show_region_handler)
        self.realtime_chkbox.stateChanged.connect(self.realtime_checkbox_handler)
        self.confirm_region_btn.clicked.connect(self.confirm_region_selection_btn_handler)
        self.simulation_chkbox.stateChanged.connect(self.simulation_checkbox_handler)

        self.worker = None
        self.save_data_btn.clicked.connect(self.save_data_btn_handler)
        self.start_sensor_btn.clicked.connect(self.start_sensor_btn_handler)
        self.stop_sensor_btn.clicked.connect(self.stop_sensor_btn_handler)

    # ==================== SIMULATION ====================

    def simulation_checkbox_handler(self, state):
        if state == Qt.CheckState.Checked.value:
            self.simulation_mode = True
            self.active_sensor = self.sim_sensor
            self.start_time = time.time() * 1000
            self.server_message_widget.setText("SIMULATION — Check Realtime to start")
            self.start_sensor_btn.setDisabled(True)
            self.stop_sensor_btn.setDisabled(True)
        else:
            self.simulation_mode = False
            self.active_sensor = self.rp_sensor
            self.server_message_widget.setText(self.rp_sensor.get_sensor_status_message())
            self.start_sensor_btn.setDisabled(False)
            self.stop_sensor_btn.setDisabled(False)

    # ==================== PREDICTION DISPLAY ====================

    def update_prediction_display(self, predictions):
        """Update all 3 model widgets + consensus."""
        # Update individual models
        self.transformer_widget.update_result(predictions.get('Transformer', {
            'class': 'No Model', 'confidence': 0, 'time_ms': 0,
            'probabilities': {'Human': 0, 'Chair': 0, 'Nothing': 0}
        }))
        self.resnet_widget.update_result(predictions.get('ResNet', {
            'class': 'No Model', 'confidence': 0, 'time_ms': 0,
            'probabilities': {'Human': 0, 'Chair': 0, 'Nothing': 0}
        }))
        self.cnn_widget.update_result(predictions.get('CNN', {
            'class': 'No Model', 'confidence': 0, 'time_ms': 0,
            'probabilities': {'Human': 0, 'Chair': 0, 'Nothing': 0}
        }))

        # Consensus: majority vote among loaded models
        votes = []
        for name in ['Transformer', 'ResNet', 'CNN']:
            pred = predictions.get(name, {})
            cls = pred.get('class', 'No Model')
            if cls != 'No Model':
                votes.append(cls)

        if votes:
            from collections import Counter
            vote_counts = Counter(votes)
            consensus_class = vote_counts.most_common(1)[0][0]
            consensus_count = vote_counts.most_common(1)[0][1]
            total_models = len(votes)

            color = LABEL_COLORS.get(consensus_class, '#95a5a6')

            if consensus_count == total_models:
                self.consensus_label.setText(f"Consensus: {consensus_class} ({total_models}/{total_models})")
            else:
                self.consensus_label.setText(f"Majority: {consensus_class} ({consensus_count}/{total_models})")

            self.consensus_label.setStyleSheet(f"color: {color}; padding: 8px;")
        else:
            self.consensus_label.setText("Consensus: —")
            self.consensus_label.setStyleSheet("color: #95a5a6; padding: 8px;")

    # ==================== ORIGINAL HANDLERS ====================

    def show_region_handler(self, state):
        self.server_message_widget.setText(self.active_sensor.get_sensor_status_message())
        if state == Qt.CheckState.Checked.value:
            self.realtime_chkbox.setDisabled(True)
            self.confirm_region_btn.setDisabled(False)
            self.show_region_to_select = True
            self.range_selector = pg.LinearRegionItem()
            self.range_selector.sigRegionChangeFinished.connect(self.region_changed_on_linear_region)
            self.range_selector.setRegion(self.previous_range_selector_region)
            self.plot_widget.addItem(self.range_selector)
        elif state == Qt.CheckState.Unchecked.value:
            self.reset_btn_view()
            self.plot_widget.removeItem(self.range_selector)

    def confirm_region_selection_btn_handler(self):
        if self.show_region_to_select:
            self.previous_range_selector_region = self.range_selector.getRegion()
            self.plot_adc_data()
            self.show_region_handler(self.show_region_chkbox.checkState().value)

    def reset_btn_view(self):
        self.realtime_chkbox.setDisabled(False)
        self.show_region_chkbox.setDisabled(False)
        self.confirm_region_btn.setDisabled(True)

    def region_changed_on_linear_region(self):
        pass

    def plot_adc_data(self, data=None):
        self.server_message_widget.setText(self.active_sensor.get_sensor_status_message())
        self.plot_widget.clear()
        if data is not None:
            y = data
            x = list(range(len(y)))
            self.raw_adc_data = y
            self.plot_widget.plot(x, y, pen=pg.mkPen('#00d4ff', width=1))
        if not self.realtime_chkbox_checked:
            return False

    def realtime_checkbox_handler(self, state):
        if state == Qt.CheckState.Checked.value:
            self.realtime_chkbox_checked = True
            self.show_region_chkbox.setDisabled(True)
            self.confirm_region_btn.setDisabled(True)
            self.simulation_chkbox.setDisabled(True)

            self.worker = Worker(
                self.func_is_realtime_checked,
                self.active_sensor,
                self.model_predictor
            )
            self.worker.signals.result.connect(self.plot_adc_data)
            self.worker.signals.prediction.connect(self.update_prediction_display)
            self.worker.signals.saved_signals_count_updated.connect(self.update_save_button_state)
            self.worker.signals.total_signals_count_updated.connect(self.total_signal_status_message_set)
            self.worker.signals.broken_signals_count_updated.connect(self.broken_signal_status_message_set)
            self.threadpool.start(self.worker)
        else:
            self.realtime_chkbox_checked = False
            self.reset_btn_view()
            self.simulation_chkbox.setDisabled(False)

    def func_is_realtime_checked(self):
        return self.realtime_chkbox_checked

    def app_status_message_set(self, text):
        self.app_status_message = text

    def app_status_message_get(self):
        return self.app_status_message

    def broken_signal_status_message_set(self, count):
        self.broken_signal_count_message_widget.setText(f"Broken: {count}")

    def total_signal_status_message_set(self, count):
        self.total_signal_count_message_widget.setText(f"Total: {count}")

    def save_data_btn_handler(self):
        self.dataFilePath = self.file_path_line_edit.text()
        self.worker.set_dataFilePath(self.dataFilePath)
        self.saving_number_of_signals = int(self.signal_numbers_line_edit.text())
        self.worker.set_saving_number_of_signals(self.saving_number_of_signals)
        self.app_status_message_set(f"Saving {self.saving_number_of_signals} signals...")
        self.app_message_widget.setText(self.app_status_message_get())
        self.file_path_line_edit.clear()
        self.signal_numbers_line_edit.clear()
        self.save_data_btn.setDisabled(True)

    def start_sensor_btn_handler(self):
        commands = ["cd /usr/RedPitaya/Examples/C", "./dma_with_udp_faster"]
        full_command = " && ".join(commands)
        self.rp_sensor.give_ssh_command(full_command)
        time.sleep(3)
        self.start_time, self.header_info = self.rp_sensor.get_data_info_from_server()
        time.sleep(1)

    def stop_sensor_btn_handler(self):
        command = "pidof dma_with_udp_faster"
        pid = self.rp_sensor.give_ssh_command(command)
        command1 = f"kill {pid}"
        self.rp_sensor.give_ssh_command(command1)

    def update_save_button_state(self, count):
        self.save_data_btn.setDisabled(False)
        self.app_status_message_set(f"Saved {self.saving_number_of_signals} signals ✅")
        self.app_message_widget.setText(self.app_status_message_get())
        self.saving_number_of_signals = None
        self.worker.set_saving_number_of_signals(None)
        self.dataFilePath = None
        self.worker.set_dataFilePath(None)

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            self.threadpool.waitForDone()
        event.accept()


# ==============================================================
#  MAIN
# ==============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
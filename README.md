# ⚡ BioWave EMG & IMU Armband Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-ESP32--S3%20%2B%20BNO080%20IMU-orange.svg)](https://www.espressif.com/)
[![Sampling Rate](https://img.shields.io/badge/Sampling-500Hz%20EMG%20%7C%20200Hz%20IMU-red.svg)]()
[![ML Frameworks](https://img.shields.io/badge/ML-Random%20Forest%20%7C%201D--ResNet-purple.svg)]()
[![HCI Protocol](https://img.shields.io/badge/HCI%20Standard-ISO%209241--9%20%2F%20411-green.svg)]()
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE)

> **BioWave** is an end-to-end biosignal acquisition, machine learning, human-computer interface (HCI), and research experimentation platform. It captures **8-channel surface electromyography (sEMG)** and **9-DOF BNO080 IMU orientation** via wireless ESP32-S3 or USB serial, processes signals through real-time causal DSP filters, classifies gestures using Random Forest or 1D-CNN architectures, and drives adaptive human-interface devices (HID) with ISO 9241-9 validated Fitts' Law benchmarks.

---

## 📑 Table of Contents

- [✨ System Highlights](#-system-highlights)
- [🏗️ System Architecture & Data Flow](#️-system-architecture--data-flow)
- [💻 System Requirements](#-system-requirements)
- [🚀 Quick Start & Installation](#-quick-start--installation)
  - [🍏 macOS Setup (Apple Silicon M1-M4 & Intel)](#-macos-setup-apple-silicon-m1m2m3m4--intel)
  - [🪟 Windows Setup (PowerShell & Command Prompt)](#-windows-setup-powershell--cmd)
  - [🐧 Linux / Ubuntu Setup](#-linux--ubuntu-setup)
- [🎮 How to Run the Applications](#-how-to-run-the-applications)
  - [1. Main Biosignal Studio (`main.py`)](#1-main-biosignal-studio-mainpy)
  - [2. Standalone Adaptive Mouse Controller (`mouse_controller.py`)](#2-standalone-adaptive-mouse-controller-mouse_controllerpy)
  - [3. BioWave Lab Suite & ISO 9241-9 Task (`biowave_lab_suite.py`)](#3-biowave-lab-suite--iso-9241-9-task-biowave_lab_suitepy)
  - [4. Synthetic EMG Simulator (`emg_simulator_app.py`)](#4-synthetic-emg-simulator-emg_simulator_apppy)
- [🦾 Hardware & Firmware Architecture](#-hardware--firmware-architecture)
  - [ESP32-S3 Dual-Core Firmware](#esp32-s3-dual-core-firmware)
  - [High-Throughput Binary Packet Protocol (`BWIM`)](#high-throughput-binary-packet-protocol-bwim)
  - [Wi-Fi Provisioning & HMAC Security](#wi-fi-provisioning--hmac-security)
- [📊 End-to-End Workflow Guide](#-end-to-end-workflow-guide)
- [🧠 Signal Processing & Feature Contract](#-signal-processing--feature-contract)
  - [231-Dimensional Canonical Feature Vector](#231-dimensional-canonical-feature-vector)
  - [Temporal Gating & Majority Vote Decision Engine](#temporal-gating--majority-vote-decision-engine)
- [🤖 Machine Learning Models](#-machine-learning-models)
  - [Random Forest Classifier](#random-forest-classifier)
  - [Deep 1D-ResNet / CNN Pipeline](#deep-1d-resnet--cnn-pipeline)
- [🧪 ISO 9241-9 Evaluation & Figures](#-iso-9241-9-evaluation--figures)
- [🔬 Unit Testing & Quality Assurance](#-unit-testing--quality-assurance)
- [📁 Repository File Tree](#-repository-file-tree)
- [🛠️ Troubleshooting & Diagnostic FAQ](#️-troubleshooting--diagnostic-faq)

---

## ✨ System Highlights

<table>
  <tr>
    <td width="50%">
      <h3>📡 Real-Time Biosignal Acquisition</h3>
      <ul>
        <li><b>8-Channel sEMG:</b> 500 Hz high-fidelity sampling with 12-bit resolution.</li>
        <li><b>9-DOF IMU:</b> 200 Hz fused Pitch, Roll, and Yaw orientation from Hillcrest BNO080 sensor.</li>
        <li><b>High-Throughput Wireless:</b> Custom binary UDP streaming (<code>BWIM</code> protocol) with HMAC-SHA256 authenticated control.</li>
        <li><b>USB Serial CDC & TCP Simulator:</b> Plug-and-play wired connectivity or simulated socket stream for offline research.</li>
      </ul>
    </td>
    <td width="50%">
      <h3>🧠 Resilient ML & HCI Core</h3>
      <ul>
        <li><b>231-D Feature Extraction:</b> Time-domain, spectral, entropy, band-power, and cross-channel covariance metrics.</li>
        <li><b>Zero-Drift Shared Contract:</b> Unified feature engineering shared across training, live inference, and standalone control.</li>
        <li><b>Adaptive Decision Engine:</b> Majority-voting action buffer (60% agreement), confidence gating, and debounce filtering.</li>
        <li><b>ISO 9241-9 Fitts' Law Suite:</b> Full multidirectional tapping task, throughput index (bits/s), and publication-ready vector figure generation.</li>
      </ul>
    </td>
  </tr>
</table>

---

## 🏗️ System Architecture & Data Flow

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   HARDWARE / SOURCE LAYER                              │
│   ┌──────────────────────────────────────────┐    ┌────────────────────────────────┐   │
│   │ ESP32-S3 (Dual-Core FreeRTOS)            │    │ Synthetic EMG Simulator        │   │
│   │  • Core 1: 500 Hz Timer ADC (8CH EMG)    │    │ (emg_simulator_app.py)         │   │
│   │  • Core 0: BNO080 IMU + UDP Streamer     │    │  • Multi-channel sine / noise  │   │
│   └────────────────────┬─────────────────────┘    └───────────────┬────────────────┘   │
└────────────────────────┼──────────────────────────────────────────┼────────────────────┘
                         │ UDP `BWIM` (Port 5000/5001)              │ TCP Socket (7000)
                         ▼                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              SHARED DSP & PIPELINE CORE                                │
│   ┌──────────────────────────┐ ┌───────────────────────────┐ ┌──────────────────────┐  │
│   │ realtime_pipeline.py     │ │ emg_v4_core.py            │ │ rf_features.py       │  │
│   │ • Lock-free Ring Buffer  │ │ • Rest/Flex Calibration   │ │ • 231-D Feature      │  │
│   │ • Stage-by-Stage Profiler│ │ • Causal Bandpass/Notch   │ │   Extraction Contract│  │
│   │ • Discontinuity Recovery │ │ • Signal Quality Gating   │ │ • Pairwise Covariance│  │
│   └────────────┬─────────────┘ └─────────────┬─────────────┘ └──────────┬───────────┘  │
└────────────────┼─────────────────────────────┼──────────────────────────┼──────────────┘
                 │                             │                          │
                 ▼                             ▼                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   APPLICATION SUITE                                    │
│   ┌────────────────────────┐  ┌─────────────────────────┐  ┌────────────────────────┐  │
│   │ main.py                │  │ mouse_controller.py     │  │ biowave_lab_suite.py   │  │
│   │ • Live Oscilloscope    │  │ • Standalone HID Mouse  │  │ • ISO 9241-9 Fitts Task│  │
│   │ • Guided Data Collect  │  │ • Majority Vote Filter  │  │ • Metric Analysis      │  │
│   │ • RF Model Trainer     │  │ • Sub-20ms Latency      │  │ • Publication Figures  │  │
│   │ • Live Classification  │  │ • Direct Lab Suite Link │  │ • Cross-Session Matrix │  │
│   └────────────────────────┘  └─────────────────────────┘  └────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 💻 System Requirements

| Specification | Minimum Requirement | Recommended |
| :--- | :--- | :--- |
| **Operating System** | macOS 11+ (Big Sur / Monterey / Ventura / Sonoma / Sequoia), Windows 10/11 (64-bit), Ubuntu 20.04+ | macOS Sequoia (Apple Silicon) or Windows 11 |
| **Python Version** | `Python 3.10` | `Python 3.11` or `Python 3.12` |
| **Processor** | Dual-core 2.0 GHz | Apple Silicon (M1/M2/M3/M4) or Intel/AMD 4+ Cores |
| **Memory (RAM)** | 4 GB | 8 GB or higher |
| **Display** | 1280 × 800 (Auto-scroll enabled for small laptops) | 1920 × 1080 or Retina Display |
| **Hardware Ports** | 1× USB Port (CP210x / CH340 / USB CDC) | 2.4 GHz 802.11 b/g/n Wi-Fi Router |

---

## 🚀 Quick Start & Installation

Choose your operating system below for detailed setup instructions:

<details open>
<summary><b>🍏 macOS Setup (Apple Silicon M1/M2/M3/M4 & Intel)</b></summary>
<br>

1. **Prerequisites & Homebrew:**
   Make sure you have Python 3.10+ installed:
   ```bash
   brew install python@3.12
   ```

2. **Clone & Create Virtual Environment:**
   ```bash
   cd "/path/to/BioWaveEMG_ArmBand"
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. **Grant macOS Accessibility Permissions (CRITICAL for Mouse Controller):**
   > [!IMPORTANT]
   > macOS blocks applications from controlling the mouse cursor by default. 
   > Open **System Settings** → **Privacy & Security** → **Accessibility** and ensure your **Terminal** / **iTerm2** / **IDE (VS Code / PyCharm / Antigravity)** is toggled **ON** (`Enabled`).

</details>

<details>
<summary><b>🪟 Windows Setup (PowerShell & Command Prompt)</b></summary>
<br>

1. **Install Python:**
   Download and install Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/).
   > [!WARNING]
   > Make sure to check the box **"Add python.exe to PATH"** during setup.

2. **Open PowerShell as Administrator & Set Execution Policy (if needed):**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Create & Activate Virtual Environment:**
   ```powershell
   cd "C:\path\to\BioWaveEMG_ArmBand"
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. **Install Dependencies:**
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. **Windows Firewall Rule for Wireless UDP:**
   If Windows Defender Firewall prompts you when running UDP streaming, click **"Allow Access"** for Private Networks (Port `5000` data & `5001` control).

</details>

<details>
<summary><b>🐧 Linux / Ubuntu Setup</b></summary>
<br>

1. **Install System Packages & Python:**
   ```bash
   sudo apt update
   sudo apt install -y python3-venv python3-pip libgl1-mesa-glx libegl1-mesa libxcb-xinerama0
   ```

2. **Serial Permissions (for USB CDC):**
   ```bash
   sudo usermod -a -G dialout $USER
   # Log out and log back in for group changes to take effect
   ```

3. **Create & Activate Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

</details>

---

## 🎮 How to Run the Applications

All applications are located in the `code/` directory.

### 1. Main Biosignal Studio (`main.py`)
The primary desktop application for live multi-channel signal visualization, sensor calibration, guided data collection, Random Forest model training, and real-time prediction monitoring.

```bash
# Activate your environment
source .venv/bin/activate    # macOS / Linux
# .\.venv\Scripts\Activate.ps1  # Windows

python code/main.py
```

---

### 2. Standalone Adaptive Mouse Controller (`mouse_controller.py`)
A fast, lightweight, and self-sufficient human-interface application that maps decoded gestures directly to mouse clicks, continuous directional tracking, and scrolling.

```bash
python code/mouse_controller.py
```

* **No Overhead:** Runs independently without loading heavy plotting canvases.
* **Majority-Vote Action Buffer:** Filters single-window noise using a 5-window rolling vote (`60% agreement`).
* **Instant Lab Suite Integration:** Click the **"Open Lab Suite"** button to immediately launch ISO 9241-9 benchmarks against your live connection.

---

### 3. BioWave Lab Suite & ISO 9241-9 Task (`biowave_lab_suite.py`)
The merged experimental suite for human factors and motor performance research:

```bash
python code/biowave_lab_suite.py
```

* **Experiment Manager:** Run matched controller trials across 4 standard conditions:
  1. *Standard Mouse*
  2. *EMG Mouse*
  3. *Fixed-Step Controller*
  4. *Adaptive Controller*
* **ISO 9241-9 Multidirectional Tapping Task:** Standard 13-target circular tapping task with customizable target amplitude ($A$) and width ($W$).
* **Performance Analysis:** Calculates Throughput ($TP = \frac{ID_e}{MT}$ in bits/s), Movement Time ($MT$), Error Rate ($ER$), and Target Re-entries.
* **Publication Figures:** Preview and export publication-ready plots (PDF / PNG).
* **Session Comparison:** Multi-session comparison matrix across dates and participants.

---

### 4. Synthetic EMG Simulator (`emg_simulator_app.py`)
Enables full development and testing of all desktop tools without requiring physical hardware.

```bash
python code/emg_simulator_app.py
```

**How to connect to the simulator:**
1. In `emg_simulator_app.py`, select **TCP Server (Recommended)** and click **Start Simulator** (Default port: `7000`).
2. In `main.py` or `mouse_controller.py`, set connection type to **TCP / IP** and enter:
   ```text
   socket://127.0.0.1:7000
   ```
3. Click **Connect** to begin streaming synthetic 8-channel EMG data.

---

## 🦾 Hardware & Firmware Architecture

### ESP32-S3 Dual-Core Firmware
The ESP32-S3 firmware is located at [`ESP32_S3 Setup/EMG_IMU_WIFI_SETUP.ino`](file:///Users/shimulkumarshoishob/Documents/CAPSTON_PROJECT/AntiGravity_IDE/BioWaveEMG_ArmBand/ESP32_S3%20Setup/EMG_IMU_WIFI_SETUP.ino). It leverages FreeRTOS dual-core multitasking:

* **Core 1 (Time-Critical Acquisition):** Hardware timer interrupt triggering synchronous 8-channel analog sampling via ADC at exactly **500 Hz** (2 ms period).
* **Core 0 (Sensors, Comm & Network):** SPI/I2C communication with Hillcrest BNO080 IMU at **200 Hz**, USB serial command parser, Wi-Fi packet assembly, and low-latency UDP broadcast.

### High-Throughput Binary Packet Protocol (`BWIM`)
To eliminate transmission overhead, frames are batched (5 frames per packet):

| Field | Type | Size | Description |
| :--- | :--- | :--- | :--- |
| **Magic** | `char[4]` | 4 Bytes | Protocol identifier: `"BWIM"` |
| **Version** | `uint8_t` | 1 Byte | Protocol version (currently `1`) |
| **Frame Count** | `uint8_t` | 1 Byte | Number of frames in payload (default: `5`) |
| **Frame Size** | `uint16_t` | 2 Bytes | Size of each sub-frame (default: `48` bytes) |
| **Sequence** | `uint32_t` | 4 Bytes | Monotonically increasing packet sequence counter |
| **Frame Payload** | `Frame[5]` | 240 Bytes | 5 consecutive 48-byte frames |

#### 48-Byte Frame Payload Structure:
```text
┌─────────────────────────┬──────────┬───────────────────────────────────────────┐
│ Field                   │ Type     │ Description                               │
├─────────────────────────┼──────────┼───────────────────────────────────────────┤
│ frameId                 │ uint32_t │ Frame index counter                       │
│ frameTimestampUs        │ uint32_t │ Microsecond acquisition timestamp         │
│ imuSampleId             │ uint32_t │ IMU sequence identifier                   │
│ imuTimestampUs          │ uint32_t │ IMU sample timestamp                      │
│ emg[8]                  │ uint16_t │ 8 raw EMG ADC values (0–4095)             │
│ roll, pitch, yaw        │ float[3] │ Fused Euler orientation angles (degrees)  │
│ imuFresh                │ uint8_t  │ 1 if new IMU sample, 0 if repeated        │
│ reserved[3]             │ uint8_t  │ 3-byte boundary alignment padding         │
└─────────────────────────┴──────────┴───────────────────────────────────────────┘
```

### Wi-Fi Provisioning & HMAC Security
* **USB Provisioning:** Connect the ESP32-S3 via USB and send AT commands over serial at `115200` baud:
  ```text
  SET_WIFI:YourSSID,YourPassword
  ```
* **HMAC-SHA256 Challenge-Response:** Commands sent over UDP port `5001` (e.g., `START_STREAM`, `STOP_STREAM`) must include an HMAC-SHA256 signature calculated with the shared `DEVICE_ACCESS_KEY` configured in the firmware.

---

## 📊 End-to-End Workflow Guide

Follow this 7-step guide to take your project from raw signals to real-time control:

```
Step 1: Connect Device   ──► Step 2: Calibrate Baselines ──► Step 3: Record Dataset
                                                                    │
Step 6: Live Control     ◄── Step 5: Validate Artifact   ◄── Step 4: Train Model
        │
Step 7: ISO 9241-9 Benchmarks
```

1. **Connect Device:**
   Launch `main.py`, select your COM Port (or UDP Wi-Fi), and click **Connect**.
2. **Channel Baseline Calibration:**
   Hold arms in a relaxed state (Rest Baseline for 3s), followed by isometric contraction (Flex Baseline for 3s). The software computes SNR, baseline mean, variance, and flags dead/noisy electrodes.
3. **Guided Data Collection:**
   Choose gestures (e.g., *Rest, Fist Close, Left, Right, Up, Down*), set repetitions (e.g., 5-10 reps), contraction duration (e.g., 2.5s), and rest duration (2.0s). Save recording bundles to `dataset/`.
4. **Train Random Forest Model:**
   Select recorded CSV datasets, set window size (e.g., `200 ms` / 100 samples) and stride (e.g., `50 ms` / 25 samples). Train the classifier and inspect the Confusion Matrix and Classification Report.
5. **Model Verification:**
   The training pipeline produces a `.joblib` bundle in `trained_model/` containing metadata, channel count, class names, feature definitions, and calibration parameters.
6. **Live Real-Time Inference:**
   Load the `.joblib` model into `main.py` or `mouse_controller.py`. Verify live classification probabilities and smooth mouse cursor tracking.
7. **Empirical Evaluation (ISO 9241-9):**
   Launch `biowave_lab_suite.py`, run multi-directional tapping experiments across controller conditions, and generate publication-ready performance curves.

---

## 🧠 Signal Processing & Feature Contract

### 231-Dimensional Canonical Feature Vector
Feature extraction is standardized across the entire ecosystem via `code/rf_features.py`. For an 11-channel input (8 EMG + 3 IMU), the feature vector dimension is calculated as:

$$\text{Total Features} = (N \times 15) + N + \frac{N(N - 1)}{2} = 165 + 11 + 55 = 231$$

For each channel, 15 time & spectral features are extracted:
1. **MAV** – Mean Absolute Value
2. **RMS** – Root Mean Square
3. **IEMG** – Integrated EMG
4. **VAR** – Variance
5. **WL** – Waveform Length
6. **ZC** – Zero Crossing Rate
7. **SSC** – Slope Sign Change Rate
8. **WAMP** – Willison Amplitude
9. **MNF** – Mean Frequency
10. **MDF** – Median Frequency
11. **PKF** – Peak Frequency
12. **SE** – Spectral Entropy
13. **BP1** – Band Power (20–60 Hz)
14. **BP2** – Band Power (60–120 Hz)
15. **BP3** – Band Power (120–220 Hz)

Plus $N$ **RMS Ratio** features and $\frac{N(N-1)}{2}$ **Pairwise Channel Covariance** features.

### Temporal Gating & Majority Vote Decision Engine
To eliminate accidental clicks and motion jitter, `code/emg_v4_core.py` and `code/mouse_controller.py` enforce a multi-tier safety pipeline:

* **Confidence Gate:** Rejects predictions below `EMG_CONFIDENCE_THRESHOLD` (default: `0.75`).
* **Majority Vote Buffer:** Requires at least 3 out of 5 consecutive predictions (`60%`) to agree before an action is dispatched.
* **Click Refractory Period:** Prevents accidental double-clicks with an enforced 250 ms debounce hold.

---

## 🤖 Machine Learning Models

### Random Forest Classifier
* **Fast & Interpretable:** Sub-millisecond CPU inference latency, making it ideal for 50 Hz control loops.
* **Auto-Tuned Ensembles:** 100 decision trees with Gini impurity splitting and bootstrap aggregation.
* **Export Artifacts:** Saved under `trained_model/rf_training_<timestamp>/` with full metadata JSON, confusion matrices, and metrics.

### Deep 1D-ResNet / CNN Pipeline
Located in [`EMG_1D_CNN/`](file:///Users/shimulkumarshoishob/Documents/CAPSTON_PROJECT/AntiGravity_IDE/BioWaveEMG_ArmBand/EMG_1D_CNN), this pipeline allows training deep 1D Residual Networks directly on raw multi-channel temporal waveforms without manual feature engineering.

* **Architecture:** 1D Convolutional blocks with batch normalization, ReLU activation, residual skip connections, and global average pooling.
* **Jupyter Notebook:** Explore training in [`EMG_1D_CNN/1D_ResNet.ipynb`](file:///Users/shimulkumarshoishob/Documents/CAPSTON_PROJECT/AntiGravity_IDE/BioWaveEMG_ArmBand/EMG_1D_CNN/1D_ResNet.ipynb).

---

## 🧪 ISO 9241-9 Evaluation & Figures

The BioWave platform implements the standard ISO 9241-9 (ISO 9241-411) multi-directional discrete pointing task.

```
       ○ 13       ○ 1
   ○ 12               ○ 2
 ○ 11                   ○ 3
○ 10          +          ○ 4    <── Circle of Targets
 ○ 9                    ○ 5
   ○ 8                ○ 6
       ○ 7        ○ 6
```

### Key Performance Metrics Computed:
* **Effective Index of Difficulty ($ID_e$):**
  $$ID_e = \log_2\left(\frac{A_e}{4.133 \times SD_x} + 1\right)$$
* **Throughput ($TP$ in bits/second):**
  $$TP = \frac{ID_e}{MT}$$
* **Path Efficiency & Target Re-entries:** Quantifies cursor wandering and overshoot during gesture-driven navigation.

---

## 🔬 Unit Testing & Quality Assurance

All critical DSP algorithms, ring buffers, temporal gates, and feature extractors are covered by unit tests.

Run the test suite using Python's built-in `unittest`:

```bash
# Run all core DSP and feature tests
PYTHONPATH=code python3 -m unittest discover -s code/tests -p "*.py"
```

Or run directly:
```bash
PYTHONPATH=code python3 code/tests/test_emg_v4_core.py
```

---

## 📁 Repository File Tree

```text
BioWaveEMG_ArmBand/
├── README.md                              # Main documentation & setup guide
├── requirements.txt                       # Python dependencies
├── LICENSE                                # Open-source MIT license
├── ESP32_S3 Setup/
│   └── EMG_IMU_WIFI_SETUP.ino             # FreeRTOS 500Hz EMG + 200Hz IMU firmware
├── code/
│   ├── main.py                            # Full Biosignal Studio (Data collect, Train, Plot)
│   ├── mouse_controller.py                # Standalone Adaptive Mouse Controller
│   ├── biowave_lab_suite.py               # ISO 9241-9 Task, Experiment Manager & Figures
│   ├── emg_simulator_app.py               # Synthetic Multi-Channel EMG TCP Simulator
│   ├── emg_v4_core.py                     # Causal DSP filters, calibration & safety core
│   ├── realtime_pipeline.py               # Lock-free Sample Ring Buffer & Profiler
│   ├── rf_features.py                     # 231-D Canonical Feature Extraction Contract
│   ├── app_theme.py                       # Cross-platform Dark Theme & HiDPI scaling
│   └── tests/
│       └── test_emg_v4_core.py            # Unit tests for core mathematical modules
├── dataset/                               # Structured CSV data bundles & metadata
├── trained_model/                         # Exported .joblib models & evaluation reports
├── EMG_1D_CNN/
│   └── 1D_ResNet.ipynb                    # Deep Learning 1D-ResNet training pipeline
├── updated_emg_sensor_and_dataset_analyser.ipynb # Exploratory data analysis notebook
└── RoboticArm_Capstone-B_FALL-26.mp4      # Demo video of robotic arm & EMG integration
```

---

## 🛠️ Troubleshooting & Diagnostic FAQ

<details>
<summary><b>❓ macOS: Mouse cursor does not move when running <code>mouse_controller.py</code></b></summary>
<br>

* **Root Cause:** macOS requires explicit Accessibility permissions for applications controlling the cursor.
* **Fix:** Open **System Settings** → **Privacy & Security** → **Accessibility**. Ensure your **Terminal** or IDE application is checked and granted access. Restart the terminal after granting permissions.

</details>

<details>
<summary><b>❓ Windows: <code>Activate.ps1 cannot be loaded because running scripts is disabled</code></b></summary>
<br>

* **Root Cause:** PowerShell execution policy restricts script execution by default.
* **Fix:** Run this command in PowerShell:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

</details>

<details>
<summary><b>❓ UDP Stream shows 0 packets received or connects but hangs</b></summary>
<br>

* **Root Cause:** Local firewall blocking incoming UDP broadcast packets on port `5000` or `5001`.
* **Fix:** 
  - Ensure your PC and ESP32-S3 are connected to the same 2.4 GHz Wi-Fi router.
  - Temporarily allow Python through the Windows Defender / macOS Firewall for private networks.
  - Check the `DEVICE_ACCESS_KEY` in `main.py` matches the key configured in `EMG_IMU_WIFI_SETUP.ino`.

</details>

<details>
<summary><b>❓ Model loading error: <code>Feature count mismatch (Expected 231, got X)</code></b></summary>
<br>

* **Root Cause:** The loaded model was trained on a different channel configuration (e.g. 8 EMG-only channels = 156 features vs 8 EMG + 3 IMU channels = 231 features).
* **Fix:** Ensure that the input channel count configured in the app matches the channel count in the model's `training_setup.json`.

</details>

<details>
<summary><b>❓ PyQt5 error: <code>Could not find the Qt platform plugin "cocoa" / "windows"</code></b></summary>
<br>

* **Fix:** Reinstall PyQt5 cleanly inside your active virtual environment:
  ```bash
  pip uninstall -y PyQt5 PyQt5-Qt5 PyQt5-sip
  pip install PyQt5
  ```

</details>

---

## 📄 Citation & License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

If you use BioWave in your academic research, please cite:
```bibtex
@misc{biowave2026,
  title={BioWave: Real-Time sEMG and IMU Human-Computer Interface and ISO 9241-9 Research Platform},
  author={BioWave Team},
  year={2026},
  howpublished={\url{https://github.com/shimulshoishob/BioWaveEMG_ArmBand}}
}
```

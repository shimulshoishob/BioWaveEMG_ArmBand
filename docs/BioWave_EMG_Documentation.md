# BioWave EMG Armband – System Documentation

| | |
|---|---|
| Document | Technical documentation for the desktop application and ESP32-S3 firmware |
| Application version | 2.0.0 (`APP_VERSION` in `code/main.py`) |
| Firmware version | `emg-imu-2.0.0` (`FW_VERSION`), wire protocol **BWIM v2** |
| Sampling rate | 500 Hz, 8 EMG channels + 3 IMU angles |
| Status | Describes branch `publication-grade-dataset`. Firmware compiled but not yet validated on hardware. |

---

## 1. Purpose and scope

BioWave is a research system for collecting, inspecting and classifying surface-EMG gesture data from an 8-channel armband with an inertial sensor. It consists of:

1. **Firmware** (`ESP32_S3 Setup/EMG_IMU_WIFI_SETUP_V2.ino`) that samples the sensors and streams them over Wi-Fi (UDP) or USB.
2. **Desktop application** (`code/main.py`, PyQt5) that receives the stream, shows it live, runs a timed data-collection protocol, records the data, and trains/runs a Random-Forest gesture classifier.
3. **Supporting modules and tools** for packet sequencing, recording, study metadata, and repair of pilot data.

The design goal for data collection is that **every saved sample is traceable to a device frame, every gap is explicit, every label boundary is sample-accurate, and no participant names are stored.**

## 2. System overview

```
 8x EMG (ADC)  ┐                               ┌─ Wi-Fi: UDP 5000 (data), UDP 5001 (control)
 BNO080 IMU    ┴─► ESP32-S3 firmware ─ BWIM v2 ┤
                   (core 1: sampling)           └─ USB serial: COBS + CRC16 framed BWIM v2
                   (core 0: network, IMU)                         │
                                                                  ▼
   Desktop app:  receiver thread ─► PacketSequencer/StreamAssembler ─► GUI (live view, analysis)
                                                  │
                                                  └─► ContinuousRecorder (writer thread) ─► raw_*.bin
                                                        events (phase changes) ─► post-processing
                                                        ─► signals_*.csv, events_*.csv, quality_*.json, metadata_*
```

### 2.1 Repository layout

| Path | Content |
|---|---|
| `ESP32_S3 Setup/EMG_IMU_WIFI_SETUP_V2.ino` | Current firmware |
| `ESP32_S3 Setup/secrets.example.h` | Template for the gitignored `secrets.h` (device key) |
| `code/main.py` | Desktop application |
| `code/emg_v4_core.py` | Pure logic: BWIM parsing, USB framing, sequencing, calibration, preprocessing |
| `code/session_recording.py` | Continuous recorder, event log, labelling, quality report |
| `code/study_meta.py` | Participant IDs, folder naming, counterbalancing, metadata JSON |
| `code/tools/repair_sessions.py` | Repair tool for pilot recordings |
| `code/gesture_game_app.py` | Separate demo game driven by predicted gestures |
| `code/tests/` | Automated tests (`pytest`) |
| `dataset/`, `trained_model/`, `experiment_results/` | Data and models; not modified by the tooling |

## 3. Firmware

### 3.1 Hardware

| Item | Detail |
|---|---|
| MCU | ESP32-S3 (build target: *ESP32S3 Dev Module*, Arduino core esp32 3.3.x) |
| EMG inputs | GPIO 1, 2, 4, 5, 6, 7, 9, 10; 12-bit ADC, 11 dB attenuation (keep signals within 0–3.3 V) |
| IMU | BNO080/BNO08x over SPI (RST 11, INT 12, CS 13, MOSI 14, MISO 15, SCK 16); rotation vector requested every 5 ms |
| Library | SparkFun BNO08x Arduino Library |

### 3.2 Task architecture

| Task | Core | Priority | Responsibility |
|---|---|---|---|
| Acquisition | 1 | 3 | Reads 8 channels every 2000 µs against an absolute 64-bit deadline, builds 5-frame packets, queues them |
| IMU | 0 | 2 | Polls the BNO08x and publishes the latest roll/pitch/yaw |
| Network | 0 | 2 | Serial commands, control packets, Wi-Fi state machine, sending packets over UDP or USB |

A FreeRTOS queue of 64 packets decouples sampling from sending. If the queue is full the oldest packet is dropped and counted. Each frame carries the latest IMU sample, so IMU values repeat between IMU updates (`imuFresh` marks new ones).

### 3.3 BWIM v2 packet format

Little-endian, packed. One packet = header + 5 frames (≈100 packets/s).

**Header (28 bytes)** – the first 12 bytes are identical to v1 so a parser can read the version first.

| Field | Type | Notes |
|---|---|---|
| `magic` | char[4] | `"BWIM"` |
| `version` | u8 | `2` |
| `frameCount` | u8 | `5` |
| `frameSize` | u16 | `56` |
| `packetSequence` | u32 | wraps at 2³² |
| `queueDrops` | u32 | cumulative since boot |
| `acqOverruns` | u32 | cumulative; frames whose deadline was missed |
| `sendFailures` | u32 | cumulative; `beginPacket`/`endPacket` failures or short USB writes |
| `wifiReconnects` | u16 | cumulative |
| `headerFlags` | u16 | reserved |

Counters are stamped when the packet is sent, so they include failures of earlier packets, not of the packet carrying them.

**Frame (56 bytes)**

| Field | Type | Notes |
|---|---|---|
| `frameId` | u32 | increments every sample, including samples that were never sent |
| `frameTimestampUs` | u64 | `esp_timer`, µs since boot, never wraps |
| `imuSampleId` | u32 | |
| `imuTimestampUs` | u64 | |
| `emg[8]` | u16 | raw ADC counts (0–4095) |
| `roll, pitch, yaw` | f32 | degrees |
| `imuFresh` | u8 | 1 if the IMU sample is new since the previous frame |
| `flags` | u8 | bit 0: frame sampled late (deadline overrun) |
| `reserved` | u16 | |

**BWIM v1** (12-byte header, 48-byte frames, 32-bit microsecond timestamps that wrap after ~71.6 min) is no longer emitted by the firmware. The desktop application still parses it and unwraps its timestamps.

### 3.4 Transports

**Wi-Fi.** The device joins the stored network, listens for control on UDP 5001 and sends data to the host address/port set by `START` (default 5000). `WiFi.setSleep(false)` is used.

**USB serial** (921600 baud). The firmware streams only after a host command:

| Command | Effect |
|---|---|
| `USBSTART` | Start streaming binary frames |
| `USBSTARTCSV` | Start the legacy CSV text stream (`e1..e8,roll,pitch,yaw`), used by the simulator |
| `USBPING` | Keepalive; the stream stops after 5 s without a command |
| `USBSTOP` | Stop streaming |
| `INFO` | Reply `INFO|id|name|imu|wifiSaved|fw|rateHz` |
| `PROVISION|ssid|pass` | Store Wi-Fi credentials (URL-encoded) and restart |

Binary USB frame: `COBS( packet ‖ CRC16 ) ‖ 0x00`, where CRC16 is CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF), little-endian, computed over the packet bytes. Text replies are preceded by a `0x00` so they never merge with a frame.

### 3.5 Wi-Fi state machine

`monitorWiFiState()` runs on every network-task pass and never blocks: it starts an attempt, checks `WiFi.status()`, times an attempt out after 15 s, and waits 10 s before retrying. A successful reconnect after the first connection increments `wifiReconnects`. While Wi-Fi is down no packets are produced, so the packet sequence stays continuous but **frame IDs jump**; the host detects the outage from the frame ID.

### 3.6 Control protocol and security

Control messages are pipe-delimited UDP datagrams to port 5001.

| Message | Reply |
|---|---|
| `DISCOVER` | `HELLO|id|name|mode|ip|imu|streaming|fw|rateHz` |
| `CHALLENGE` | `CHALLENGE|<16 hex>` |
| `START|challenge|clientIp|clientPort|auth` | `ACK|STARTED` / `ERR|…` |
| `STOP|challenge|auth` | `ACK|STOPPED` |
| `PING|challenge|auth` | `ACK|PONG` |

`auth` = HMAC-SHA256 (hex) of `command|challenge[|payload]` with the device key. The firmware keeps **4 challenge slots keyed by the sender's IP and port**, each valid for 30 s and single-use, so one client cannot invalidate another's challenge. Because the challenge is tied to the sender's port, the client must send `CHALLENGE` and the signed command **from the same socket**. If no authenticated command arrives for 15 s the stream stops.

Wi-Fi credentials are stored obfuscated in flash using a key derived from the device key.

### 3.7 Building and flashing

1. Install Arduino core *esp32* 3.3.x and the *SparkFun BNO08x* library.
2. Copy `secrets.example.h` to `secrets.h` and set `DEVICE_ACCESS_KEY_VALUE` to a random string of at least 16 characters, for example `python3 -c "import secrets; print(secrets.token_hex(24))"`.
3. Compile and upload with board *ESP32S3 Dev Module*. Compilation fails with a clear message if `secrets.h` is missing, the key is the placeholder, or it is shorter than 16 characters.
4. Provision Wi-Fi from the app (USB provisioning dialog) or with the `PROVISION` command.

## 4. Desktop application

### 4.1 Installation and start

```bash
python3 -m venv code/.venv && source code/.venv/bin/activate
pip install -r requirements.txt
cp code/device_secret.example.json code/device_secret.json   # set the same key as secrets.h
python code/main.py
```

`code/device_secret.json` is gitignored. The app refuses to talk to a device while the key is empty, the placeholder, or shorter than 16 characters.

### 4.2 Connecting

| Mode | How |
|---|---|
| Wi-Fi | *Wireless configuration* → discover → select device → Connect. The app opens UDP 5000 (4 MB receive buffer), sends `START`, and pings every 2 s. |
| USB (COM port) | *Port configuration* → select port → Connect. The app sends `USBSTART` (repeated until data flows), `USBPING` every 2 s, and `USBSTOP` on exit. Fixed 11-channel layout. |
| Simulator | A `socket://host:port` URL streams legacy CSV text; 2–11 channels. |

Control commands retry up to 3 times (0.2 / 0.5 / 1.0 s backoff) with a fresh challenge each time. After 2 consecutive failed keepalives the app re-issues `START` on a background thread and records the event in the quality report.

### 4.3 Receive pipeline

1. **Receiver thread** (time-critical priority) drains the socket/serial port and decodes packets with NumPy views.
2. **`PacketSequencer`** releases packets in sequence order, holding out-of-order packets for up to 5 packets or 50 ms. Packets that never arrive are declared lost; later arrivals are counted as *late*, repeats as *duplicates*, and neither is inserted. A sequence far behind the stream is treated as a device restart.
3. **`StreamAssembler`** adds frame-ID continuity: any jump in frame ID becomes a gap of NaN frames with the correct frame IDs, `Valid=0`, and timestamp `-1` (never invented). v1 timestamps are unwrapped to 64 bits.
4. Segments are merged for about 20 ms and delivered to the GUI as one batch. The GUI's live filters and plots see only valid rows; the recorder receives everything.

The **Packet Tracer** window shows packet rate, sample rate, missing/late/duplicate/invalid counts, jitter and the device counters.

### 4.4 Calibration and live analysis

Calibration captures a REST and a FLEX phase to derive per-channel baselines and activation scales (`compute_calibration`). Live analysis shows RMS/MAV, dominant frequency, band power, mains-noise score (50 Hz and 100/150/200 Hz harmonics, display only), correlation and other metrics. These displays never alter saved data, which is always raw.

### 4.5 Data-collection protocol

Opened from *Data Collection*. Per participant and per hand, each of the 10 gesture classes is repeated (default 9 repetitions). One cycle is:

| Phase | Default | Notes |
|---|---|---|
| Prepare | 4 s | gesture picture shown |
| Perform | 3 s | label = gesture |
| Rest | 3 s | label = rest |

Cycles are grouped in blocks of at most 30, separated by a break (default 210 s). When the armband changes arm there is a longer hand-switch break (default 300 s). Timing uses a monotonic deadline clock. The operator can **Pause** (for example an emergency): recording continues but is labelled *paused*; on resume a mid-cycle pause restarts the cycle from Prepare and the interrupted attempt is relabelled *aborted*, while a pause during a break simply continues the countdown.

**Study rules**

* Participant ID must match `^S\d{2,3}$` (a pseudonym; names are never stored).
* Consent must be given; the consent form version string is saved.
* Randomization of gesture order is mandatory in study mode and can only be disabled in *Practice / demo mode*, whose sessions are saved in a `practice_` folder.
* A random seed is generated per session and saved.
* Default hand order alternates by participant number (odd: left first, even: right first). A manual override is allowed and recorded.
* Temperature and humidity are required before a session can start. Participant details and electrode placement (landmark, distance, channel-1 muscle, skin preparation, electrode type, photo file name) are optional and saved as null if empty.

### 4.6 Recording model

Frames are recorded **continuously** from the start to the end of the protocol (prepare, rest, breaks and pauses included). The GUI never labels samples. Instead:

1. A writer thread appends every frame to `raw_<session>.bin` (fixed 80-byte records) and keeps counters (valid rows, clipping, device counters). Memory stays bounded and nothing is discarded.
2. The protocol logs **events** (phase changes) with a monotonic host time.
3. After the session, and as a checkpoint at every break, a background worker resolves each event to a **device frame ID**: the newest frame received at that host time, plus an estimated transport latency (median minus minimum of *arrival time − device time*). Both the corrected and uncorrected values and the latency are stored.
4. The raw file is streamed through the event table: a frame takes the label of the last event whose frame ID is not greater than its own, so labels switch at an exact sample.

Host times use `time.monotonic()`; a wall-clock UTC start time appears only in the metadata.

### 4.7 Output files

Folder: `<ID>_day<NN>_<hand-order>_<UTC>`, for example `S03_day01_left-right_20261103T012608Z`.

| File | Content |
|---|---|
| `signals_<session>.csv` | One row per device frame (raw) |
| `events_<session>.csv` | Phase-change table with resolved frame IDs |
| `quality_<session>.json` | Quality report |
| `metadata_<session>.json` / `.txt` | Machine-readable / legacy text metadata |
| `raw_<session>.bin` | Unlabelled frame stream the other files derive from |

**`signals` columns:** `Device_Frame_ID, Device_Timestamp_us, Packet_Sequence, Valid, Late_Flag, Ch1–Ch8, IMU_roll, IMU_pitch, IMU_yaw, IMU_Sample_ID, IMU_Fresh, Host_Received_s, Label, Label_Code, Phase, Hand, Trial_ID`.

* `Device_Timestamp_us` is relative to the first valid frame (int64); `-1` on gap rows.
* `Ch1–Ch8` are raw ADC counts; `nan` on gap rows (`Valid=0`). No filtering or baseline correction is applied.
* `Host_Received_s` is monotonic seconds since recording start.

**`events` columns:** `event_index, phase, label, label_code, hand, trial_id, block, cycle_no, host_monotonic_s, device_frame_id_at_event, device_frame_id_uncorrected, latency_estimate_s, planned_ms, aborted`.

**`quality` contents:** valid-sample ratio and loss %, whether the 0.98 threshold is met, packet counters (received, missing, late, duplicate), the gap list (start frame, length, label, phase), device counters, keepalive restarts, per-channel clipping % (≤5 or ≥4090 counts), late frames, and expected versus actual frames per phase. If fewer than 98 % of frames are valid the app warns the operator.

### 4.8 Label map

| Name | Code | Previous name |
|---|---|---|
| rest | 0 | Rest |
| left | 1 | Left |
| right | 2 | Right |
| fist_close | 3 | fist_close |
| up | 4 | Up |
| down | 5 | Down |
| double_fist_close | 6 | Double_Closed_Fist |
| victory_sign | 7 | Victory_Sign |
| thumbs_up | 8 | Thumbs_Up |
| pinch_thumb_index | 9 | Pinch_using_thumb_and_index_finger |
| settle | 98 | Settle |
| prepare | 99 | Prepare |
| paused | 97 | (recorder) |
| aborted | 96 | (recorder) |

Break and hand-switch phases carry label `rest`; the `Phase` column distinguishes them. Training uses only `perform` and `rest` rows with `Valid=1`; a gap ends the current segment.

### 4.9 Model training and classification

*Train Model* opens a three-step window.

1. **Recordings.** *Scan Dataset Folder* lists every `signals_*.csv` (current) and `emg_data_*.csv` (pilot) under the dataset folder with participant, hand, frames, valid share and channel count. Current recordings are ticked, pilot files are listed unticked. Only `perform` and `rest` frames with `Valid=1` are used; a gap ends the current segment, so gaps are never bridged. All ticked files must have the same channel count (current files have 8 EMG channels, pilot files have 11).
2. **Models.** Tick any number of models and adjust their settings. Data preparation: window length, stride, edge trim (cuts the reaction-time onset from each segment), test share, and whether to hold out segments or whole recordings.
3. **Results.** All models are trained on the same windows and split and compared side by side (accuracy, balanced accuracy, macro F1, training time, time for one live prediction), with the per-class report and a confusion matrix whose off-diagonal cells are shown in red.

| Model | Folder | Input |
|---|---|---|
| Random Forest | `random_forest` | features |
| Extra Trees | `extra_trees` | features |
| SVM (RBF kernel) | `svm_rbf` | features, standardised |
| SVM (linear) | `svm_linear` | features, standardised |
| k-Nearest Neighbours | `knn` | features, standardised |
| Logistic Regression | `logistic_regression` | features, standardised |
| Linear Discriminant Analysis | `lda` | features |
| Gradient Boosting | `gradient_boosting` | features |
| Neural Network (MLP) | `mlp` | features, standardised |
| 1D CNN | `cnn_1d` | raw window (needs PyTorch) |

Each trained model is saved in its own folder: `trained_model/<model folder>/<run name>_<date_time>/` containing `<model folder>_model.joblib`, `training_setup.json`, `training_results.json`, `classification_report.txt`, `confusion_matrix.csv` and `training_summary.txt`. Classical models learn from the window feature vector of `rf_features.py`; the 1D CNN learns from the mean-centred raw window. SVM probabilities are a softmax of the decision scores (not calibrated).

The real-time window runs the loaded model at most 12 times per second, smooths with a 5-window majority vote, and can publish the predicted gesture to the demo game over localhost UDP (port 5005). The main window shows a workflow bar (connect, calibrate, collect, train, classify) with the next step.

Note: holding out whole recordings gives an honest estimate of how a model behaves on a new session; holding out segments from the same recording is optimistic.

### 4.9a User interface

* **Layout.** A branded header with four live status pills (link and packet rate, calibration, loaded model, recording), command cards (Connection, Tools), a pipeline banner that shows where you are (Connect, Calibrate, Collect, Train, Classify) with the next action and the four stage buttons, the colour-coded channel plots, a side panel (channel status, detection and display settings) and a status bar with the live statistics.
* **Laptop screens.** Windows size themselves to the usable screen (menu bar, dock and taskbar excluded) and the base font scales with the screen height. Every dialog has a scrollable body, and dialogs with primary actions (for example Model Training) keep those buttons pinned at the bottom, so no option can be pushed off-screen on a 13" MacBook, a 1366x768 laptop or a 150 %-scaled Windows laptop. Inputs and buttons have a guaranteed minimum height derived from the font.
* **Checking it.** `python code/tools/ui_audit.py --profile mac13 --shots out/` opens every window at an emulated laptop screen (`mac13`, `mac_small`, `win125`, `win150`, `win_hd`), reports clipped or off-screen controls and saves screenshots. The automated tests run it for the main profiles.

### 4.10 Pilot-data repair

```bash
python code/tools/repair_sessions.py            # reads dataset/, writes dataset_repaired/
```

For each pilot session it sorts rows by packet number within each trial, classifies packet-number jumps (inside a cycle: real loss; between cycles: the session's regular unrecorded-prepare jump, a smaller jump is real loss, a different larger one is an unrecorded interval), inserts NaN rows with `Valid=0` for real loss only, and writes a gap table and quality JSON marked `pilot`. Timestamps are not invented. Outputs use aliases `PILOT_001…`; the alias-to-source mapping goes to the gitignored `code/participant_map.json`.

## 5. Testing

Run `pytest` from the repository root or from `code/`. Coverage includes BWIM v1/v2 parsing, COBS/CRC, sequencing (no loss, single loss, burst loss, reordering, duplicates, 32-bit sequence and timestamp wrap, device restart), control-protocol retries against a fake device, the wired and wireless workers against fake sources, event-to-frame labelling, gap tracking, quality reporting, participant/ID/counterbalancing rules, and the repair tool.

Not covered by automated tests: real radio and USB timing, the IMU, and firmware behaviour on hardware.

## 6. Hardware acceptance procedure

1. Flash the firmware with a real key and set the same key in `code/device_secret.json`.
2. Connect over USB; confirm firmware version and 500 Hz in the metadata. Repeat over Wi-Fi.
3. Record a short session and check column names, frame IDs, phase changes and a valid ratio near 100 %.
4. Record a session and cut the router power for 10 s in a Perform phase. Expect one gap of about 5000 frames (a block of `Valid=0` rows) in `quality_*.json`, and possibly a keepalive restart entry.
5. Perform a sharp contraction at a cue and confirm the EMG onset aligns with the `perform` label edge (validates the latency estimate).
6. Unplug and replug the USB cable mid-session and confirm the gap is recorded.

## 7. Known limitations

* If the device reboots mid-session the event is flagged in the quality report, but timestamps and frame IDs are not stitched across the restart.
* The latency correction is an estimate because device and host clocks are not synchronised.
* The initial `START` at connect time can block the GUI for several seconds when retries are needed.
* Sampling is software-timed `analogRead` at 500 Hz; hardware-timed (DMA) 1 kHz sampling is not implemented.
* The `ESP32_S3 Setup` folder also contains the older sketch `EMG_IMU_WIFI_SETUP.ino`, which does not implement BWIM v2.

## 8. Glossary

| Term | Meaning |
|---|---|
| BWIM | BioWave IMU+EMG packet format |
| Frame | One 500 Hz sample set (8 EMG + IMU) |
| Packet | 5 consecutive frames sent together |
| Gap | Frames the device sampled but the host never received; stored as `Valid=0` rows |
| COBS | Consistent Overhead Byte Stuffing; makes `0x00` a safe frame delimiter |
| Event | A protocol phase change with a host time, resolved later to a frame ID |

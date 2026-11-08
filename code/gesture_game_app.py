"""BioWave Gesture Game - standalone Lane Dash game driven by EMG gestures.

Run the main BioWave app (main.py) with a model loaded and a calibrated stream; it
publishes each smoothed gesture prediction to 127.0.0.1:GESTURE_BRIDGE_PORT over UDP
(a few dozen bytes, ~12 Hz). This app only listens, so it never touches the EMG
transport. Without the main app (or a model) the keyboard fallback still works:
Left/Right arrows move, Space = fist_close, Down = re-arm.

    python gesture_game_app.py
"""
import json
import random
import socket
import sys
import time

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app_theme import (
    THEME_COLORS,
    apply_dark_theme,
    apply_dark_title_bar,
    configure_high_dpi,
    fit_window_to_screen,
    themed_button_style,
    themed_label_style,
)

GESTURE_BRIDGE_HOST = "127.0.0.1"
GESTURE_BRIDGE_PORT = 5005
GESTURE_STALE_S = 1.5              # Treat the bridge as offline after this much silence

GAME_LANES = 3
GAME_LOOP_MS = 33                  # ~30 FPS render/update loop
GAME_GESTURE_POLL_MS = 60          # How often we drain the bridge socket
GAME_MIN_CONFIDENCE_PCT = 12.0     # Low bar so gesture control feels responsive, not laggy
GAME_ACTION_COOLDOWN_S = 0.45      # Debounce between accepted gesture actions


class GestureBridgeClient:
    """Non-blocking UDP listener that keeps only the newest published prediction."""

    def __init__(self, host=GESTURE_BRIDGE_HOST, port=GESTURE_BRIDGE_PORT):
        self.latest = {"label": "N/A", "conf_pct": 0.0, "model": "", "ts": 0.0}
        self._sock = None
        self.host = host
        self.port = int(port)
        self.error = ""
        self.packets = 0
        self.rebind(host, port)

    def rebind(self, host, port):
        """(Re)open the listening socket; returns True on success, else sets self.error."""
        self.close()
        self.host, self.port, self.error = host, int(port), ""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, int(port)))
            sock.setblocking(False)
            self._sock = sock
            return True
        except (OSError, OverflowError) as exc:
            self.error = str(exc)
            self._sock = None
            return False

    @property
    def listening(self):
        return self._sock is not None

    def poll(self):
        """Drain the socket and return (label, conf_pct, model_text, online)."""
        if self._sock is not None:
            while True:
                try:
                    data, _addr = self._sock.recvfrom(1024)
                except (BlockingIOError, OSError):
                    break
                try:
                    msg = json.loads(data.decode("utf-8"))
                    self.packets += 1
                    self.latest = {
                        "label": str(msg.get("label", "N/A")),
                        "conf_pct": float(msg.get("conf_pct", 0.0)),
                        "model": str(msg.get("model", "")),
                        "ts": time.monotonic(),
                    }
                except (ValueError, TypeError):
                    continue
        online = (time.monotonic() - self.latest["ts"]) < GESTURE_STALE_S
        return self.latest["label"], self.latest["conf_pct"], self.latest["model"], online

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


class GestureGameWidget(QWidget):
    """Lightweight 3-lane runner controlled entirely by EMG gesture predictions.

    Left / Right gestures shift the player between 3 lanes, Rest returns
    control to neutral (arming the next move), and fist_close triggers a
    'smash' that clears obstacles in the player's lane and grants a brief
    shield. The confidence gate for accepting a gesture is kept low so the
    game feels responsive even with a noisy classifier.
    """

    score_changed = pyqtSignal(int)
    lives_changed = pyqtSignal(int)
    game_over = pyqtSignal(int)

    LANE_COUNT = GAME_LANES

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 480)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.running = False
        self.paused = False
        self.player_lane = 1
        self.lives = 3
        self.score = 0
        self.high_score = 0
        self.objects = []  # list of dicts: lane, y (0..1 down the track), kind
        self.spawn_accum_ms = 0.0
        self.spawn_interval_ms = 950.0
        self.fall_speed = 0.34  # fraction of track height per second
        self.elapsed_s = 0.0
        self.invulnerable_until = 0.0
        self.smash_flash_until = 0.0
        self.flash_color = None
        self.current_gesture_label = "N/A"
        self.current_gesture_conf = 0.0

        self._last_action_label = None
        self._last_action_ts = 0.0

        self.loop_timer = QTimer(self)
        self.loop_timer.timeout.connect(self._tick)

    # --- lifecycle -----------------------------------------------------
    def start_game(self):
        self.reset_state()
        self.running = True
        self.paused = False
        self.loop_timer.start(GAME_LOOP_MS)
        self.setFocus()
        self.update()

    def pause_game(self, paused):
        self.paused = bool(paused)

    def stop_game(self):
        self.running = False
        self.loop_timer.stop()
        self.update()

    def reset_state(self):
        self.player_lane = 1
        self.lives = 3
        self.score = 0
        self.objects = []
        self.spawn_accum_ms = 0.0
        self.spawn_interval_ms = 950.0
        self.fall_speed = 0.34
        self.elapsed_s = 0.0
        self.invulnerable_until = 0.0
        self.smash_flash_until = 0.0
        self._last_action_label = None
        self._last_action_ts = 0.0
        self.lives_changed.emit(self.lives)
        self.score_changed.emit(self.score)
        self.update()

    # --- gesture input ---------------------------------------------------
    def update_gesture(self, label, confidence_pct):
        self.current_gesture_label = str(label or "N/A")
        self.current_gesture_conf = float(confidence_pct or 0.0)
        if not self.running or self.paused:
            return

        norm = self.current_gesture_label.strip().lower()
        if not norm or norm == "n/a":
            return
        if self.current_gesture_conf < GAME_MIN_CONFIDENCE_PCT:
            return

        now = time.perf_counter()
        if (now - self._last_action_ts) < GAME_ACTION_COOLDOWN_S:
            return
        if norm == self._last_action_label:
            return  # require the gesture to change (e.g. relax to Rest) before it retriggers

        acted = False
        if "left" in norm:
            self._move_player(-1)
            acted = True
        elif "right" in norm:
            self._move_player(1)
            acted = True
        elif "fist" in norm or "close" in norm:
            self._smash()
            acted = True
        elif "rest" in norm:
            acted = True  # arms the next gesture without moving

        if acted:
            self._last_action_label = norm
            self._last_action_ts = now

    def handle_key_action(self, key):
        """Keyboard fallback so the game can be demoed without hardware."""
        if key == Qt.Key_Left:
            self._last_action_label = None
            self.update_gesture("Left", 100.0)
        elif key == Qt.Key_Right:
            self._last_action_label = None
            self.update_gesture("Right", 100.0)
        elif key == Qt.Key_Space:
            self._last_action_label = None
            self.update_gesture("fist_close", 100.0)
        elif key == Qt.Key_Down:
            self._last_action_label = None  # manual "Rest" to re-arm

    def keyPressEvent(self, event):
        self.handle_key_action(event.key())
        super().keyPressEvent(event)

    # --- game mechanics --------------------------------------------------
    def _move_player(self, delta):
        self.player_lane = int(np.clip(self.player_lane + delta, 0, self.LANE_COUNT - 1))

    def _smash(self):
        now = time.perf_counter()
        cleared = 0
        remaining = []
        for obj in self.objects:
            if obj["lane"] == self.player_lane and obj["kind"] == "obstacle" and obj["y"] > 0.45:
                cleared += 1
            else:
                remaining.append(obj)
        self.objects = remaining
        if cleared > 0:
            self._add_score(cleared * 15)
        self.invulnerable_until = now + 0.6
        self.smash_flash_until = now + 0.25
        self.flash_color = QColor(THEME_COLORS.get("accent", "#3B9797"))

    def _add_score(self, amount):
        if amount == 0:
            return
        self.score = max(0, int(self.score + amount))
        self.score_changed.emit(self.score)

    def _tick(self):
        if not self.running or self.paused:
            return
        dt_s = GAME_LOOP_MS / 1000.0
        self.elapsed_s += dt_s

        # Difficulty ramps smoothly with survival time.
        self.fall_speed = min(0.75, 0.34 + (self.elapsed_s * 0.012))
        self.spawn_interval_ms = max(360.0, 950.0 - (self.elapsed_s * 14.0))

        self.spawn_accum_ms += dt_s * 1000.0
        if self.spawn_accum_ms >= self.spawn_interval_ms:
            self.spawn_accum_ms = 0.0
            self._spawn_object()

        now = time.perf_counter()
        survivors = []
        for obj in self.objects:
            obj["y"] += self.fall_speed * dt_s
            if obj["y"] >= 0.90 and not obj.get("resolved", False):
                if obj["lane"] == self.player_lane:
                    obj["resolved"] = True
                    if obj["kind"] == "coin":
                        self._add_score(10)
                        continue
                    elif obj["kind"] == "obstacle":
                        if now < self.invulnerable_until:
                            continue
                        self._lose_life()
                        continue
            if obj["y"] < 1.05:
                survivors.append(obj)
            elif obj["kind"] == "obstacle" and not obj.get("resolved", False):
                self._add_score(2)  # reward for dodging cleanly
        self.objects = survivors

        self.update()

    def _spawn_object(self):
        lane = random.randint(0, self.LANE_COUNT - 1)
        kind = "coin" if random.random() < 0.30 else "obstacle"
        self.objects.append({"lane": lane, "y": -0.05, "kind": kind, "resolved": False})

    def _lose_life(self):
        self.lives -= 1
        self.lives_changed.emit(self.lives)
        self.invulnerable_until = time.perf_counter() + 1.0
        if self.lives <= 0:
            self.running = False
            self.loop_timer.stop()
            self.high_score = max(self.high_score, self.score)
            self.game_over.emit(self.score)

    # --- rendering ---------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect()

        bg = QColor(THEME_COLORS.get("bg", "#0A182C"))
        painter.fillRect(rect, bg)

        track_margin = 16
        track_rect = QRectF(
            track_margin, track_margin + 20,
            rect.width() - (2 * track_margin), rect.height() - (2 * track_margin) - 44,
        )
        panel = QColor(THEME_COLORS.get("panel", "#132440"))
        painter.setPen(QPen(QColor(THEME_COLORS.get("accent", "#3B9797")), 2))
        painter.setBrush(QBrush(panel))
        painter.drawRoundedRect(track_rect, 12, 12)

        lane_w = track_rect.width() / self.LANE_COUNT
        painter.setPen(QPen(QColor(255, 255, 255, 30), 1, Qt.DashLine))
        for i in range(1, self.LANE_COUNT):
            x = track_rect.left() + (lane_w * i)
            painter.drawLine(QPointF(x, track_rect.top()), QPointF(x, track_rect.bottom()))

        # Falling objects (obstacles + coins)
        for obj in self.objects:
            cx = track_rect.left() + (lane_w * (obj["lane"] + 0.5))
            cy = track_rect.top() + (obj["y"] * track_rect.height())
            if obj["kind"] == "coin":
                painter.setPen(QPen(QColor("#FFD166"), 2))
                painter.setBrush(QBrush(QColor("#FFB703")))
                r = min(lane_w, track_rect.height()) * 0.09
                painter.drawEllipse(QPointF(cx, cy), r, r)
            else:
                painter.setPen(QPen(QColor("#8C1128"), 2))
                painter.setBrush(QBrush(QColor("#BF092F")))
                s = min(lane_w, track_rect.height()) * 0.11
                painter.drawRect(QRectF(cx - s, cy - s, s * 2, s * 2))

        # Player
        player_cx = track_rect.left() + (lane_w * (self.player_lane + 0.5))
        player_cy = track_rect.top() + (track_rect.height() * 0.90)
        player_r = min(lane_w, track_rect.height()) * 0.10
        now = time.perf_counter()
        shielded = now < self.invulnerable_until
        player_color = QColor(THEME_COLORS.get("success", "#4CAF50")) if shielded else QColor(THEME_COLORS.get("accent", "#3B9797"))
        painter.setPen(QPen(QColor("#FFFFFF"), 2))
        painter.setBrush(QBrush(player_color))
        painter.drawEllipse(QPointF(player_cx, player_cy), player_r, player_r)
        if shielded:
            painter.setPen(QPen(QColor(255, 255, 255, 140), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(player_cx, player_cy), player_r * 1.6, player_r * 1.6)

        if now < self.smash_flash_until and self.flash_color is not None:
            flash = QColor(self.flash_color)
            flash.setAlpha(60)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(flash))
            painter.drawRoundedRect(track_rect, 12, 12)

        # HUD
        hud_font = QFont()
        hud_font.setPointSize(11)
        hud_font.setBold(True)
        painter.setFont(hud_font)
        painter.setPen(QColor(THEME_COLORS.get("text", "#E8EEF0")))
        painter.drawText(
            QRectF(track_rect.left(), 2, track_rect.width() * 0.6, track_margin + 16),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"Score: {self.score}   Best: {self.high_score}",
        )
        painter.drawText(
            QRectF(track_rect.left() + track_rect.width() * 0.4, 2, track_rect.width() * 0.6, track_margin + 16),
            Qt.AlignRight | Qt.AlignVCenter,
            ("Lives: " + ("* " * max(0, self.lives))).rstrip(),
        )

        gesture_font = QFont()
        gesture_font.setPointSize(10)
        painter.setFont(gesture_font)
        painter.setPen(QColor(THEME_COLORS.get("muted", "#A9C2CF")))
        painter.drawText(
            QRectF(track_rect.left(), track_rect.bottom() + 4, track_rect.width(), track_margin + 16),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"Gesture: {self.current_gesture_label}  ({self.current_gesture_conf:.0f}%)",
        )

        if not self.running:
            painter.setPen(QColor(255, 255, 255, 230))
            title_font = QFont()
            title_font.setPointSize(18)
            title_font.setBold(True)
            painter.setFont(title_font)
            msg = "Press Start" if self.score == 0 and self.lives == 3 else f"Game Over - Score {self.score}"
            painter.drawText(track_rect, Qt.AlignCenter, msg)




class GestureGameWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Game - Lane Dash")
        fit_window_to_screen(self, 520, 720)
        self.setMinimumSize(380, 520)
        apply_dark_title_bar(self)
        self.bridge = GestureBridgeClient()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)

        intro = QLabel(
            "Control the runner with your gestures: Left/Right switch lanes, Rest arms the "
            "next move, and fist_close smashes obstacles in your lane for bonus points. "
            "Keyboard fallback: Left/Right arrows, Space = fist_close, Down = re-arm."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(themed_label_style("muted"))
        layout.addWidget(intro)

        self.lbl_link = QLabel("")
        self.lbl_link.setWordWrap(True)
        layout.addWidget(self.lbl_link)

        # --- connectivity panel: where to listen for the BioWave app's gesture stream ---
        conn_row = QHBoxLayout()
        conn_row.addWidget(QLabel("Listen on:"))
        self.txt_bridge_host = QLineEdit(GESTURE_BRIDGE_HOST)
        self.txt_bridge_host.setToolTip("127.0.0.1 = same PC only; 0.0.0.0 = accept from other PCs on the network")
        conn_row.addWidget(self.txt_bridge_host, 1)
        self.spin_bridge_port = QSpinBox()
        self.spin_bridge_port.setRange(1024, 65535)
        self.spin_bridge_port.setValue(GESTURE_BRIDGE_PORT)
        conn_row.addWidget(self.spin_bridge_port)
        self.btn_bridge_apply = QPushButton("Connect")
        self.btn_bridge_apply.setStyleSheet(themed_button_style("accent"))
        self.btn_bridge_apply.clicked.connect(self.apply_bridge_settings)
        conn_row.addWidget(self.btn_bridge_apply)
        layout.addLayout(conn_row)
        self.lbl_conn_help = QLabel(
            "Steps: 1) start mouse_controller.py or main.py, connect the EMG device, load a model and "
            "calibrate; 2) keep the host/port here equal to the sender's (default 127.0.0.1:5005); "
            "3) press Connect to re-open the link.")
        self.lbl_conn_help.setWordWrap(True)
        self.lbl_conn_help.setStyleSheet(themed_label_style("muted"))
        layout.addWidget(self.lbl_conn_help)

        self.game_widget = GestureGameWidget()
        self.game_widget.game_over.connect(self._on_game_over)
        self.game_widget.lives_changed.connect(self._on_lives_changed)
        layout.addWidget(self.game_widget, 1)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.setStyleSheet(themed_button_style("success"))
        self.btn_start.clicked.connect(self.on_start_clicked)
        btn_row.addWidget(self.btn_start)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setStyleSheet(themed_button_style("accent"))
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.on_pause_clicked)
        btn_row.addWidget(self.btn_pause)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setStyleSheet(themed_button_style("muted"))
        self.btn_reset.clicked.connect(self.on_reset_clicked)
        btn_row.addWidget(self.btn_reset)
        layout.addLayout(btn_row)

        self.lbl_status = QLabel("Press Start, then use your gestures (or arrow keys) to play.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(themed_label_style("muted"))
        layout.addWidget(self.lbl_status)

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_gesture)
        self.poll_timer.start(GAME_GESTURE_POLL_MS)
        self.poll_gesture()
        self.game_widget.setFocus()

    def apply_bridge_settings(self):
        host = self.txt_bridge_host.text().strip() or GESTURE_BRIDGE_HOST
        if not self.bridge.rebind(host, self.spin_bridge_port.value()):
            self.lbl_status.setText(f"Could not listen on {host}:{self.spin_bridge_port.value()} - {self.bridge.error}")
        self.poll_gesture()
        self.game_widget.setFocus()

    def on_start_clicked(self):
        self.game_widget.start_game()
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("Pause")
        self.lbl_status.setText("Game running - use Left/Right to move, fist_close to smash.")
        self.game_widget.setFocus()

    def on_pause_clicked(self):
        is_paused = not self.game_widget.paused
        self.game_widget.pause_game(is_paused)
        self.btn_pause.setText("Resume" if is_paused else "Pause")
        self.lbl_status.setText("Paused." if is_paused else "Game running.")
        self.game_widget.setFocus()

    def on_reset_clicked(self):
        self.game_widget.stop_game()
        self.game_widget.reset_state()
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.lbl_status.setText("Press Start, then use your gestures (or arrow keys) to play.")
        self.game_widget.setFocus()

    def _on_lives_changed(self, lives):
        if lives <= 0:
            self.lbl_status.setText("Out of lives! Press Reset to play again.")

    def _on_game_over(self, final_score):
        self.btn_pause.setEnabled(False)
        self.lbl_status.setText(f"Game over - final score {final_score}. Press Reset to play again.")

    def poll_gesture(self):
        label, conf_pct, model, online = self.bridge.poll()
        if not self.bridge.listening:
            text, kind = (f"Gesture link: cannot listen on {self.bridge.host}:{self.bridge.port} "
                          f"({self.bridge.error or 'port busy'}) - keyboard only. Close the other game "
                          f"instance or pick another port."), "muted"
        elif online:
            text, kind = f"Gesture link: live | {model or 'model'} | {label} ({conf_pct:.0f}%)", "success"
        else:
            text, kind = (f"Gesture link: listening on {self.bridge.host}:{self.bridge.port}, waiting for "
                          f"the BioWave app (no gestures received yet - connect the device, load a model and "
                          f"calibrate there; keyboard works)."), "muted"
        self.lbl_link.setText(text)
        self.lbl_link.setStyleSheet(themed_label_style(kind))
        if online:
            self.game_widget.update_gesture(label, conf_pct)

    def closeEvent(self, event):
        self.poll_timer.stop()
        self.game_widget.stop_game()
        self.bridge.close()
        super().closeEvent(event)


if __name__ == "__main__":
    configure_high_dpi()
    app = QApplication(sys.argv)
    apply_dark_theme(app, font_size=16)
    window = GestureGameWindow()
    window.show()
    sys.exit(app.exec_())

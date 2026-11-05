"""Model zoo and training pipeline for gesture classification (no Qt, no GUI).

* ``load_segments`` reads both the current continuous-recording format (``Valid`` / ``Phase`` columns, gap
  frames as NaN) and the older pilot format, and returns one array per uninterrupted task segment.
* ``MODEL_SPECS`` lists the selectable models (Random Forest, Extra Trees, SVM, k-NN, ... and a 1D CNN) with
  their tunable parameters, so the GUI can build its forms from data.
* ``prepare_dataset`` / ``split_train_test`` / ``fit_and_evaluate`` are the model-independent steps.

Classical models learn from the window feature vector of ``rf_features``; the 1D CNN learns directly from the
raw window (``input_kind == "window"``). Every fitted model exposes ``predict_proba`` and ``classes_`` (integer
class indices), which is all the real-time classifier needs.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from rf_features import build_windows_from_sequence, extract_window_features

try:  # scikit-learn is required for the classical models, but importing this module must not fail without it
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                                 f1_score)
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False
    BaseEstimator = ClassifierMixin = object

TRAINING_PHASES = ("perform", "rest")        # prepare / settle / break / paused / aborted are not training data


# ---------------------------------------------------------------------------------------------
# Reading recordings
# ---------------------------------------------------------------------------------------------
def _header_index(header) -> dict:
    return {str(name).strip().lower(): i for i, name in enumerate(header)}


def _channel_columns(index: dict) -> list:
    cols = sorted((int(name[2:]), i) for name, i in index.items() if name.startswith("ch") and name[2:].isdigit())
    return [i for _n, i in cols]


def detect_channel_count(path: str, probe_rows: int = 4000) -> int:
    """Number of channel columns that actually carry data (older files pad unused channels with blanks)."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV has no header.")
        index = _header_index(header)
        if "label" not in index:
            raise ValueError("CSV missing required column: Label")
        cols = _channel_columns(index)
        if not cols:
            raise ValueError("CSV must contain channel columns (Ch1..ChN)")
        active = set()
        for n, row in enumerate(reader):
            for c in cols:
                if c < len(row) and row[c].strip() != "":
                    active.add(c)
            if n >= probe_rows and len(active) == len(cols):
                break
        return len(active) if active else len(cols)


def load_segments(path: str):
    """Read one recording into ``[(label, array(n, channels)), ...]`` and the channel count.

    A segment is an uninterrupted run of valid frames with the same label, trial, hand and phase. Frames that
    are invalid (``Valid == 0`` or NaN), and, when a ``Phase`` column exists, frames outside the perform / rest
    phases, end the current segment and are dropped, so gaps are never bridged.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV has no header.")
        index = _header_index(header)
        if "label" not in index:
            raise ValueError("CSV missing required column: Label")
        label_c = index["label"]
        trial_c, phase_c = index.get("trial_id"), index.get("phase")
        hand_c, valid_c = index.get("hand"), index.get("valid")
        all_cols = _channel_columns(index)
        if not all_cols:
            raise ValueError("CSV must contain at least one channel column (Ch1..ChN).")

        rows = iter(reader)
        # decide which channel columns are in use from the first rows, then keep them for the whole file
        buffered, active = [], set()
        for row in rows:
            buffered.append(row)
            for c in all_cols:
                if c < len(row) and row[c].strip() != "":
                    active.add(c)
            if len(buffered) >= 4000:
                break
        cols = [c for c in all_cols if c in active] or list(all_cols)

        segments, buf, key, label = [], [], None, None

        def flush():
            nonlocal buf, key, label
            if buf and label:
                segments.append((label, np.asarray(buf, dtype=np.float32)))
            buf, key, label = [], None, None

        def consume(row):
            nonlocal key, label
            if len(row) <= label_c:
                return
            name = row[label_c].strip()
            phase = row[phase_c].strip().lower() if phase_c is not None and phase_c < len(row) else ""
            invalid = valid_c is not None and valid_c < len(row) and row[valid_c].strip() == "0"
            if not name or invalid or (phase and phase not in TRAINING_PHASES):
                flush()
                return
            try:
                sample = [float(row[c]) if c < len(row) and row[c].strip() != "" else 0.0 for c in cols]
            except ValueError:
                flush()
                return
            if not np.isfinite(sample).all():
                flush()
                return
            new_key = (name, row[trial_c] if trial_c is not None and trial_c < len(row) else "1",
                       row[hand_c] if hand_c is not None and hand_c < len(row) else "", phase)
            if key is not None and new_key != key:
                flush()
            key, label = new_key, name
            buf.append(sample)

        for row in buffered:
            consume(row)
        for row in rows:
            consume(row)
        flush()
    return segments, len(cols)


def describe_session(csv_path: str) -> dict:
    """Cheap summary of one recording for the dataset list (reads metadata, not the whole CSV)."""
    folder = os.path.dirname(csv_path)
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    session = re.sub(r"^(signals_|emg_data_)", "", stem)
    info = {"path": csv_path, "session": session, "format": "legacy", "participant": "", "hand": "", "day": "",
            "rows": 0, "valid_pct": None, "labels": "", "size_mb": os.path.getsize(csv_path) / 1e6}
    meta_json = os.path.join(folder, f"metadata_{session}.json")
    quality_json = os.path.join(folder, f"quality_{session}.json")
    try:
        with open(meta_json, encoding="utf-8") as fh:
            meta = json.load(fh)
        info["format"] = "continuous"
        info["participant"] = str(meta.get("participant_id") or "")
        info["hand"] = str(meta.get("session_info", {}).get("hand_order") or "")
        info["day"] = str(meta.get("session_info", {}).get("session_day") or "")
        info["labels"] = ", ".join(meta.get("protocol", {}).get("labels") or [])
        info["rows"] = int(meta.get("result", {}).get("rows") or 0)
        ratio = meta.get("result", {}).get("valid_sample_ratio")
        info["valid_pct"] = None if ratio is None else 100.0 * float(ratio)
    except (OSError, ValueError, AttributeError):
        pass
    if info["format"] == "legacy":
        # legacy files: take labels and sample count from the plain-text metadata, never the contributor name
        try:
            with open(os.path.join(folder, f"metadata_{session}.txt"), encoding="utf-8") as fh:
                kv = dict(line.rstrip("\n").split("=", 1) for line in fh if "=" in line)
            info["labels"] = kv.get("labels", "").replace(",", ", ")
            info["rows"] = int(float(kv.get("num_samples", 0) or 0))
            info["hand"] = kv.get("hand_order", "")
            info["day"] = kv.get("session_day", "")
        except (OSError, ValueError):
            pass
    try:
        with open(quality_json, encoding="utf-8") as fh:
            q = json.load(fh)
        info["valid_pct"] = 100.0 * float(q.get("valid_sample_ratio", 0.0))
    except (OSError, ValueError):
        pass
    return info


def find_recordings(root: str) -> list:
    """Signal CSVs under ``root``: ``signals_*.csv`` (current) and ``emg_data_*.csv`` (pilot)."""
    found = []
    for folder, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".csv") and (name.startswith("signals_") or name.startswith("emg_data_")):
                found.append(os.path.join(folder, name))
    return sorted(found)


# ---------------------------------------------------------------------------------------------
# Windows, features, split
# ---------------------------------------------------------------------------------------------
@dataclass
class Dataset:
    classes: list
    y: np.ndarray                      # class index per window
    groups: np.ndarray                 # source segment per window (never split across train/test)
    features: Optional[np.ndarray] = None
    windows: Optional[np.ndarray] = None   # (n, samples, channels)
    probe_window: Optional[np.ndarray] = None   # one window, for the latency measurement
    channels: int = 0
    window_samples: int = 0
    stride_samples: int = 0
    skipped_segments: int = 0

    @property
    def class_counts(self) -> np.ndarray:
        return np.bincount(self.y, minlength=len(self.classes))


def prepare_dataset(segments: list, *, window_samples: int, stride_samples: int, sample_rate: int,
                    trim_samples: int = 0, want_features: bool = True, want_windows: bool = False,
                    progress: Optional[Callable[[str], None]] = None) -> Dataset:
    """Cut segments into overlapping windows and (optionally) extract features. Needs >= 2 classes."""
    channels = {int(np.asarray(seq).shape[1]) for _label, seq in segments if np.asarray(seq).ndim == 2}
    if len(channels) != 1:
        raise ValueError(f"Segments must all have the same channel count (found {sorted(channels) or 'none'}).")
    n_ch = channels.pop()
    windows, labels, groups, skipped = [], [], [], 0
    for seg_id, (label, seq) in enumerate(segments):
        arr = np.asarray(seq, dtype=np.float32)
        if trim_samples > 0 and arr.shape[0] > 2 * trim_samples + window_samples:
            arr = arr[trim_samples:arr.shape[0] - trim_samples]    # drop the onset/offset transient
        cut = build_windows_from_sequence(arr, window_samples, stride_samples)
        if len(cut) == 0:
            skipped += 1
            continue
        windows.extend(cut)
        labels.extend([label] * len(cut))
        groups.extend([seg_id] * len(cut))
    if len(windows) < 20:
        raise ValueError(f"Too few windows for training ({len(windows)}). Add more data or reduce the window size.")
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes for classification.")
    to_idx = {c: i for i, c in enumerate(classes)}
    y = np.asarray([to_idx[v] for v in labels], dtype=np.int32)
    if int(np.bincount(y).min()) < 2:
        raise ValueError("Each class needs at least 2 windows for a stratified split.")

    ds = Dataset(classes=classes, y=y, groups=np.asarray(groups, dtype=np.int64), channels=n_ch,
                 window_samples=window_samples, stride_samples=stride_samples, skipped_segments=skipped,
                 probe_window=np.asarray(windows[0], dtype=np.float32))
    if want_windows:
        ds.windows = np.ascontiguousarray(np.stack(windows).astype(np.float32))
    if want_features:
        if progress:
            progress("Extracting features...")
        try:
            from joblib import Parallel, delayed
            if len(windows) >= 256:
                feats = Parallel(n_jobs=max(1, (os.cpu_count() or 2) - 1), prefer="threads")(
                    delayed(extract_window_features)(w, sample_rate) for w in windows)
            else:
                raise RuntimeError("sequential")
        except Exception:
            feats = [extract_window_features(w, sample_rate=sample_rate) for w in windows]
        ds.features = np.asarray(feats, dtype=np.float32)
    return ds


def split_train_test(y: np.ndarray, groups: np.ndarray, n_classes: int, test_size: float, seed: int):
    """Group-aware split: overlapping windows of one segment never straddle train and test.

    Falls back to a stratified split when a grouped one cannot keep every class on both sides.
    Returns (train_index, test_index).
    """
    idx = np.arange(len(y))
    if int(np.unique(groups).size) >= 2:
        try:
            train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(idx, y, groups))
            if (len(train) and len(test) and np.unique(y[train]).size == n_classes
                    and np.unique(y[test]).size == n_classes):
                return train, test
        except Exception:
            pass
    train, test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    return train, test


# ---------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Param:
    key: str
    label: str
    kind: str                  # "int" | "float" | "choice"
    default: object
    minimum: float = 0
    maximum: float = 1_000_000
    step: float = 1
    choices: tuple = ()
    tooltip: str = ""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    folder: str                # sub-folder of trained_model/ that holds this model's runs
    family: str
    input_kind: str            # "features" or "window"
    summary: str
    params: tuple = ()
    requires: str = ""         # importable module needed (e.g. "torch")
    install_hint: str = ""

    def availability(self) -> tuple:
        if not HAS_SKLEARN and self.input_kind == "features":
            return False, "scikit-learn is not installed (pip install scikit-learn)"
        if self.requires and importlib.util.find_spec(self.requires) is None:
            return False, self.install_hint or f"Python package '{self.requires}' is not installed"
        return True, ""

    def default_params(self) -> dict:
        return {p.key: p.default for p in self.params}


class DecisionProbaClassifier(BaseEstimator, ClassifierMixin):
    """Gives margin-based classifiers (SVMs) a ``predict_proba`` via a softmax of their decision scores.

    Real SVM probability calibration needs an internal cross-validation that is far too slow for tens of
    thousands of windows; the softmax is not calibrated but ranks classes correctly and drives the live
    confidence bar.
    """
    def __init__(self, estimator=None, temperature: float = 1.0):
        self.estimator = estimator
        self.temperature = temperature

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_
        self.n_features_in_ = int(np.asarray(X).shape[1])
        return self

    def _scores(self, X):
        d = np.asarray(self.estimator.decision_function(X), dtype=np.float64)
        return np.stack([-d, d], axis=1) if d.ndim == 1 else d

    def predict_proba(self, X):
        z = self._scores(X) / max(1e-6, float(self.temperature))
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.estimator.predict(X)


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


class CNN1DClassifier:
    """Small 1D convolutional network on raw windows (samples x channels).

    Each window is mean-centred per channel (the ADC baseline is irrelevant) and divided by the training
    standard deviation. Three conv blocks, global average pooling, dropout, linear head. Trained with Adam,
    light amplitude/noise augmentation and early stopping on a validation split.
    """
    input_kind = "window"

    def __init__(self, n_channels: int, n_classes: int, window_samples: int, epochs: int = 30, batch_size: int = 128,
                 lr: float = 1e-3, filters: int = 32, dropout: float = 0.3, weight_decay: float = 1e-4,
                 seed: int = 42, balanced: bool = True, patience: int = 6):
        self.config = dict(n_channels=int(n_channels), n_classes=int(n_classes), window_samples=int(window_samples),
                           epochs=int(epochs), batch_size=int(batch_size), lr=float(lr), filters=int(filters),
                           dropout=float(dropout), weight_decay=float(weight_decay), seed=int(seed),
                           balanced=bool(balanced), patience=int(patience))
        self.classes_ = np.arange(int(n_classes))
        self.scale_ = np.ones((1, 1, int(n_channels)), dtype=np.float32)
        self.history: list = []
        self._net = None
        self._state_bytes: Optional[bytes] = None

    # -- network -----------------------------------------------------------------------------
    def _build(self):
        import torch.nn as nn
        c, f = self.config["n_channels"], self.config["filters"]

        def block(cin, cout, k):
            return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU())

        return nn.Sequential(
            block(c, f, 7), nn.MaxPool1d(2),
            block(f, 2 * f, 5), nn.MaxPool1d(2),
            block(2 * f, 4 * f, 3),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(self.config["dropout"]),
            nn.Linear(4 * f, self.config["n_classes"]))

    def _ensure_net(self):
        if self._net is None:
            if not torch_available():
                raise RuntimeError("PyTorch is required to run this model: pip install torch")
            import torch
            self._net = self._build()
            if self._state_bytes is not None:
                self._net.load_state_dict(torch.load(io.BytesIO(self._state_bytes), map_location="cpu"))
            self._net.eval()
        return self._net

    def _prep(self, windows):
        x = np.asarray(windows, dtype=np.float32)
        if x.ndim == 2:
            x = x[None]
        x = x - x.mean(axis=1, keepdims=True)
        return np.ascontiguousarray((x / self.scale_).transpose(0, 2, 1))     # (n, channels, samples)

    # -- training ----------------------------------------------------------------------------
    def fit(self, windows, y, val_windows=None, val_y=None, progress=None, cancel=None):
        if not torch_available():
            raise RuntimeError("PyTorch is required to train this model: pip install torch")
        import torch
        import torch.nn as nn
        cfg = self.config
        torch.manual_seed(cfg["seed"])
        rng = np.random.default_rng(cfg["seed"])
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

        raw = np.asarray(windows, dtype=np.float32)
        centred = raw - raw.mean(axis=1, keepdims=True)
        self.scale_ = (centred.reshape(-1, centred.shape[2]).std(axis=0) + 1e-6).reshape(1, 1, -1).astype(np.float32)
        x_train = torch.from_numpy(self._prep(raw))
        y_train = torch.from_numpy(np.asarray(y, dtype=np.int64))
        x_val = torch.from_numpy(self._prep(val_windows)) if val_windows is not None else None
        y_val = np.asarray(val_y, dtype=np.int64) if val_y is not None else None

        net = self._build()
        weights = None
        if cfg["balanced"]:
            counts = np.bincount(np.asarray(y), minlength=cfg["n_classes"]).astype(np.float32)
            weights = torch.from_numpy(counts.sum() / (cfg["n_classes"] * np.maximum(counts, 1.0)))
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        best_state, best_score, stale = None, -1.0, 0
        n = len(y_train)
        for epoch in range(1, cfg["epochs"] + 1):
            if cancel is not None and cancel.is_set():
                break
            net.train()
            order = torch.from_numpy(rng.permutation(n))
            total = 0.0
            for start in range(0, n, cfg["batch_size"]):
                idx = order[start:start + cfg["batch_size"]]
                xb = x_train[idx]
                xb = xb * (1.0 + 0.1 * torch.randn(len(idx), 1, 1)) + 0.02 * torch.randn_like(xb)   # augmentation
                opt.zero_grad()
                loss = loss_fn(net(xb), y_train[idx])
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
            net.eval()
            val_acc = None
            if x_val is not None:
                with torch.no_grad():
                    pred = net(x_val).argmax(dim=1).numpy()
                val_acc = float((pred == y_val).mean())
            score = val_acc if val_acc is not None else -total / n
            self.history.append({"epoch": epoch, "loss": total / n, "val_acc": val_acc})
            if progress:
                progress(epoch, cfg["epochs"], total / n, val_acc)
            if score > best_score:
                best_score, stale = score, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= cfg["patience"]:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self._net = net
        self._state_bytes = None
        return self

    # -- inference ---------------------------------------------------------------------------
    def predict_proba(self, windows):
        import torch
        net = self._ensure_net()
        x = torch.from_numpy(self._prep(windows))
        out = []
        with torch.no_grad():
            for start in range(0, len(x), 1024):
                out.append(torch.softmax(net(x[start:start + 1024]), dim=1).numpy())
        return np.concatenate(out, axis=0)

    def predict(self, windows):
        return self.predict_proba(windows).argmax(axis=1)

    # -- pickling: the torch module itself is stored as a state_dict -------------------------------
    def __getstate__(self):
        state = dict(self.__dict__)
        if self._net is not None:
            import torch
            buf = io.BytesIO()
            torch.save(self._net.state_dict(), buf)
            state["_state_bytes"] = buf.getvalue()
        state["_net"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def _p(key, label, kind, default, lo=0, hi=1_000_000, step=1, choices=(), tip=""):
    return Param(key, label, kind, default, lo, hi, step, tuple(choices), tip)


MODEL_SPECS = {s.key: s for s in [
    ModelSpec("random_forest", "Random Forest", "random_forest", "Ensemble of trees", "features",
              "Robust, fast to train and to run; a strong default for EMG features.",
              (_p("n_estimators", "Trees", "int", 150, 20, 2000, 10, tip="Fewer trees = faster live prediction."),
               _p("max_depth", "Max depth (0 = none)", "int", 16, 0, 200),
               _p("min_samples_leaf", "Min samples per leaf", "int", 2, 1, 50))),
    ModelSpec("extra_trees", "Extra Trees", "extra_trees", "Ensemble of trees", "features",
              "Like Random Forest with randomised splits; often faster and similarly accurate.",
              (_p("n_estimators", "Trees", "int", 200, 20, 2000, 10),
               _p("max_depth", "Max depth (0 = none)", "int", 0, 0, 200))),
    ModelSpec("svm_rbf", "SVM (RBF kernel)", "svm_rbf", "Support vector machine", "features",
              "Non-linear margin classifier on standardised features; good with small datasets. Slower to train on very large ones.",
              (_p("c", "C (margin strictness)", "float", 10.0, 0.01, 1000.0, 1.0),
               _p("gamma", "Gamma", "choice", "scale", choices=("scale", "auto")))),
    ModelSpec("svm_linear", "SVM (linear)", "svm_linear", "Support vector machine", "features",
              "Fast linear SVM on standardised features; a good quick baseline.",
              (_p("c", "C (margin strictness)", "float", 1.0, 0.001, 1000.0, 0.5),)),
    ModelSpec("knn", "k-Nearest Neighbours", "knn", "Instance based", "features",
              "Classifies by the closest training windows; simple, but slower at prediction time on large data.",
              (_p("k", "Neighbours (k)", "int", 7, 1, 100),
               _p("weights", "Vote weighting", "choice", "distance", choices=("distance", "uniform")))),
    ModelSpec("logistic_regression", "Logistic Regression", "logistic_regression", "Linear model", "features",
              "Linear model with true probabilities; very fast and easy to interpret.",
              (_p("c", "C (inverse regularisation)", "float", 1.0, 0.001, 1000.0, 0.5),)),
    ModelSpec("lda", "Linear Discriminant Analysis", "lda", "Linear model", "features",
              "Classic, extremely fast EMG classifier; works well when classes are roughly Gaussian.", ()),
    ModelSpec("gradient_boosting", "Gradient Boosting", "gradient_boosting", "Ensemble of trees", "features",
              "Histogram gradient boosting; often the most accurate tree model, slower to tune.",
              (_p("max_iter", "Boosting rounds", "int", 200, 20, 2000, 10),
               _p("learning_rate", "Learning rate", "float", 0.1, 0.005, 1.0, 0.01),
               _p("max_depth", "Max depth (0 = none)", "int", 6, 0, 50))),
    ModelSpec("mlp", "Neural Network (MLP)", "mlp", "Neural network", "features",
              "Small fully connected network on standardised features (scikit-learn).",
              (_p("hidden", "Hidden units", "int", 128, 8, 1024, 8),
               _p("max_iter", "Max epochs", "int", 300, 20, 2000, 20))),
    ModelSpec("cnn1d", "1D CNN (deep learning)", "cnn_1d", "Deep learning", "window",
              "Convolutional network on the raw EMG window; learns its own features. Needs PyTorch; slower to train.",
              (_p("epochs", "Epochs", "int", 30, 1, 500, 5),
               _p("batch_size", "Batch size", "int", 128, 8, 2048, 8),
               _p("lr", "Learning rate", "float", 0.001, 0.00001, 0.1, 0.0005),
               _p("filters", "Base filters", "int", 32, 8, 256, 8),
               _p("dropout", "Dropout", "float", 0.3, 0.0, 0.8, 0.05)),
              requires="torch", install_hint="PyTorch is not installed: run  pip install torch"),
]}
DEFAULT_MODEL_KEY = "random_forest"


def build_estimator(spec: ModelSpec, params: dict, *, n_classes: int, n_channels: int, window_samples: int,
                    seed: int, balanced: bool):
    """Create an unfitted estimator for ``spec`` with ``params`` (missing params use their defaults)."""
    p = {**spec.default_params(), **(params or {})}
    cw = "balanced" if balanced else None
    jobs = max(1, (os.cpu_count() or 2) - 1)
    key = spec.key
    if key == "random_forest":
        return RandomForestClassifier(n_estimators=int(p["n_estimators"]), max_depth=int(p["max_depth"]) or None,
                                      min_samples_leaf=int(p["min_samples_leaf"]), class_weight=cw,
                                      random_state=seed, n_jobs=jobs)
    if key == "extra_trees":
        return ExtraTreesClassifier(n_estimators=int(p["n_estimators"]), max_depth=int(p["max_depth"]) or None,
                                    class_weight=cw, random_state=seed, n_jobs=jobs)
    if key == "svm_rbf":
        return make_pipeline(StandardScaler(), DecisionProbaClassifier(
            SVC(kernel="rbf", C=float(p["c"]), gamma=p["gamma"], class_weight=cw, decision_function_shape="ovr")))
    if key == "svm_linear":
        return make_pipeline(StandardScaler(), DecisionProbaClassifier(
            LinearSVC(C=float(p["c"]), class_weight=cw, random_state=seed, max_iter=5000)))
    if key == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=int(p["k"]), weights=p["weights"], n_jobs=jobs))
    if key == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=float(p["c"]), class_weight=cw, max_iter=2000))
    if key == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if key == "gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=int(p["max_iter"]), learning_rate=float(p["learning_rate"]),
                                              max_depth=int(p["max_depth"]) or None, class_weight=cw,
                                              random_state=seed)
    if key == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(int(p["hidden"]), max(8, int(p["hidden"]) // 2)), max_iter=int(p["max_iter"]),
            early_stopping=True, random_state=seed))
    if key == "cnn1d":
        return CNN1DClassifier(n_channels, n_classes, window_samples, epochs=int(p["epochs"]),
                               batch_size=int(p["batch_size"]), lr=float(p["lr"]), filters=int(p["filters"]),
                               dropout=float(p["dropout"]), seed=seed, balanced=balanced)
    raise KeyError(f"Unknown model '{key}'")


@dataclass
class FitResult:
    spec: ModelSpec
    params: dict
    estimator: object
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    report_dict: dict
    confusion: np.ndarray
    train_windows: int
    test_windows: int
    train_seconds: float
    latency_ms: float                 # one window, end to end (feature extraction + inference)
    num_features: int = 0
    history: list = field(default_factory=list)


def fit_and_evaluate(spec: ModelSpec, params: dict, ds: Dataset, train_idx, test_idx, *, seed: int, balanced: bool,
                     sample_rate: int, progress=None, cancel=None) -> FitResult:
    """Train ``spec`` on the train windows and score it on the held-out windows."""
    ok, why = spec.availability()
    if not ok:
        raise RuntimeError(f"{spec.name} is not available: {why}")
    n_classes = len(ds.classes)
    est = build_estimator(spec, params, n_classes=n_classes, n_channels=ds.channels,
                          window_samples=ds.window_samples, seed=seed, balanced=balanced)
    windowed = spec.input_kind == "window"
    data = ds.windows if windowed else ds.features
    if data is None:
        raise ValueError(f"The dataset was not prepared for {spec.name} ({spec.input_kind} input).")
    y = ds.y
    t0 = time.perf_counter()
    if windowed:
        # hold out 12 % of the training windows (group-aware) for early stopping
        sub_tr, sub_val = split_train_test(y[train_idx], ds.groups[train_idx], n_classes, 0.12, seed)
        tr, va = np.asarray(train_idx)[sub_tr], np.asarray(train_idx)[sub_val]
        est.fit(data[tr], y[tr], data[va], y[va], progress=progress, cancel=cancel)
    else:
        est.fit(data[train_idx], y[train_idx])
    train_seconds = time.perf_counter() - t0

    y_test = y[test_idx]
    y_pred = np.asarray(est.predict(data[test_idx])).astype(int)
    labels = np.arange(n_classes)
    report = classification_report(y_test, y_pred, labels=labels, target_names=ds.classes, digits=4,
                                   zero_division=0, output_dict=True)
    # latency of one live prediction: feature extraction (if any) + model
    sample = ds.probe_window
    probe = data[test_idx[:1]]
    t1 = time.perf_counter()
    reps = 15
    for _ in range(reps):
        if not windowed and sample is not None:
            extract_window_features(sample, sample_rate=sample_rate)
        est.predict_proba(probe) if hasattr(est, "predict_proba") else est.predict(probe)
    latency_ms = (time.perf_counter() - t1) * 1000.0 / reps
    return FitResult(
        spec=spec, params={**spec.default_params(), **(params or {})}, estimator=est,
        accuracy=float(accuracy_score(y_test, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
        macro_f1=float(f1_score(y_test, y_pred, average="macro", labels=labels, zero_division=0)),
        report_dict=report, confusion=confusion_matrix(y_test, y_pred, labels=labels),
        train_windows=int(len(train_idx)), test_windows=int(len(test_idx)), train_seconds=train_seconds,
        latency_ms=float(latency_ms), num_features=int(ds.features.shape[1]) if ds.features is not None else 0,
        history=list(getattr(est, "history", [])))

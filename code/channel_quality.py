"""Per-channel electrode-contact assessment from the calibration REST and FLEX captures (no Qt).

A dry-electrode armband either touches the skin well or it does not, and a bad channel quietly ruins
a whole session. These checks use only what calibration already records (a relaxed phase and a
flexed phase, raw ADC counts) and rate every EMG channel GOOD / FAIR / POOR / NO SIGNAL with a 0-100
score, the measured numbers behind it, and plain advice on what to fix.

What is measured (and why it says something about contact):
  * Flex response (SNR, dB)   RMS during FLEX versus RMS at REST. A well-coupled electrode over an active
                              muscle rises clearly; a lifted or dry one barely moves. The strongest cue.
  * Rest noise (RMS, counts)  Residual noise of the relaxed arm after removing the DC level and trend.
  * Mains hum (%)             Share of REST power at 50/60 Hz and harmonics. High hum = poor skin coupling.
  * Baseline drift (counts)   Change of the REST level between the first and second half (still settling).
  * Clipping (%)              Samples at the ADC limits.
  * Rail / flat line          DC level stuck at 0 or 4095, or no variation at all = floating / disconnected.
"""
from __future__ import annotations

import numpy as np

ADC_MAX = 4095
CLIP_LOW, CLIP_HIGH = 5, 4090
RAIL_MARGIN = 60                 # DC level within this many counts of 0 / 4095 = input pinned to a rail
FLAT_STD = 0.6                   # counts; below this a channel is not seeing any signal at all
MAINS_HZ = (50.0, 60.0)
MAINS_BAND_HZ = 2.5
BAND_LOW_HZ, BAND_HIGH_HZ = 5.0, 240.0
SNR_GOOD_DB, SNR_FAIR_DB = 12.0, 6.0

GOOD, FAIR, POOR, NO_SIGNAL = "GOOD", "FAIR", "POOR", "NO SIGNAL"
GRADE_RANK = {GOOD: 0, FAIR: 1, POOR: 2, NO_SIGNAL: 3}

TIPS = {
    "no_signal": "No signal. Electrode is not touching the skin or the lead is loose: re-seat the armband and wipe the skin.",
    "rail": "Input is stuck at the ADC limit (floating electrode). Press the band onto the skin and check the connection.",
    "weak_flex": "Barely reacts to the flex. Tighten the band, clean the skin with alcohol, or move it over the muscle belly.",
    "noisy_rest": "Noisy while relaxed. Ask the participant to relax fully and keep the cable still; check the reference contact.",
    "hum": "Mains hum is high. Move away from chargers / power cables and improve skin contact (dry skin couples badly).",
    "drift": "Baseline is still settling. Wait 30-60 s after fitting the band, then recalibrate.",
    "clipping": "Signal hits the ADC limits. Ease the flex or check that the band is not pressing unevenly.",
    "low_freq": "Slow movement artefact. Keep the forearm still and the band snug during rest.",
}


def _welch_power(x: np.ndarray, fs: float, nseg: int = 256):
    """Mean one-sided power spectrum of ``x`` (samples,) with 50 % overlapping Hann segments."""
    n = len(x)
    seg = min(nseg, n)
    if seg < 32:
        return np.zeros(1), np.zeros(1)
    win = np.hanning(seg)
    step = max(1, seg // 2)
    acc, count = None, 0
    for start in range(0, n - seg + 1, step):
        part = x[start:start + seg]
        part = (part - part.mean()) * win
        p = np.abs(np.fft.rfft(part)) ** 2
        acc = p if acc is None else acc + p
        count += 1
    return np.fft.rfftfreq(seg, 1.0 / fs), acc / max(1, count)


def _detrend(x: np.ndarray) -> np.ndarray:
    t = np.arange(len(x), dtype=np.float64)
    if len(x) < 3:
        return x - np.mean(x)
    slope, intercept = np.polyfit(t, x, 1)
    return x - (slope * t + intercept)


def _ramp(value: float, lo: float, hi: float, max_penalty: float) -> float:
    """0 at ``lo`` rising linearly to ``max_penalty`` at ``hi`` (clamped)."""
    if hi == lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0) * max_penalty)


def assess_channel(rest: np.ndarray, flex: np.ndarray | None, fs: float, name: str = "") -> dict:
    """Rate one channel. ``rest`` / ``flex`` are 1-D arrays of raw ADC counts (flex may be None or empty)."""
    rest = np.asarray(rest, dtype=np.float64)
    rest = rest[np.isfinite(rest)]
    flex = np.asarray(flex if flex is not None else [], dtype=np.float64)
    flex = flex[np.isfinite(flex)]
    out = {"name": name, "grade": NO_SIGNAL, "score": 0, "snr_db": None, "rest_rms": None, "flex_rms": None,
           "hum_pct": None, "drift": None, "clip_pct": None, "dc": None, "issues": [], "advice": ""}
    if rest.size < 16:
        out["issues"] = [("no_signal", TIPS["no_signal"])]
        out["advice"] = TIPS["no_signal"]
        return out

    dc = float(np.median(rest))
    rest_ac = _detrend(rest)
    rest_rms = float(np.sqrt(np.mean(rest_ac ** 2)))
    flex_rms = float(np.sqrt(np.mean((flex - np.median(rest)) ** 2))) if flex.size >= 16 else None
    half = rest.size // 2
    drift = float(abs(np.median(rest[half:]) - np.median(rest[:half])))
    both = np.concatenate((rest, flex)) if flex.size else rest
    clip_pct = float(100.0 * np.mean((both <= CLIP_LOW) | (both >= CLIP_HIGH)))

    freqs, power = _welch_power(rest, fs)
    band = (freqs >= BAND_LOW_HZ) & (freqs <= min(BAND_HIGH_HZ, fs / 2.0 - 1.0))
    total = float(power[band].sum()) if band.any() else 0.0
    hum_pct = 0.0
    low_pct = 0.0
    if total > 1e-9:
        # Mains is 50 Hz or 60 Hz depending on the country: score whichever family carries the energy.
        fam = [float(power[band & (np.abs(freqs - f0 * np.arange(1, 5)[:, None]) <= MAINS_BAND_HZ).any(axis=0)].sum())
               for f0 in MAINS_HZ]
        hum_pct = 100.0 * max(fam) / total
        low_pct = 100.0 * float(power[(freqs >= BAND_LOW_HZ) & (freqs < 15.0)].sum()) / total

    snr_db = None
    if flex_rms is not None:
        snr_db = float(20.0 * np.log10((flex_rms + 1e-6) / (max(rest_rms, 0.5))))

    out.update(dc=dc, rest_rms=rest_rms, flex_rms=flex_rms, snr_db=snr_db, hum_pct=float(hum_pct),
               drift=drift, clip_pct=clip_pct)

    issues = []          # (code, penalty)
    rail = dc <= RAIL_MARGIN or dc >= ADC_MAX - RAIL_MARGIN
    flat = float(np.std(rest)) < FLAT_STD and (flex.size < 16 or float(np.std(flex)) < FLAT_STD)
    if rail:
        issues.append(("rail", 100.0))
    elif flat:
        issues.append(("no_signal", 100.0))
    else:
        if snr_db is not None:
            pen = _ramp(snr_db, SNR_GOOD_DB, SNR_FAIR_DB, 35.0) if snr_db >= SNR_FAIR_DB \
                else 35.0 + _ramp(snr_db, SNR_FAIR_DB, 0.0, 30.0)
            if pen > 0:
                issues.append(("weak_flex", pen))
        pen = _ramp(rest_rms, 15.0, 60.0, 20.0) + _ramp(rest_rms, 60.0, 160.0, 20.0)
        if pen > 0:
            issues.append(("noisy_rest", pen))
        pen = _ramp(hum_pct, 20.0, 50.0, 20.0) + _ramp(hum_pct, 50.0, 90.0, 10.0)
        if pen > 0:
            issues.append(("hum", pen))
        pen = _ramp(drift, 15.0, 60.0, 15.0)
        if pen > 0:
            issues.append(("drift", pen))
        pen = _ramp(clip_pct, 0.3, 1.0, 25.0) + _ramp(clip_pct, 1.0, 5.0, 25.0)
        if pen > 0:
            issues.append(("clipping", pen))
        pen = _ramp(low_pct, 60.0, 90.0, 10.0)
        if pen > 0:
            issues.append(("low_freq", pen))

    score = int(round(max(0.0, 100.0 - sum(p for _c, p in issues))))
    if flat or rail:
        grade = NO_SIGNAL
        score = 0
    elif score >= 75:
        grade = GOOD
    elif score >= 50:
        grade = FAIR
    else:
        grade = POOR
    issues.sort(key=lambda item: -item[1])
    # A GOOD channel only lists problems that are still noticeable; a bad one lists everything it found.
    shown = [(c, TIPS[c]) for c, p in issues if p >= (8.0 if grade == GOOD else 3.0)]
    out.update(grade=grade, score=score, issues=shown,
               advice=(shown[0][1] if shown else ("Good contact." if grade == GOOD else "")))
    return out


def assess_channels(rest: np.ndarray, flex: np.ndarray | None, fs: float, names=None) -> dict:
    """Rate every channel. ``rest`` / ``flex`` have shape (samples, channels) of raw ADC counts."""
    rest = np.asarray(rest, dtype=np.float64)
    if rest.ndim == 1:
        rest = rest[:, None]
    flex_arr = None if flex is None or np.size(flex) == 0 else np.asarray(flex, dtype=np.float64)
    if flex_arr is not None and flex_arr.ndim == 1:
        flex_arr = flex_arr[:, None]
    channels = []
    for ch in range(rest.shape[1]):
        label = names[ch] if names and ch < len(names) else f"Ch{ch + 1}"
        f = flex_arr[:, ch] if flex_arr is not None and ch < flex_arr.shape[1] else None
        item = assess_channel(rest[:, ch], f, fs, label)
        item["index"] = ch
        channels.append(item)

    counts = {g: sum(1 for c in channels if c["grade"] == g) for g in GRADE_RANK}
    bad = [c for c in channels if c["grade"] in (POOR, NO_SIGNAL)]
    weak = [c for c in channels if c["grade"] == FAIR]
    ready = not bad
    if ready and not weak:
        message = f"All {len(channels)} channels have good contact. Ready to record."
    elif ready:
        message = (f"{counts[GOOD]} good, {counts[FAIR]} fair ({', '.join(c['name'] for c in weak)}). "
                   "Usable; improving the fair channels will give cleaner data.")
    else:
        message = (f"Fix before recording: {', '.join(c['name'] for c in bad)}. "
                   "Adjust the band / skin contact, then recalibrate.")
    return {"channels": channels, "counts": counts, "ready": ready, "message": message,
            "mean_score": float(np.mean([c["score"] for c in channels])) if channels else 0.0,
            "fs": float(fs)}


def summary_for_metadata(report: dict | None) -> dict | None:
    """Compact, JSON-friendly copy of a report for the session metadata."""
    if not report:
        return None

    def num(value, nd=2):
        return None if value is None else round(float(value), nd)

    return {
        "ready": bool(report["ready"]), "message": report["message"], "counts": dict(report["counts"]),
        "channels": [{"channel": c["index"] + 1, "grade": c["grade"], "score": c["score"],
                      "flex_snr_db": num(c["snr_db"], 1), "rest_rms": num(c["rest_rms"], 2),
                      "hum_pct": num(c["hum_pct"], 1), "drift_counts": num(c["drift"], 1),
                      "clip_pct": num(c["clip_pct"], 2)} for c in report["channels"]],
    }

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.fft import rfft, rfftfreq
from scipy.signal import stft


@dataclass
class AudioData:
    path: Path | None
    data: np.ndarray
    samplerate: int


def save_wav(path: str | Path, data: np.ndarray, samplerate: int):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, int(samplerate))


def load_wav(path: str | Path):
    path = Path(path)
    data, sr = sf.read(path, always_2d=True)
    return AudioData(path, data[:, 0].astype(np.float64), int(sr))


def remove_dc(data):
    return data - np.mean(data) if data.size else data


def dbfs(value):
    return float(20 * np.log10(max(float(value), 1e-12)))


def rms(data):
    return float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0


def rms_dbfs(data):
    return dbfs(rms(remove_dc(data)))


def peak_dbfs(data):
    return dbfs(np.max(np.abs(data))) if data.size else -240.0


def compute_fft(data, sr):
    data = remove_dc(data)
    if data.size == 0:
        return np.array([]), np.array([])
    window = np.hanning(len(data))
    spec = np.abs(rfft(data * window))
    freqs = rfftfreq(len(data), 1 / sr)
    return freqs, 20 * np.log10(spec + 1e-12)


def dominant_frequency(data, sr, min_freq=80):
    data = remove_dc(data)
    if data.size == 0:
        return 0.0
    data = data[: min(len(data), int(sr * 2))]
    f, sdb = compute_fft(data, sr)
    valid = f >= min_freq
    if not np.any(valid):
        return 0.0
    slin = np.power(10, sdb[valid] / 20)
    return float(f[valid][int(np.argmax(slin))])


def analyze_array(data, sr):
    return {
        "sample_rate": sr,
        "duration": len(data) / sr if sr else 0,
        "peak_dbfs": peak_dbfs(data),
        "rms_dbfs": rms_dbfs(data),
        "dominant_frequency": dominant_frequency(data, sr),
        "clipping": bool(np.max(np.abs(data)) >= 0.999) if data.size else False,
    }


def harmonic_level(data, sr, target_freq, search_width_hz=8.0):
    data = remove_dc(data)
    if data.size == 0:
        return {"frequency": target_freq, "amplitude": 0.0, "db": -240.0}

    max_samples = min(len(data), int(sr * 4))
    data = data[:max_samples]

    freqs, spec_db = compute_fft(data, sr)
    if freqs.size == 0:
        return {"frequency": target_freq, "amplitude": 0.0, "db": -240.0}

    valid = (freqs >= target_freq - search_width_hz) & (
        freqs <= target_freq + search_width_hz
    )
    if not valid.any():
        return {"frequency": target_freq, "amplitude": 0.0, "db": -240.0}

    local_freqs = freqs[valid]
    local_db = spec_db[valid]
    idx = int(np.argmax(local_db))
    peak_db = float(local_db[idx])
    amp = float(10 ** (peak_db / 20.0))
    return {"frequency": float(local_freqs[idx]), "amplitude": amp, "db": peak_db}


def analyze_thd(data, sr, fundamental=None, max_harmonic=5):
    if data.size == 0:
        return {"fundamental_hz": 0.0, "thd_percent": 0.0, "harmonics": []}

    if fundamental is None or fundamental <= 0:
        fundamental = dominant_frequency(data, sr)

    harmonics = []
    for n in range(1, max_harmonic + 1):
        target = fundamental * n
        if target >= sr / 2:
            break
        h = harmonic_level(data, sr, target)
        h["order"] = n
        h["target_hz"] = target
        harmonics.append(h)

    if not harmonics or harmonics[0]["amplitude"] <= 0:
        thd = 0.0
    else:
        v1 = harmonics[0]["amplitude"]
        harmonic_power = sum(h["amplitude"] ** 2 for h in harmonics[1:])
        thd = (harmonic_power**0.5) / v1 * 100.0

    return {
        "fundamental_hz": float(fundamental),
        "thd_percent": float(thd),
        "harmonics": harmonics,
    }


def analyze_thdn_sinad(
    data, sr, fundamental=None, bandwidth_hz=20000.0, notch_width_hz=20.0
):
    data = remove_dc(data)
    if data.size == 0:
        return {
            "fundamental_hz": 0.0,
            "thdn_percent": 0.0,
            "sinad_db": 0.0,
            "signal_db": -240.0,
            "noise_distortion_db": -240.0,
        }

    if fundamental is None or fundamental <= 0:
        fundamental = dominant_frequency(data, sr)

    max_samples = min(len(data), int(sr * 4))
    data = data[:max_samples]

    window = np.hanning(len(data))
    spectrum = np.abs(rfft(data * window))
    freqs = rfftfreq(len(data), 1 / sr)

    valid = (freqs >= 20.0) & (freqs <= min(bandwidth_hz, sr / 2))
    signal_band = (
        (freqs >= fundamental - notch_width_hz)
        & (freqs <= fundamental + notch_width_hz)
        & valid
    )
    nd_band = valid & (~signal_band)

    signal_power = float(np.sum(spectrum[signal_band] ** 2))
    nd_power = float(np.sum(spectrum[nd_band] ** 2))

    if signal_power <= 0:
        thdn_percent = 0.0
        sinad_db = 0.0
    else:
        signal_mag = signal_power**0.5
        nd_mag = max(nd_power**0.5, 1e-12)
        thdn_percent = nd_mag / signal_mag * 100.0
        sinad_db = 20.0 * np.log10(signal_mag / nd_mag)

    return {
        "fundamental_hz": float(fundamental),
        "thdn_percent": float(thdn_percent),
        "sinad_db": float(sinad_db),
        "signal_db": float(20.0 * np.log10(max(signal_power**0.5, 1e-12))),
        "noise_distortion_db": float(20.0 * np.log10(max(nd_power**0.5, 1e-12))),
    }


def analyze_sweep_stft(data, sr, min_freq=20.0, max_freq=20000.0, bins_per_octave=12):
    """Estimate a relative frequency-response curve from a recorded sweep.

    This is an automatic sweep tracker based on STFT peak tracking. It is meant for
    practical measurement-console use when the speaker generates a sweep but the
    exact excitation signal is not available. It returns a relative response curve
    normalized to the level near 1 kHz.
    """
    data = remove_dc(np.asarray(data, dtype=np.float64))
    if data.size == 0 or sr <= 0:
        return {
            "ok": False,
            "reason": "empty audio",
            "frequency_hz": np.array([]),
            "level_db": np.array([]),
            "relative_db": np.array([]),
            "tracked_frequency_hz": np.array([]),
            "tracked_level_db": np.array([]),
            "duration_s": 0.0,
            "detected_start_hz": 0.0,
            "detected_stop_hz": 0.0,
            "detected_direction": "unknown",
            "points": 0,
        }

    min_freq = max(10.0, float(min_freq))
    max_freq = min(float(max_freq), sr / 2.0 * 0.95)
    if max_freq <= min_freq:
        max_freq = min(sr / 2.0 * 0.95, max(min_freq * 2.0, 20000.0))

    # About 85 ms at 96 kHz. This gives acceptable low-frequency resolution while still tracking most practical speaker sweeps.
    nperseg = int(min(max(4096, sr * 0.085), max(4096, data.size)))
    nperseg = min(nperseg, data.size)
    noverlap = int(nperseg * 0.75)
    if nperseg < 256:
        return {
            "ok": False,
            "reason": "audio too short for sweep analysis",
            "frequency_hz": np.array([]),
            "level_db": np.array([]),
            "relative_db": np.array([]),
            "tracked_frequency_hz": np.array([]),
            "tracked_level_db": np.array([]),
            "duration_s": data.size / sr,
            "detected_start_hz": 0.0,
            "detected_stop_hz": 0.0,
            "detected_direction": "unknown",
            "points": 0,
        }

    freqs, times, z = stft(
        data,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    mag = np.abs(z)
    valid = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(valid) or mag.size == 0:
        return {
            "ok": False,
            "reason": "no frequencies inside requested sweep band",
            "frequency_hz": np.array([]),
            "level_db": np.array([]),
            "relative_db": np.array([]),
            "tracked_frequency_hz": np.array([]),
            "tracked_level_db": np.array([]),
            "duration_s": data.size / sr,
            "detected_start_hz": 0.0,
            "detected_stop_hz": 0.0,
            "detected_direction": "unknown",
            "points": 0,
        }

    band_freqs = freqs[valid]
    band_mag = mag[valid, :]
    peak_idx = np.argmax(band_mag, axis=0)
    peak_freqs = band_freqs[peak_idx]
    peak_mags = band_mag[peak_idx, np.arange(band_mag.shape[1])]
    peak_db = 20.0 * np.log10(np.maximum(peak_mags, 1e-12))

    # Keep the stronger frames  ambient noise between/after sweeps does not dominate.
    if peak_db.size:
        threshold = np.percentile(peak_db, 35.0)
        keep = peak_db >= threshold
    else:
        keep = np.array([], dtype=bool)
    peak_freqs = peak_freqs[keep]
    peak_db = peak_db[keep]

    if peak_freqs.size < 5:
        return {
            "ok": False,
            "reason": "too few reliable sweep frames detected",
            "frequency_hz": np.array([]),
            "level_db": np.array([]),
            "relative_db": np.array([]),
            "tracked_frequency_hz": peak_freqs,
            "tracked_level_db": peak_db,
            "duration_s": data.size / sr,
            "detected_start_hz": float(peak_freqs[0]) if peak_freqs.size else 0.0,
            "detected_stop_hz": float(peak_freqs[-1]) if peak_freqs.size else 0.0,
            "detected_direction": "unknown",
            "points": int(peak_freqs.size),
        }

    # Bin tracked peaks on a log-frequency axis, using median level per bin.
    octaves = np.log2(max_freq / min_freq)
    n_bins = int(max(12, min(240, round(octaves * float(bins_per_octave)))))
    edges = np.geomspace(min_freq, max_freq, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    level = np.full(n_bins, np.nan, dtype=np.float64)
    for i in range(n_bins):
        inside = (peak_freqs >= edges[i]) & (peak_freqs < edges[i + 1])
        if np.any(inside):
            level[i] = float(np.median(peak_db[inside]))

    good = np.isfinite(level)
    if np.count_nonzero(good) >= 2:
        level = np.interp(np.log(centers), np.log(centers[good]), level[good])
    else:
        good_peak = np.isfinite(peak_db)
        if np.count_nonzero(good_peak) >= 2:
            order = np.argsort(peak_freqs[good_peak])
            level = np.interp(
                np.log(centers),
                np.log(peak_freqs[good_peak][order]),
                peak_db[good_peak][order],
            )
        else:
            level = np.nan_to_num(level, nan=-240.0)

    ref_idx = int(np.argmin(np.abs(centers - 1000.0)))
    ref_level = float(level[ref_idx])
    relative = level - ref_level

    detected_start = float(peak_freqs[0])
    detected_stop = float(peak_freqs[-1])
    if detected_stop > detected_start * 1.15:
        direction = "up"
    elif detected_start > detected_stop * 1.15:
        direction = "down"
    else:
        direction = "mixed/unknown"

    return {
        "ok": True,
        "reason": "ok",
        "frequency_hz": centers,
        "level_db": level,
        "relative_db": relative,
        "tracked_frequency_hz": peak_freqs,
        "tracked_level_db": peak_db,
        "duration_s": data.size / sr,
        "detected_start_hz": detected_start,
        "detected_stop_hz": detected_stop,
        "detected_direction": direction,
        "points": int(peak_freqs.size),
        "reference_level_db": ref_level,
    }


def _moving_average_nan_safe(values, window):
    values = np.asarray(values, dtype=np.float64)
    window = int(max(1, window))
    if window <= 1 or values.size == 0:
        return values
    out = np.empty_like(values)
    half = window // 2
    for i in range(values.size):
        lo = max(0, i - half)
        hi = min(values.size, i + half + 1)
        out[i] = float(np.nanmedian(values[lo:hi]))
    return out


def analyze_sweep_known(
    data,
    sr,
    start_freq=20.0,
    stop_freq=20000.0,
    duration_s=14.0,
    direction="Up",
    sweep_type="Log",
    trim_start_s=0.20,
    trim_end_s=0.20,
    bins_per_octave=24,
    smooth=True,
):
    """Estimate relative response from a known sweep mapping.

    This is intended for sources such as Bedrock BTB65 where the sweep is known:
    20 Hz -> 20 kHz, 14 s, logarithmic sweep. Direction is selectable. Instead of following the
    loudest STFT peak, it maps each time frame to the expected sweep frequency
    and samples the spectrum there. This is more robust in rooms where resonances
    or noise can be louder than the instantaneous sweep tone.
    """
    data = remove_dc(np.asarray(data, dtype=np.float64))
    if data.size == 0 or sr <= 0:
        return {
            "ok": False,
            "reason": "empty audio",
            "frequency_hz": np.array([]),
            "level_db": np.array([]),
            "relative_db": np.array([]),
            "tracked_frequency_hz": np.array([]),
            "tracked_level_db": np.array([]),
            "duration_s": 0.0,
            "detected_start_hz": 0.0,
            "detected_stop_hz": 0.0,
            "detected_direction": "known",
            "points": 0,
        }

    duration_actual = data.size / float(sr)
    expected_duration = float(duration_s)
    if expected_duration <= 0:
        expected_duration = duration_actual

    # v15: choose the sweep segment more robustly. v14 always used the last
    # requested duration, which failed if the sweep did not end exactly at capture.
    # When the buffer is longer than the expected sweep, choose the window with
    # maximum RMS energy. This keeps tone modes unchanged and only affects sweep.
    original_size = data.size
    target_samples = int(min(data.size, max(1, round(expected_duration * sr))))
    segment_start_sample = 0
    if data.size > target_samples:
        # Downsampled energy search: 50 ms blocks, then find the best contiguous
        # target-duration region. This is fast and avoids repeated large convolutions.
        block = max(1, int(round(0.05 * sr)))
        n_blocks = data.size // block
        if n_blocks > 2:
            trimmed_for_blocks = data[: n_blocks * block]
            block_energy = np.mean(
                trimmed_for_blocks.reshape(n_blocks, block) ** 2, axis=1
            )
            win_blocks = max(1, int(round(target_samples / block)))
            win_blocks = min(win_blocks, n_blocks)
            kernel = np.ones(win_blocks, dtype=np.float64)
            scores = np.convolve(block_energy, kernel, mode="valid")
            best_block = int(np.argmax(scores))
            segment_start_sample = min(best_block * block, data.size - target_samples)
        else:
            segment_start_sample = data.size - target_samples
        data = data[segment_start_sample : segment_start_sample + target_samples]
    else:
        data = data[-target_samples:]
        segment_start_sample = max(0, original_size - target_samples)
    segment_end_sample = segment_start_sample + data.size
    duration_actual = data.size / float(sr)

    start_freq = max(5.0, float(start_freq))
    stop_freq = min(float(stop_freq), sr / 2.0 * 0.95)
    f_low = min(start_freq, stop_freq)
    f_high = max(start_freq, stop_freq)
    if f_high <= f_low:
        return {
            "ok": False,
            "reason": "invalid sweep frequency range",
            "frequency_hz": np.array([]),
            "level_db": np.array([]),
            "relative_db": np.array([]),
            "tracked_frequency_hz": np.array([]),
            "tracked_level_db": np.array([]),
            "duration_s": duration_actual,
            "detected_start_hz": start_freq,
            "detected_stop_hz": stop_freq,
            "detected_direction": "known",
            "points": 0,
        }

    trim_start_s = max(0.0, float(trim_start_s))
    trim_end_s = max(0.0, float(trim_end_s))
    if trim_start_s + trim_end_s >= duration_actual * 0.75:
        trim_start_s = 0.0
        trim_end_s = 0.0

    # ~85 ms windows; enough for practical sweep display while keeping time tracking.
    nperseg = int(min(max(4096, sr * 0.085), max(4096, data.size)))
    nperseg = min(nperseg, data.size)
    noverlap = int(nperseg * 0.75)
    if nperseg < 256:
        return {
            "ok": False,
            "reason": "audio too short for known sweep analysis",
            "frequency_hz": np.array([]),
            "level_db": np.array([]),
            "relative_db": np.array([]),
            "tracked_frequency_hz": np.array([]),
            "tracked_level_db": np.array([]),
            "duration_s": duration_actual,
            "detected_start_hz": start_freq,
            "detected_stop_hz": stop_freq,
            "detected_direction": "known",
            "points": 0,
        }

    freqs, times, z = stft(
        data,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    mag = np.abs(z)
    valid_time = (times >= trim_start_s) & (
        times <= max(trim_start_s, duration_actual - trim_end_s)
    )
    if not np.any(valid_time):
        valid_time = np.ones_like(times, dtype=bool)

    times_v = times[valid_time]
    mag_v = mag[:, valid_time]
    usable_duration = max(1e-9, (duration_actual - trim_start_s - trim_end_s))
    p = np.clip((times_v - trim_start_s) / usable_duration, 0.0, 1.0)

    direction_norm = str(direction).strip().lower()
    type_norm = str(sweep_type).strip().lower()
    if direction_norm.startswith("down"):
        f0, f1 = f_high, f_low
        direction_label = "known down"
    else:
        f0, f1 = f_low, f_high
        direction_label = "known up"

    if type_norm.startswith("lin"):
        expected_freqs = f0 + (f1 - f0) * p
        sweep_type_label = "linear"
    else:
        expected_freqs = f0 * np.power(f1 / f0, p)
        sweep_type_label = "logarithmic"

    levels = []
    used_freqs = []
    for col, target in enumerate(expected_freqs):
        if target < f_low or target > f_high:
            continue
        # Use a narrow local band around expected frequency. Wider at high freq,
        # but not so wide that room peaks far away dominate.
        half_width = max(8.0, target * 0.03)
        band = (freqs >= target - half_width) & (freqs <= target + half_width)
        if not np.any(band):
            idx = int(np.argmin(np.abs(freqs - target)))
            amp = mag_v[idx, col]
        else:
            amp = np.max(mag_v[band, col])
        levels.append(20.0 * np.log10(max(float(amp), 1e-12)))
        used_freqs.append(float(target))

    used_freqs = np.asarray(used_freqs, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    if used_freqs.size < 5:
        return {
            "ok": False,
            "reason": "too few mapped sweep frames",
            "frequency_hz": np.array([]),
            "level_db": np.array([]),
            "relative_db": np.array([]),
            "tracked_frequency_hz": used_freqs,
            "tracked_level_db": levels,
            "duration_s": duration_actual,
            "detected_start_hz": float(f0),
            "detected_stop_hz": float(f1),
            "detected_direction": direction_label,
            "points": int(used_freqs.size),
        }

    octaves = np.log2(f_high / f_low)
    n_bins = int(max(24, min(360, round(octaves * float(bins_per_octave)))))
    edges = np.geomspace(f_low, f_high, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    binned = np.full(n_bins, np.nan, dtype=np.float64)
    for i in range(n_bins):
        inside = (used_freqs >= edges[i]) & (used_freqs < edges[i + 1])
        if np.any(inside):
            binned[i] = float(np.median(levels[inside]))

    good = np.isfinite(binned)
    if np.count_nonzero(good) >= 2:
        binned = np.interp(np.log(centers), np.log(centers[good]), binned[good])
    else:
        order = np.argsort(used_freqs)
        binned = np.interp(np.log(centers), np.log(used_freqs[order]), levels[order])

    if smooth:
        # Roughly 1/6-octave display smoothing when bins_per_octave=24.
        smooth_window = max(3, int(round(float(bins_per_octave) / 6.0)))
        if smooth_window % 2 == 0:
            smooth_window += 1
        binned = _moving_average_nan_safe(binned, smooth_window)

    ref_idx = int(np.argmin(np.abs(centers - 1000.0)))
    ref_level = float(binned[ref_idx])
    relative = binned - ref_level

    warnings = []
    if duration_actual < expected_duration * 0.95:
        warnings.append("Captured sweep segment is shorter than expected duration.")

    return {
        "ok": True,
        "reason": "ok",
        "frequency_hz": centers,
        "level_db": binned,
        "relative_db": relative,
        "tracked_frequency_hz": used_freqs,
        "tracked_level_db": levels,
        "duration_s": duration_actual,
        "detected_start_hz": float(f0),
        "detected_stop_hz": float(f1),
        "detected_direction": direction_label,
        "sweep_type": sweep_type_label,
        "points": int(used_freqs.size),
        "reference_level_db": ref_level,
        "trim_start_s": float(trim_start_s),
        "trim_end_s": float(trim_end_s),
        "segment_start_s": float(segment_start_sample / float(sr)),
        "segment_end_s": float(segment_end_sample / float(sr)),
        "warnings": warnings,
    }


def analyze_external_sweep_response(
    data,
    sr,
    start_hz=20.0,
    end_hz=20000.0,
    duration_s=None,
    bins=180,
    sweep_type="log",
    direction="up",
    window_ms=50.0,
):
    """Simple RMS-vs-time sweep response used as an alternative sweep method.

    This method maps frequency to time and measures RMS in a small time window.
    It intentionally does NOT follow the loudest FFT peak. This is useful for
    externally generated sweeps when the generator timing is known and the old
    sweep method gave stable practical results. Output is normalized to 1 kHz.
    """
    data = remove_dc(np.asarray(data, dtype=np.float64))
    if data.size == 0 or sr <= 0:
        return {
            "ok": False,
            "reason": "empty audio",
            "frequency_hz": np.array([]),
            "level_db": np.array([]),
            "relative_db": np.array([]),
            "tracked_frequency_hz": np.array([]),
            "tracked_level_db": np.array([]),
            "duration_s": 0.0,
            "detected_start_hz": 0.0,
            "detected_stop_hz": 0.0,
            "detected_direction": "known",
            "points": 0,
            "sweep_type": str(sweep_type),
        }

    if duration_s is None or float(duration_s) <= 0:
        duration_s = len(data) / float(sr)
    duration_s = float(duration_s)
    n = min(len(data), int(round(duration_s * sr)))
    data = data[:n]
    duration_s = n / float(sr)
    if duration_s <= 0:
        return {
            "ok": False,
            "reason": "invalid duration",
            "frequency_hz": np.array([]),
            "level_db": np.array([]),
            "relative_db": np.array([]),
            "tracked_frequency_hz": np.array([]),
            "tracked_level_db": np.array([]),
            "duration_s": 0.0,
            "detected_start_hz": 0.0,
            "detected_stop_hz": 0.0,
            "detected_direction": "known",
            "points": 0,
            "sweep_type": str(sweep_type),
        }

    start_hz = max(float(start_hz), 1.0)
    end_hz = min(float(end_hz), sr / 2.0 * 0.95)
    if end_hz <= start_hz:
        end_hz = min(sr / 2.0 * 0.95, max(start_hz * 2.0, 20000.0))

    freqs = np.geomspace(start_hz, end_hz, int(max(8, bins)))
    direction_norm = str(direction).strip().lower()
    sweep_type_norm = str(sweep_type).strip().lower()

    if sweep_type_norm.startswith("lin"):

        def progress_for_f(f):
            return (float(f) - start_hz) / max(end_hz - start_hz, 1e-12)

        sweep_type_label = "linear"
    else:
        log_ratio = np.log(end_hz / start_hz)

        def progress_for_f(f):
            return np.log(float(f) / start_hz) / max(log_ratio, 1e-12)

        sweep_type_label = "logarithmic"

    levels = []
    half = max(1, int((float(window_ms) / 1000.0) * sr / 2.0))
    for f in freqs:
        p = np.clip(progress_for_f(f), 0.0, 1.0)
        if direction_norm.startswith("down"):
            p = 1.0 - p
        idx = int(round(p * (len(data) - 1)))
        a = max(0, idx - half)
        b = min(len(data), idx + half)
        levels.append(rms_dbfs(data[a:b]) if b > a else -240.0)

    levels = np.asarray(levels, dtype=np.float64)
    ref_idx = int(np.argmin(np.abs(freqs - 1000.0)))
    ref = float(levels[ref_idx]) if levels.size else -240.0
    relative = levels - ref

    if direction_norm.startswith("down"):
        detected_start = end_hz
        detected_stop = start_hz
        direction_label = "known down"
    else:
        detected_start = start_hz
        detected_stop = end_hz
        direction_label = "known up"

    return {
        "ok": True,
        "reason": "ok",
        "frequency_hz": freqs,
        "level_db": levels,
        "relative_db": relative,
        "tracked_frequency_hz": freqs,
        "tracked_level_db": levels,
        "duration_s": duration_s,
        "detected_start_hz": float(detected_start),
        "detected_stop_hz": float(detected_stop),
        "detected_direction": direction_label,
        "points": int(freqs.size),
        "reference_level_db": ref,
        "sweep_type": sweep_type_label,
        "warnings": [],
    }

import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
import mne


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

WINDOW_SECONDS = 30

TARGET_SAMPLE_RATES = {
    'eeg': 128,
    'ecg': 128,
    'emg': 128,
}

EXPECTED_CHANNELS = {
    'eeg': 2,
    'ecg': 1,
    'emg': 1,
}


# ═══════════════════════════════════════════════════════════════════════
# FILTERING
# ═══════════════════════════════════════════════════════════════════════

def apply_filters(data, sample_rate, modality):
    """
    EEG : 0.5–40 Hz + 50 Hz notch
    ECG : 0.5–40 Hz + 50 Hz notch
    EMG : 20–60 Hz
    """

    if data.shape[1] < 8:
        return data.astype(np.float32, copy=False)

    nyq = sample_rate / 2.0

    if modality == 'eeg':
        low, high = 0.5, min(40.0, nyq - 1.0)
        apply_notch = True

    elif modality == 'ecg':
        low, high = 0.5, min(40.0, nyq - 1.0)
        apply_notch = True

    elif modality == 'emg':
        low, high = 20.0, min(60.0, nyq - 1.0)
        apply_notch = False

    else:
        return data.astype(np.float32, copy=False)

    if low >= high:
        return data.astype(np.float32, copy=False)

    b, a = butter(
        4,
        [low / nyq, high / nyq],
        btype='band'
    )

    filtered = filtfilt(b, a, data, axis=1)

    if apply_notch:
        notch_freq = 50.0

        if notch_freq < nyq:
            bn, an = iirnotch(notch_freq / nyq, Q=30)
            filtered = filtfilt(bn, an, filtered, axis=1)

    return filtered.astype(np.float32, copy=False)


# ═══════════════════════════════════════════════════════════════════════
# NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════

def normalize_window(data):
    mean = data.mean(axis=1, keepdims=True)
    std  = data.std(axis=1, keepdims=True)

    std = np.where(std < 1e-6, 1.0, std)

    return (data - mean) / std


# ═══════════════════════════════════════════════════════════════════════
# RESAMPLING
# ═══════════════════════════════════════════════════════════════════════

def resample_if_needed(data, source_rate, target_rate):

    if int(round(source_rate)) == int(target_rate):
        return data.astype(np.float32, copy=False)

    return resample_poly(
        data,
        up=target_rate,
        down=int(round(source_rate)),
        axis=1
    ).astype(np.float32, copy=False)


# ═══════════════════════════════════════════════════════════════════════
# CHANNEL HANDLING
# ═══════════════════════════════════════════════════════════════════════

def pad_or_trim_channels(data, expected_channels):

    c = data.shape[0]

    if c == expected_channels:
        return data

    if c > expected_channels:
        return data[:expected_channels]

    padded = np.zeros(
        (expected_channels, data.shape[1]),
        dtype=data.dtype
    )

    padded[:c] = data

    return padded


def empty_modality_array(modality):

    ch = EXPECTED_CHANNELS[modality]
    sr = TARGET_SAMPLE_RATES[modality]

    return np.zeros(
        (ch, sr * WINDOW_SECONDS),
        dtype=np.float32
    )


# ═══════════════════════════════════════════════════════════════════════
# EDF LOADING
# ═══════════════════════════════════════════════════════════════════════

def open_raw(path):

    if not path:
        return None

    p = Path(path)

    if not p.exists():
        return None

    return mne.io.read_raw_edf(
        str(p),
        preload=False,
        verbose='ERROR'
    )


# ═══════════════════════════════════════════════════════════════════════
# WINDOW EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def extract_window(raw, modality, start_seconds):

    if raw is None:
        return empty_modality_array(modality)

    source_rate = float(raw.info['sfreq'])

    target_rate = TARGET_SAMPLE_RATES[modality]

    exp_ch = EXPECTED_CHANNELS[modality]

    exp_samples = target_rate * WINDOW_SECONDS

    start_idx = int(round(start_seconds * source_rate))

    stop_idx = start_idx + int(round(WINDOW_SECONDS * source_rate))

    data = raw.get_data(
        start=start_idx,
        stop=stop_idx
    )

    if data.shape[1] == 0:
        return empty_modality_array(modality)

    data = resample_if_needed(
        data,
        source_rate,
        target_rate
    )

    data = apply_filters(
        data,
        target_rate,
        modality
    )

    if data.shape[1] > exp_samples:
        data = data[:, :exp_samples]

    elif data.shape[1] < exp_samples:

        pad = np.zeros(
            (data.shape[0], exp_samples - data.shape[1]),
            dtype=np.float32
        )

        data = np.hstack([data, pad])

    data = pad_or_trim_channels(data, exp_ch)

    return normalize_window(data)

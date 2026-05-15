import torch
import numpy as np
from pathlib import Path

from models import (
    SingleSignalPredictor,
    MultiModalFusionNetwork,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent

CHECKPOINT_DIR = BASE_DIR / 'checkpoints'

EEG_CHECKPOINT = CHECKPOINT_DIR / 'eeg_best.pt'
FUSION_CHECKPOINT = CHECKPOINT_DIR / 'fusion_best.pt'

# ═══════════════════════════════════════════════════════════════════════
# THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════

EEG_THRESHOLD = 0.62
FUSION_THRESHOLD = 0.6184

# ═══════════════════════════════════════════════════════════════════════
# LOAD EEG MODEL
# ═══════════════════════════════════════════════════════════════════════

print('Loading EEG trigger model...')

eeg_model = SingleSignalPredictor(
    modality='eeg',
    kernel_size=13,
    base_filters=16,
    lstm_hidden=64,
    dropout=0.5,
).to(DEVICE)

eeg_model.load_state_dict(
    torch.load(EEG_CHECKPOINT, map_location=DEVICE)
)

eeg_model.eval()

print('EEG trigger model loaded.')

# ═══════════════════════════════════════════════════════════════════════
# LOAD FUSION MODEL
# ═══════════════════════════════════════════════════════════════════════

print('Loading multimodal fusion model...')

fusion_model = MultiModalFusionNetwork().to(DEVICE)

fusion_model.load_state_dict(
    torch.load(FUSION_CHECKPOINT, map_location=DEVICE)
)

fusion_model.eval()

print('Fusion model loaded.')

# ═══════════════════════════════════════════════════════════════════════
# EEG TRIGGER INFERENCE
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_trigger_inference(eeg_sequence):
    """
    eeg_sequence shape:
        (seq_len, channels, samples)

    Example:
        (8, 2, 3840)
    """

    x = torch.tensor(
        eeg_sequence,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    logits = eeg_model(x)

    probs = torch.softmax(logits, dim=1)

    preictal_prob = probs[0, 1].item()

    prediction = int(preictal_prob >= EEG_THRESHOLD)

    return {
        'probability': float(preictal_prob),
        'prediction': prediction,
        'threshold': EEG_THRESHOLD,
    }

# ═══════════════════════════════════════════════════════════════════════
# EEG EMBEDDING EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_eeg_embedding(eeg_sequence):
    """
    Returns:
        embedding shape = (128,)
    """

    x = torch.tensor(
        eeg_sequence,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    embedding, logits = eeg_model(
        x,
        return_embedding=True
    )

    embedding = embedding.squeeze(0)

    return embedding

# ═══════════════════════════════════════════════════════════════════════
# MULTIMODAL FUSION INFERENCE
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_fusion_inference(
    eeg_sequence,
    ecg_sequence,
    emg_sequence
):
    """
    Inputs:
        eeg_sequence : (8, 2, 3840)
        ecg_sequence : (8, 1, 3840)
        emg_sequence : (8, 1, 3840)
    """

    eeg_embedding = extract_eeg_embedding(
        eeg_sequence
    )

    ecg_tensor = torch.tensor(
        ecg_sequence,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    emg_tensor = torch.tensor(
        emg_sequence,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    eeg_embedding = eeg_embedding.unsqueeze(0)

    logits = fusion_model(
        eeg_embedding,
        ecg_tensor,
        emg_tensor
    )

    probability = torch.sigmoid(
        logits
    )[0].item()

    prediction = int(
        probability >= FUSION_THRESHOLD
    )

    return {
        'probability': float(probability),
        'prediction': prediction,
        'threshold': FUSION_THRESHOLD,
    }

# ═══════════════════════════════════════════════════════════════════════
# COMPLETE PIPELINE
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(
    eeg_sequence,
    ecg_sequence=None,
    emg_sequence=None
):
    """
    Full prediction pipeline.

    Step 1:
        EEG trigger model

    Step 2:
        If trigger positive →
        run multimodal fusion

    Returns:
        dictionary
    """

    trigger_result = run_trigger_inference(
        eeg_sequence
    )

    trigger_positive = (
        trigger_result['prediction'] == 1
    )

    if not trigger_positive:

        return {
            'triggered': False,
            'final_prediction': 0,
            'trigger_probability': trigger_result['probability'],
            'message': 'No seizure risk detected.',
        }

    if ecg_sequence is None or emg_sequence is None:

        return {
            'triggered': True,
            'final_prediction': 1,
            'trigger_probability': trigger_result['probability'],
            'message': 'EEG trigger activated but ECG/EMG missing.',
        }

    fusion_result = run_fusion_inference(
        eeg_sequence,
        ecg_sequence,
        emg_sequence
    )

    return {
        'triggered': True,
        'trigger_probability': trigger_result['probability'],
        'fusion_probability': fusion_result['probability'],
        'final_prediction': fusion_result['prediction'],
        'message': (
            'High seizure risk detected.'
            if fusion_result['prediction'] == 1
            else 'Fusion model rejected seizure risk.'
        )
    }

# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def load_models():
    return {
        'eeg_model': eeg_model,
        'fusion_model': fusion_model,
    }


def predict_seizure(
    eeg_sequence,
    ecg_sequence=None,
    emg_sequence=None
):
    return predict(
        eeg_sequence,
        ecg_sequence,
        emg_sequence
    )
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

import numpy as np

from inference import predict

app = FastAPI(
    title='SeizeIT2 API',
    version='1.0.0'
)

# ═══════════════════════════════════════════════════════════════════════
# REQUEST FORMAT
# ═══════════════════════════════════════════════════════════════════════

class PredictionRequest(BaseModel):

    eeg_sequence: List

    ecg_sequence: Optional[List] = None

    emg_sequence: Optional[List] = None

# ═══════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════

@app.get('/')
def home():

    return {
        'message': 'SeizeIT2 API is running.'
    }

# ═══════════════════════════════════════════════════════════════════════
# PREDICTION ENDPOINT
# ═══════════════════════════════════════════════════════════════════════

@app.post('/predict')
def predict_endpoint(request: PredictionRequest):

    eeg_sequence = np.array(
        request.eeg_sequence,
        dtype=np.float32
    )

    ecg_sequence = None
    emg_sequence = None

    if request.ecg_sequence is not None:

        ecg_sequence = np.array(
            request.ecg_sequence,
            dtype=np.float32
        )

    if request.emg_sequence is not None:

        emg_sequence = np.array(
            request.emg_sequence,
            dtype=np.float32
        )

    result = predict(
        eeg_sequence=eeg_sequence,
        ecg_sequence=ecg_sequence,
        emg_sequence=emg_sequence,
    )

    return result
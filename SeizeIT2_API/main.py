from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from interface import load_models, predict_seizure

app = FastAPI(title="SeizeIT2 API")

print("Loading models...")
models = load_models()
print("Models loaded successfully.")


class PredictionRequest(BaseModel):
    eeg: list
    ecg: list
    emg: list


@app.get("/")
def root():
    return {
        "message": "SeizeIT2 API is running"
    }


@app.post("/predict")
def predict(req: PredictionRequest):

    eeg = np.array(req.eeg, dtype=np.float32)
    ecg = np.array(req.ecg, dtype=np.float32)
    emg = np.array(req.emg, dtype=np.float32)

    print("EEG shape:", eeg.shape)
    print("ECG shape:", ecg.shape)
    print("EMG shape:", emg.shape)

    result = predict_seizure(
        eeg_sequence=eeg,
        ecg_sequence=ecg,
        emg_sequence=emg,
    )

    return result
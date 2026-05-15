import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

EXPECTED_CHANNELS = {
    'eeg': 2,
    'ecg': 1,
    'emg': 1,
}

ECG_KERNEL_SIZE   = 7
EMG_KERNEL_SIZE   = 7

ECG_BASE_FILTERS  = 16
EMG_BASE_FILTERS  = 16

ECG_LSTM_HIDDEN   = 32
EMG_LSTM_HIDDEN   = 32

FUSION_HIDDEN_DIM = 128
FUSION_DROPOUT    = 0.4


# ═══════════════════════════════════════════════════════════════════════════
# TRIGGER MODEL (PHASE 1)
# ═══════════════════════════════════════════════════════════════════════════

class ResBlock1D(nn.Module):
    def __init__(self, channels, dropout=0.35):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
        )

        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class ConvEncoder1D(nn.Module):

    def __init__(self,
                 in_channels,
                 base_filters=16,
                 kernel_size=7,
                 dropout=0.35):

        super().__init__()

        f   = base_filters
        pad = kernel_size // 2

        self.net = nn.Sequential(

            nn.Conv1d(
                in_channels,
                f,
                kernel_size,
                stride=2,
                padding=pad
            ),

            nn.BatchNorm1d(f),
            nn.GELU(),

            nn.MaxPool1d(
                3,
                stride=2,
                padding=1
            ),

            nn.Conv1d(
                f,
                f * 2,
                5,
                stride=2,
                padding=2
            ),

            nn.BatchNorm1d(f * 2),
            nn.GELU(),

            ResBlock1D(f * 2, dropout=dropout),
            ResBlock1D(f * 2, dropout=dropout),

            nn.Dropout(dropout),
        )

        self.pool = nn.AdaptiveAvgPool1d(4)

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x)
        return x.flatten(start_dim=1)


class SingleSignalPredictor(nn.Module):

    def __init__(self,
                 modality,
                 kernel_size=7,
                 base_filters=16,
                 lstm_hidden=64,
                 dropout=0.35):

        super().__init__()

        in_ch = EXPECTED_CHANNELS[modality]

        self.encoder = ConvEncoder1D(
            in_channels=in_ch,
            base_filters=base_filters,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        enc_dim = base_filters * 2 * 4

        self.lstm = nn.LSTM(
            input_size=enc_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.attn = nn.Linear(lstm_hidden * 2, 1)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(lstm_hidden * 2, 2)

    def forward(self, x, return_embedding=False):

        # x shape:
        # (batch, seq_len, channels, samples)

        B, T, C, S = x.shape

        x_flat = x.view(B * T, C, S)

        enc = self.encoder(x_flat)

        enc = enc.view(B, T, -1)

        out, _ = self.lstm(enc)

        attn_w = torch.softmax(
            self.attn(out),
            dim=1
        )

        pooled = (out * attn_w).sum(dim=1)

        logits = self.fc(
            self.dropout(pooled)
        )

        if return_embedding:
            return pooled, logits

        return logits


# ═══════════════════════════════════════════════════════════════════════════
# PHYSIO ENCODER (ECG / EMG)
# ═══════════════════════════════════════════════════════════════════════════

class PhysioEncoder(nn.Module):

    def __init__(self,
                 in_channels=1,
                 kernel_size=7,
                 base_filters=16,
                 lstm_hidden=32,
                 dropout=0.4):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv1d(
                in_channels,
                base_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),

            nn.BatchNorm1d(base_filters),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(
                base_filters,
                base_filters * 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),

            nn.BatchNorm1d(base_filters * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.AdaptiveAvgPool1d(32),
        )

        cnn_out_dim = base_filters * 2 * 32

        self.window_proj = nn.Linear(
            cnn_out_dim,
            lstm_hidden * 2
        )

        self.bilstm = nn.LSTM(
            input_size=lstm_hidden * 2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.out_dim = lstm_hidden * 2

    def forward(self, x):

        # x:
        # (batch, seq_len, channels, time)

        B, S, C, T = x.shape

        x = x.view(B * S, C, T)

        x = self.cnn(x)

        x = x.view(B * S, -1)

        x = self.window_proj(x)

        x = x.view(B, S, -1)

        out, _ = self.bilstm(x)

        out = self.dropout(out)

        mean_pool = out.mean(dim=1)

        max_pool = out.max(dim=1).values

        return mean_pool + max_pool


# ═══════════════════════════════════════════════════════════════════════════
# MULTIMODAL FUSION NETWORK (PHASE 2)
# ═══════════════════════════════════════════════════════════════════════════

class MultiModalFusionNetwork(nn.Module):

    def __init__(self,
                 eeg_dim=128,
                 ecg_kernel=ECG_KERNEL_SIZE,
                 emg_kernel=EMG_KERNEL_SIZE,
                 ecg_filters=ECG_BASE_FILTERS,
                 emg_filters=EMG_BASE_FILTERS,
                 ecg_lstm=ECG_LSTM_HIDDEN,
                 emg_lstm=EMG_LSTM_HIDDEN,
                 fusion_hidden=FUSION_HIDDEN_DIM,
                 dropout=FUSION_DROPOUT):

        super().__init__()

        self.ecg_encoder = PhysioEncoder(
            in_channels=1,
            kernel_size=ecg_kernel,
            base_filters=ecg_filters,
            lstm_hidden=ecg_lstm,
            dropout=dropout,
        )

        self.emg_encoder = PhysioEncoder(
            in_channels=1,
            kernel_size=emg_kernel,
            base_filters=emg_filters,
            lstm_hidden=emg_lstm,
            dropout=dropout,
        )

        fusion_in_dim = (
            eeg_dim
            + self.ecg_encoder.out_dim
            + self.emg_encoder.out_dim
        )

        self.fusion_head = nn.Sequential(

            nn.LayerNorm(fusion_in_dim),

            nn.Linear(
                fusion_in_dim,
                fusion_hidden
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                fusion_hidden,
                fusion_hidden // 2
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                fusion_hidden // 2,
                1
            ),
        )

    def forward(self, eeg_emb, ecg_seq, emg_seq):

        ecg_feat = self.ecg_encoder(ecg_seq)

        emg_feat = self.emg_encoder(emg_seq)

        fused = torch.cat(
            [eeg_emb, ecg_feat, emg_feat],
            dim=-1
        )

        logits = self.fusion_head(fused).squeeze(-1)

        return logits

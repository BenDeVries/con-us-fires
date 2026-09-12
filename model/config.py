"""Hyperparameters and paths for the spatio-temporal wildfire model."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

DATA_DIR = Path("output/data")
CKPT_DIR = Path("output/model")


@dataclass
class Config:
    # data / windowing
    lookback: int = 36          # encoder history length (months)
    horizon: int = 12           # decoder forecast length (months) = 1 year ahead
    # Base predictors excluded from the panel. Default = the 5 lowest-value predictors
    # per xgb TreeSHAP (model/xgb/importance.py): base ranks 39-50 of 51 in both hurdle
    # heads, stable across seeds and splits (Spearman >= 0.97).
    # Ignored when n_pca is set (all features enter PCA).
    drop_features: list[str] = field(default_factory=lambda: [
        "built_frac", "days_vpd_p90", "evc_mean", "pop_density", "spi1y"])
    n_pca: int | None = None    # None = raw features; int = number of PCs to keep
    n_harmonics: int = 1        # Fourier pairs k=1..n_harmonics for seasonal encoding
    # Hold data.STATIC_COLS (the 12 near-constant predictors) out of the PCA rotation and feed
    # them raw, tail-transformed. Defaults off so every pre-existing checkpoint rebuilds the
    # panel it was trained on; new studies opt in with --static-bypass.
    static_bypass: bool = False
    # Strict forecast inputs: fixed terrain, calendar, county identity and past fire only.
    # Persisted in every checkpoint so prediction cannot rebuild a conditional panel.
    forecast_safe: bool = False

    # graph conv
    gcn_hidden: int = 64
    gcn_layers: int = 2
    gcn_dropout: float = 0.1
    lc_embed_dim: int = 8       # embedding dim for lc_dominant categorical
    # Learned per-county vector concatenated into the GCN input — the model's only carrier of
    # county identity. Entering before A-hat means identity is smoothed over queen neighbours
    # and can interact with covariates and season through the LSTM. 0 builds no module at all,
    # so the state_dict is unchanged and pre-existing checkpoints still load under strict=True.
    county_embed_dim: int = 0

    # temporal
    lstm_hidden: int = 128
    lstm_layers: int = 1
    autoregressive_decoder: bool = True   # feed prev predicted E[y] into next decoder step

    # teacher forcing (training). off -> current free-running (detached mean) rollout.
    teacher_forcing: bool = False
    tf_ratio_start: float = 1.0          # scheduled-sampling ratio at epoch 1
    tf_ratio_end: float = 0.0            # ratio at tf_anneal_epochs (linear anneal)
    tf_anneal_epochs: int = 0            # epochs to anneal over; 0 -> use max_epochs

    # ancestral Monte-Carlo sampling (inference)
    n_samples: int = 500                 # ZIB sample trajectories per origin
    sample_chunk: int = 50               # trajectories processed per chunk (memory bound)
    sample_seed: int = 0                 # generator seed for reproducible rollouts

    # heads / loss
    head_hidden: int = 0        # 0 -> bare Linear heads; >0 -> one-hidden-layer MLP of this width
    phi_min: float = 1.0        # floor on Beta precision for stability
    eps: float = 1e-6           # clamp on the Beta MEAN head (mu), not on the response
    link: str = "logit"         # "logit" | "cloglog" — link for the gate and Beta-mean heads

    # optimisation
    batch_size: int = 4         # number of forecast origins per batch (each spans all 3108 nodes)
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 200
    grad_clip: float = 5.0
    patience: int = 20          # early-stop patience on test-split NLL

    seed: int = 0
    device: str = "cuda"        # falls back to cpu in train.py if unavailable

    def validate_forecast_safe(self):
        if self.forecast_safe and (self.n_pca is not None or self.static_bypass
                                   or self.teacher_forcing or self.lc_embed_dim != 0):
            raise ValueError("forecast_safe requires no PCA/static bypass, no teacher "
                             "forcing, and lc_embed_dim=0")

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

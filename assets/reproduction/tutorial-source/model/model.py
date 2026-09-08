"""GCN -> seq2seq LSTM spatio-temporal model with zero-inflated Beta heads.

Per month, a graph conv net mixes information across queen-adjacent counties to
produce node embeddings; an LSTM encodes 36 months of these embeddings per county;
an LSTM decoder rolls out 12 future months conditioned on known future covariates
(optionally autoregressive on its own predicted expected burned fraction). Three
heads emit the zero-inflated Beta parameters (pi gate logit, mu, phi) per
county x horizon-month.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .zib import gate_fire_prob, sample_zib


def _mu_link(mu_lin: torch.Tensor, link: str, eps: float) -> torch.Tensor:
    """Map the Beta-mean head's linear output to (0,1) under the chosen link."""
    if link == "cloglog":
        mu = -torch.expm1(-torch.exp(mu_lin.clamp(max=30.0)))   # 1 - exp(-exp(eta))
    else:
        mu = torch.sigmoid(mu_lin)
    return mu.clamp(eps, 1 - eps)


def build_norm_adj(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Symmetric-normalized adjacency with self-loops D^-1/2 (A+I) D^-1/2, as sparse COO."""
    A = torch.zeros(n_nodes, n_nodes)
    src, dst = edge_index
    A[src, dst] = 1.0
    A = ((A + A.t()) > 0).float()          # force symmetry (queen graph is symmetric)
    A.fill_diagonal_(1.0)                   # self-loops
    deg = A.sum(1).clamp(min=1.0)
    dinv = deg.pow(-0.5)
    norm = dinv.unsqueeze(1) * A * dinv.unsqueeze(0)
    return norm.to_sparse_coo().coalesce()


class GCN(nn.Module):
    """Stacked graph convolution over a fixed sparse normalized adjacency.

    Aggregation A @ x is done with sparse mm over all (batch*time) slices at once by
    packing them into the column dimension -- memory scales with edges, not N^2.
    """

    def __init__(self, in_dim: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers
        self.lins = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: [..., N, F] ; A: sparse [N, N]
        lead = x.shape[:-2]
        N, Fin = x.shape[-2], x.shape[-1]
        h = x.reshape(-1, N, Fin)                      # [M, N, F]
        M = h.shape[0]
        for i, lin in enumerate(self.lins):
            Fprev = h.shape[-1]
            packed = h.permute(1, 0, 2).reshape(N, M * Fprev)   # [N, M*F]
            agg = torch.sparse.mm(A, packed)                    # neighbor aggregation
            agg = agg.reshape(N, M, Fprev).permute(1, 0, 2)     # [M, N, F]
            h = lin(agg)
            if i < len(self.lins) - 1:
                h = F.relu(h)
                h = F.dropout(h, self.dropout, self.training)
        return h.reshape(*lead, N, h.shape[-1])


def _make_head(in_dim: int, hidden: int) -> nn.Module:
    """Output head. hidden == 0 keeps the bare Linear, so its state_dict keys
    (`head_*.weight` / `head_*.bias`) stay loadable from pre-MLP checkpoints."""
    if hidden <= 0:
        return nn.Linear(in_dim, 1)
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))


class SpatioTemporalZIB(nn.Module):
    def __init__(self, cfg: Config, n_cov: int, n_lc_classes: int,
                 n_nodes: int | None = None):
        super().__init__()
        self.cfg = cfg
        # A zero-width CUDA embedding can fail in its backward kernel. Build no module
        # when land cover is excluded; positive widths keep all legacy checkpoint keys.
        self.lc_embed = (nn.Embedding(n_lc_classes, cfg.lc_embed_dim)
                         if cfg.lc_embed_dim > 0 else None)

        if cfg.county_embed_dim > 0 and n_nodes is None:
            raise ValueError("county_embed_dim > 0 requires n_nodes")
        self.county_embed = (nn.Embedding(n_nodes, cfg.county_embed_dim)
                             if cfg.county_embed_dim > 0 else None)
        if self.county_embed is not None:
            # persistent=False keeps the index out of the state_dict, so checkpoint
            # compatibility depends only on county_embed_dim.
            self.register_buffer("node_ids", torch.arange(n_nodes), persistent=False)
        # Read the width back off the built module so the two can never disagree.
        d_county = self.county_embed.embedding_dim if self.county_embed is not None else 0

        enc_in = n_cov + cfg.lc_embed_dim + d_county + 2            # + autoregressive [burn, occ]
        dec_in = n_cov + cfg.lc_embed_dim + d_county + (1 if cfg.autoregressive_decoder else 0)
        self.enc_gcn = GCN(enc_in, cfg.gcn_hidden, cfg.gcn_layers, cfg.gcn_dropout)
        self.dec_gcn = GCN(dec_in, cfg.gcn_hidden, cfg.gcn_layers, cfg.gcn_dropout)

        self.enc_lstm = nn.LSTM(cfg.gcn_hidden, cfg.lstm_hidden,
                                cfg.lstm_layers, batch_first=True)
        self.dec_lstm = nn.LSTM(cfg.gcn_hidden, cfg.lstm_hidden,
                                cfg.lstm_layers, batch_first=True)

        self.head_pi = _make_head(cfg.lstm_hidden, cfg.head_hidden)
        self.head_mu = _make_head(cfg.lstm_hidden, cfg.head_hidden)
        self.head_phi = _make_head(cfg.lstm_hidden, cfg.head_hidden)

    def _encode(self, batch: dict, A: torch.Tensor):
        """Run the encoder once. Returns the decoder init state, AR seed, decoder covariate
        tensors, and the (B, N, H) shape. Independent of any rollout feedback policy, so a
        single encode can seed many sampled trajectories."""
        cfg = self.cfg
        enc_cov, enc_cat, enc_ar = batch["enc_cov"], batch["enc_cat"], batch["enc_ar"]
        dec_cov, dec_cat = batch["dec_cov"], batch["dec_cat"]
        B, L, N, _ = enc_cov.shape
        H = dec_cov.shape[1]

        # Time-invariant county embedding, broadcast over batch and time. It enters *before*
        # the graph conv, so A-hat smooths each county's identity over its queen neighbours
        # and the LSTM can modulate it by covariates and season.
        ce = None if self.county_embed is None else self.county_embed(self.node_ids)  # [N,d]

        # ---- encoder: GCN per month, then LSTM over time per node ----
        enc_parts = [enc_cov]
        if self.lc_embed is not None:
            enc_parts.append(self.lc_embed(enc_cat))
        if ce is not None:
            enc_parts.append(ce.expand(B, L, -1, -1))
        enc_parts.append(enc_ar)
        enc_feat = torch.cat(enc_parts, dim=-1)                                   # [B,L,N,*]
        enc_emb = self.enc_gcn(enc_feat, A)                                       # [B,L,N,Hs]
        enc_seq = enc_emb.permute(0, 2, 1, 3).reshape(B * N, L, cfg.gcn_hidden)
        _, (h_n, c_n) = self.enc_lstm(enc_seq)                                    # state inits decoder

        # last observed burned fraction seeds the autoregressive decoder
        prev_ey = enc_ar[:, -1, :, 0].reshape(B * N, 1)                           # [B*N,1]
        # Folded into the land-cover embedding tensor so _decode and sample() -- whose
        # exp_dec already expands this buffer per trajectory -- need no changes.
        dec_emb_lc = (self.lc_embed(dec_cat) if self.lc_embed is not None else
                      dec_cov.new_empty(B, H, N, 0))                            # [B,H,N,E]
        if ce is not None:
            dec_emb_lc = torch.cat([dec_emb_lc, ce.expand(B, H, -1, -1)], dim=-1)
        return (h_n, c_n), prev_ey, dec_cov, dec_emb_lc, B, N, H

    def _decode(self, state, prev_ey, dec_cov, dec_emb_lc, A, B, N, H, feedback):
        """Roll out H decoder steps. `feedback(pi_l, mu, phi, t) -> next prev_ey [B*N,1]`
        supplies the autoregressive input for step t+1 (mean, teacher-forced, or sampled).
        B is the *effective* batch (windows for training; windows*samples for sampling)."""
        cfg = self.cfg
        pis, mus, phis = [], [], []
        for t in range(H):
            feats = [dec_cov[:, t], dec_emb_lc[:, t]]                             # [B,N,*]
            if cfg.autoregressive_decoder:
                feats.append(prev_ey.reshape(B, N, 1))
            node_feat = torch.cat(feats, dim=-1)                                  # [B,N,Fin]
            emb = self.dec_gcn(node_feat, A).reshape(B * N, 1, cfg.gcn_hidden)
            out, state = self.dec_lstm(emb, state)                               # [B*N,1,Hl]
            o = out[:, 0]
            pi_l = self.head_pi(o)
            mu = _mu_link(self.head_mu(o), cfg.link, cfg.eps)
            phi = cfg.phi_min + F.softplus(self.head_phi(o))
            pis.append(pi_l); mus.append(mu); phis.append(phi)
            prev_ey = feedback(pi_l, mu, phi, t)

        def stack(parts):
            # parts: H tensors [B*N, 1] with row m = b*N + n -> buffer is (B, N, H).
            return torch.stack(parts, dim=1).reshape(B, N, H).transpose(1, 2).contiguous()
        return {"pi_logit": stack(pis), "mu": stack(mus), "phi": stack(phis)}

    def forward(self, batch: dict, A: torch.Tensor, tf_ratio: float = 0.0) -> dict:
        """Deterministic (mean-path) rollout, or teacher-forced when training.

        tf_ratio > 0 (training only) mixes the true previous target into the AR feedback via
        a per-element Bernoulli(tf_ratio) mask; the complement feeds the detached mean E[y].
        Default tf_ratio=0 with eval mode reproduces the free-running detached-mean rollout."""
        cfg = self.cfg
        state, prev_ey, dec_cov, dec_emb_lc, B, N, H = self._encode(batch, A)

        def mean_feedback(pi_l, mu, phi, t):
            return (gate_fire_prob(pi_l, cfg.link) * mu).detach()  # E[y]=(1-pi)*mu

        if self.training and tf_ratio > 0.0:
            y_true = batch["y"]                                    # [B,H,N] true horizon targets
            def feedback(pi_l, mu, phi, t):
                mean = mean_feedback(pi_l, mu, phi, t)
                true_prev = y_true[:, t].reshape(B * N, 1)         # y_{o+1+t}, feeds step t+1
                mask = (torch.rand_like(mean) < tf_ratio).to(mean.dtype)
                return mask * true_prev + (1.0 - mask) * mean
        else:
            feedback = mean_feedback

        return self._decode(state, prev_ey, dec_cov, dec_emb_lc, A, B, N, H, feedback)

    @torch.no_grad()
    def sample(self, batch: dict, A: torch.Tensor, n_samples: int,
               chunk: int | None = None) -> torch.Tensor:
        """Ancestral Monte-Carlo rollout: draw y ~ ZIB(pi,mu,phi) at each step and feed the
        sample forward. The ensemble over trajectories is the origin-conditioned predictive
        p(y_{o+1:o+H} | F_o). Encoder runs once; samples are processed in chunks (each
        multiplies only the current decoder step, not the graph). Returns [B, S, H, N] on CPU.

        Reproducibility relies on the global torch RNG (see sample_zib) — seed once upstream."""
        cfg = self.cfg
        state, prev_ey, dec_cov, dec_emb_lc, B, N, H = self._encode(batch, A)
        (h_n, c_n) = state
        layers = h_n.shape[0]
        chunk = chunk or n_samples

        def exp_state(x, s):
            # [layers, B*N, Hl] (row b*N+n) -> [layers, B*s*N, Hl] (row (b*s+j)*N+n)
            Hl = x.shape[-1]
            return (x.reshape(layers, B, N, Hl).unsqueeze(2)
                    .expand(layers, B, s, N, Hl).reshape(layers, B * s * N, Hl).contiguous())

        def exp_bn1(x, s):
            # [B*N,1] -> [B*s*N,1]
            return (x.reshape(B, N, 1).unsqueeze(1)
                    .expand(B, s, N, 1).reshape(B * s * N, 1).contiguous())

        def exp_dec(x, s):
            # [B,H,N,F] -> [B*s,H,N,F]
            Bx, Hx, Nx, Fx = x.shape
            return (x.unsqueeze(1).expand(Bx, s, Hx, Nx, Fx)
                    .reshape(Bx * s, Hx, Nx, Fx).contiguous())

        outs, done = [], 0
        while done < n_samples:
            s = min(chunk, n_samples - done)
            drawn: list[torch.Tensor] = []

            def sample_feedback(pi_l, mu, phi, t):
                y = sample_zib(pi_l, mu, phi, cfg.link, cfg.eps)   # [B*s*N,1]
                drawn.append(y)
                return y

            st = (exp_state(h_n, s), exp_state(c_n, s))
            self._decode(st, exp_bn1(prev_ey, s),
                         exp_dec(dec_cov, s), exp_dec(dec_emb_lc, s),
                         A, B * s, N, H, sample_feedback)
            # drawn: H tensors [B*s*N,1] with row (b*s+j)*N+n -> [B,s,H,N]
            Y = torch.stack(drawn, dim=1).reshape(B, s, N, H).permute(0, 1, 3, 2)
            outs.append(Y.cpu())
            done += s
        return torch.cat(outs, dim=1)                              # [B, S, H, N]

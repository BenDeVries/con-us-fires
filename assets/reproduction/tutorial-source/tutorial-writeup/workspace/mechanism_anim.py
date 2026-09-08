"""Schematic animation of the GCN -> seq2seq LSTM mechanism.

No fitted model and no panel data: the point is the *shape* of the computation, so the
node field is synthetic. The one thing that is not synthetic is the propagation operator
-- `build_norm_adj` is imported from `model.model`, so the neighbourhood mixing shown on
the left panel is the same D^-1/2 (A+I) D^-1/2 the trained model uses.

Writes assets/fig/gcn_lstm_mechanism.gif. The last frame also goes to the workspace as a
PNG for review; it stays out of final-product, which carries only referenced artifacts.

    conda run -n pytorch python tutorial-writeup/workspace/mechanism_anim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model" / "config.py").exists())
sys.path.insert(0, str(ROOT))
from model.model import build_norm_adj  # noqa: E402

OUT = ROOT / "tutorial-writeup" / "final-product" / "assets" / "fig"
REVIEW = Path(__file__).resolve().parent / "preview"

DEVRIES = {
    "observed_data": "#006400",
    "model_fit":     "#FFA500",
    "highlight":     "#FF00FF",
    "secondary":     "#CD950C",
}
CMAP = plt.get_cmap("YlOrRd")

L_SHOW, H_SHOW = 4, 3          # encoder / decoder slots drawn; the real model uses 48 and 12
SUB = 3                        # sub-frames per month: features, aggregate, embed
N_HID = 4                      # cells drawn for the hidden embedding z_t


# --------------------------------------------------------------------------- graph
def hex_lattice(ncol: int = 5, nrow: int = 5, seed: int = 3):
    """Jittered offset lattice. Interior nodes get 6 neighbours, matching the queen county
    graph's median degree (18,448 directed edges over 3,108 counties)."""
    rng = np.random.default_rng(seed)
    pts = [(c + 0.5 * (r % 2) + rng.normal(0, 0.05), r * 0.87 + rng.normal(0, 0.05))
           for r in range(nrow) for c in range(ncol)]
    P = np.asarray(pts)
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    A = ((D > 0) & (D < 1.08)).astype(int)
    return P, A


P, A = hex_lattice()
N = len(P)
EDGES = [(i, j) for i in range(N) for j in range(i + 1, N) if A[i, j]]
AHAT = build_norm_adj(torch.tensor(np.array(np.nonzero(A))), N).to_dense().numpy()

FOCUS = 12                                          # an interior node, degree 6
HOP1 = sorted(np.flatnonzero(A[FOCUS]))
HOP2 = sorted(set(np.flatnonzero(A[HOP1].sum(0))) - set(HOP1) - {FOCUS})

# synthetic node field, AR(1) in time so consecutive months look related rather than
# independently reshuffled -- the animation is about persistence as much as mixing
_rng = np.random.default_rng(11)
_x = _rng.random(N)
FIELD = []
for _ in range(L_SHOW + H_SHOW):
    _x = 0.72 * _x + 0.28 * _rng.random(N)
    FIELD.append(_x.copy())
FIELD = np.asarray(FIELD)
FIELD = (FIELD - FIELD.min()) / np.ptp(FIELD)
MIXED = FIELD @ AHAT.T

# a fixed random read-out of the focus node's aggregated patch: this is what one GCN layer
# computes at that node, so the drawn embedding varies with the neighbourhood, not at random
_W = np.random.default_rng(5).normal(0, 1.0, size=(N_HID, len(HOP1) + 1))
PATCH = MIXED[:, [FOCUS] + HOP1]
ZVEC = 1 / (1 + np.exp(-(PATCH @ _W.T)))

# schematic head outputs; the gate tracks the aggregated field so the three bars move together
_pi = 1 - MIXED[L_SHOW:, FOCUS]
HEADS = np.stack([_pi, 0.25 + 0.5 * MIXED[L_SHOW:, FOCUS], 0.35 + 0.45 * _pi], axis=1)


# ------------------------------------------------------------------- strip geometry
X0, X1 = 0.17, 0.985
SLOT_W = (X1 - X0) / (L_SHOW + H_SHOW)
SLOT = [X0 + (i + 0.5) * SLOT_W for i in range(L_SHOW + H_SHOW)]
ORIGIN_X = X0 + L_SHOW * SLOT_W
ROW = {"in": 0.84, "gcn": 0.645, "lstm": 0.44, "head": 0.235}
BW = SLOT_W * 0.62
HBW = BW / 4.2                                      # width of one head bar
SEGS = [(DEVRIES["observed_data"], "terrain + calendar"),
        (DEVRIES["secondary"], "county identity"),
        (DEVRIES["highlight"], "fire state / fed-back $E[y]$")]


def _box(ax, cx, cy, w, h, *, ec, fc="white", lw=1.1, r=0.02, z=3):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def _arrow(ax, p0, p1, *, color, lw=1.1, style="-|>", ls="-", z=2, rad=0.0, alpha=1.0,
           scale=9, shrink=(0.0, 0.0)):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=scale,
                                 color=color, lw=lw, ls=ls, zorder=z, alpha=alpha,
                                 shrinkA=shrink[0], shrinkB=shrink[1],
                                 connectionstyle=f"arc3,rad={rad}"))


def draw_graph(ax, step: int, sub: int, decoder: bool):
    """Left panel: one month of the county graph, and what the convolution does to it."""
    ax.clear()
    ax.set_axis_off()
    ax.set_xlim(P[:, 0].min() - 0.6, P[:, 0].max() + 0.6)
    ax.set_ylim(P[:, 1].min() - 0.75, P[:, 1].max() + 0.62)
    ax.set_aspect("equal")

    m = step if not decoder else L_SHOW + step
    vals = MIXED[m] if sub == 2 else FIELD[m]

    for i, j in EDGES:
        ax.plot(*zip(P[i], P[j]), color="0.78", lw=0.8, zorder=1)

    if sub == 1:                                    # the aggregation step itself
        # shrink in points, so the head lands on the circle's edge rather than under it
        for j in HOP1:
            _arrow(ax, P[j], P[FOCUS], color=DEVRIES["observed_data"], lw=1.6, z=6,
                   scale=13, shrink=(7.0, 9.0))
        for j in HOP2:
            ax.add_patch(Circle(P[j], 0.225, fc="none", ec=DEVRIES["observed_data"],
                                lw=0.8, ls=":", alpha=0.65, zorder=4))

    for i in range(N):
        edge = DEVRIES["highlight"] if i == FOCUS else "0.35"
        ax.add_patch(Circle(P[i], 0.155, fc=CMAP(vals[i]), ec=edge,
                            lw=2.0 if i == FOCUS else 0.7, zorder=5))

    cap = {0: "county features arrive",
           1: r"mix over neighbours:  $\hat{A}=\tilde{D}^{-1/2}(A+I)\tilde{D}^{-1/2}$",
           2: "node embedding $z_{t,i}$"}[sub]
    ax.set_title(cap, fontsize=9.5, pad=6)
    note = ("dotted ring: reached by GCN layer 2" if sub == 1 else
            "ringed node: the county we follow")
    ax.text(0.5, -0.045, note, transform=ax.transAxes, ha="center", fontsize=8,
            color=DEVRIES["observed_data"] if sub == 1 else "0.45")


def draw_strip(ax, step: int, sub: int, decoder: bool):
    """Right panel: the whole sequence, filled in as far as the current step."""
    ax.clear()
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    cur = step if not decoder else L_SHOW + step
    ax.plot([ORIGIN_X] * 2, [0.03, 0.93], color="0.35", lw=1.0, ls=":", zorder=1)
    ax.text(ORIGIN_X, 0.955, "forecast origin $o$", fontsize=8.5, ha="center", color="0.25")
    ax.text((X0 + ORIGIN_X) / 2, 0.995, "encoder — observed history",
            fontsize=9, ha="center", color=DEVRIES["observed_data"])
    ax.text((ORIGIN_X + X1) / 2, 0.995, "decoder — 12-month horizon",
            fontsize=9, ha="center", color=DEVRIES["model_fit"])

    for key, lab in [("in", "inputs"), ("gcn", "GCN"), ("lstm", "LSTM"), ("head", "heads")]:
        ax.text(X0 - 0.025, ROW[key], lab, fontsize=9, ha="right", va="center")
    ax.text(X0 - 0.025, 0.925, "months $o-47 \\dots$", fontsize=7.5, ha="right",
            va="center", color="0.45")

    for k, (c, lab) in enumerate(SEGS):
        lx = X0 - 0.02 + k * 0.28
        ax.add_patch(Rectangle((lx, 0.022), 0.018, 0.030, fc=c, ec="white", lw=0.5))
        ax.text(lx + 0.026, 0.037, lab, fontsize=8, va="center")

    for s in range(L_SHOW + H_SHOW):
        dec = s >= L_SHOW
        col = DEVRIES["model_fit"] if dec else DEVRIES["observed_data"]
        cx = SLOT[s]
        live = s < cur or (s == cur and sub >= 0)
        on = s < cur
        lab = f"$o{s - L_SHOW + 1:+d}$" if dec else ("$o$" if s == L_SHOW - 1 else f"$o{s - L_SHOW + 1:+d}$")
        ax.text(cx, 0.925, lab, fontsize=8, ha="center", va="center", color=col)

        # ---- inputs: terrain/calendar | county identity | autoregressive state
        if live:
            w = BW / len(SEGS)
            for k, (c, _) in enumerate(SEGS):
                ax.add_patch(Rectangle((cx - BW / 2 + k * w, ROW["in"] - 0.035), w, 0.07,
                                       fc=c, ec="white", lw=0.6, zorder=3))
            _arrow(ax, (cx, ROW["in"] - 0.038), (cx, ROW["gcn"] + 0.035), color="0.5", lw=0.9)

        # ---- GCN box
        if live:
            hot = (s == cur and sub == 1)
            _box(ax, cx, ROW["gcn"], BW, 0.068,
                 ec=DEVRIES["highlight"] if hot else col, lw=1.8 if hot else 1.1,
                 fc="#fff0fb" if hot else "white")
            ax.text(cx, ROW["gcn"], r"$\hat{A}$", fontsize=9, ha="center", va="center")

        # ---- embedding vector then LSTM cell
        if on or (s == cur and sub == 2):
            zw = BW / N_HID
            for k in range(N_HID):
                ax.add_patch(Rectangle((cx - BW / 2 + k * zw, ROW["gcn"] - 0.088), zw, 0.032,
                                       fc=CMAP(ZVEC[s, k]), ec="white", lw=0.5, zorder=3))
            _arrow(ax, (cx, ROW["gcn"] - 0.09), (cx, ROW["lstm"] + 0.04), color="0.5", lw=0.9)
            hot = (s == cur and sub == 2)
            _box(ax, cx, ROW["lstm"], BW, 0.078,
                 ec=DEVRIES["highlight"] if hot else col, lw=1.8 if hot else 1.1,
                 fc="#fff0fb" if hot else "white")
            ax.text(cx, ROW["lstm"], "LSTM", fontsize=7.5, ha="center", va="center")
            if s > 0:
                prev = SLOT[s - 1]
                cross = s == L_SHOW
                _arrow(ax, (prev + BW / 2, ROW["lstm"]), (cx - BW / 2, ROW["lstm"]),
                       color=DEVRIES["highlight"] if cross else "0.45",
                       lw=1.8 if cross else 1.0)
                if cross:
                    ax.text((prev + cx) / 2, ROW["lstm"] + 0.045, "$(h_L, c_L)$",
                            fontsize=7.5, ha="center", color=DEVRIES["highlight"])

        # ---- heads, decoder only
        if dec and (on or (s == cur and sub == 2)):
            _arrow(ax, (cx, ROW["lstm"] - 0.04), (cx, ROW["head"] + 0.062), color="0.5", lw=0.9)
            vals = HEADS[s - L_SHOW]
            for k, (v, nm) in enumerate(zip(vals, [r"$\pi$", r"$\mu$", r"$\phi$"])):
                bx = cx - BW / 2 + (k + 0.6) * HBW * 1.35
                ax.add_patch(Rectangle((bx - HBW / 2, ROW["head"] - 0.03), HBW, 0.052 * v,
                                       fc=DEVRIES["model_fit"], ec="0.3", lw=0.5, zorder=3))
                if s == L_SHOW:
                    ax.text(bx, ROW["head"] - 0.052, nm, fontsize=8, ha="center", va="center")

        # ---- autoregressive feedback into the next decoder input. Routed out to the right
        # of the column and up through the inter-slot gap, so it crosses no box.
        if dec and s + 1 < L_SHOW + H_SHOW and (on or (s == cur and sub == 2)):
            gap = (SLOT[s] + SLOT[s + 1]) / 2
            top = ROW["in"] - 0.072
            pts = [(cx + BW * 0.30, ROW["head"] - 0.005), (gap, ROW["head"] - 0.005),
                   (gap, top), (SLOT[s + 1] + BW / 3, top)]
            for a, b in zip(pts, pts[1:]):
                _arrow(ax, a, b, color=DEVRIES["highlight"], lw=1.1, ls="--", style="-", z=6)
            _arrow(ax, pts[-1], (SLOT[s + 1] + BW / 3, ROW["in"] - 0.037),
                   color=DEVRIES["highlight"], lw=1.1, ls="--", z=6)
            if s == L_SHOW:
                ax.text(gap + 0.006, top - 0.015, r"$E[y]$ fed forward",
                        fontsize=7.5, color=DEVRIES["highlight"], rotation=90,
                        ha="left", va="top")


def main():
    fig = plt.figure(figsize=(12.0, 5.2))
    ax_g = fig.add_axes((0.012, 0.055, 0.275, 0.845))
    ax_s = fig.add_axes((0.320, 0.020, 0.672, 0.880))
    sup = fig.suptitle("", fontsize=12.5, y=0.975)

    plan = [(s, sub, False) for s in range(L_SHOW) for sub in range(SUB)] + \
           [(s, sub, True) for s in range(H_SHOW) for sub in range(SUB)]
    plan += [plan[-1]] * 3                                   # hold the final state

    def update(i):
        step, sub, dec = plan[i]
        draw_graph(ax_g, step, sub, dec)
        draw_strip(ax_s, step, sub, dec)
        phase = "Decoder" if dec else "Encoder"
        lab = f"$o{step + 1:+d}$" if dec else f"$o{step - L_SHOW + 1:+d}$"
        what = {0: "read terrain, calendar, county identity and observed fire history",
                1: "one graph convolution: every county mixes with its neighbours",
                2: "the LSTM consumes the embedding and advances its state"}[sub]
        if dec and sub == 0:
            what = "read terrain, calendar and the previous expected response"
        if dec and sub == 2:
            what = "the LSTM step emits the zero-augmented Beta triple $(\\pi,\\mu,\\phi)$"
        sup.set_text(f"{phase}, month {lab}  —  {what}")
        return []

    anim = FuncAnimation(fig, update, frames=len(plan), interval=1000, blit=False)
    OUT.mkdir(parents=True, exist_ok=True)
    anim.save(OUT / "gcn_lstm_mechanism.gif", writer=PillowWriter(fps=1), dpi=100)
    update(len(plan) - 1)
    REVIEW.mkdir(parents=True, exist_ok=True)
    fig.savefig(REVIEW / "gcn_lstm_mechanism.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / "gcn_lstm_mechanism.gif")


if __name__ == "__main__":
    main()

"""
architectures.py — two-stream and single-stream models.

Adaptation arms supported here:

  Run B         two-stream, physio FROZEN, treat+fusion trainable
  Run B-mirror  two-stream, treat+fusion FROZEN, physio trainable   [NEW]
  Run C         two-stream, everything trainable
  Run C-L2SP    Run C plus an L2 anchor to the source weights       [NEW]
  Run D         single-stream, LSTM FROZEN, head trainable

B-mirror and C-L2SP exist to close the two obvious reviewer objections to the
Run B result: "freezing any large block would have helped" and "Run B is just
regularization wearing an architecture costume". Without them, a B > C result
is not attributable to the decomposition.
"""
import torch
import torch.nn as nn

from utils.constants import HIDDEN_DIM, TREAT_DIM, LSTM_LAYERS, DROPOUT, LABEL_COLS

N_TARGETS_DEFAULT = len(LABEL_COLS)


# ── BUILDING BLOCKS ────────────────────────────────────────────────────────────
class PhysiologyStream(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, n_layers=LSTM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.norm(h[-1])


class TreatmentStream(nn.Module):
    def __init__(self, input_dim, output_dim=TREAT_DIM, dropout=DROPOUT):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, output_dim), nn.ReLU(), nn.LayerNorm(output_dim))

    def forward(self, x):
        return self.mlp(x)


class FusionHead(nn.Module):
    def __init__(self, physio_dim=HIDDEN_DIM, treat_dim=TREAT_DIM,
                 n_targets=N_TARGETS_DEFAULT, dropout=0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(physio_dim + treat_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout * 0.75),
            nn.Linear(32, n_targets))

    def forward(self, p, t):
        return self.head(torch.cat([p, t], dim=1))


# ── HELPERS ────────────────────────────────────────────────────────────────────
def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all(model) -> int:
    return sum(p.numel() for p in model.parameters())


class LateFusionModel(nn.Module):
    """
    Single encoder, LATE fusion. The control that separates "two streams help"
    from "don't push static features through an RNN".

    SingleStreamModel tiles the 61 static treatment features across all 6
    timesteps and feeds them into the LSTM (see SingleStreamDataset). That is a
    weak baseline by modern standards, and a reviewer will say so: much of any
    two-stream advantage could simply be avoiding that mistake, which is
    ordinary engineering rather than a contribution.

    This model fixes exactly that and nothing else — one LSTM over the
    sequential channels only, statics concatenated to the final hidden state
    before the head. It has NO separate treatment encoder, so:

        D_late ~= two-stream  -> the advantage was the static-feature handling
        two-stream > D_late   -> the second encoder earns its place
    """

    def __init__(self, seq_dim, treat_dim, n_targets=N_TARGETS_DEFAULT,
                 hidden_dim=HIDDEN_DIM, n_layers=LSTM_LAYERS, dropout=DROPOUT,
                 head_hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.lstm_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + treat_dim, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 32), nn.ReLU(), nn.Dropout(dropout * 0.75),
            nn.Linear(32, n_targets))

    def forward(self, x_seq, x_treat):
        _, (h, _) = self.lstm(x_seq)
        return self.head(torch.cat([self.lstm_norm(h[-1]), x_treat], dim=-1))

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def unfreeze_head_only(self):
        self.freeze_all()
        for p in self.head.parameters():
            p.requires_grad = True


def solve_late_head_hidden(target_params, in_dim, mid=32, n_targets=N_TARGETS_DEFAULT):
    """Head width H making the late-fusion head match `target_params`."""
    best, bh = None, 8
    for H in range(8, 2048):
        n = in_dim * H + H + H * mid + mid + mid * n_targets + n_targets
        d = abs(n - target_params)
        if best is None or d < best:
            best, bh = d, H
    return bh


def solve_head_hidden(target_params: int, lstm_hidden: int = HIDDEN_DIM,
                      mid: int = 32, n_targets: int = N_TARGETS_DEFAULT) -> int:
    """
    Width H of the first head layer such that a
    Linear(lstm_hidden, H) → Linear(H, mid) → Linear(mid, n_targets) stack has
    approximately `target_params` trainable parameters.

    Used to capacity-match Run D's adaptable head to Run B's adaptable
    (treat + fusion) block. Without this, Run D adapts 6,339 parameters against
    Run B's 12,259 — a 1.93x handicap that confounds "decomposition helps" with
    "more adaptable capacity helps".
    """
    # params(H) = (lstm_hidden*H + H) + (H*mid + mid) + (mid*n_targets + n_targets)
    #           = H*(lstm_hidden + 1 + mid) + (mid + mid*n_targets + n_targets)
    per_h = lstm_hidden + 1 + mid
    const = mid + mid * n_targets + n_targets
    return max(1, round((target_params - const) / per_h))


def l2sp_penalty(model, source_state, device, only_trainable=True) -> torch.Tensor:
    """
    L2-SP: squared distance from the source weights, summed over parameters.
    Anchors a full fine-tune to its starting point, which is the standard
    regularized-full-FT baseline. Run C-L2SP uses this.
    """
    pen = torch.zeros((), device=device)
    for name, p in model.named_parameters():
        if only_trainable and not p.requires_grad:
            continue
        ref = source_state.get(name)
        if ref is None:
            continue
        pen = pen + ((p - ref.to(p.device)) ** 2).sum()
    return pen


# ── TWO-STREAM ─────────────────────────────────────────────────────────────────
class TwoStreamModel(nn.Module):
    def __init__(self, seq_input_dim, treat_input_dim, n_targets=N_TARGETS_DEFAULT):
        super().__init__()
        self.physio = PhysiologyStream(seq_input_dim, HIDDEN_DIM, LSTM_LAYERS, DROPOUT)
        self.treat = TreatmentStream(treat_input_dim, TREAT_DIM, DROPOUT)
        self.fusion = FusionHead(HIDDEN_DIM, TREAT_DIM, n_targets, DROPOUT)

    def forward(self, x_seq, x_treat):
        return self.fusion(self.physio(x_seq), self.treat(x_treat))

    # ── freeze strategies ──
    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        """Run C — full fine-tune."""
        for p in self.parameters():
            p.requires_grad = True

    def freeze_physio(self):
        for p in self.physio.parameters():
            p.requires_grad = False

    def unfreeze_adaptive(self):
        """Run B — physio frozen, treat + fusion trainable."""
        self.freeze_all()
        for p in self.treat.parameters():
            p.requires_grad = True
        for p in self.fusion.parameters():
            p.requires_grad = True

    def unfreeze_mirror(self):
        """
        Run B-mirror — treat + fusion frozen, physio trainable.

        The direct control for Run B. If B beats its own mirror, the claim
        "the physiology stream is the stable component and the treatment stream
        is the one that should absorb drift" has evidence. If the mirror does
        equally well, all that was demonstrated is that freezing *something*
        helps, and the decomposition argument does not hold.
        """
        self.freeze_all()
        for p in self.physio.parameters():
            p.requires_grad = True


# ── SINGLE-STREAM (Run D) ──────────────────────────────────────────────────────
class SingleStreamModel(nn.Module):
    """
    Monolithic LSTM over all (seq + treatment) channels with a dense head.

    `head_hidden` is exposed so Run D's adaptable head can be capacity-matched
    to Run B's adaptable block; see solve_head_hidden(). Run both the matched
    and unmatched widths and report each.
    """

    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, n_layers=LSTM_LAYERS,
                 dropout=DROPOUT, n_targets=N_TARGETS_DEFAULT, head_hidden=64):
        super().__init__()
        self.head_hidden = head_hidden
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0)
        self.lstm_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 32), nn.ReLU(), nn.Dropout(dropout * 0.75),
            nn.Linear(32, n_targets))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(self.lstm_norm(h[-1]))

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def unfreeze_head_only(self):
        """Run D — LSTM frozen, head trainable."""
        self.freeze_all()
        for p in self.head.parameters():
            p.requires_grad = True

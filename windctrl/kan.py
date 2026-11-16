"""Kolmogorov-Arnold layers in the form used by the reference KAN-PPO IPC paper.

Every edge is a learnable 1-D function

    phi_ij(x) = w_ij * x + sum_{k=1..8} c_ijk * B_k(clip(x, -3, 3)),

with B_k the eight uniform cubic B-spline basis functions on five intervals over [-3, 3], and
a layer computes  z_i = b_i + sum_j phi_ij(s_j).  The linear term keeps the law defined for
rare inputs beyond the spline domain. A single layer is an additive model, so each edge can be
plotted and read directly as "how input j moves output i".
"""
import torch
import torch.nn as nn


class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, intervals=5, order=3, x_range=(-3.0, 3.0),
                 sigma_w=0.01, sigma_c=0.01, bias=0.0):
        super().__init__()
        self.in_features, self.out_features, self.order, self.x_range = in_features, out_features, order, x_range
        lo, hi = x_range
        h = (hi - lo) / intervals
        knots = torch.arange(-order, intervals + order + 1, dtype=torch.float32) * h + lo
        self.register_buffer("knots", knots)
        self.w = nn.Parameter(torch.randn(out_features, in_features) * sigma_w)
        self.c = nn.Parameter(torch.randn(out_features, in_features, intervals + order) * sigma_c)
        self.b = nn.Parameter(torch.full((out_features,), float(bias)))

    def basis(self, x):
        """Cubic B-spline basis at x (already clipped), shape (..., intervals + order)."""
        lo, hi = self.x_range
        x = x.clamp(lo, hi - 1e-6).unsqueeze(-1)   # the top knot belongs to the last interval
        t = self.knots
        b = ((x >= t[:-1]) & (x < t[1:])).to(x.dtype)
        for k in range(1, self.order + 1):
            b = (x - t[:-(k + 1)]) / (t[k:-1] - t[:-(k + 1)]) * b[..., :-1] \
                + (t[k + 1:] - x) / (t[k + 1:] - t[1:-k]) * b[..., 1:]
        return b

    def edges(self, x):
        """phi_ij(x_j) for every edge, shape (batch, out_features, in_features)."""
        spline = torch.einsum("bjk,ijk->bij", self.basis(x), self.c)
        return spline + x.unsqueeze(1) * self.w

    def forward(self, x):
        return self.edges(x).sum(-1) + self.b


class KAN(nn.Module):
    """Stack of KAN layers; `widths = [n_in, n_out]` is the single additive layer."""

    def __init__(self, widths, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(KANLayer(a, b, **layer_kwargs) for a, b in zip(widths[:-1], widths[1:]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

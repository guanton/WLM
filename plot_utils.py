import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import matplotlib.cm as cm
from typing import Any, Dict, Optional, List, Tuple

# ── tikz-style rc overrides (call plt.rcParams.update(TIKZ_RC) in notebook) ──
TIKZ_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "font.family": "serif",
    "font.size": 10,
    "mathtext.fontset": "cm",
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

C_BOID = "#5BA7D9"  # light blue — used for all boids scatter


# ── internal helpers ──

def _get_lims(X_np, pad_frac=0.08):
    """Axis limits from (N, T, 2) array."""
    pts = X_np.reshape(-1, 2)
    lo, hi = pts.min(0), pts.max(0)
    pad = pad_frac * np.maximum(1e-6, hi - lo)
    return (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])


def _tikz_scatter(ax, x, y, color=C_BOID, s=8, alpha=0.6, label=None):
    """Single scatter with tikz-clean styling."""
    ax.scatter(x, y, s=s, alpha=alpha, c=color, edgecolors="none",
               linewidths=0, rasterized=True, label=label)



def snapshot_grid(X_np, times, n_cols=5, title="", color=C_BOID):
    """
    Static grid of snapshots at evenly-spaced timesteps.

    Args:
        X_np: (N, T, 2) numpy array
        times: (T,) time values
        n_cols: number of snapshot panels
        title: optional suptitle
        color: scatter color (default: C_BOID light blue)
    """
    N, T, _ = X_np.shape
    idxs = np.linspace(0, T - 1, n_cols, dtype=int)
    xlim, ylim = _get_lims(X_np)

    fig, axes = plt.subplots(1, n_cols, figsize=(2.8 * n_cols, 2.8),
                              sharex=True, sharey=True)
    for i, t_idx in enumerate(idxs):
        ax = axes[i]
        _tikz_scatter(ax, X_np[:, t_idx, 0], X_np[:, t_idx, 1], color=color)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        t_val = float(times[t_idx]) if times is not None else t_idx
        ax.set_title(f"$t={t_val:.1f}$", fontsize=9)
        if i > 0:
            ax.tick_params(labelleft=False)

    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


def make_gif(X_dict, times, save_path="boids.gif",
             frame_skip=2, fps=10, subsample=None, color=None,
             train_time=None):
    """
    Animated GIF with tikz-clean style. All panels use the same color.

    Args:
        X_dict: {label: (N, T, 2) np.array}. One panel per key.
                GT-only: {"Ground Truth": X_gt}
                Compare:  {"Ground Truth": X_gt, "WLM": X_wlm}
        times:  (T,) time values
        save_path: output path
        frame_skip: render every k-th frame
        fps: gif framerate
        subsample: max particles per dataset (None = all)
        color: scatter color for all panels (default: C_BOID)
        train_time: if set, frames past this time show "Forecast" tag and
                    W1 is displayed between first two datasets
    """
    labels = list(X_dict.keys())
    if color is None:
        _colors = ["tab:blue"] + ["tab:orange"] * (len(labels) - 1)
    elif isinstance(color, list):
        _colors = color
    else:
        _colors = [color] * len(labels)
    datasets = [v.copy() for v in X_dict.values()]
    n_rows = len(labels)
    is_comparison = (n_rows >= 2)

    # Subsample
    if subsample is not None:
        rng = np.random.default_rng(0)
        for i in range(n_rows):
            N = datasets[i].shape[0]
            if subsample < N:
                idx = rng.choice(N, subsample, replace=False)
                datasets[i] = datasets[i][idx]

    # Axis limits from FIRST dataset only (GT)
    ref_pts = datasets[0].reshape(-1, 2)
    ref_pts = ref_pts[np.isfinite(ref_pts).all(axis=1)]
    lo = np.quantile(ref_pts, 0.005, axis=0)
    hi = np.quantile(ref_pts, 0.995, axis=0)
    pad = 0.1 * np.maximum(0.5, hi - lo)
    xlim = (lo[0] - pad[0], hi[0] + pad[0])
    ylim = (lo[1] - pad[1], hi[1] + pad[1])

    # Optional: try importing OT for W1, fall back to sliced W2
    _w1_fn = None
    if is_comparison:
        try:
            import ot as pot
            def _w1_fn(a, b):
                from scipy.spatial.distance import cdist
                M = cdist(a, b, metric="euclidean").astype(np.float64)
                n, m = M.shape
                return pot.emd2(np.ones(n) / n, np.ones(m) / m, M)
        except ImportError:
            # Fallback: sliced Wasserstein (no extra deps)
            def _w1_fn(a, b, n_proj=50):
                d = a.shape[1]
                rng = np.random.default_rng(42)
                dirs = rng.normal(size=(n_proj, d))
                dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                total = 0.0
                for v in dirs:
                    pa = np.sort(a @ v)
                    pb = np.sort(b @ v)
                    # match sizes by subsampling the larger
                    n = min(len(pa), len(pb))
                    total += np.mean(np.abs(pa[:n] - pb[:n]))
                return total / n_proj

    T = datasets[0].shape[1]
    frames = []

    fig, axes = plt.subplots(1, n_rows, figsize=(3.2 * n_rows, 3.2),
                              squeeze=False)
    axes = axes[0]

    # Set up all axes once so tight_layout computes stable positions
    for r, label in enumerate(labels):
        ax = axes[r]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=9, fontweight="medium")
        ax.tick_params(labelsize=7)
        if r > 0:
            ax.tick_params(labelleft=False)
    fig.suptitle("$t = 0.00$", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # Freeze subplot positions — no more tight_layout calls
    for k in range(0, T, frame_skip):
        t_val = float(times[k]) if times is not None else k
        is_forecast = (train_time is not None) and (t_val > float(train_time))

        for r, (label, data) in enumerate(zip(labels, datasets)):
            ax = axes[r]
            ax.clear()
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            _tikz_scatter(ax, data[:, k, 0], data[:, k, 1],
                          color=_colors[r], s=10, alpha=0.55)
            ax.set_title(label, fontsize=9, fontweight="medium")
            ax.tick_params(labelsize=7)
            if r > 0:
                ax.tick_params(labelleft=False)

        # Build suptitle
        phase = "Forecast" if is_forecast else "Train"
        title_parts = [f"$t = {t_val:.1f}$"]

        if is_comparison and train_time is not None:
            title_parts.append(f"[{phase}]")

        if is_comparison and _w1_fn is not None:
            gt_k = datasets[0][:, k, :]
            pred_k = datasets[1][:, k, :]
            # filter NaN/Inf
            gt_k = gt_k[np.isfinite(gt_k).all(axis=1)]
            pred_k = pred_k[np.isfinite(pred_k).all(axis=1)]
            if gt_k.shape[0] > 0 and pred_k.shape[0] > 0:
                w1 = _w1_fn(gt_k, pred_k)
                title_parts.append(f"   $W_1 = {w1:.2f}$")

        fig.suptitle("  ".join(title_parts), fontsize=10, y=0.98)

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())
        frames.append(frame.copy())

    plt.close(fig)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    print(f"Saved: {save_path}  ({len(frames)} frames)")
    return save_path

C_FORECAST = "#9B5DE5"  # purple for forecast labels/divider
def compare_snapshots(X_gt, X_wlm, times, n_cols=6, subsample=None,
                      train_time=None):
    """Two-row snapshot grid: Ground Truth on top, WLM on bottom.

    If train_time is set, a dashed vertical line separates train/forecast
    columns, and forecast time labels are colored purple.
    """
    N_gt, T, _ = X_gt.shape
    idxs = np.linspace(0, T - 1, n_cols, dtype=int)

    if subsample and subsample < N_gt:
        rng = np.random.default_rng(0)
        si = rng.choice(N_gt, subsample, replace=False)
        X_gt_s, X_wlm_s = X_gt[si], X_wlm[si]
    else:
        X_gt_s, X_wlm_s = X_gt, X_wlm

    # Axis limits from GT only
    ref_pts = X_gt_s.reshape(-1, 2)
    lo = np.quantile(ref_pts, 0.005, axis=0)
    hi = np.quantile(ref_pts, 0.995, axis=0)
    pad = 0.1 * np.maximum(0.5, hi - lo)
    xlim = (lo[0] - pad[0], hi[0] + pad[0])
    ylim = (lo[1] - pad[1], hi[1] + pad[1])

    fig, axes = plt.subplots(2, n_cols, figsize=(2.6 * n_cols, 5.2),
                              sharex=True, sharey=True)

    for col, t_idx in enumerate(idxs):
        t_val = float(times[t_idx])
        is_forecast = (train_time is not None) and (t_val > float(train_time))
        title_color = C_FORECAST if is_forecast else "black"

        for row, (data, label) in enumerate([(X_gt_s, "GT"), (X_wlm_s, "WLM")]):
            ax = axes[row, col]
            c = "tab:blue" if row == 0 else "tab:orange"
            ax.scatter(data[:, t_idx, 0], data[:, t_idx, 1],
                       s=10, alpha=0.55, c=c, edgecolors="none", rasterized=True)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            if col > 0:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.set_title(f"$t = {t_val:.1f}$", fontsize=9, color=title_color)

            # Purple border for forecast panels
            if is_forecast:
                for spine in ax.spines.values():
                    spine.set_edgecolor(C_FORECAST)
                    spine.set_linewidth(1.2)

        if col == 0:
            axes[0, 0].set_ylabel("Ground Truth", fontsize=9)
            axes[1, 0].set_ylabel("WLM", fontsize=9)

    fig.tight_layout(h_pad=0.4, w_pad=0.3)

    # Draw dashed vertical divider between last train and first forecast column
    if train_time is not None:
        # Find the split: last column where t <= train_time
        t_vals = [float(times[idxs[c]]) for c in range(n_cols)]
        split_after = None
        for c in range(n_cols - 1):
            if t_vals[c] <= float(train_time) < t_vals[c + 1]:
                split_after = c
                break

        if split_after is not None:
            # Get figure-space x midpoint between the two columns
            bbox_left = axes[0, split_after].get_position()
            bbox_right = axes[0, split_after + 1].get_position()
            x_mid = (bbox_left.x1 + bbox_right.x0) / 2.0

            fig.add_artist(plt.Line2D(
                [x_mid, x_mid], [0.0, 1.0],
                transform=fig.transFigure, clip_on=False,
                color=C_FORECAST, linewidth=1.5, linestyle="--", alpha=0.8,
            ))

    return fig


def plot_single_holdout_scatter(
        x_pred,
        y_true,
        holdout_idx,
        time_val,
        w1_value,
        max_points=1000,
        title_prefix="",
):
    """
    Clean scatter: true marginal (blue dots) vs predicted (orange x).
    W1 displayed in the title. Projects to first 2 dims (PC1/PC2 for EB).

    Args:
        x_pred: (N_pred, d) predicted positions (numpy or tensor)
        y_true: (N_true, d) ground truth positions (numpy or tensor)
        holdout_idx: integer index of the held-out marginal
        time_val: float, physical time of the holdout
        w1_value: float, W1 distance
        max_points: subsample cap for clarity
        title_prefix: optional string prepended to title

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if hasattr(x_pred, "detach"):
        x_pred = x_pred.detach().cpu().numpy()
    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()

    # Subsample
    if x_pred.shape[0] > max_points:
        idx = np.random.choice(x_pred.shape[0], max_points, replace=False)
        x_pred = x_pred[idx]
    if y_true.shape[0] > max_points:
        idx = np.random.choice(y_true.shape[0], max_points, replace=False)
        y_true = y_true[idx]

    # Project to 2D
    xp = x_pred[:, :2]
    yt = y_true[:, :2]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt[:, 0], yt[:, 1], s=8, alpha=0.45, c="tab:blue", label="True", zorder=2)
    ax.scatter(xp[:, 0], xp[:, 1], s=8, alpha=0.45, c="tab:orange", marker="x", label="Pred", zorder=3)

    w1_str = f"{w1_value:.4f}" if np.isfinite(w1_value) else "NaN"
    title = f"{title_prefix}Holdout t={time_val:.2f} (idx={holdout_idx})  |  W₁ = {w1_str}"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig



def plot_multi_holdout_scatter(
        all_true: List[Tuple[torch.Tensor, int]],
        all_pred: List[Tuple[torch.Tensor, int]],
        time_grid: torch.Tensor,
        title: str = "Multi-Holdout Interpolation",
        max_points: int = 500,
):
    """
    Plot all holdout marginals: true (o) vs predicted (x), colored by time.

    Args:
        all_true: List of (positions, holdout_idx) for ground truth
        all_pred: List of (positions, holdout_idx) for predictions
        time_grid: Full time grid tensor
        title: Plot title
        max_points: Max points to plot per marginal (for clarity)

    Returns:
        matplotlib Figure
    """
    n_holdouts = len(all_true)

    # Get data dimensionality
    d = all_true[0][0].shape[-1] if len(all_true) > 0 else 2

    # Create colormap
    cmap = cm.get_cmap('viridis', n_holdouts)
    colors = [cmap(i) for i in range(n_holdouts)]

    if d == 2:
        fig, ax = plt.subplots(figsize=(8, 8))

        for i, ((y_true, h_idx), (y_pred, _)) in enumerate(zip(all_true, all_pred)):
            y_true_np = y_true.numpy()
            y_pred_np = y_pred.numpy()

            # Subsample if needed
            if y_true_np.shape[0] > max_points:
                idx = np.random.choice(y_true_np.shape[0], max_points, replace=False)
                y_true_np = y_true_np[idx]
            if y_pred_np.shape[0] > max_points:
                idx = np.random.choice(y_pred_np.shape[0], max_points, replace=False)
                y_pred_np = y_pred_np[idx]

            t_val = float(time_grid[h_idx].item())
            label_true = f"t={t_val:.1f} true"
            label_pred = f"t={t_val:.1f} pred"

            # True: circles (o)
            ax.scatter(y_true_np[:, 0], y_true_np[:, 1],
                       c=[colors[i]], marker='o', s=20, alpha=0.6,
                       label=label_true, edgecolors='none')
            # Predicted: x markers
            ax.scatter(y_pred_np[:, 0], y_pred_np[:, 1],
                       c=[colors[i]], marker='x', s=20, alpha=0.6,
                       label=label_pred)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.set_aspect('equal', adjustable='box')

    elif d >= 3:
        # Use first 2 PCs or first 2 dims
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax_idx, (dim1, dim2) in enumerate([(0, 1), (0, 2) if d > 2 else (0, 1)]):
            ax = axes[ax_idx]

            for i, ((y_true, h_idx), (y_pred, _)) in enumerate(zip(all_true, all_pred)):
                y_true_np = y_true.numpy()
                y_pred_np = y_pred.numpy()

                if y_true_np.shape[0] > max_points:
                    idx = np.random.choice(y_true_np.shape[0], max_points, replace=False)
                    y_true_np = y_true_np[idx]
                if y_pred_np.shape[0] > max_points:
                    idx = np.random.choice(y_pred_np.shape[0], max_points, replace=False)
                    y_pred_np = y_pred_np[idx]

                t_val = float(time_grid[h_idx].item())

                ax.scatter(y_true_np[:, dim1], y_true_np[:, dim2],
                           c=[colors[i]], marker='o', s=15, alpha=0.5)
                ax.scatter(y_pred_np[:, dim1], y_pred_np[:, dim2],
                           c=[colors[i]], marker='x', s=15, alpha=0.5)

            ax.set_xlabel(f'dim {dim1}')
            ax.set_ylabel(f'dim {dim2}')
            ax.set_title(f'{title} (dims {dim1}-{dim2})')

        # Add legend to first subplot
        from matplotlib.lines import Line2D
        legend_elements = []
        for i, (_, h_idx) in enumerate(all_true):
            t_val = float(time_grid[h_idx].item())
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=colors[i], markersize=8,
                                          label=f't={t_val:.1f} true'))
            legend_elements.append(Line2D([0], [0], marker='x', color=colors[i],
                                          markersize=8, linestyle='None',
                                          label=f't={t_val:.1f} pred'))
        axes[0].legend(handles=legend_elements, loc='upper right', fontsize=7, ncol=2)
    else:
        # 1D case
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, ((y_true, h_idx), (y_pred, _)) in enumerate(zip(all_true, all_pred)):
            y_true_np = y_true.numpy().flatten()
            y_pred_np = y_pred.numpy().flatten()

            t_val = float(time_grid[h_idx].item())
            ax.hist(y_true_np, bins=50, alpha=0.5, color=colors[i], label=f't={t_val:.1f} true')
            ax.hist(y_pred_np, bins=50, alpha=0.3, color=colors[i], histtype='step',
                    linewidth=2, label=f't={t_val:.1f} pred')
        ax.legend()
        ax.set_title(title)

    plt.tight_layout()
    return fig


def make_compare_gif(
        X_true, X_learned, dt,
        true_label="true", est_label="est",
        grid_size=None,
        save_path="temp.gif",
        always_show=True,
        X_null=None,
        show_null=True,
        frame_skip=5,
        fps=5,
        filter_outliers=False,
        times=None,  # NEW feature
        projection="auto",  # NEW feature
        render="auto",  # NEW feature
        bins=80,  # NEW feature
        subsample=None,  # NEW feature
):
    """
    Compare X_true vs X_learned (and optional X_null) as a GIF.

    Hybrid Behavior:
      - If d=2 (or PCA reduced to 2D): Uses "Old Style" (Single plot, overlapping scatter, legend).
      - If d=3: Uses "New Style" (Grid of subplots, rows=datasets, cols=projections).

    Supports:
      - (N,T,d) or (T,N,d) inputs.
      - Outlier filtering, custom timestamps, and subsampling.
    """
    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import imageio

    try:
        from IPython.display import display, Image
        _HAS_IPY = True
    except Exception:
        _HAS_IPY = False

    # --- Setup Directories ---
    save_path = str(save_path)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # --- Helpers ---
    def to_np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else x

    def canon_shape(A):
        """Return array shaped (N, T, d)."""
        A = to_np(A)
        if A is None:
            return None
        if A.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {A.shape}")
        # Heuristic: time dimension is typically much smaller than N
        # If first axis looks like time (and N is large), transpose.
        if A.shape[0] < A.shape[1] and A.shape[0] <= 1024 and A.shape[1] >= 256:
            return np.transpose(A, (1, 0, 2))
        return A

    # --- Data Loading & Shape Checking ---
    X_true_np = canon_shape(X_true)
    X_learned_np = canon_shape(X_learned)
    X_null_np = canon_shape(X_null) if (X_null is not None) else None

    can_show_null = bool(show_null) and (X_null_np is not None)

    if X_true_np.shape != X_learned_np.shape:
        raise ValueError(f"Shape mismatch true={X_true_np.shape} learned={X_learned_np.shape}")
    if can_show_null and X_null_np.shape != X_true_np.shape:
        raise ValueError(f"X_null shape {X_null_np.shape} must match {X_true_np.shape}")

    N, T, d = X_true_np.shape
    # --- EB-friendly: if d>3, visualize using first two coordinates (PC1/PC2) ---
    if d > 3:
        X_true_np = X_true_np[..., :2]
        X_learned_np = X_learned_np[..., :2]
        if can_show_null:
            X_null_np = X_null_np[..., :2]
        d = 2
        # ensure we don't later run PCA again
        if projection == "auto":
            proj_mode = "xy"
        else:
            proj_mode = projection

    # --- Subsampling (Optimization) ---
    if subsample is not None and int(subsample) < N:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=int(subsample), replace=False)
        X_true_np = X_true_np[idx]
        X_learned_np = X_learned_np[idx]
        if can_show_null:
            X_null_np = X_null_np[idx]
        N = X_true_np.shape[0]

    # --- Dimensionality Reduction (PCA) if needed ---
    if "proj_mode" not in locals():
        if projection == "auto":
            proj_mode = "pca" if d > 3 else ("orth3" if d == 3 else "xy")
        else:
            proj_mode = projection

    if proj_mode == "pca":
        # Compute global PCA over all data/times to ensure stable projection
        mats = [X_true_np.reshape(-1, d), X_learned_np.reshape(-1, d)]
        if can_show_null:
            mats.append(X_null_np.reshape(-1, d))
        Z = np.concatenate(mats, axis=0)
        mask = np.isfinite(Z).all(axis=1)
        Z = Z[mask]

        if Z.shape[0] > 0:
            mu = Z.mean(axis=0, keepdims=True)
            Zc = Z - mu
            C = (Zc.T @ Zc) / max(1, Zc.shape[0])
            evals, evecs = np.linalg.eigh(C)
            W = evecs[:, -2:]  # Top 2 eigenvectors

            def pca2(A):
                return (A - mu) @ W

            X_true_np = pca2(X_true_np)
            X_learned_np = pca2(X_learned_np)
            if can_show_null:
                X_null_np = pca2(X_null_np)
            d = 2  # Now we are in 2D mode

    # --- Calculate Axis Limits ---
    # We do this for both modes to ensure robust outlier handling
    data_list = [X_true_np, X_learned_np]
    if can_show_null:
        data_list.append(X_null_np)
    all_pts = np.concatenate(data_list, axis=0).reshape(-1, d)
    all_pts = all_pts[np.isfinite(all_pts).all(axis=1)]

    if grid_size is None:
        if all_pts.shape[0] == 0:
            lims = [(-1.0, 1.0)] * d
        else:
            if filter_outliers:
                q = 0.995
                lo = np.quantile(all_pts, 1 - q, axis=0)
                hi = np.quantile(all_pts, q, axis=0)
            else:
                lo = np.min(all_pts, axis=0)
                hi = np.max(all_pts, axis=0)
            pad = 0.05 * np.maximum(1e-12, hi - lo)
            lims = [(float(lo[k] - pad[k]), float(hi[k] + pad[k])) for k in range(d)]
    else:
        gs = float(grid_size)
        lims = [(-gs, gs)] * d

    # --- Time Labels ---
    dtf = float(dt)
    if times is not None:
        times = np.asarray(times).reshape(-1)
        # Allow slight mismatches (T vs T+1) commonly found in ODE solvers
        if len(times) not in [T, T + 1]:
            raise ValueError(f"`times` length {len(times)} mismatch with T={T}")

    frames = []

    # ==========================================
    # BRANCH: 2D (Old Behavior)
    # Single plot, overlapping scatters, legend
    # ==========================================
    if d == 2:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=110)

        for k in range(0, T, int(frame_skip)):
            ax.clear()
            ax.set_xlim(*lims[0])
            ax.set_ylim(*lims[1])
            ax.set_aspect("equal")

            # True
            ax.scatter(X_true_np[:, k, 0], X_true_np[:, k, 1],
                       s=5, alpha=0.4, label=true_label, c="tab:blue")

            # Learned
            ax.scatter(X_learned_np[:, k, 0], X_learned_np[:, k, 1],
                       s=5, alpha=0.4, label=est_label, c="tab:orange")

            # Null
            if can_show_null:
                ax.scatter(X_null_np[:, k, 0], X_null_np[:, k, 1],
                           s=5, alpha=0.12, label="Null", c="grey")

            # Title & Legend
            tval = float(times[k]) if times is not None else (k * dtf)
            ax.set_title(f"t = {tval:.3f}")
            ax.grid(alpha=0.3)
            # Legend only needs to be added once, but adding every frame is safe in loop
            ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor='white', framealpha=0.8)

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())
            frames.append(frame.copy())

        plt.close(fig)

    # ==========================================
    # BRANCH: 3D (New Behavior)
    # Grid of subplots, separate rows, projections
    # ==========================================
    else:
        # 3 Projections
        panels = [(0, 1), (0, 2), (1, 2)]
        col_titles = ["(x,y)", "(x,z)", "(y,z)"]

        # Rows: true, learned, (optional null)
        row_specs = [(true_label, X_true_np), (est_label, X_learned_np)]
        if can_show_null:
            row_specs.append(("Null", X_null_np))

        nrows = len(row_specs)
        ncols = len(panels)

        render_mode = "hist2d" if (render == "auto" or render == "hist2d") else "scatter"

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 4.0 * nrows), dpi=110)
        # Ensure axes is always 2D array
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        elif ncols == 1:
            axes = np.array([[ax] for ax in axes])

        for k in range(0, T, int(frame_skip)):
            for r, (rlabel, Xr) in enumerate(row_specs):
                for c, (i, j) in enumerate(panels):
                    ax = axes[r, c]
                    ax.clear()
                    ax.set_aspect("equal", adjustable="box")
                    ax.set_xlim(*lims[i])
                    ax.set_ylim(*lims[j])
                    ax.grid(alpha=0.25)

                    pts = Xr[:, k, :]
                    x = pts[:, i]
                    y = pts[:, j]

                    # Basic NaN filtering
                    m = np.isfinite(x) & np.isfinite(y)
                    x = x[m];
                    y = y[m]

                    if render_mode == "hist2d":
                        ax.hist2d(x, y, bins=int(bins), range=[lims[i], lims[j]])
                    else:
                        ax.scatter(x, y, s=4, alpha=0.35)

                    if r == 0:
                        ax.set_title(col_titles[c], fontsize=10)
                    if c == 0:
                        ax.set_ylabel(rlabel)

            tval = float(times[k]) if times is not None else (k * dtf)
            fig.suptitle(f"t = {tval:.3f}", y=0.98, fontsize=12)

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())
            frames.append(frame.copy())

        plt.close(fig)

    # --- Save and Display ---
    imageio.mimsave(save_path, frames, fps=int(fps), loop=0)

    if always_show and _HAS_IPY:
        display(Image(filename=save_path))

    return save_path

from typing import Any, Optional
from pathlib import Path
from mechanics import pick_integrator
from potential_energy_models import make_accel_from_potential
import wandb

@torch.no_grad()
def maybe_gif(
    step_idx: int,
    *,
    gif_every: int,
    gif_p0_idx: int,
    particles_gif: Optional[int],
    gif_frame_skip: int,
    gif_fps: int,
    substeps_per_dt: int,
    integrator_name: str,
    max_force: Optional[float],
    model: torch.nn.Module,
    X_em: torch.Tensor,               # (num_p0,N,T+1,d)
    time_grid: torch.Tensor,          # (T+1,)
    dt_base: float,
    vel_provider,
    vel_mode: str,
    V_em: Optional[torch.Tensor],
    friction: Any,
    outdir: Path,
    device: torch.device,
    wb_run: Optional[Any] = None,
) -> None:
    if int(gif_every) <= 0:
        return
    if int(step_idx) % int(gif_every) != 0:
        return

    model.eval()

    p0_idx = int(gif_p0_idx)
    X_gt = X_em[p0_idx]  # (N,T+1,d)
    x0_full = X_gt[:, 0, :].detach()

    # --- FIX: Filter NaNs immediately at t=0 ---
    # If we don't do this, Transformer/Attention will crash or output all NaNs
    valid_mask = torch.isfinite(x0_full).all(dim=-1)

    # Apply mask to Ground Truth (so we only track valid particles)
    X_gt = X_gt[valid_mask]
    x0 = x0_full[valid_mask]

    # Now subsample from the VALID set
    idx: Optional[torch.Tensor] = None
    if particles_gif is not None and int(particles_gif) < x0.shape[0]:
        idx = torch.randint(0, x0.shape[0], (int(particles_gif),), device=device)
        X_gt = X_gt[idx]
        x0 = x0[idx]
        # Note: We must also filter V_em later using the same logic if it exists

    t0 = float(time_grid[0].item())
    vel_mode_l = str(vel_mode).lower()

    if vel_provider is None or vel_mode_l == "zero":
        v0 = torch.zeros_like(x0)
    elif vel_mode_l == "bundle":
        if V_em is None:
            raise ValueError("maybe_gif: vel_mode='bundle' requires V_em loaded from bundle.")
        m = int(torch.argmin((time_grid - float(t0)).abs()).item())
        m = max(0, min(m, time_grid.numel() - 1))

        # We must filter V_em exactly how we filtered X_em above
        v0_all = V_em[int(p0_idx), :, m, :]
        v0_valid = v0_all[valid_mask]  # Apply the NaN mask

        v0 = v0_valid if idx is None else v0_valid[idx]  # Apply the subsample mask
        v0 = v0.to(device=x0.device, dtype=x0.dtype)
    else:
        v0 = vel_provider(p0_idx, x0, t0).detach()

    dt_train = float(dt_base) / int(substeps_per_dt)
    total_micro = (X_gt.shape[1] - 1) * int(substeps_per_dt)

    integrator, _ = pick_integrator(str(integrator_name))
    accel_eval = make_accel_from_potential(model, create_graph=False, max_force=max_force)

    X_pred = integrator(
        x0=x0,
        v0=v0,
        accel=accel_eval,
        dt=dt_train,
        steps=int(total_micro),
        friction=friction,
        return_all=True,
        t_start=float(t0),
    )

    steps_macro = X_gt.shape[1] - 1
    macro_idx = (torch.arange(0, steps_macro + 1, device=device) * int(substeps_per_dt)).long()
    X_pred_macro = X_pred[:, macro_idx, :]

    gif_path = str(outdir / f"compare_step{int(step_idx):07d}.gif")
    make_compare_gif(
        X_true=X_gt.detach().cpu(),
        X_learned=X_pred_macro.detach().cpu(),
        dt=float(dt_base),
        times=time_grid.detach().cpu().numpy(),
        save_path=gif_path,
        frame_skip=int(gif_frame_skip),
        fps=int(gif_fps),
        always_show=False,
        projection="auto",
        render="auto",
        subsample=int(particles_gif) if particles_gif is not None else None,
    )

    # if wb_run is not None:
    #     import wandb
    #     wandb.log({"gif": wandb.Video(gif_path, fps=int(gif_fps), format="gif")}, step=int(step_idx))

    model.train()
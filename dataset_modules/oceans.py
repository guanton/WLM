# dataset_modules/oceans.py
"""
Ocean Currents Dataset Loader

Loads ocean currents data from HYCOM reanalysis (Gulf of Mexico vortex).
Data format from curly-flow-matching repo: positions and velocities over time.

The data consists of ~111 particles tracked over 9 timesteps in a vortex.
This is a good test case for interpolation (hold out middle timesteps).
"""
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def generate_oceans(
        *,
        npz_path: Path,
        num_p0: int = 1,
        train_ts: Optional[List[int]] = None,  # If provided, only keep these timesteps
        seed: int = 0,
        device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Load ocean currents data from NPZ file.

    Expected NPZ structure (from curly-flow-matching):
        positions: (T, N, d) - particle positions over time
        velocities: (T, N, d) - particle velocities over time

    Outputs (matching your WLM pipeline format):
        X_em_torch: (num_p0=1, N, T, d) float32
        V_em_torch: (num_p0=1, N, T, d) float32
        time_grid:  (T,) float32
        meta: dict with dataset info

    Args:
        npz_path: Path to oceans.npz file
        num_p0: Number of populations (enforced to 1 for this dataset)
        train_ts: Optional list of timestep indices to keep (for pre-filtering)
        seed: Random seed for any shuffling
        device: Target device
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Ocean currents npz not found: {npz_path}")

    if int(num_p0) != 1:
        raise ValueError("Ocean currents loader supports num_p0=1 only (single trajectory set).")

    # Load data
    data = np.load(npz_path, allow_pickle=True)
    positions = np.asarray(data["positions"], dtype=np.float32)  # (T, N, d)
    velocities = np.asarray(data["velocities"], dtype=np.float32)  # (T, N, d)

    if positions.ndim != 3 or velocities.ndim != 3:
        raise ValueError(f"Expected 3D arrays; got positions={positions.shape}, velocities={velocities.shape}")
    if positions.shape != velocities.shape:
        raise ValueError(f"Shape mismatch: positions={positions.shape} vs velocities={velocities.shape}")

    T, N, d = positions.shape
    print(f"[Oceans] Loaded: T={T} timesteps, N={N} particles, d={d} dimensions")

    # Optional: filter to specific timesteps
    if train_ts is not None:
        train_ts = sorted([int(t) for t in train_ts])
        if not all(0 <= t < T for t in train_ts):
            raise ValueError(f"train_ts indices out of range [0, {T}): {train_ts}")
        positions = positions[train_ts]
        velocities = velocities[train_ts]
        T = len(train_ts)
        print(f"[Oceans] Filtered to timesteps {train_ts}: T={T}")
    else:
        train_ts = list(range(T))

    rng = np.random.default_rng(int(seed))

    # Shuffle particle order ONCE (same permutation for all timesteps)
    # This preserves particle identity across time (important for trajectories)
    perm = rng.permutation(N)
    for t in range(T):
        positions[t] = positions[t][perm]
        velocities[t] = velocities[t][perm]

    # Normalization (like EB dataset)
    norm_mean = None
    norm_std = None

    # Convert to torch in your pipeline format: (num_p0, N, T, d)
    # Input is (T, N, d), need (1, N, T, d)
    X_np = np.transpose(positions, (1, 0, 2))  # (N, T, d)
    V_np = np.transpose(velocities, (1, 0, 2))  # (N, T, d)

    X_em_torch = torch.from_numpy(X_np).unsqueeze(0).to(device=device)  # (1, N, T, d)
    V_em_torch = torch.from_numpy(V_np).unsqueeze(0).to(device=device)  # (1, N, T, d)

    # Time grid: unit spacing (dt=1.0 in normalized time)
    # Original data: delta_t = 0.9 physical units, but we normalize to 1.0
    dt = 1.0
    time_grid = torch.arange(T, device=device, dtype=torch.float32) * float(dt)

    meta: Dict[str, Any] = {
        "mode": "oceans",
        "dataset": "ocean_currents_hycom",
        "npz_path": str(npz_path),
        "num_p0": 1,
        "N": int(N),
        "T": int(T),
        "steps": int(T - 1),
        "dt": float(dt),
        "d": int(d),
        "original_timesteps": train_ts,
        "has_vel": True,
        "vel_kind": "ground_truth",
        "seed": int(seed),
    }

    if norm_mean is not None:
        meta["norm_mean"] = norm_mean
        meta["norm_std"] = norm_std

    print(
        f"[Oceans] Output shapes: X={tuple(X_em_torch.shape)}, V={tuple(V_em_torch.shape)}, t={tuple(time_grid.shape)}")

    return X_em_torch, V_em_torch, time_grid, meta


def add_oceans_parser(subparsers) -> argparse.ArgumentParser:
    """Add oceans subcommand to data_generator.py parser."""
    p = subparsers.add_parser("oceans", help="Ocean currents dataset from HYCOM reanalysis")
    p.add_argument("--npz-path", type=str, default="datasets/oceans_with_v0.npz",
                   help="Path to oceans.npz file")
    p.add_argument("--train-ts", type=int, nargs="*", default=None,
                   help="Specific timestep indices to keep (default: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-p0", type=int, default=1)  # Enforced to 1
    return p


# ============================================================
# Utility: Download ocean currents data from GitHub
# ============================================================

def download_oceans_data(
        output_path: Path = Path("datasets/oceans.npz"),
        repo_url: str = "https://raw.githubusercontent.com/kpetrovicc/curly-flow-matching/main/data/oceans/oceans.npz",
) -> Path:
    """
    Download ocean currents data from curly-flow-matching repo.

    Note: This requires network access. If unavailable, you'll need to
    manually download the file from:
    https://github.com/kpetrovicc/curly-flow-matching/tree/main/data/oceans
    """
    import urllib.request

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"[Oceans] Data already exists: {output_path}")
        return output_path

    print(f"[Oceans] Downloading from {repo_url}...")
    try:
        urllib.request.urlretrieve(repo_url, output_path)
        print(f"[Oceans] Saved to {output_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download ocean currents data: {e}\n"
            f"Please manually download from: {repo_url}"
        )

    return output_path


# ============================================================
# Quick test / inspection
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test ocean currents data loader")
    parser.add_argument("--npz-path", type=str, default="datasets/oceans.npz")
    parser.add_argument("--download", action="store_true", help="Download data if missing")
    parser.add_argument("--plot", action="store_true", help="Plot data visualization")
    args = parser.parse_args()

    npz_path = Path(args.npz_path)

    if args.download and not npz_path.exists():
        download_oceans_data(npz_path)

    if not npz_path.exists():
        print(f"Data file not found: {npz_path}")
        print("Use --download to fetch from GitHub, or manually place the file.")
        exit(1)

    # Test loader
    X, V, time_grid, meta = generate_oceans(
        npz_path=npz_path,
        seed=42,
    )

    print("\n=== Data Summary ===")
    print(f"X shape: {X.shape}")
    print(f"V shape: {V.shape}")
    print(f"Time grid: {time_grid}")
    print(f"Meta: {meta}")

    # Quick stats
    X_np = X[0].numpy()  # (N, T, d)
    V_np = V[0].numpy()

    print(f"\nPosition stats:")
    print(f"  min: {X_np.min():.3f}, max: {X_np.max():.3f}")
    print(f"  mean: {X_np.mean():.3f}, std: {X_np.std():.3f}")

    print(f"\nVelocity stat:")
    print(f"  min: {V_np.min():.3f}, max: {V_np.max():.3f}")
    print(f"  mean: {V_np.mean():.3f}, std: {V_np.std():.3f}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot positions colored by time
        ax = axes[0]
        T = X_np.shape[1]
        colors = plt.cm.viridis(np.linspace(0, 1, T))
        for t in range(T):
            ax.scatter(X_np[:, t, 0], X_np[:, t, 1], c=[colors[t]], s=10, alpha=0.7, label=f't={t}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Particle Positions by Time')
        ax.legend(markerscale=2, fontsize=8)

        # Plot velocity field at t=0
        ax = axes[1]
        t_idx = 0
        ax.quiver(X_np[:, t_idx, 0], X_np[:, t_idx, 1],
                  V_np[:, t_idx, 0], V_np[:, t_idx, 1],
                  alpha=0.7)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Velocity Field at t={t_idx}')

        plt.tight_layout()
        plt.savefig('oceans_data_preview.png', dpi=150)
        print("\nSaved plot to oceans_data_preview.png")
        plt.show()
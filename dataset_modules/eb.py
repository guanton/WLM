# dataset_modules/eb.py
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _choose_labels(sorted_unique_labels: np.ndarray, num_times: int) -> np.ndarray:
    """Pick `num_times` labels evenly across sorted unique labels."""
    L = len(sorted_unique_labels)
    if L <= num_times:
        return sorted_unique_labels

    idx = np.linspace(0, L - 1, num=num_times)
    idx = np.unique(np.round(idx).astype(int))
    if len(idx) < num_times:
        missing = num_times - len(idx)
        candidates = [i for i in range(L) if i not in set(idx)]
        idx = np.concatenate([idx, np.array(candidates[:missing], dtype=int)])
        idx = np.sort(idx)
    return sorted_unique_labels[idx[:num_times]]


def generate_eb(
    *,
    npz_path: Path,
    num_times: int = 5,
    pca_dim: int = 5,                 # keep only top 5 PCA
    num_p0: int = 1,
    label_subset: Optional[List[int]] = None,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    EB NPZ -> padded marginals (NaN padding; NO duplication)

    Outputs:
      X_em_torch: (num_p0=1, Nmax, T, d=pca_dim) float32 with NaN padding
      V_em_torch: (num_p0=1, Nmax, T, d=pca_dim) float32 with NaN padding (pcs_delta)
      time_grid:  (T,) float32 (selected label ids)
      meta: dict (Ns_per_time, selected_labels, dt, steps, ...)
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"EB npz not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    pcs = np.asarray(data["pcs"])              # (Ncells, 100)
    pcs_delta = np.asarray(data["pcs_delta"])  # (Ncells, 100)
    labels = np.asarray(data["sample_labels"]) # (Ncells,)

    if pcs.ndim != 2 or pcs_delta.ndim != 2:
        raise ValueError(f"Expected pcs/pcs_delta to be 2D; got {pcs.shape}, {pcs_delta.shape}")
    if pcs.shape != pcs_delta.shape:
        raise ValueError(f"pcs and pcs_delta must match shape; got {pcs.shape} vs {pcs_delta.shape}")
    if labels.shape[0] != pcs.shape[0]:
        raise ValueError(f"labels length must match pcs rows; got {labels.shape[0]} vs {pcs.shape[0]}")

    if pca_dim < 1 or pca_dim > pcs.shape[1]:
        raise ValueError(f"pca_dim must be in [1, {pcs.shape[1]}], got {pca_dim}")
    if int(num_p0) != 1:
        raise ValueError("EB loader supports num_p0=1 only (one population).")

    uniq_sorted = np.sort(np.unique(labels))

    if label_subset is not None and len(label_subset) > 0:
        chosen = np.array([int(x) for x in label_subset], dtype=labels.dtype)
        chosen = chosen[np.isin(chosen, uniq_sorted)]
        if len(chosen) == 0:
            raise ValueError("label_subset provided, but none found in sample_labels.")
        if len(chosen) != int(num_times):
            raise ValueError(f"label_subset must have exactly num_times={num_times} labels; got {len(chosen)}.")
        selected_labels = np.sort(chosen)
    else:
        selected_labels = _choose_labels(uniq_sorted, int(num_times))

    idx_by_label: Dict[int, np.ndarray] = {int(lab): np.where(labels == lab)[0] for lab in selected_labels}
    Ns_per_time = [int(idx_by_label[int(lab)].shape[0]) for lab in selected_labels]
    if any(n == 0 for n in Ns_per_time):
        raise ValueError(f"Some selected labels have zero cells: {list(zip(selected_labels.tolist(), Ns_per_time))}")

    T = int(len(selected_labels))
    Nmax = int(max(Ns_per_time))
    d = int(pca_dim)

    rng = np.random.default_rng(int(seed))

    pcs_d = pcs[:, :d].astype(np.float32, copy=False)
    dpcs_d = pcs_delta[:, :d].astype(np.float32, copy=False)

    # NaN-padded rectangular arrays (T, Nmax, d)
    X_np = np.full((T, Nmax, d), np.nan, dtype=np.float32)
    V_np = np.full((T, Nmax, d), np.nan, dtype=np.float32)

    for t, lab in enumerate(selected_labels):
        idx = idx_by_label[int(lab)]
        Xt = pcs_d[idx]  # (n_t, d)
        Vt = dpcs_d[idx]  # (n_t, d)

        # shuffle within each marginal to avoid any index-based structure
        perm = rng.permutation(Xt.shape[0])
        Xt = Xt[perm]
        Vt = Vt[perm]

        n = Xt.shape[0]
        X_np[t, :n, :] = Xt
        V_np[t, :n, :] = Vt

        # =========================================================
        # TRAJECTORYNET PREPROCESSING MATCH
        # =========================================================
        # 1. Compute Global Stats (ignoring NaNs)
    flat_X = X_np.reshape(-1, d)
    mask_valid = ~np.isnan(flat_X).any(axis=1)

    mu = np.mean(flat_X[mask_valid], axis=0)  # (d,)
    sigma = np.std(flat_X[mask_valid], axis=0)  # (d,)
    sigma[sigma < 1e-6] = 1.0  # Safety

    print(f"[EB Normalization] Global Mean: {mu}")
    print(f"[EB Normalization] Global Std : {sigma}")

    # 2. Reshape for broadcasting (T, N, d) -> need (1, 1, d)
    mu_bc = mu.reshape(1, 1, d)
    sigma_bc = sigma.reshape(1, 1, d)

    # 3. Standardize X: (X - mu) / sigma
    X_np = (X_np - mu_bc) / sigma_bc

    # 4. Scale V: V / sigma (Do not subtract mean from velocity!)
    V_np = V_np / sigma_bc
    # =========================================================

    # Convert to torch in boids-style layout: (num_p0, N, T, d)
    X_em_torch = torch.from_numpy(np.transpose(X_np, (1, 0, 2))).unsqueeze(0).to(device=device)
    V_em_torch = torch.from_numpy(np.transpose(V_np, (1, 0, 2))).unsqueeze(0).to(device=device)

    dt = 1.0
    time_grid = torch.arange(T, device=device, dtype=torch.float32) * float(dt)

    meta: Dict[str, Any] = {
        "mode": "eb",
        "dataset": "eb_velocity_v5",
        "npz_path": str(npz_path),
        "num_p0": 1,
        "Nmax": int(Nmax),
        "Ns_per_time": Ns_per_time,
        "T": int(T),
        "steps": int(T - 1),
        "dt": float(dt),
        "d": int(d),
        "selected_labels": [int(x) for x in selected_labels.tolist()],
        "has_vel": True,
        "vel_kind": "pcs_delta_topk",
        "padding": "nan",
        "seed": int(seed),
        # Save normalization constants in case we need to plot in original space later
        "norm_mean": mu,
        "norm_std": sigma,
    }

    return X_em_torch, V_em_torch, time_grid, meta


def add_eb_parser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("eb", help="EB dataset loader from eb_velocity_v5.npz")
    p.add_argument("--npz-path", type=str, default="datasets/eb_velocity_v5.npz")
    p.add_argument("--num-times", type=int, default=5)
    p.add_argument("--pca-dim", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label-subset", type=int, nargs="*", default=None)
    p.add_argument("--num-p0", type=int, default=1)  # parity; enforced to 1
    return p
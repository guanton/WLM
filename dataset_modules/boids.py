import torch
from typing import Dict, Any, Tuple, List, Optional
import argparse
import numpy as np


# ============================================================
# Boids (Vicsek-style) generator
# ============================================================

def _boids_get_dv(
        X: torch.Tensor,
        V: torch.Tensor,
        *,
        outer_radius: float,
        inner_radius: float,
        w_cohesion: float,
        w_separation: float,
        w_alignment: float,
        w_boundary: float,
        boundary: float,
) -> torch.Tensor:
    """Compute acceleration-like update dV for a simple boids model."""
    # Pairwise position differences: (N,N,d) with [i,j] = X_j - X_i
    dX = X[None, :, :] - X[:, None, :]
    dist2 = (dX * dX).sum(dim=-1, keepdim=True)  # (N,N,1)

    inner_mask = (dist2 < float(inner_radius) ** 2).to(X.dtype)
    outer_mask = (dist2 < float(outer_radius) ** 2).to(X.dtype)

    # Cohesion: average displacement towards neighbors
    denom = outer_mask.sum(dim=1).clamp_min(1.0)  # (N,1)
    v1 = (outer_mask * dX).sum(dim=1) / denom

    # Separation: repel from close neighbors
    v2 = -(inner_mask * dX).sum(dim=1)

    # Alignment: steer towards neighbors' velocities
    dV = V[None, :, :] - V[:, None, :]  # (N,N,d) = V_j - V_i
    v3 = (outer_mask * dV).sum(dim=1) / denom

    # Boundary force: pushes back if coordinate exceeds boundary
    v4 = -torch.sign(X) * (torch.abs(X) > float(boundary)).to(X.dtype)

    return (
            float(w_cohesion) * v1
            + float(w_separation) * v2
            + float(w_alignment) * v3
            + float(w_boundary) * v4
    )


def _sample_gmm_positions(
        N: int,
        d: int,
        device: torch.device,
        n_components_min: int = 1,
        n_components_max: int = 5,
        mean_range: float = 3.0,
        std_min: float = 0.3,
        std_max: float = 1.5,
        custom_means: Optional[List[List[float]]] = None,
        custom_stds: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Sample positions from a Gaussian Mixture Model.

    If custom_means/custom_stds are provided, use those exact clusters.
    Otherwise, randomly generate cluster parameters.
    """
    if custom_means is not None:
        # Use provided clusters
        means = torch.tensor(custom_means, device=device, dtype=torch.float32)
        n_components = means.shape[0]
        if custom_stds is not None:
            stds = torch.tensor(custom_stds, device=device, dtype=torch.float32)
        else:
            stds = torch.ones(n_components, device=device) * std_min
    else:
        # Random clusters
        n_components = np.random.randint(n_components_min, n_components_max + 1)
        means = (torch.rand(n_components, d, device=device) * 2 - 1) * mean_range
        stds = torch.rand(n_components, device=device) * (std_max - std_min) + std_min

    # Assign particles to components (roughly equal split)
    counts = np.random.multinomial(N, [1.0 / n_components] * n_components)
    X_parts = []
    for i, count in enumerate(counts):
        if count > 0:
            cluster = torch.randn(count, d, device=device) * stds[i] + means[i]
            X_parts.append(cluster)

    X = torch.cat(X_parts, dim=0)
    return X[torch.randperm(N)]  # Shuffle to mix clusters


def generate_boids(
        *,
        N: int,
        steps: int,
        dt: float,
        d: int,
        num_p0: int,
        outer_radius: float,
        inner_radius: float,
        w_cohesion: float,
        w_separation: float,
        w_alignment: float,
        w_boundary: float,
        boundary: float,
        init_pos_std: float,
        init_vel_std: float,
        sigma: float,
        device: torch.device,
        init_mode: str = "gaussian",
        gmm_n_components_min: int = 1,
        gmm_n_components_max: int = 5,
        gmm_mean_range: float = 3.0,
        gmm_std_min: float = 0.3,
        gmm_std_max: float = 1.5,
        custom_means: Optional[List[List[float]]] = None,
        custom_stds: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Generate boids trajectories.

    Output:
      X_em_torch: (num_p0, N, steps+1, d)
      V_em_torch: (num_p0, N, steps+1, d)
      time_grid:  (steps+1,)

    Args:
        init_mode: "gaussian" for single Gaussian, "gmm" for random mixture,
                   "custom_gmm" for user-specified clusters
        custom_means: List of [x, y] cluster centers (for custom_gmm mode)
        custom_stds: List of cluster standard deviations (for custom_gmm mode)
    """
    if d != 2:
        raise ValueError("boids mode currently assumes d=2")

    all_pops_X: List[torch.Tensor] = []
    all_pops_V: List[torch.Tensor] = []  # FIX: This was missing!

    for _k in range(int(num_p0)):
        # Initialize positions
        if init_mode in ("gmm", "custom_gmm"):
            X = _sample_gmm_positions(
                N=int(N), d=int(d), device=device,
                n_components_min=int(gmm_n_components_min),
                n_components_max=int(gmm_n_components_max),
                mean_range=float(gmm_mean_range),
                std_min=float(gmm_std_min),
                std_max=float(gmm_std_max),
                custom_means=custom_means,
                custom_stds=custom_stds,
            )
        else:  # gaussian
            X = torch.randn(int(N), int(d), device=device) * float(init_pos_std)

        # Initialize velocities
        V = torch.randn(int(N), int(d), device=device) * float(init_vel_std)
        V = V - V.mean(dim=0, keepdim=True)  # Zero mean
        V = V / (V.norm(dim=1, keepdim=True).clamp_min(1e-12))  # Unit norm

        # Allocate trajectory storage
        X_traj = torch.empty((int(N), int(steps) + 1, int(d)), device=device, dtype=torch.float32)
        V_traj = torch.empty((int(N), int(steps) + 1, int(d)), device=device, dtype=torch.float32)

        X_traj[:, 0, :] = X.to(torch.float32)
        V_traj[:, 0, :] = V.to(torch.float32)

        # Simulate
        for t in range(int(steps)):
            dV = _boids_get_dv(
                X, V,
                outer_radius=float(outer_radius),
                inner_radius=float(inner_radius),
                w_cohesion=float(w_cohesion),
                w_separation=float(w_separation),
                w_alignment=float(w_alignment),
                w_boundary=float(w_boundary),
                boundary=float(boundary),
            )
            V = V + float(dt) * dV
            V = V / (V.norm(dim=1, keepdim=True).clamp_min(1e-12))  # Renormalize

            X = X + float(dt) * V
            if float(sigma) > 0.0:
                X = X + (float(dt) ** 0.5) * float(sigma) * torch.randn_like(X)

            X_traj[:, t + 1, :] = X.to(torch.float32)
            V_traj[:, t + 1, :] = V.to(torch.float32)

        all_pops_X.append(X_traj)
        all_pops_V.append(V_traj)

    X_em_torch = torch.stack(all_pops_X, dim=0)
    V_em_torch = torch.stack(all_pops_V, dim=0)
    time_grid = torch.arange(int(steps) + 1, device=device, dtype=torch.float32) * float(dt)

    meta = {
        "mode": "boids",
        "N": int(N),
        "steps": int(steps),
        "dt": float(dt),
        "d": int(d),
        "num_p0": int(num_p0),
        "outer_radius": float(outer_radius),
        "inner_radius": float(inner_radius),
        "w_cohesion": float(w_cohesion),
        "w_separation": float(w_separation),
        "w_alignment": float(w_alignment),
        "w_boundary": float(w_boundary),
        "boundary": float(boundary),
        "init_pos_std": float(init_pos_std),
        "init_vel_std": float(init_vel_std),
        "sigma": float(sigma),
        "has_vel": True,
        "vel_kind": "boids_sim",
        "init_mode": str(init_mode),
        "gmm_n_components_min": int(gmm_n_components_min),
        "gmm_n_components_max": int(gmm_n_components_max),
        "gmm_mean_range": float(gmm_mean_range),
        "gmm_std_min": float(gmm_std_min),
        "gmm_std_max": float(gmm_std_max),
    }
    return X_em_torch, V_em_torch, time_grid, meta


def add_boids_parser(subparsers) -> argparse.ArgumentParser:
    """Add boids/flocking subparser."""
    pbo = subparsers.add_parser("boids", help="Boids / flocking (from boids.ipynb)")
    pbo.add_argument("--N", type=int, required=False, default=None)
    pbo.add_argument("--steps", type=int, required=False, default=None)
    pbo.add_argument("--dt", type=float, required=False, default=None)
    pbo.add_argument("--d", type=int, default=2)
    pbo.add_argument("--num-p0", type=int, default=1)

    pbo.add_argument("--outer-radius", type=float, default=1.0)
    pbo.add_argument("--inner-radius", type=float, default=0.3)

    pbo.add_argument("--w-cohesion", type=float, default=0.005)
    pbo.add_argument("--w-separation", type=float, default=0.1)
    pbo.add_argument("--w-alignment", type=float, default=0.3)
    pbo.add_argument("--w-boundary", type=float, default=0.5)

    pbo.add_argument("--boundary", type=float, default=5.0)
    pbo.add_argument("--init-pos-std", type=float, default=1.0)
    pbo.add_argument("--init-vel-std", type=float, default=1.0)
    pbo.add_argument("--sigma", type=float, default=0.0,
                     help="Optional isotropic positional noise")

    pbo.add_argument("--vel-mode", type=str, default="bundle",
                     choices=["bundle"],
                     help="Boids always uses bundle velocities from simulation")

    # GMM initialization
    pbo.add_argument("--init-mode", type=str, default="gaussian",
                     choices=["gaussian", "gmm", "custom_gmm"],
                     help="Position initialization mode")
    pbo.add_argument("--gmm-n-components-min", type=int, default=1)
    pbo.add_argument("--gmm-n-components-max", type=int, default=5)
    pbo.add_argument("--gmm-mean-range", type=float, default=3.0)
    pbo.add_argument("--gmm-std-min", type=float, default=0.3)
    pbo.add_argument("--gmm-std-max", type=float, default=1.5)

    return pbo
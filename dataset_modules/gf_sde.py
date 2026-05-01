import torch
from typing import Dict, Any, Optional, Tuple, List, Callable
from mechanics import euler_maruyama_generalized
from functions import get_potential_grad_as_torch
import argparse
import torch.nn as nn
import torch.optim as optim
import numpy as np

# score estimation helpers for computing initial velocity (under conservative system)


def compute_score_blob_adaptive(x):
    """
    Approximates ∇ log ρ(x) using a Gaussian kernel with Adaptive Bandwidth.
    Prevents collapse by scaling sigma with the particle spread.
    """
    N, d = x.shape
    device = x.device

    # 1. Compute pairwise distances (N, N)
    # Note: For N > 5000 this might be memory heavy; usually fine for N=2000
    x_diff = x.unsqueeze(1) - x.unsqueeze(0)  # (N, N, d)
    dist_sq = (x_diff ** 2).sum(dim=-1)  # (N, N)

    # 2. Heuristic: Median Distance
    # Add large val to diagonal to ignore self-distance in min/median
    off_diag_dist = dist_sq + torch.eye(N, device=device) * 1e6
    median_dist_sq = torch.median(off_diag_dist)

    # Bandwidth rule of thumb (Silverman-like or simple median)
    # h^2 = median_dist / (2 * log(N)) is a common heuristic for stability
    h_sq = median_dist_sq / (2.0 * np.log(N + 1))

    # Safety: Clamp bandwidth to avoid division by zero if particles collapse perfectly
    h_sq = torch.clamp(h_sq, min=1e-3)

    # 3. Compute Kernel and Gradient
    # K(x,y) = exp( - ||x-y||^2 / h^2 )
    K_matrix = torch.exp(-dist_sq / h_sq)  # (N, N)

    # Gradient of Kernel Sum: ∇_x Σ_y K(x,y)
    # ∇ K = - 2(x-y)/h^2 * K
    grad_K_sum = - (2.0 / h_sq) * torch.einsum('ijd,ij->id', x_diff, K_matrix)  # (N, d)

    sum_K = K_matrix.sum(dim=1, keepdim=True)  # (N, 1)

    # Score = (Σ ∇ K) / (Σ K)
    score = grad_K_sum / (sum_K + 1e-10)

    return score


# === 2. NEURAL SCORE MATCHING IMPLEMENTATION ===

class ScoreNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),  # MUST be smooth (twice differentiable), ReLU won't work!
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)


def score_matching_loss(model, x):
    """
    Hyvärinen Score Matching Loss:
    L = E[ 1/2 ||s(x)||^2 + div(s(x)) ]
    """
    x.requires_grad_(True)
    score_pred = model(x)

    # 1. Norm squared term
    norm_sq = 0.5 * torch.sum(score_pred ** 2, dim=1)

    # 2. Divergence term (Trace of Jacobian)
    # For low dimensions (d=2), we can compute exact trace efficiently.
    divergence = 0.0
    for i in range(x.shape[1]):
        # Grad of i-th output w.r.t i-th input
        grad_i = torch.autograd.grad(
            outputs=score_pred[:, i].sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0][:, i]
        divergence += grad_i

    loss = (norm_sq + divergence).mean()
    return loss


class NeuralScoreEstimator:
    """
    Stateful wrapper for the Neural Score Network to manage model and optimizer.
    """

    def __init__(self, input_dim=2, hidden_dim=64, lr=1e-3, device='cpu'):
        self.device = device
        self.model = ScoreNetwork(input_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self, x_batch, steps=5):
        """Updates the score network on the current particle distribution."""
        self.model.train()
        # We freeze the particles for the training step (detach)
        x_input = x_batch.detach().clone()

        loss_val = 0.0
        for _ in range(steps):
            self.optimizer.zero_grad()
            loss = score_matching_loss(self.model, x_input)
            loss.backward()
            self.optimizer.step()
            loss_val = loss.item()
        return loss_val

    def compute_score(self, x):
        """Inference wrapper."""
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

# ============================================================
# Gradient-flow SDE generator
# ============================================================
def random_centers_uniform_box(
    num_p0: int,
    d: int,
    device: torch.device,
    center_range: float,
) -> torch.Tensor:
    """Centers uniform in [-center_range, center_range]^d."""
    return (2.0 * torch.rand((num_p0, d), device=device) - 1.0) * float(center_range)

def gaussian_blob(
    N: int,
    d: int,
    device: torch.device,
    mean: torch.Tensor,
    var: float,
) -> torch.Tensor:
    """X ~ N(mean, var * I_d). RNG controlled by set_seed(args.seed)."""
    mean = mean.to(device=device, dtype=torch.float32).view(1, d)
    return mean + (var ** 0.5) * torch.randn((N, d), device=device)

def generate_potential_sde(
    *,
    N: int,
    steps: int,
    dt: float,
    d: int,
    num_p0: int,
    pot: str,
    sigma: float,
    p0: str,
    p0_mean: Optional[List[float]],
    p0_var: float,
    center_uniform: bool,
    center_range: float,
    kill_condition: bool,
    device: torch.device,
    score_method: str = "kernel",
    score_hidden: int = 64
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Dict[str, Any]]:
    """
    Ground truth SDE:
      dX_t = -∇ψ(X_t, t) dt + sigma dW_t.
    """
    grad_psi = get_potential_grad_as_torch(pot)

    def drift_sde(x: torch.Tensor, t: float) -> torch.Tensor:
        return -grad_psi(x, t)

    if p0 != "gaussian":
        raise ValueError(f"Unknown p0={p0}. Currently supported: gaussian")

    if p0_mean is None:
        base_mean = torch.zeros(d, device=device)
    else:
        if len(p0_mean) != d:
            raise ValueError(f"--p0-mean must have length d={d}.")
        base_mean = torch.tensor(p0_mean, device=device, dtype=torch.float32)

    if center_uniform:
        centers = random_centers_uniform_box(num_p0=num_p0, d=d, device=device, center_range=center_range)
    else:
        centers = base_mean.view(1, d).repeat(num_p0, 1)

    all_pops: List[torch.Tensor] = []
    for i in range(num_p0):
        x0 = gaussian_blob(N=N, d=d, device=device, mean=centers[i], var=float(p0_var))
        X_pop = euler_maruyama_generalized(
            x0,
            drift_sde,
            sigma=float(sigma),
            dt=float(dt),
            steps=int(steps),
            kill_condition=kill_condition
        )  # (N, steps+1, d)
        all_pops.append(X_pop)

    X_em_torch = torch.stack(all_pops, dim=0)  # (num_p0, N, steps+1, d)
    time_grid = torch.arange(steps + 1, device=device, dtype=torch.float32) * float(dt)

    # Compute velocities
    all_vels = []
    for i in range(num_p0):
        V_pop = compute_gradient_flow_velocity(
            X_traj=X_em_torch[i],
            time_grid=time_grid,
            sigma=float(sigma),
            grad_psi_func=grad_psi,
            score_method=score_method,
            score_hidden=score_hidden,
            device=device
        )
        all_vels.append(V_pop)

    V_em_torch = torch.stack(all_vels, dim=0)  # (num_p0, N, steps+1, 2)

    meta = {
        "mode": "potential_sde",
        "N": N,
        "steps": steps,
        "dt": dt,
        "d": d,
        "num_p0": num_p0,
        "pot": pot,
        "sigma": sigma,
        "p0": p0,
        "p0_mean": centers.detach().to("cpu").tolist(),
        "p0_var": p0_var,
        "center_uniform": center_uniform,
        "center_range": center_range,
        "has_vel": V_em_torch,
    }
    return X_em_torch, V_em_torch, time_grid, meta


def compute_gradient_flow_velocity(
        X_traj: torch.Tensor,  # (N, T+1, d)
        time_grid: torch.Tensor,  # (T+1,)
        sigma: float,
        grad_psi_func: Callable[[torch.Tensor, float], torch.Tensor],
        score_method: str = "kernel",
        score_hidden: int = 64,
        score_train_steps: int = 5,
        device: torch.device = None,
) -> torch.Tensor:
    """
    Compute velocity from gradient flow of free energy.

    Free energy: F[p] = ∫ Ψ(x) p(x)dx + (σ²/2) ∫ p(x) log p(x) dx

    Velocity: v(x,t) = -∇Ψ(x,t) + (σ²/2) ∇log p(x,t)
                     = -∇Ψ(x,t) + (σ²/2) score(x,t)

    Args:
        X_traj: Particle trajectories (N, T+1, d)
        time_grid: Time values (T+1,)
        sigma: Diffusion coefficient
        grad_psi_func: Gradient of potential Ψ, signature (x, t) -> grad
        score_method: "kernel" or "neural"
        score_hidden: Hidden dim for neural estimator
        score_train_steps: Training steps per timestep for neural
        device: torch device

    Returns:
        V_traj: Velocity trajectories (N, T+1, d)
    """
    if device is None:
        device = X_traj.device

    N, T_plus_1, d = X_traj.shape
    V_traj = torch.zeros_like(X_traj)

    # Initialize neural estimator if needed
    neural_estimator = None
    if score_method == "neural":
        neural_estimator = NeuralScoreEstimator(
            input_dim=d,
            hidden_dim=score_hidden,
            lr=1e-3,
            device=device
        )

    for t_idx in range(T_plus_1):
        x = X_traj[:, t_idx, :]  # (N, d)
        t = float(time_grid[t_idx].item())

        # 1. Potential force: -∇Ψ(x,t)
        g_psi = grad_psi_func(x, t)

        # 2. Score (entropic force): ∇log p(x,t)
        if score_method == "neural":
            # Online training
            for _ in range(score_train_steps):
                neural_estimator.train_step(x, steps=1)
            score = neural_estimator.compute_score(x)
        elif score_method == "kernel":
            score = compute_score_blob_adaptive(x)
        else:
            raise ValueError(f"Unknown score_method: {score_method}")

        # Total drift: -∇Ψ + (σ²/2) * score
        V_traj[:, t_idx, :] = -g_psi + (sigma ** 2 / 2.0) * score

    return V_traj

def add_potential_sde_parser(subparsers) -> argparse.ArgumentParser:
    """Add potential-driven SDE subparser."""
    pp = subparsers.add_parser("potential_sde", help="Potential-driven SDE dataset")
    pp.add_argument("--N", type=int, required=False, default=None)
    pp.add_argument("--steps", type=int, required=False, default=None)
    pp.add_argument("--dt", type=float, required=False, default=None)
    pp.add_argument("--d", type=int, default=2)
    pp.add_argument("--num-p0", type=int, default=1)

    pp.add_argument("--kill-condition", action="store_true", default=False)

    pp.add_argument("--pot", type=str, required=False, default=None,
                    help="Potential name for get_potential_grad_as_torch")
    pp.add_argument("--sigma", type=float, required=False, default=None)

    pp.add_argument("--p0", type=str, default="gaussian", choices=["gaussian"])
    pp.add_argument("--p0-mean", type=float, nargs="*", default=None,
                    help="d floats; if omitted => zero")
    pp.add_argument("--p0-var", type=float, default=0.1)

    pp.add_argument("--center-uniform", action="store_true",
                    help="If set, each population center ~ Unif([-R,R]^d).")
    pp.add_argument("--center-range", type=float, default=4.0)

    # Velocity mode
    pp.add_argument("--vel-mode", type=str, default="gradient_flow",
                    choices=["gradient_flow", "zero"],
                    help="Velocity computation mode")

    # Gradient flow specific
    pp.add_argument("--score-method", type=str, default="kernel",
                    choices=["kernel", "neural"],
                    help="Score estimation method for gradient flow")
    pp.add_argument("--score-hidden", type=int, default=64,
                    help="Hidden dim for neural score estimator")
    pp.add_argument("--score-train-steps", type=int, default=5,
                    help="Training steps per timestep for neural score")

    return pp
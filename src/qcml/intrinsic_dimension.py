import torch
from torch import Tensor
from torch.linalg import eigh
from typing import List
from .model import get_eigenstates, QuantumCognitionModel


def quantum_metric(A: List[Tensor], D: int, x: Tensor) -> Tensor:
    """
    Calculates Quantum Metric

    Args :
        A : Matrix configuration
        D : Dimention of dataset
        x : Batch of points

    Returns :
        Quantum metric with shape batch * D * D
    """
    N = A[0].shape[0]
    g = torch.zeros((x.shape[0], D, D))
    eigenstates, eigenvalues = get_eigenstates(A, x, return_ev=True)
    psi0 = eigenstates[0, :, :, :]
    e0 = eigenvalues[0, :]
    for mu in range(D):
        for nu in range(D):
            res = []
            for n in range(1, N):
                psin = eigenstates[n, :, :, :]
                en = eigenvalues[n, :]
                res.append(
                    (
                        (psi0.transpose(1, 2).conj() @ A[mu] @ psin).squeeze()
                        * (psin.transpose(1, 2).conj() @ A[nu] @ psi0).squeeze()
                        / (en - e0)
                    ).real
                )
            g[:, mu, nu] = 2 * sum(res)
    return g


def intrinsic_dimension(model: QuantumCognitionModel, X_A: Tensor) -> float:
    """
    Calculates intrinsic dimension of dataset X_A

    Args :
        model : Trained quantum cognition model
        X_A : Point cloud. Projection of X on guessed manifold.

    Returns :
        Average of intrinsic dimension calculated on each point
    """
    g_X_A = quantum_metric(model.A, X_A.shape[1], X_A)
    eigenvalues, _ = eigh(g_X_A)
    gaps = eigenvalues / eigenvalues.roll(1, 1)
    gaps[:, 0] = -1
    d = X_A.shape[1] - torch.argmax(gaps, dim=1)

    return torch.mean(d, dtype=float).item()

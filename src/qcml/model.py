import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Linear
from torch.linalg import eigh
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from typing import Tuple, List, Dict, Union
from tqdm.auto import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_eigenstates(
    A: List[Tensor], x: Tensor, return_ev: bool = False
) -> Union[Tensor, List[Tensor]]:
    """
    Calculation of eigenstates and eigenvectors of error Hamiltonian H

    Args:
        A : Matrix configuration
        x : Batch of points
        return_ev : Should return eigenvalues ?

    Returns:
        Eigenstates for each data point (shape : N * batch * N * 1, if N * N is the
        shape of A) and optionnaly the associated eigenvalues (shape : N * batch)
    """

    H = 0.5 * sum(
        [
            (
                diff_k := Ak.unsqueeze(0)
                - torch.stack(
                    [
                        x[i, k] * torch.eye(Ak.shape[0], device=x.device)
                        for i in range(x.shape[0])
                    ]
                )
            )
            @ diff_k
            for k, Ak in enumerate(A)
        ]
    )
    eigenvalues, eigenstates = eigh(H)
    eigenvalues = eigenvalues.transpose(0, 1)
    eigenstates = (
        (torch.exp(-1j * torch.angle(eigenstates[:, 0, :])).unsqueeze(1) * eigenstates)
        .permute((2, 0, 1))
        .unsqueeze(3)
    )

    if return_ev:
        return eigenstates, eigenvalues
    return eigenstates


class QuantumCognitionModel(Module):
    """
    A class to train a matrix configuration A on a data set X with Quantum Cognition
    Machine Learning

    Attributes:
        A : Matrix configuration
        w : Quantum fluctuation weight

    """

    def __init__(self, D: int, N: int, w: float):
        """
        Args:
            D : Dimension of dataset
            N : Dimension of the Hilbert space
            w : Quantum fluctuation weight
        """
        super().__init__()
        self.D = D
        self.N = N
        self.w = w
        self.B = ModuleList(
            [Linear(N, N, bias=False, dtype=torch.cfloat) for _ in range(D)]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        """Forward pass

        Args:
            x : a batch of data points

        Returns:
            state positions and weighted variances
        """
        # Make hermitian matrix from randomly initialized weights
        self.A = [(layer.weight + layer.weight.conj().T) / 2 for layer in self.B]

        psi0 = get_eigenstates(self.A, x)[0, :, :, :]
        psi0HAks = [psi0.transpose(1, 2).conj() @ Ak for Ak in self.A]

        pos = torch.stack([(psi0HAk @ psi0).squeeze() for psi0HAk in psi0HAks], dim=1)
        wvar = self.w * torch.sum(
            torch.stack(
                [
                    (psi0HAk @ Ak @ psi0).squeeze()
                    for Ak, psi0HAk in zip(self.A, psi0HAks)
                ],
                dim=1,
            )
            - pos**2,
            dim=1,
        )

        return pos, wvar


def train(
    model: QuantumCognitionModel, X: Tensor, config: Dict
) -> QuantumCognitionModel:
    """
    Train model on dataset X

    Args :
        model : Quantum Cognition Model
        X : Dataset of points of shape batch * D, with D the dimension of the dataset

    Returns :
        Trained model
    """
    lr, n_epochs = config["lr"], config["n_epochs"]
    train_dataloader = DataLoader(
        X,
        shuffle=True,
        batch_size=32,
    )

    optimizer = AdamW(model.parameters(), lr=lr)

    num_training_steps = n_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    print(num_training_steps)

    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for _ in range(n_epochs):
        for batch in train_dataloader:
            batch = batch.to(device)
            positions, _ = model(batch)
            loss = torch.mean(torch.linalg.vector_norm(positions - batch, dim=1))
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            print(loss)
            progress_bar.update(1)

    return model

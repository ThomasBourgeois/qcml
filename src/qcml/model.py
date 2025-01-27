import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Linear
from torch.linalg import eigh
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from typing import Tuple
from tqdm.auto import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class QuantumCognitionModel(Module):
    """
    A class to train a matrix configuration A on a data set X with Quantum Cognition
    Machine Learning

    Attributes:
        A : matrix configuration
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
        A = [(layer.weight + layer.weight.conj().T) / 2 for layer in self.B]

        def diff(k, Ak):
            diff_k = Ak.unsqueeze(0) - torch.stack(
                [
                    x[i, k] * torch.eye(self.N, device=x.device)
                    for i in range(x.shape[0])
                ]
            )
            return diff_k

        H = 0.5 * sum(
            [diff_k @ diff_k for k, Ak in enumerate(A) if (diff_k := diff(k, Ak)).any()]
        )
        _, eigenvectors = eigh(H)
        phi0 = eigenvectors[:, :, 0]
        phi0 = (torch.exp(-1j * torch.angle(phi0[:, 0])).unsqueeze(1) * phi0).unsqueeze(
            2
        )
        phi0HAks = [phi0.transpose(1, 2).conj() @ Ak for Ak in A]

        pos = torch.stack([(phi0HAk @ phi0).squeeze() for phi0HAk in phi0HAks], dim=1)
        wvar = self.w * torch.sum(
            torch.stack(
                [(phi0HAk @ Ak @ phi0).squeeze() for Ak, phi0HAk in zip(A, phi0HAks)],
                dim=1,
            )
            - pos**2,
            dim=1,
        )

        return pos, wvar


def train(model, X):
    train_dataloader = DataLoader(
        X,
        shuffle=True,
        batch_size=32,
    )

    optimizer = AdamW(model.parameters(), lr=1e-3)

    num_epochs = 50
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    print(num_training_steps)

    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for _ in range(num_epochs):
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

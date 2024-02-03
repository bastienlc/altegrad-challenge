import torch
from torch.nn import CrossEntropyLoss, Module, Parameter
from torch.utils.data import Dataset

from .similarities import load_similarities


class MeanFuseWeights(Module):
    def __init__(self, n_models: int):
        super(MeanFuseWeights, self).__init__()
        self.weights = Parameter(torch.ones(n_models) / n_models)

    def forward(self, similarities: torch.Tensor):
        similarities = similarities - similarities.min(dim=1).values[:, :, None]
        similarities = similarities / similarities.max(dim=1).values[:, :, None]

        return torch.sum(similarities * self.weights[:, None, None], dim=0)


class SimilaritiesDataset(Dataset):
    def __init__(self, split: str, n_models: int):
        self.split = split
        self.n_models = n_models
        self.similarities = torch.Tensor(load_similarities(self.split, self.n_models))
        self.n_samples = self.similarities.shape[1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.similarities[:, idx, :]


def train(
    model: Module,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epochs: int,
):
    model = model.to(device)
    CEL = CrossEntropyLoss()
    for e in range(1, epochs + 1):
        batch = dataset.similarities
        batch = batch.to(device)
        output = model(batch)
        loss = CEL(output, torch.arange(output.shape[0], device=output.device)) + CEL(
            output.T, torch.arange(output.shape[0], device=output.device)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if e % 10 == 0:
            print(
                f"Epoch {e}/{epochs} - Loss: {loss.item() / len(batch):.4f}", end="\r"
            )

    print()
    return model.weights.detach().cpu().numpy()

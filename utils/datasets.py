import torch
from torch.utils.data import Dataset


class DummyPositiveNegativeDataset(Dataset):
    def __init__(self, num_samples: int, threshold: float = 0.5) -> None:
        super().__init__()

        self.num_samples = num_samples
        self.num_features = 2
        self.threshold = threshold

        is_positive = torch.randn((num_samples,)) < threshold
        targets = is_positive.float()
        positive_masks = targets.unsqueeze(dim=-1)
        negative_masks = 1 - positive_masks
        targets = 2 * targets - 1

        positive_samples = 0.2 * torch.randn((num_samples, self.num_features)) - 0.5
        negative_samples = 0.5 * torch.randn((num_samples, self.num_features)) + 0.5

        self.inputs = (
            positive_masks * positive_samples + negative_masks * negative_samples
        )
        self.targets = targets.long()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input = self.inputs[index]
        target = self.targets[index]

        return input, target

    def __len__(self) -> int:
        return self.num_samples

import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, inputs: torch.Tensor) -> None:
        super().__init__()

        num_gaussians = inputs.size(0)
        self.std = 1

        weights = torch.empty((num_gaussians,))
        self.weights = nn.Parameter(weights, requires_grad=True)
        self.register_buffer("inputs", inputs.data.clone())

        self.weights.data.normal_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of classifier.

        Args:
            input (torch.Tensor): Batched features of shape (batch_size, 2).

        """
        std = self.std
        weights = self.weights
        inputs = self.inputs

        dot_prod = torch.matmul(input, inputs.transpose(0, 1))
        norm_input = torch.sum(input**2, dim=-1, keepdim=True)
        norm_inputs = torch.sum(inputs**2, dim=-1)
        distance = norm_input + norm_inputs - 2 * dot_prod
        output = weights * torch.exp(-distance / (2 * std**2))
        output = output.sum(dim=-1)

        return output

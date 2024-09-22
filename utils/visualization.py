import matplotlib.pyplot as plt
import torch

from utils.datasets import DummyPositiveNegativeDataset


def plot_distribution(
    labeled_dataset: DummyPositiveNegativeDataset,
    unlabeled_dataset: DummyPositiveNegativeDataset | None = None,
    path: str | None = None,
) -> None:
    labeled_positive_inputs = []
    labeled_negative_inputs = []

    for labeled_input, labeled_target in labeled_dataset:
        if labeled_target.item() == 1:
            labeled_positive_inputs.append(labeled_input)
        else:
            labeled_negative_inputs.append(labeled_input)

    labeled_positive_inputs = torch.stack(labeled_positive_inputs, dim=0)
    labeled_negative_inputs = torch.stack(labeled_negative_inputs, dim=0)

    if unlabeled_dataset is None:
        unlabeled_inputs = None
    else:
        unlabeled_inputs = []

        for unlabeled_input, _ in unlabeled_dataset:
            unlabeled_inputs.append(unlabeled_input)

        unlabeled_inputs = torch.stack(unlabeled_inputs, dim=0)

    fig, axis = plt.subplots(figsize=(8, 8))

    if unlabeled_inputs is not None:
        # plot unlabeld inputs at first
        axis.scatter(
            unlabeled_inputs[:, 0],
            unlabeled_inputs[:, 1],
            s=50,
            alpha=0.5,
            color="black",
            marker="s",
            edgecolors="black",
        )

    axis.scatter(
        labeled_positive_inputs[:, 0],
        labeled_positive_inputs[:, 1],
        s=50,
        alpha=0.5,
        color="red",
        marker="o",
        edgecolors="red",
    )
    axis.scatter(
        labeled_negative_inputs[:, 0],
        labeled_negative_inputs[:, 1],
        s=50,
        alpha=0.5,
        color="blue",
        marker="^",
        edgecolors="blue",
    )
    fig.savefig(path, bbox_inches="tight")

    plt.close()


def plot_loss(loss: list[float], path: str | None = None) -> None:
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.plot(loss)
    fig.savefig(path, bbox_inches="tight")

    plt.close()

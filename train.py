import os
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from utils.criterion import PNURisk, SquaredLoss
from utils.datasets import DummyPositiveNegativeDataset
from utils.models import Classifier
from utils.visualization import plot_distribution, plot_loss


def main() -> None:
    setup()

    args = parse_args()

    loss_dir = "figures/loss"
    accuracy_dir = "figures/accuracy"
    distribution_dir = "figures/distribution"

    positive_rate = args.positive_rate
    num_labeled_training = 100
    num_unlabeled_training = args.num_unlabeled_training
    num_labeled_validation = 50
    eta = args.eta
    epochs = args.epochs

    labeled_training_dataset = DummyPositiveNegativeDataset(
        num_labeled_training, threshold=positive_rate
    )
    unlabeled_training_dataset = DummyPositiveNegativeDataset(
        num_unlabeled_training, threshold=positive_rate
    )
    labeled_validation_dataset = DummyPositiveNegativeDataset(
        num_labeled_validation, threshold=positive_rate
    )

    os.makedirs(distribution_dir, exist_ok=True)

    plot_distribution(
        labeled_training_dataset,
        unlabeled_dataset=unlabeled_training_dataset,
        path=f"{distribution_dir}/positive{positive_rate}.png",
    )

    # dataset and dataloader
    labeled_training_dataloader = DataLoader(
        labeled_training_dataset, batch_size=10, shuffle=True
    )
    unlabeled_training_dataloader = DataLoader(
        unlabeled_training_dataset, batch_size=100, shuffle=True
    )
    labeled_validation_dataloader = DataLoader(
        labeled_validation_dataset, batch_size=100, shuffle=True
    )

    # model
    inputs = []

    for input, _ in labeled_training_dataset:
        inputs.append(input)

    for input, _ in unlabeled_training_dataset:
        inputs.append(input)

    inputs = torch.stack(inputs)

    classifier = Classifier(inputs)

    # optimizer
    optimizer = SGD(classifier.parameters(), weight_decay=1)

    # criterion
    margin_loss = SquaredLoss()
    criterion = PNURisk(margin_loss, positive_rate=positive_rate, eta=eta)

    training_loss = []
    validation_loss = []
    validation_accuracy = []

    for epoch in range(epochs):
        classifier.train()
        criterion.train()

        for (labeled_input, labeled_target), (unlabeled_input, _) in zip(
            labeled_training_dataloader, unlabeled_training_dataloader
        ):
            labeled_output = classifier(labeled_input)
            unlabeled_output = classifier(unlabeled_input)
            loss = criterion(labeled_output, unlabeled_output, labeled_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss.append(loss.item())

        loss = 0
        accuracy = 0

        classifier.eval()
        criterion.eval()

        for labeled_input, labeled_target in labeled_validation_dataloader:
            with torch.inference_mode():
                labeled_output = classifier(labeled_input)

                labeled_positive_output = labeled_output[labeled_target == 1]
                labeled_negative_output = labeled_output[labeled_target == -1]

                # PN risk
                if labeled_positive_output.size(0) == 0:
                    risk_p = 0
                else:
                    loss_p = margin_loss(labeled_positive_output)
                    risk_p = loss_p.mean()

                if labeled_negative_output.size(0) == 0:
                    risk_n = 0
                else:
                    loss_n = margin_loss(labeled_negative_output)
                    risk_n = loss_n.mean()

                risk_pn = positive_rate * risk_p + (1 - positive_rate) * risk_n
                loss += risk_pn.item()

                num_correct = torch.sum(labeled_output * labeled_target > 0)
                accuracy += num_correct.item() / labeled_output.size(0)

        validation_loss.append(loss / len(labeled_validation_dataloader))
        validation_accuracy.append(accuracy / len(labeled_validation_dataloader))

    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(accuracy_dir, exist_ok=True)

    plot_loss(training_loss, path=f"{loss_dir}/eta{eta:.2f}_training.png")
    plot_loss(validation_loss, path=f"{loss_dir}/eta{eta:.2f}_validation.png")
    plot_loss(validation_accuracy, path=f"{accuracy_dir}/eta{eta:.2f}_validation.png")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train Model.")

    parser.add_argument(
        "--num-unlabeled-training",
        type=int,
        default=1000,
        help="Number of unlabeled samples for training.",
    )
    parser.add_argument(
        "--positive-rate",
        type=float,
        default=0.5,
        help="Rate of positive samples, i.e., p(t=+1).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0,
        help="Trade off parameter between PN risk and PU risk (NU risk).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs for training.",
    )

    return parser.parse_args()


def setup() -> None:
    plt.rcParams["font.size"] = 15

    torch.manual_seed(0)


if __name__ == "__main__":
    main()

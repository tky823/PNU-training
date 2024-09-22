import torch
import torch.nn as nn


class SquaredLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = (1 - input) ** 2

        return output


class PNURisk(nn.Module):
    def __init__(
        self, criterion: nn.Module, positive_rate: float, eta: float = 0
    ) -> None:
        super().__init__()

        self.criterion = criterion
        self.positive_rate = positive_rate
        self.eta = eta

    def forward(
        self,
        labeled_input: torch.Tensor,
        unlabeled_input: torch.Tensor,
        labeled_target: torch.Tensor,
    ) -> torch.Tensor:
        positive_rate = self.positive_rate
        eta = self.eta

        labeled_positive_output = labeled_input[labeled_target == 1]
        labeled_negative_output = labeled_input[labeled_target == -1]

        # PN risk
        if labeled_positive_output.size(0) == 0:
            risk_p = 0
        else:
            loss_p = self.criterion(labeled_positive_output)
            risk_p = loss_p.mean()

        if labeled_negative_output.size(0) == 0:
            risk_n = 0
        else:
            loss_n = self.criterion(-labeled_negative_output)
            risk_n = loss_n.mean()

        risk_pn = positive_rate * risk_p + (1 - positive_rate) * risk_n

        if eta > 0:
            # PU risk
            if labeled_positive_output.size(0) == 0:
                risk_p_c = 0
            else:
                loss_p_c = self.criterion(labeled_positive_output) - self.criterion(
                    -labeled_positive_output
                )
                risk_p_c = loss_p_c.mean()

            loss_nu = self.criterion(-unlabeled_input)
            risk_nu = loss_nu.mean()
            risk_pu = positive_rate * risk_p_c + risk_nu
            output = (1 - eta) * risk_pn + eta * risk_pu
        elif eta < 0:
            # NU risk
            if labeled_negative_output.size(0) == 0:
                risk_n_c = 0
            else:
                loss_n_c = self.criterion(labeled_positive_output) - self.criterion(
                    -labeled_positive_output
                )
                risk_n_c = loss_n_c.mean()

            loss_pu = self.criterion(unlabeled_input)
            risk_pu = loss_pu.mean()
            risk_nu = (1 - positive_rate) * risk_n_c + risk_pu
            output = (1 + eta) * risk_pn - eta * risk_pu
        else:
            output = risk_pn

        return output

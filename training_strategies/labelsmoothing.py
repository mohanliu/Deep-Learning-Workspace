"""Source:
https://github.com/rwightman/pytorch-image-models/blob/b669f4a5881d17fe3875656ec40138c1ef50c4c9/timm/loss/cross_entropy.py#L11

Original Paper: 
    Rethinking the Inception Architecture for Computer Vision (Inception V3)
    https://arxiv.org/abs/1512.00567
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.

    NLL:
        - negative log likelihood
        - NLL Loss = - \sum_{K} log(p(x)) x q(x)
        - p(x) is prediction and q(x) is ground truth
        - Multi-class classification (with softmax activation function)
        - considers only the output for the corresponding class

    CE:
        - cross entropy
        - CE loss = - \sum_{K} (log(p(x)) x q(x) + log(1 - p(x)) x (1 - q(x)))
        - Binary classification (with sigmoid activation function)
        - considers the other outputs as well
    """

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing  # LS
        self.confidence = 1.0 - smoothing  # 1 - LS

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Basic NLL Loss function:

            NLL(p(x), q(x)) = - \sum_{K} log(p(x)) x q(x),

            where, p(x) is prediction and q(x) is ground truth

        Label Smoothing:

            q'(x) = (1 - LS) x q(x) + LS/K

        NLL-CE:

            NLL(p(x), q'(x)) = - \sum_{K} log(p(x)) x q'(x),
                           = (1 - LS) * \sum_{K} (- log(p(x)) x q(x)) + LS/K * \sum_{K} (- log(p(x)))
                           = (1 - LS) * NLL(p(x), q(x)) + LS * \sum_{K} (- log(p(x))) / K
                           = (1 - LS) * NLL-loss + LS * Smoothed-lss
        """
        # get softmax and then calculate logrithm: log(p(x))
        logprobs = F.log_softmax(x, dim=-1)

        # get NLL loss
        nll_loss = -logprobs.gather(
            dim=-1,  # gather data at axis=1 (axis=0 is sample axis)
            index=target.unsqueeze(1),  # gather only the corresponding label (NLL loss)
        )
        nll_loss = nll_loss.squeeze(1)  # reshape to squeezed shape

        # get smoothed loss (class average of log-softmax probability)
        smooth_loss = -logprobs.mean(dim=-1)

        # calculate the final NLL loss with label smoothing
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        return loss.mean()  # return sample mean


if __name__ == "__main__":
    LSLoss = LabelSmoothingCrossEntropy()

    pred = torch.tensor(
        [
            [0.6, 0.5, 0.8, 0.3, 0.1],
            [0.1, 0.9, 0.4, 0.2, 0.2],
            [0.1, 0.3, 0.4, 0.2, 0.9],
        ]
    )  # raw predictions of each sample (across all 5 classes)

    target_gt = torch.tensor([2, 1, 4])  # targets with class id

    print(LSLoss(pred, target_gt))

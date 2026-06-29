import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = True,
        from_logits: bool = True,
        squared_pred: bool = False,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        reduction: str = "mean",
    ):
        """
        Dice Loss.

        Args:
            include_background:
                If False, ignores channel 0.

            to_onehot_y:
                Convert target labels (B,H,W[,D]) to one-hot.
                If True -> sigmoid is applied to predictions.
                Otherwise -> softmax is applied.

            squared_pred:
                Square predictions and targets in denominator.

            smooth_nr:
                Smoothing constant added to numerator.

            smooth_dr:
                Smoothing constant added to denominator.

            reduction:
                "mean", "sum", or "none".
        """
        super().__init__()

        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")

        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.from_logits = from_logits
        self.squared_pred = squared_pred
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds:
                (B,C,H,W) or (B,C,H,W,D)

            target:
                If to_onehot_y=True:
                    (B,1,H,W) or (B,1,H,W,D)
                Else:
                    it is already on-hot vector
                    (B,C,H,W) or (B,C,H,W,D)

        Returns:
            Dice loss.
        """
        if self.from_logits:
            probs = torch.softmax(preds, dim=1)
        else:
            probs = preds
            
        if self.to_onehot_y:
            num_classes = probs.shape[1]

            if target.ndim == probs.ndim:
                # (B,1,H,W) -> (B,H,W)
                target = target.squeeze(1)

            target = F.one_hot(
                target.long(),
                num_classes=num_classes
            )

            # Move channel dimension to dim=1
            dims = list(range(target.ndim))
            target = target.permute(0, dims[-1], *dims[1:-1]).float()

        else:
            target = target.float()

        if probs.shape != target.shape:
            raise ValueError(
                f"probs shape {probs.shape} does not match target shape {target.shape}"
            )

        if not self.include_background:
            probs = probs[:, 1:]
            target = target[:, 1:]

        # Sum over spatial dimensions only
        reduce_dims = tuple(range(2, probs.ndim))

        intersection = torch.sum(probs * target, dim=reduce_dims)

        if self.squared_pred:
            pred_sum = torch.sum(probs ** 2, dim=reduce_dims)
            target_sum = torch.sum(target ** 2, dim=reduce_dims)
        else:
            pred_sum = torch.sum(probs, dim=reduce_dims)
            target_sum = torch.sum(target, dim=reduce_dims)

        dice = (
            2.0 * intersection + self.smooth_nr
        ) / (
            pred_sum + target_sum + self.smooth_dr
        )

        loss = 1.0 - dice

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss
    

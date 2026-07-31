from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        from_logits: bool = True,
        sigmoid: bool = False,
        softmax: bool = True,
        squared_pred: bool = False,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        reduction: Literal["mean", "sum", "none"] = "mean",
        generalized: bool = False,
        class_weights: torch.Tensor | list | tuple | None = None,
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
        if sigmoid and softmax:
            raise ValueError("Only one of sigmoid or softmax can be True.")

        self.sigmoid = sigmoid
        self.softmax = softmax
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.from_logits = from_logits
        self.squared_pred = squared_pred
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.reduction = reduction
        self.generalized = generalized
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)

        if not self.include_background:
            class_weights = class_weights[1:]
            # Normalize so overall magnitude remains comparable
            class_weights = class_weights / class_weights.sum()

        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else None,
        )

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
            if self.sigmoid:
                probs = torch.sigmoid(preds)
            elif self.softmax:
                probs = torch.softmax(preds, dim=1)
            else:
                probs = preds
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
            if target.ndim == probs.ndim - 1:
                target = target.unsqueeze(1)

        if probs.shape != target.shape:
            raise ValueError(
                f"probs shape {probs.shape} does not match target shape {target.shape} - set to_one_hot_y to True"
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
        
        # Generalized Dice
        if self.generalized:
            # class volumes from ground truth
            class_volume = target_sum

            # inverse squared volume weighting
            weights = 1.0 / (class_volume.pow(2) + self.smooth_dr)

            # avoid inf if a class is absent
            weights = torch.where(
                torch.isfinite(weights),
                weights,
                torch.zeros_like(weights)
            )

            numerator = 2.0 * torch.sum(
                weights * intersection,
                dim=1
            )

            denominator = torch.sum(
                weights * (pred_sum + target_sum),
                dim=1
            )

            dice = (
                numerator + self.smooth_nr
            ) / (
                denominator + self.smooth_dr
            )

            loss = 1.0 - dice

            if self.reduction == "mean":
                return loss.mean()

            if self.reduction == "sum":
                return loss.sum()

            return loss
        
        else: # Standard Dice
            dice = (
                2.0 * intersection + self.smooth_nr
            ) / (
                pred_sum + target_sum + self.smooth_dr
            )

            loss = 1.0 - dice
            
            # Apply class weights
            if self.class_weights is not None:
                weights = self.class_weights

                if weights.numel() != loss.shape[1]:
                    raise ValueError(
                        f"class_weights has {weights.numel()} values but "
                        f"loss has {loss.shape[1]} channels."
                    )

                # (C,) -> (1,C)
                weights = weights.view(1, -1)

                loss = loss * weights

                if self.reduction == "mean":
                    return loss.sum(dim=1).mean()

                if self.reduction == "sum":
                    return loss.sum()

                return loss
    
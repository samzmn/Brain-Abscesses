import warnings
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
        smooth_nr: float = 0,
        smooth_dr: float = 1e-6,
        reduction: Literal["mean", "sum", "none"] = "mean",
        generalized: bool = False,
        class_weights: torch.Tensor | list | tuple | None = None,
        ignore_empty: bool = False,
        log_dice: bool = False,
        tversky: bool = False,
        alpha: float = 0.5,
        beta: float = 0.5,
        focal_tversky: bool = False,
        gamma: float = 1.33,
        batch_dice: bool = False,
        device: str = "cuda",
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
        if generalized and class_weights is not None:
            warnings.warn("class_weights are ignored when generalized=True.", stacklevel=2,)
        if generalized and ignore_empty:
            warnings.warn("ignore_empty has no effect on Generalized Dice.", stacklevel=2,)
        if generalized and tversky:
            raise ValueError("Generalized Dice and Tversky cannot be enabled simultaneously.")
        if generalized and focal_tversky:
            raise ValueError("Generalized Dice and Focal Tversky cannot be enabled simultaneously.")
        if tversky and log_dice:
            warnings.warn("log_dice has no effect when Tversky/Focal-Tversky is enabled.", stacklevel=2,)
        if focal_tversky and not tversky:
            warnings.warn("focal_tversky=True automatically enables Tversky.", stacklevel=2,)
            tversky = True
        if squared_pred and tversky:
            warnings.warn("squared_pred has no effect when Tversky loss is enabled.", stacklevel=2,)
        if batch_dice and generalized:
            warnings.warn("Using batch_dice together with generalized Dice is usually unnecessary because both reduce class imbalance.", stacklevel=2,)
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be in [0,1].")
        if not (0 <= beta <= 1):
            raise ValueError("beta must be in [0,1].")
        if alpha + beta == 0:
            raise ValueError("alpha and beta cannot both be zero.")
        if gamma <= 0:
            raise ValueError("gamma must be >0.")

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
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        self.ignore_empty = ignore_empty
        self.log_dice = log_dice
        self.tversky = tversky or focal_tversky
        self.alpha = alpha
        self.beta = beta
        self.focal_tversky = focal_tversky
        self.gamma = gamma
        self.batch_dice = batch_dice

        if class_weights is not None:
            if not include_background:
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

        if self.batch_dice:
            # Sum over batch + spatial dimensions
            reduce_dims = (0,) + tuple(range(2, probs.ndim))
        else:
            # Sum over spatial dimensions only
            reduce_dims = tuple(range(2, probs.ndim))

        intersection = torch.sum(probs * target, dim=reduce_dims)

        if self.tversky:
            false_positive = torch.sum(
                probs * (1.0 - target),
                dim=reduce_dims,
            )
            false_negative = torch.sum(
                (1.0 - probs) * target,
                dim=reduce_dims,
            )

        if self.squared_pred:
            pred_sum = torch.sum(probs ** 2, dim=reduce_dims)
            target_sum = torch.sum(target ** 2, dim=reduce_dims)
        else:
            pred_sum = torch.sum(probs, dim=reduce_dims)
            target_sum = torch.sum(target, dim=reduce_dims)

        if self.ignore_empty:
            valid_mask = target_sum > 0
        
        # Generalized Dice
        if self.generalized:
            # class volumes from ground truth
            class_volume = target_sum

            # inverse squared volume weighting
            weights = 1.0 / (class_volume.pow(2) + self.smooth_dr)
            # weights = torch.where(
            #     class_volume > 0,
            #     1.0 / class_volume.pow(2),
            #     torch.zeros_like(class_volume),
            # )

            # avoid inf if a class is absent
            weights = torch.where(
                torch.isfinite(weights),
                weights,
                torch.zeros_like(weights)
            )

            if self.batch_dice:
                numerator = 2.0 * torch.sum(weights * intersection)

                denominator = torch.sum(
                    weights * (pred_sum + target_sum)
                )
            else:
                numerator = 2.0 * torch.sum(
                    weights * intersection,
                    dim=1,
                )

                denominator = torch.sum(
                    weights * (pred_sum + target_sum),
                    dim=1,
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
            if self.tversky:
                score = (
                    intersection + self.smooth_nr
                ) / (
                    intersection
                    + self.alpha * false_positive
                    + self.beta * false_negative
                    + self.smooth_dr
                )

                if self.focal_tversky:
                    loss = (1.0 - score).pow(self.gamma)
                else:
                    loss = 1.0 - score

            else: 
                score = (
                    2.0 * intersection + self.smooth_nr
                ) / (
                    pred_sum + target_sum + self.smooth_dr
                )

                if self.log_dice:
                    loss = -torch.log(
                        score.clamp(min=self.smooth_nr, max=1.0)
                    )
                else:
                    loss = 1.0 - score

            if self.ignore_empty:
                loss = loss * valid_mask.float()
                if self.batch_dice:
                    normalizer = valid_mask.sum().clamp(min=1)
                else:
                    normalizer = valid_mask.sum(dim=1).clamp(min=1)

                if self.class_weights is None:
                    if self.batch_dice:
                        loss = loss.sum() / normalizer
                    else:
                        loss = loss.sum(dim=1) / normalizer

                    if self.reduction == "mean":
                        return loss.mean()

                    if self.reduction == "sum":
                        return loss.sum()

                    return loss

            # Apply class weights
            if self.class_weights is not None:
                weights = self.class_weights

                if weights.numel() != loss.shape[-1]:
                    raise ValueError(
                        f"class_weights has {weights.numel()} values but "
                        f"loss has {loss.shape[-1]} channels."
                    )

                if self.batch_dice:
                    weights = weights.view(-1)
                else:
                    weights = weights.view(1,-1) # (C,) -> (1,C)

                if self.ignore_empty:
                    if self.batch_dice:
                        weights = weights.squeeze(0)
                        weights = weights * valid_mask.float()
                        weights = weights / weights.sum().clamp(min=1e-8)
                        weights = weights.unsqueeze(0)
                    else:
                        weights = weights * valid_mask.float()
                        weights = weights / weights.sum(
                            dim=1,
                            keepdim=True,
                        ).clamp(min=1e-8)
                    
                loss = loss * weights

                if self.reduction == "mean":
                    if self.batch_dice:
                        return loss.sum()
                    else:
                        return loss.sum(dim=1).mean()

                if self.reduction == "sum":
                    return loss.sum()

                return loss
            
            else:

                if self.reduction == "mean":
                    return loss.mean()

                if self.reduction == "sum":
                    return loss.sum()

                return loss
    
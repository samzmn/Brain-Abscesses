import torch
import torch.nn.functional as F


class DiceMetric:
    def __init__(
        self,
        include_background: bool = True,
        reduction: str = "mean",
        ignore_empty: bool = True,
        num_classes: int | None = None,
    ):
        self.include_background = include_background
        self.reduction = reduction
        self.ignore_empty = ignore_empty
        self.num_classes = num_classes

        self.reset()

    def reset(self):
        self._buffer = []

    @torch.no_grad()
    def update(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
    ):
        """
        Parameters
        ----------
        y_pred
            Either:
                (B,C,H,W[,D]) one-hot prediction
                (B,1,H,W[,D]) label map

        y
            Either:
                (B,C,H,W[,D]) one-hot labels
                (B,1,H,W[,D]) label map
        """

        if y_pred.ndim < 4:
            raise ValueError("Expected BCHW or BCHWD tensor.")

        # -------------------------------------------------------
        # Convert label maps -> one-hot
        # -------------------------------------------------------

        if y_pred.shape[1] == 1:
            n_classes = self.num_classes
            if n_classes is None:
                raise ValueError("num_classes must be provided for label maps.")

            y_pred = (
                F.one_hot(y_pred.squeeze(1).long(), n_classes)
                .permute(0, -1, *range(1, y_pred.ndim - 1))
                .float()
            )

        if y.shape[1] == 1:
            n_classes = self.num_classes or y_pred.shape[1]

            y = (
                F.one_hot(y.squeeze(1).long(), n_classes)
                .permute(0, -1, *range(1, y.ndim - 1))
                .float()
            )

        if not self.include_background:
            y_pred = y_pred[:, 1:]
            y = y[:, 1:]

        spatial_dims = tuple(range(2, y.ndim))

        intersection = (y_pred * y).sum(dim=spatial_dims)

        pred_sum = y_pred.sum(dim=spatial_dims)
        target_sum = y.sum(dim=spatial_dims)

        denominator = pred_sum + target_sum

        dice = (2.0 * intersection) / denominator.clamp_min(1e-8)

        if self.ignore_empty:
            empty = target_sum == 0
            dice[empty] = torch.nan
        else:
            empty = target_sum == 0
            dice[empty] = (pred_sum[empty] == 0).float()

        self._buffer.append(dice.cpu())

    @torch.no_grad()
    def aggregate(self):

        if len(self._buffer) == 0:
            raise RuntimeError("No samples have been added.")

        scores = torch.cat(self._buffer, dim=0)

        if self.reduction == "none":
            return scores

        if self.reduction == "mean_batch":
            return torch.nanmean(scores, dim=0)

        if self.reduction == "mean":
            return torch.nanmean(scores)

        raise ValueError(f"Unknown reduction '{self.reduction}'")
    
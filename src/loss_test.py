import torch
import torch.nn.functional as F

# ----------------------------
# Label maps (B,H,W)
# ----------------------------

target = torch.tensor(
[
    [[
        [0,0,1,1],
        [0,1,1,2],
        [0,2,2,2],
        [3,3,3,0],
    ]],
    [[
        [0,1,1,1],
        [0,1,2,2],
        [3,3,2,0],
        [0,0,0,0],
    ]],
], dtype=torch.long)

prediction = torch.tensor(
[
    [
        [0,0,1,1],
        [0,1,1,2],
        [0,2,1,2],   # one edema predicted as brain
        [3,3,0,0],   # one abscess missed
    ],
    [
        [0,1,1,1],
        [0,1,2,0],   # edema -> background
        [3,3,2,0],
        [0,0,0,0],
    ],
], dtype=torch.long)

print("target: ",target.shape)
# print("pred: ", prediction.shape)

prediction_onehot = (
    F.one_hot(prediction, num_classes=4)
    .permute(0,3,1,2)
    .float()
)

logits = prediction_onehot * 8.0 - 4.0
print("logits: ",logits.shape)

target_onehot = F.one_hot(
    target.squeeze(1).long(),
    num_classes=4
).permute(0,3,1,2).float()
print("target one-hot: ", target_onehot.shape)
print("pred one-hot: ", prediction_onehot.shape)

from utils.losses import DiceLoss

my_loss = DiceLoss(
    include_background=True,
    to_onehot_y=True,
    from_logits=True,
    squared_pred=False,
    reduction="mean",
)

loss2 = my_loss(logits, target)
print(loss2)

from utils.metrics import DiceMetric

metric = DiceMetric(
    include_background=True,
    reduction="mean_batch",
    ignore_empty=True,
    num_classes=4
)

metric.update(prediction_onehot, target)

print(metric.aggregate())

pred = torch.argmax(prediction_onehot, dim=1, keepdim=True)
print(prediction.shape, pred.shape, target.shape)
metric.update(pred, target)

print(metric.aggregate())

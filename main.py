# """AverageLoss metric tests."""

# import torch
# from torchmetrics import segmentation


# BATCH_SIZE = 32
# NUM_CLASSES = 5
# NUM_BATCHES = 4

# preds = torch.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 16))
# target = torch.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 16))
# generalized_dice = segmentation.GeneralizedDiceScore(num_classes=NUM_CLASSES)

# for batch_preds, batch_target in zip(preds, target, strict=False):
#     generalized_dice.update(batch_preds, batch_target)  # type: ignore
# actual = generalized_dice.compute()
# print(actual)


import torch

x = torch.Tensor(
    [
        [0, 1, 0],
        [0, 2, 0],
        [0, 0, 1],
    ],
)
is_index_tensor = (x.bool() != x).any()

print()

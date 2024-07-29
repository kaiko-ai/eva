import torch

from eva.vision.models.networks.decoders.segmentation.mask2former import Mask2Former
from eva.vision.losses import Mask2formerLoss
from eva.vision.losses.mask2former.matcher import Mask2formerMatcher
# from transformers.models.mask2former import modeling_mask2former


from typing import Tuple, List

from monai.networks import one_hot  # type: ignore

import torch

def _convert_semantic_label(semantic_labels: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    mask_labels, class_labels = [], []
    for batch in semantic_labels:
        masks, labels = [], []
        for label_id in batch.unique():
            masks.append((batch == label_id))
            labels.append(label_id)
        mask_labels.append(torch.stack(masks))
        class_labels.append(torch.stack(labels))
    return mask_labels, class_labels

# [tensor([0, 1, 2, 3, 4]), tensor([0, 1, 2, 3, 4])]

features = [torch.rand(2, 192, 16, 16)]
targets = torch.randint(0, 10, (2, 224, 224))

mask_labels, class_labels = _convert_semantic_label(targets)


# targets_one_hot = torch.randint(0, 2, (2, 5, 224, 224)).float()

decoder = Mask2Former(in_features=192, num_classes=10)
loss = Mask2formerLoss(num_labels=10)
matcher = Mask2formerMatcher()


mask_logits_per_layer, class_logits_per_layer = decoder(features)
print(mask_logits_per_layer[0].mean(), class_logits_per_layer[0].mean())
quit()

out = loss._layer_loss(mask_logits_per_layer[0], mask_labels, class_logits_per_layer[0], class_labels)
print(out)

out = loss.forward(mask_logits_per_layer, mask_labels, class_logits_per_layer, class_labels)
print(out)

# semantic label: (h,w)
# one-hot: (n,h,w)

# matcher(
#     mask_logits_per_layer[0]
# )

# num_classes = class_logits_per_layer[0].shape[-1]
# print(class_logits_per_layer[0].shape)

# loss(
#     mask_logits_per_layer,
#     class_logits_per_layer,
# )
# print(mask_logits_per_layer)

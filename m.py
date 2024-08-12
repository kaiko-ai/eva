from typing import Dict, List, Tuple

import torch
import transformers

def _convert_semantic_label(
    semantic_labels: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Converts semantic labels to mask and class labels.

    Args:
        semantic_labels: Tensor of shape `(batch_size, height, width)`
            representing the semantic labels.

    Returns:
        A tuple containing the list of mask labels and list of class labels.
    """
    mask_labels, class_labels = [], []
    for batch in semantic_labels:
        masks, labels = [], []
        for label_id in batch.unique():
            masks.append((batch == label_id).float())
            labels.append(label_id)
        mask_labels.append(torch.stack(masks))
        class_labels.append(torch.stack(labels))
    return mask_labels, class_labels


model = transformers.Mask2FormerForUniversalSegmentation(
    config=transformers.Mask2FormerConfig(
        use_timm_backbone=True,
        backbone="vit_small_patch16_224",
        backbone_kwargs={
            "out_indices": 3
        },
        num_labels=10,
    ),
)
# print(model.model.pixel_level_module.encoder)

x = torch.rand(1, 3, 224, 224)
out = model(x)

x = torch.rand(2, 3, 224, 224)
t = torch.randint(0, 10, (2, 224, 224))
mask_labels, class_labels = _convert_semantic_label(t)
out = model(x, mask_labels, class_labels)


import transformers

postprocess = transformers.Mask2FormerImageProcessor()
seg = postprocess.post_process_semantic_segmentation(out, (224, 224))
seg = torch.stack(seg)
print(seg.shape)

# pred_semantic_map = image_processor.post_process_semantic_segmentation(
#     outputs, target_sizes=[image.size[::-1]]
# )[0]



# print(out.masks_queries_logits[0].shape, out.class_queries_logits[0].shape, out.loss)

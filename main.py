from eva.vision.data.datasets import TotalSegmentator2D
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader

dataset = TotalSegmentator2D(
    root="data/total_segmentator_2d",
    split="train",
    download=True,
    image_target_transforms=v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=(224, 224)),
            v2.ToDtype(torch.float32, scale=True),
        ]
    ),
)
dataset.prepare_data()
dataset.setup()

# for i in range(len(dataset)):
#     image, mask = dataset[i]
#     print(image.shape, mask.shape)
def indices(h,w):
    return torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)))

def checkerboard(shape, k):
    """
        shape: dimensions of output tensor
        k: edge size of square
    """
    h, w = shape
    base = indices(h//k, w//k).sum(dim=0) % 2
    x = base.repeat_interleave(k, 0).repeat_interleave(k, 1)
    return 1-x


data = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
for image, mask in data:
    # mask = 1.0 * mask
    mask = mask[0][0].unsqueeze(dim=0)
    print(mask.shape)
    # predictions = torch.full_like(mask, fill_value=1).to(device=mask.device, dtype=torch.int)
    predictions = checkerboard((mask.shape[-2], mask.shape[-1]), 2)
    predictions = predictions.unsqueeze(dim=0)
    print(predictions, predictions.shape)
    loss = torch.nn.CrossEntropyLoss(predictions, mask)
    print(loss)
    quit()
    break


# data = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
# for image, mask in data:
#     print(image.shape, mask.shape)
#     break

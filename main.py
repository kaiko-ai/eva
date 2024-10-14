from eva.vision.data.datasets import KiTS23


dataset = KiTS23(root="data/kits23", split="train", download=True)
dataset.prepare_data()
dataset.setup()
# dataset._download()

index = 300
image = dataset.load_image(index)
mask = dataset.load_mask(index)

print(image)
print(mask.unique())

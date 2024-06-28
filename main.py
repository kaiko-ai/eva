from eva.vision import datasets

dataset = datasets.MoNuSAC("data/monusac", split="train")

maximum, minimum = None, None
for image, mask in dataset:
    print(image.shape)
    maximum = max(maximum or image.shape, image.shape)
    minimum = min(minimum or image.shape, image.shape)

print(maximum , minimum)

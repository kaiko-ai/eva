from eva.vision import datasets

dataset = datasets.Bach(root="data/batch", split=None, download=False)
dataset.setup()

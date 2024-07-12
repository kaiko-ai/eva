from eva.vision import datasets

dataset = datasets.LiTS(root="data/lits", split="val")
dataset.setup()

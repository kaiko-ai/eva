from eva.vision.data.datasets.segmentation import BCSS


dataset = BCSS("data/bcss", split="train", download=True)
dataset.prepare_data()

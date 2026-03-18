from eva.vision.data.datasets.segmentation.flare22 import FLARE22

dataset = FLARE22(root="./data/flare22", split="train", download=True)
dataset.prepare_data()
dataset.setup()

# data = dataset[0]
for i, data in enumerate(dataset):
    print(i)


# from eva.vision.data.datasets.segmentation.word import WORD

# dataset = WORD(root="./data/word", split="train", download=True)
# dataset.prepare_data()
# dataset.setup()

# # data = dataset[0]
# for i, data in enumerate(dataset):
#     print(i)


# from eva.vision.data.datasets.segmentation.kits23 import KiTS23

# dataset = KiTS23(root="./data/kits23", split="train", download=True)
# dataset.prepare_data()
# dataset.setup()

# # data = dataset[0]
# for i, data in enumerate(dataset):
#     print(data)
#     quit()

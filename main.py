from eva.vision import datasets

data = datasets.TotalSegmentator(root="data/total_segmentator", split=None, download=False)
data.prepare_data()
data.setup()

# image = data.load_image(0)
# print(image.shape)

mask = data.load_mask(0)
# print(mask.shape)
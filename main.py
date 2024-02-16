from eva.vision import datasets

data = datasets.TotalSegmentator2D(root="data/total_segmentator", split=None, download=False)
data.prepare_data()
data.setup()

image, mask = data.load_image_and_mask(0)
print(image.shape)
print(mask.shape)

from eva.vision.data import datasets

dataset = datasets.TotalSegmentatorClassification(
    root="data/total_segmentator",
    split="val",
)
dataset.setup()
quit()

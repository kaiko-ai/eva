from multiprocessing import freeze_support

from eva.vision.data.datasets import segmentation


def main():
    dataset = segmentation.TotalSegmentator2D("data/total_segmentator", split="train")

    dataset.prepare_data()
    dataset.setup()

    for i in range(0, 2000, 500):
        dataset[i]
        print(i)


if __name__ == "__main__":
    freeze_support()
    main()

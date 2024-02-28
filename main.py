from eva.vision.data.datasets import CRC_HE, CRC_HE_NONORM

dataset = CRC_HE("data/crc_he", split="train", download=True)
dataset.prepare_data()

dataset = CRC_HE_NONORM("data/crc_he", split="train", download=True)
dataset.prepare_data()

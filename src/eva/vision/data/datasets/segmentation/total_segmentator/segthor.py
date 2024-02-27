"""SegTHOR dataset."""

from typing import List

from typing_extensions import override

from eva.vision.data.datasets.segmentation.total_segmentator import base


class SegTHOR(base.TotalSegmentator2DBase):
    """SegTHOR: Segmentation of THoracic Organs at Risk in CT images.

    References:
      - https://arxiv.org/abs/1912.05950
      - https://competitions.codalab.org/competitions/21145
      - https://ceur-ws.org/Vol-2349/
    """

    @property
    @override
    def classes(self) -> List[str]:
        return ["aorta", "esophagus", "heart", "trachea"]

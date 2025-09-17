# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import mmcv
import numpy as np
from mmengine.fileio import get as fileio_get

@DATASETS.register_module()
class FLAIRDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('building', 'pervious surface', 'impervious surface', 'bare soil', 'water',
           'coniferous','deciduous','brushwood','vineyard','herbaceous vegetation','agricultural land','plowed land'),
        palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0)])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            reduce_zero_label=reduce_zero_label,
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


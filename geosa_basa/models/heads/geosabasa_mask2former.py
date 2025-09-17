from .rein_mask2former import ReinMask2FormerHead
from mmseg.registry import MODELS
from mmseg.utils import SampleList
from torch import Tensor
from typing import List, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from mmdet.models.utils.misc import multi_apply
from mmseg.models.builder import MODELS
from mmseg.utils import ConfigType, SampleList
from mmseg.structures.seg_data_sample import SegDataSample
from mmengine.structures import InstanceData, PixelData
from typing import Dict, List, Optional, Tuple, Union
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptMultiConfig, reduce_mean)
from mmseg.utils import add_prefix

@MODELS.register_module()
class GeoSABaSAMask2FormerHead(ReinMask2FormerHead):
    def __init__(self, loss_mse=None,
                 **kwargs):
        super().__init__(**kwargs)
        if loss_mse is not None:
            self.loss_mse = MODELS.build(loss_mse)
        else:
            self.loss_mse = None

    def _loss_consist_by_feat_single(self,mask_preds: Tensor,mask_preds_style: Tensor)-> Tuple[Tensor]:
        loss_con = self.loss_mse(mask_preds_style,mask_preds.detach())+self.loss_mse(mask_preds,mask_preds_style.detach())
        return (loss_con,)


    def loss_2(self, x: Tuple[Tensor], x_style: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        all_cls_scores_style, all_mask_preds_style = self(x_style, batch_data_samples)
        # loss
        loss_dict = dict()
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds, batch_gt_instances, batch_img_metas)
        loss_dict.update(add_prefix(losses, 'decode'))

        losses_style = self.loss_by_feat(all_cls_scores_style, all_mask_preds_style, batch_gt_instances, batch_img_metas)
        loss_dict.update(add_prefix(losses_style, 'decode_style'))

        if self.loss_mse is not None:
            # print(len(all_mask_preds))
            losses_con= multi_apply(self._loss_consist_by_feat_single, all_mask_preds, all_mask_preds_style)
            losses_con1 = losses_con[0]
            loss_dict_con = dict()
            # print(losses_con1)
            # loss from the last decoder layer
            loss_dict_con['loss_con'] = losses_con1[-1]
            # loss from other decoder layers
            num_dec_layer = 0
            for loss_con_i in losses_con1[:-1]:
                # print(loss_con_i)
                loss_dict_con[f'd{num_dec_layer}.loss_con'] = loss_con_i
                num_dec_layer += 1
            loss_dict.update(add_prefix(loss_dict_con, 'decode_con'))

        return loss_dict

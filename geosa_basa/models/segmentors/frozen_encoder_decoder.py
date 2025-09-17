from typing import List
import torch
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
from typing import Iterable


def detach_everything(everything):
    if isinstance(everything, Tensor):
        return everything.detach()
    elif isinstance(everything, Iterable):
        return [detach_everything(x) for x in everything]
    else:
        return everything


@MODELS.register_module()
class FrozenBackboneEncoderDecoder(EncoderDecoder):
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        # calculate_param(self)


    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        with torch.no_grad():
            x = self.backbone(inputs)
            x = detach_everything(x)
        if self.with_neck:
            x = self.neck(x)
        return x


from mmengine.logging import MMLogger

def calculate_param(model):
    num_params = 0
    num_trainable = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
        if param.requires_grad == True:
            num_trainable += param.numel()
    logger = MMLogger.get_current_instance()
    logger.info(
        f"Total trainable params--{num_trainable}, All model params--{num_params}"
    )

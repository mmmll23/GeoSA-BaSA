import time
import os
from typing import List
import torch
from torch import Tensor
import numpy as np
from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
from typing import Iterable
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from PIL import Image

def untransform_image2(output):
    style_out = output.detach().permute(1, 2, 0).to('cpu').numpy()
    style_out = np.uint8(style_out)
    style_out = Image.fromarray(style_out)
    return style_out


def untransform_label(output):
    label_out = output.detach().to('cpu').numpy()
    label_out = np.uint8(label_out)
    label_out = Image.fromarray(label_out)
    return label_out

def FDA_source_to_target(src_img, trg_img, L=0.1):
    fft_src = torch.fft.fft2(src_img.clone(), dim=(-2, -1))
    fft_src = torch.stack((fft_src.real, fft_src.imag), -1)
    fft_trg = torch.fft.fft2(trg_img.clone(), dim=(-2, -1))
    fft_trg = torch.stack((fft_trg.real, fft_trg.imag), -1)

    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())

    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    fft_src_ = torch.zeros(fft_src.size(), dtype=torch.float)
    fft_src_[..., 0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[..., 1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.fft.irfft2(torch.complex(fft_src_[..., 0], fft_src_[..., 1]), s=[imgH, imgW], dim=(-2, -1))

    return src_in_trg

def extract_ampl_phase(fft_im):
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2(fft_im[:,:,:,:,1], fft_im[:,:,:,:,0])
    return fft_amp, fft_pha

def low_freq_mutate(amp_src, amp_trg, L=0.1):
    _, _, h, w = amp_src.size()
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    amp_src[:,:,0:b,0:b] = amp_trg[:,:,0:b,0:b]
    amp_src[:,:,0:b,w-b:w] = amp_trg[:,:,0:b,w-b:w]
    amp_src[:,:,h-b:h,0:b] = amp_trg[:,:,h-b:h,0:b]
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]
    return amp_src

@MODELS.register_module()
class GeoSABaSAEncoderDecoder(EncoderDecoder):
    def basa(self, inputs, data_samples) -> Tensor:
        mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(inputs.device)
        std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(inputs.device)
        inputs_denorm = inputs * std + mean
        inputs_B_denorm = torch.flip(inputs_denorm,[0])
        L = 0.01 + (0.1 - 0.01) * torch.rand(1).item()
        inputs_C = FDA_source_to_target(inputs_denorm, inputs_B_denorm, L=L).to(inputs.device)
        inputs_C = torch.clamp(inputs_C, 0.0, 255.0)
        inputs_C_norm = (inputs_C - mean) / std

        return inputs_C_norm

    def _decode_head_forward_train_2(self, inputs: List[Tensor],inputs_style: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss_2(inputs,inputs_style, data_samples,
                                            self.train_cfg)
        losses.update(loss_decode)
        return losses


    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        with torch.no_grad():
            style_inputs = self.basa(inputs,data_samples)


        x = self.extract_feat(inputs)
        x_style = self.extract_feat(style_inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train_2(x,x_style, data_samples)
        losses.update(loss_decode)


        return losses
from mmseg.models.builder import BACKBONES, MODELS
# from .dino_v2 import DinoVisionTransformer
from .eva_02 import EVA2
from .utils import set_requires_grad2, set_train2
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from .enhancer import Enhancer
from .bridge import Bridge
from .dino_layers import MemEffAttention
import torch.utils.checkpoint as checkpoint

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W, prompt_len):
        B, N, C = x.shape
        # print(x.shape)
        n = (N-prompt_len) // 21
        x_prompt = x[:, :prompt_len, :]
        x1 = x[:, prompt_len:prompt_len+16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, prompt_len+16 * n:prompt_len+20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, prompt_len+20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x_prompt,x1, x2, x3], dim=1)
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W, prompt_len):
        x = self.fc1(x)
        # print("x:",x.shape)
        x = self.dwconv(x, H, W, prompt_len)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class InteractionBlock_cls_efficient(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, proj_bias=False,norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25,use_checkpoint=False):
        super().__init__()
        self.self_attn = MemEffAttention(dim,num_heads,qkv_bias=qkv_bias,proj_bias=proj_bias)
        self.with_cffn = with_cffn
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_checkpoint=use_checkpoint

    def forward(self, x, c, blocks, index, enhancer, bridge, H, W,rel_pos_bias):
        for idx, blk in enumerate(blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias)

        x = enhancer.forward(
            x,
            index,
            batch_first=True,
            has_cls_token=True,
        )

        cls, x = x[:, :1, ], x[:, 1:, ]

        prompts = bridge.forward(x,index)
        # print("prompts:",prompts.shape)
        # print("c:",c.shape)
        prompted_c = torch.cat([prompts, c], dim=1)
        prompted_c = self.self_attn(prompted_c)
        prompt_len = prompts.shape[1]
        # print("prompted_c",prompted_c.shape)
        if self.with_cffn:
            prompted_c = prompted_c + self.drop_path(self.ffn(self.ffn_norm(prompted_c), H, W,prompt_len))
        c = prompted_c[:,prompt_len:]

        x = torch.cat((cls, x), dim=1)
        return x, c

class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4

@BACKBONES.register_module()
class GeoSAEVAVisionTransformer(EVA2):
    def __init__(
        self,
        conv_inplane,
        use_extra_extractor=True,
        add_vit_feature=True,
        side_dims = 512,
        enhancer_config=None,
        bridge_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.side_dims = side_dims
        self.add_vit_feature = add_vit_feature
        self.spm = SpatialPriorModule(inplanes=conv_inplane,
                                      embed_dim=self.side_dims)
        global_attn_indexes = self.out_indices
        interaction_indexes1 = [0, global_attn_indexes[0]]
        interaction_indexes2 = [global_attn_indexes[0] + 1, global_attn_indexes[1]]
        interaction_indexes3 = [global_attn_indexes[1] + 1, global_attn_indexes[2]]
        interaction_indexes4 = [global_attn_indexes[2] + 1, global_attn_indexes[3]]

        interaction_indexes = [interaction_indexes1, interaction_indexes2,
                               interaction_indexes3, interaction_indexes4]
        self.interaction_indexes = interaction_indexes
        self.interactions = nn.Sequential(*[
            InteractionBlock_cls_efficient(dim=self.side_dims, num_heads=16,qkv_bias=self.qkv_bias,proj_bias=True,
                                           norm_layer=norm_layer,with_cffn=True, cffn_ratio=0.5,use_checkpoint=self.use_checkpoint)
            for i in range(len(interaction_indexes))
        ])
        # qkv_bias=False, proj_bias=False,norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #                  drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25
        self.level_embed = nn.Parameter(torch.zeros(3, self.side_dims))
        self.up = nn.ConvTranspose2d(self.side_dims, self.side_dims, 2, 2)
        self.norm1 = nn.SyncBatchNorm(self.side_dims)
        self.norm2 = nn.SyncBatchNorm(self.side_dims)
        self.norm3 = nn.SyncBatchNorm(self.side_dims)
        self.norm4 = nn.SyncBatchNorm(self.side_dims)
        self.conv1 = nn.Conv2d(self.embed_dim, self.side_dims, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(self.embed_dim, self.side_dims, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(self.embed_dim, self.side_dims, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv4 = nn.Conv2d(self.embed_dim, self.side_dims, kernel_size=1, stride=1, padding=0, bias=True)

        self.enhancer: Enhancer = MODELS.build(enhancer_config)
        self.bridge: Bridge = MODELS.build(bridge_config)


    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward_features(self, x, masks=None):
        # SPM forward
        c1, c2, c3, c4 = self.spm(x)  # c2-c4:bs,128*128(h*2,w*2)/64*64(h,w)/32*32(h/2,w/2),c
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)  # bs,128*128+64*64+32*32,c

        B, _, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        # print(x.shape)  #torch.Size([4, 1025, 1024])

        outs_multilayer = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]  # e.g.,[0,5]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],i,self.enhancer,self.bridge,Hp, Wp,rel_pos_bias)
            outs_multilayer.append(x[:, 1:, :].permute(0, 2, 1).view(B, -1, Hp, Wp).contiguous())
        feats, querys = self.enhancer.return_auto(outs_multilayer)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.contiguous().transpose(1, 2).view(B, -1, Hp * 2, Wp * 2)  #bs,dim,128,128
        c3 = c3.contiguous().transpose(1, 2).view(B, -1, Hp, Wp)  #bs,dim,64,64
        c4 = c4.contiguous().transpose(1, 2).view(B, -1, Hp // 2, Wp // 2)  ##bs,dim,32,32
        c1 = self.up(c2) + c1  ##bs,dim,256,256

        if self.add_vit_feature:
            x1, x2, x3, x4 = feats
            x1 = self.conv1(x1)
            x2 = self.conv2(x2)
            x3 = self.conv3(x3)
            x4 = self.conv4(x4)
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4],querys

    def forward(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad2(self, ["blocks","cls_token","pos_embed","pos_drop","patch_embed","rope"])
        set_train2(self, ["blocks","cls_token","pos_embed","pos_drop","patch_embed","rope"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "blocks" in k or "cls_token" in k or "pos_embed" in k or "pos_drop" in k or "patch_embed" in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state


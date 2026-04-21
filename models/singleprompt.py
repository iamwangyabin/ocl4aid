import logging
from functools import partial
from typing import Iterable

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit

logger = logging.getLogger()


class PreAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt=None):
        bsz, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            bsz, num_tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if prompt is not None:
            prompt_len = prompt.size(2)
            prompt = prompt.view(
                bsz, 2, prompt_len, self.num_heads, channels // self.num_heads
            ).permute(1, 0, 3, 2, 4).contiguous()
            key_prefix = prompt[0]
            value_prefix = prompt[1]
            k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SinglePromptBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PreAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = vit.LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = vit.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = vit.Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = vit.LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = vit.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, prompt=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), prompt=prompt)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def _build_norm_factory(norm_module: nn.Module):
    if isinstance(norm_module, nn.LayerNorm):
        return partial(nn.LayerNorm, eps=norm_module.eps)
    return type(norm_module)


class SinglePrompt(nn.Module):
    def __init__(
        self,
        task_num: int = 10,
        num_classes: int = 100,
        backbone_name: str = None,
        len_prompt: int = 20,
        pos_prompt: Iterable[int] = (0, 1, 2, 3, 4),
        logit_type: str = "cos_sim",
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()
        del task_num
        del kwargs

        if backbone_name is None:
            raise ValueError("backbone_name must be specified")

        if hasattr(vit, backbone_name):
            logger.info("Using custom ViT model: %s", backbone_name)
            backbone = getattr(vit, backbone_name)(
                pretrained=pretrained,
                num_classes=num_classes,
            )
        else:
            logger.info("Using timm model: %s", backbone_name)
            backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=num_classes,
            )

        if not hasattr(backbone, "fc"):
            if hasattr(backbone, "head"):
                backbone.fc = backbone.head
            else:
                raise RuntimeError("SinglePrompt requires a ViT backbone with an fc/head classifier")

        self.backbone = backbone
        self.embed_dim = self.backbone.num_features
        self.len_prompt = int(len_prompt)
        self.logit_type = logit_type
        self.logit_temperature = 0.1

        pos_prompt = list(pos_prompt)
        if self.len_prompt <= 0 or not pos_prompt:
            raise ValueError("SinglePrompt requires non-empty len_prompt and pos_prompt")
        self.register_buffer("pos_prompt", torch.tensor(pos_prompt, dtype=torch.int64))

        self._replace_backbone_blocks()

        if self.logit_type == "linear":
            self.backbone.fc = nn.Linear(self.embed_dim, num_classes)
        elif self.logit_type == "cos_sim":
            self.backbone.fc = nn.Linear(self.embed_dim, num_classes, bias=False)
        else:
            raise ValueError(f"Unsupported logit_type: {self.logit_type}")

        self.prompt = nn.Parameter(
            torch.empty(len(pos_prompt), 2, self.len_prompt, self.embed_dim)
        )
        nn.init.uniform_(self.prompt, -1.0, 1.0)

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        if self.backbone.fc.bias is not None:
            self.backbone.fc.bias.requires_grad = True

    def _replace_backbone_blocks(self):
        prompt_blocks = []
        for orig_block in self.backbone.blocks:
            dim = orig_block.norm1.normalized_shape[0]
            num_heads = orig_block.attn.num_heads
            mlp_ratio = orig_block.mlp.fc1.out_features / dim
            qkv_bias = orig_block.attn.qkv.bias is not None
            drop = float(orig_block.attn.proj_drop.p)
            attn_drop = float(orig_block.attn.attn_drop.p)
            init_values = None
            if hasattr(orig_block.ls1, "gamma"):
                init_values = float(orig_block.ls1.gamma.detach().mean().item())
            drop_path = float(getattr(orig_block.drop_path1, "drop_prob", 0.0))
            norm_layer = _build_norm_factory(orig_block.norm1)
            act_layer = type(orig_block.mlp.act)

            new_block = SinglePromptBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                init_values=init_values,
                drop_path=drop_path,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            new_block.load_state_dict(orig_block.state_dict(), strict=True)
            prompt_blocks.append(new_block)

        self.backbone.blocks = nn.Sequential(*prompt_blocks)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(inputs)
        x = torch.cat((self.backbone.cls_token.expand(x.size(0), -1, -1), x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed[:, : x.size(1), :])

        bsz, _, channels = x.size()
        prompt = self.prompt.unsqueeze(0).expand(bsz, -1, -1, -1, -1)
        pos_bias = self.backbone.pos_embed[:, :1, :].unsqueeze(1).unsqueeze(2)
        prompt = prompt + pos_bias.expand(bsz, self.prompt.size(0), 2, self.len_prompt, channels)

        for block_idx, block in enumerate(self.backbone.blocks):
            prompt_idx = (self.pos_prompt == block_idx).nonzero(as_tuple=False).flatten()
            if prompt_idx.numel() > 0:
                x = block(x, prompt=prompt[:, prompt_idx.item()])
            else:
                x = block(x)

        x = self.backbone.norm(x)
        return x[:, 0]

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.logit_type == "linear":
            return self.backbone.fc(x)
        return F.normalize(x, dim=1) @ F.normalize(self.backbone.fc.weight, dim=1).T / self.logit_temperature

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        x = self.forward_features(inputs)
        return self.forward_head(x)

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target)

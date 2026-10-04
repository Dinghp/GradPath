from __future__ import annotations

import math
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape,
        dtype=x.dtype,
        device=x.device,
    )
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_c: int = 3,
        embed_dim: int = 768,
        norm_layer=None,
    ):
        super().__init__()

        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_c,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if (height, width) != self.img_size:
            raise ValueError(
                f"Input image size ({height}x{width}) does not match "
                f"model size ({self.img_size[0]}x{self.img_size[1]})."
            )

        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale=None,
        attn_drop_ratio: float = 0.0,
        proj_drop_ratio: float = 0.0,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_num, channel_num = x.shape

        qkv = (
            self.qkv(x)
            .reshape(
                batch_size,
                token_num,
                3,
                self.num_heads,
                channel_num // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)

        x = (
            (attention @ value)
            .transpose(1, 2)
            .reshape(batch_size, token_num, channel_num)
        )
        x = self.proj(x)
        return self.proj_drop(x)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale=None,
        drop_ratio: float = 0.0,
        attn_drop_ratio: float = 0.0,
        drop_path_ratio: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_ratio=attn_drop_ratio,
            proj_drop_ratio=drop_ratio,
        )
        self.drop_path = (
            DropPath(drop_path_ratio)
            if drop_path_ratio > 0.0
            else nn.Identity()
        )
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop_ratio,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _PartialRowLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        selected_rows: torch.Tensor,
        selected_bias: Optional[torch.Tensor],
        selected_indices: torch.Tensor,
        scaling: float,
    ) -> torch.Tensor:
        ctx.scaling = float(scaling)
        ctx.row_parameter_dtype = selected_rows.dtype
        ctx.bias_parameter_dtype = (
            selected_bias.dtype
            if selected_bias is not None
            else None
        )
        ctx.has_selected_bias = selected_bias is not None
        ctx.save_for_backward(x, weight, selected_indices)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, selected_indices = ctx.saved_tensors

        grad_x = None
        grad_selected_rows = None
        grad_selected_bias = None

        if ctx.needs_input_grad[0]:
            grad_x = torch.matmul(
                grad_output,
                weight.to(dtype=grad_output.dtype),
            )

        selected_grad_output = None
        if ctx.needs_input_grad[3] or ctx.needs_input_grad[4]:
            selected_grad_output = grad_output.index_select(
                dim=-1,
                index=selected_indices,
            )

        if ctx.needs_input_grad[3]:
            selected_grad_output_2d = selected_grad_output.reshape(
                -1,
                selected_grad_output.shape[-1],
            )
            x_2d = x.reshape(-1, x.shape[-1])

            grad_selected_rows = (
                selected_grad_output_2d.float().transpose(0, 1)
                @ x_2d.float()
            )
            grad_selected_rows.mul_(ctx.scaling)
            grad_selected_rows = grad_selected_rows.to(
                ctx.row_parameter_dtype
            )

        if ctx.needs_input_grad[4] and ctx.has_selected_bias:
            reduce_dims = tuple(
                range(selected_grad_output.ndim - 1)
            )
            grad_selected_bias = selected_grad_output.float().sum(
                dim=reduce_dims
            )
            grad_selected_bias.mul_(ctx.scaling)
            grad_selected_bias = grad_selected_bias.to(
                ctx.bias_parameter_dtype
            )

        return (
            grad_x,
            None,
            None,
            grad_selected_rows,
            grad_selected_bias,
            None,
            None,
        )

class _PartialColumnLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        selected_columns: torch.Tensor,
        selected_indices: torch.Tensor,
        scaling: float,
    ) -> torch.Tensor:
        selected_x = x.index_select(dim=-1, index=selected_indices)

        ctx.scaling = float(scaling)
        ctx.parameter_dtype = selected_columns.dtype
        ctx.save_for_backward(selected_x, weight)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        selected_x, weight = ctx.saved_tensors

        grad_x = None
        grad_selected_columns = None

        if ctx.needs_input_grad[0]:
            grad_x = torch.matmul(
                grad_output,
                weight.to(dtype=grad_output.dtype),
            )

        if ctx.needs_input_grad[3]:
            grad_output_2d = grad_output.reshape(
                -1,
                grad_output.shape[-1],
            )
            selected_x_2d = selected_x.reshape(
                -1,
                selected_x.shape[-1],
            )

            grad_selected_columns = (
                grad_output_2d.float().transpose(0, 1)
                @ selected_x_2d.float()
            )
            grad_selected_columns.mul_(ctx.scaling)
            grad_selected_columns = grad_selected_columns.to(
                ctx.parameter_dtype
            )

        return (
            grad_x,
            None,
            None,
            grad_selected_columns,
            None,
            None,
        )


class PartialRowLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        selected_indices: torch.Tensor,
        bias: bool = True,
        update_selected_bias: bool = False,
        alpha: Optional[float] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        indices = torch.as_tensor(
            selected_indices,
            dtype=torch.long,
            device="cpu",
        ).flatten()
        if indices.numel() == 0:
            raise ValueError("selected_indices cannot be empty")
        if indices.unique().numel() != indices.numel():
            raise ValueError("selected_indices must not contain duplicates")
        if indices.min().item() < 0 or indices.max().item() >= out_features:
            raise ValueError("selected row index is out of range")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(indices.numel())
        self.alpha = float(alpha if alpha is not None else self.rank)
        self.scaling = self.alpha / self.rank
        self.update_selected_bias = bool(
            update_selected_bias and bias
        )

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        self.register_buffer(
            "selected_indices",
            indices,
            persistent=True,
        )
        self.selected_rows = nn.Parameter(
            torch.empty(self.rank, in_features, **factory_kwargs),
            requires_grad=True,
        )

        if self.update_selected_bias:
            self.selected_bias = nn.Parameter(
                torch.empty(self.rank, **factory_kwargs),
                requires_grad=True,
            )
        else:
            self.register_parameter("selected_bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        with torch.no_grad():
            indices = self.selected_indices.to(self.weight.device)
            selected = self.weight.index_select(dim=0, index=indices)
            self.selected_rows.copy_(selected / self.scaling)

            if self.selected_bias is not None:
                selected_bias = self.bias.index_select(
                    dim=0,
                    index=indices,
                )
                self.selected_bias.copy_(
                    selected_bias / self.scaling
                )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        selected_indices: torch.Tensor,
        update_selected_bias: bool = False,
        alpha: Optional[float] = None,
    ) -> "PartialRowLinear":
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            selected_indices=selected_indices,
            bias=linear.bias is not None,
            update_selected_bias=update_selected_bias,
            alpha=alpha,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)

            indices = layer.selected_indices.to(layer.weight.device)
            selected = layer.weight.index_select(dim=0, index=indices)
            layer.selected_rows.copy_(selected / layer.scaling)

            if layer.selected_bias is not None:
                selected_bias = layer.bias.index_select(
                    dim=0,
                    index=indices,
                )
                layer.selected_bias.copy_(
                    selected_bias / layer.scaling
                )

            layer.sync_weight()

        return layer

    @torch.no_grad()
    def sync_weight(self) -> None:
        indices = self.selected_indices.to(self.weight.device)
        updated_rows = self.scaling * self.selected_rows
        self.weight.index_copy_(
            dim=0,
            index=indices,
            source=updated_rows.to(
                device=self.weight.device,
                dtype=self.weight.dtype,
            ),
        )

        if self.selected_bias is not None:
            updated_bias = self.scaling * self.selected_bias
            self.bias.index_copy_(
                dim=0,
                index=indices,
                source=updated_bias.to(
                    device=self.bias.device,
                    dtype=self.bias.dtype,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.sync_weight()
        indices = self.selected_indices.to(x.device)
        return _PartialRowLinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.selected_rows,
            self.selected_bias,
            indices,
            self.scaling,
        )

    @torch.no_grad()
    def to_merged_linear(self) -> nn.Linear:
        self.sync_weight()
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        merged.weight.copy_(self.weight)
        if self.bias is not None:
            merged.bias.copy_(self.bias)
        return merged

class PartialColumnLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        selected_indices: torch.Tensor,
        bias: bool = True,
        alpha: Optional[float] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        indices = torch.as_tensor(
            selected_indices,
            dtype=torch.long,
            device="cpu",
        ).flatten()
        if indices.numel() == 0:
            raise ValueError("selected_indices cannot be empty")
        if indices.unique().numel() != indices.numel():
            raise ValueError("selected_indices must not contain duplicates")
        if indices.min().item() < 0 or indices.max().item() >= in_features:
            raise ValueError("selected column index is out of range")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(indices.numel())
        self.alpha = float(alpha if alpha is not None else self.rank)
        self.scaling = self.alpha / self.rank

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        self.register_buffer(
            "selected_indices",
            indices,
            persistent=True,
        )
        self.selected_columns = nn.Parameter(
            torch.empty(out_features, self.rank, **factory_kwargs),
            requires_grad=True,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        with torch.no_grad():
            indices = self.selected_indices.to(self.weight.device)
            selected = self.weight.index_select(dim=1, index=indices)
            self.selected_columns.copy_(selected / self.scaling)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        selected_indices: torch.Tensor,
        alpha: Optional[float] = None,
    ) -> "PartialColumnLinear":
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            selected_indices=selected_indices,
            bias=linear.bias is not None,
            alpha=alpha,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)

            indices = layer.selected_indices.to(layer.weight.device)
            selected = layer.weight.index_select(dim=1, index=indices)
            layer.selected_columns.copy_(selected / layer.scaling)
            layer.sync_weight()

        return layer

    @torch.no_grad()
    def sync_weight(self) -> None:
        indices = self.selected_indices.to(self.weight.device)
        updated_columns = self.scaling * self.selected_columns
        self.weight.index_copy_(
            dim=1,
            index=indices,
            source=updated_columns.to(
                device=self.weight.device,
                dtype=self.weight.dtype,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.sync_weight()
        indices = self.selected_indices.to(x.device)
        return _PartialColumnLinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.selected_columns,
            indices,
            self.scaling,
        )

    @torch.no_grad()
    def to_merged_linear(self) -> nn.Linear:
        self.sync_weight()
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        merged.weight.copy_(self.weight)
        if self.bias is not None:
            merged.bias.copy_(self.bias)
        return merged


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_c: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        representation_size=None,
        distilled: bool = False,
        drop_ratio: float = 0.0,
        attn_drop_ratio: float = 0.0,
        drop_path_ratio: float = 0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_c=in_c,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim))
            if distilled
            else None
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                num_patches + self.num_tokens,
                embed_dim,
            )
        )
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [
            value.item()
            for value in torch.linspace(0, drop_path_ratio, depth)
        ]

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_ratio=drop_ratio,
                    attn_drop_ratio=attn_drop_ratio,
                    drop_path_ratio=dpr[index],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for index in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "fc",
                            nn.Linear(embed_dim, representation_size),
                        ),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.classifier = nn.Linear(
            self.num_features,
            num_classes,
        )

        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(
                self.embed_dim,
                num_classes,
            )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.apply(_init_vit_weights)

    def forward_features(self, x: torch.Tensor):
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            dist_token = self.dist_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if self.dist_token is None:
            return self.pre_logits(x[:, 0])

        return x[:, 0], x[:, 1]

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)

        if self.head_dist is None:
            return self.classifier(x)

        cls_x, dist_x = x
        cls_logits = self.classifier(cls_x)
        dist_logits = self.head_dist(dist_x)

        if self.training:
            return cls_logits, dist_logits

        return (cls_logits + dist_logits) / 2


def _normalize_target_blocks(
    depth: int,
    target_blocks: Optional[Union[str, Sequence[int]]],
) -> list[int]:
    if target_blocks is None or target_blocks == "all":
        return list(range(depth))

    if isinstance(target_blocks, str):
        value = target_blocks.lower().replace(" ", "")
        if value.startswith("last"):
            count = int(value[4:])
            if not 1 <= count <= depth:
                raise ValueError(
                    f"Invalid target_blocks={target_blocks}"
                )
            return list(range(depth - count, depth))

        raise ValueError(
            "target_blocks must be 'all', 'lastN', "
            "or a sequence of block indices"
        )

    indices = sorted({int(index) for index in target_blocks})
    if not indices:
        raise ValueError("target_blocks cannot be empty")
    if indices[0] < 0 or indices[-1] >= depth:
        raise ValueError(
            f"target_blocks must be within [0, {depth - 1}]"
        )

    return indices


class GradientActivationImportanceCollector:
    def __init__(
        self,
        model: VisionTransformer,
        block_indices: Sequence[int],
        token_mode: str = "all",
    ):
        token_mode = token_mode.lower()
        if token_mode not in {"all", "cls", "patch"}:
            raise ValueError(
                "token_mode must be one of: 'all', 'cls', 'patch'"
            )

        self.model = model
        self.block_indices = list(block_indices)
        self.token_mode = token_mode
        self.scores: Dict[int, torch.Tensor] = {}
        self.counts: Dict[int, int] = {}
        self.handles = []

        for block_index in self.block_indices:
            fc2 = self.model.blocks[block_index].mlp.fc2
            if not isinstance(fc2, nn.Linear):
                raise TypeError(
                    "Importance must be collected before paired paths "
                    f"are injected. Block {block_index} fc2 is "
                    f"{type(fc2).__name__}."
                )

            hidden_dim = fc2.in_features
            self.scores[block_index] = torch.zeros(
                hidden_dim,
                dtype=torch.float64,
                device="cpu",
            )
            self.counts[block_index] = 0

            handle = fc2.register_forward_hook(
                self._make_forward_hook(block_index)
            )
            self.handles.append(handle)

    def _select_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3 or self.token_mode == "all":
            return tensor
        if self.token_mode == "cls":
            return tensor[:, :1, :]
        if tensor.shape[1] <= 1:
            return tensor
        return tensor[:, 1:, :]

    def _make_forward_hook(self, block_index: int):
        def forward_hook(module, inputs, output):
            del module, output
            hidden = inputs[0]
            if not hidden.requires_grad:
                raise RuntimeError(
                    "FFN hidden activations do not require gradients. "
                    "During importance collection, set input images with "
                    "requires_grad_(True)."
                )

            activation = hidden.detach()

            def gradient_hook(gradient: torch.Tensor):
                selected_activation = self._select_tokens(activation)
                selected_gradient = self._select_tokens(gradient.detach())

                reduce_dims = tuple(
                    range(selected_activation.ndim - 1)
                )
                batch_score = (
                    selected_activation.float()
                    * selected_gradient.float()
                ).abs().sum(dim=reduce_dims)

                sample_count = (
                    selected_activation.numel()
                    // selected_activation.shape[-1]
                )

                self.scores[block_index].add_(
                    batch_score.double().cpu()
                )
                self.counts[block_index] += int(sample_count)

            hidden.register_hook(gradient_hook)

        return forward_hook

    def normalized_scores(self) -> Dict[int, torch.Tensor]:
        normalized = {}
        for block_index in self.block_indices:
            count = max(self.counts[block_index], 1)
            normalized[block_index] = (
                self.scores[block_index] / count
            ).float()
        return normalized

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _extract_images_labels(batch):
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]

    if isinstance(batch, dict):
        image = None
        label = None
        for key in ("image", "images", "input", "inputs", "x"):
            if key in batch:
                image = batch[key]
                break
        for key in ("label", "labels", "target", "targets", "y"):
            if key in batch:
                label = batch[key]
                break
        if image is not None and label is not None:
            return image, label

    raise TypeError(
        "Each batch must be (images, labels) or a dictionary containing "
        "image and label tensors."
    )


def build_balanced_class_weights(
    labels: Sequence[int],
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    label_tensor = torch.as_tensor(
        labels,
        dtype=torch.long,
    )
    counts = torch.bincount(
        label_tensor,
        minlength=num_classes,
    )

    if counts.numel() != num_classes:
        raise RuntimeError(
            "Unexpected class-count vector length"
        )
    if torch.any(counts == 0):
        missing = torch.nonzero(
            counts == 0,
            as_tuple=False,
        ).flatten().tolist()
        raise ValueError(
            "Cannot build class-balanced gradient weights because "
            f"the training set has no samples for classes {missing}"
        )

    total = counts.sum().float()
    weights = total / (
        float(num_classes) * counts.float()
    )

    return counts, weights


def collect_gradient_activation_importance(
    model: VisionTransformer,
    data_loader: Iterable,
    criterion: nn.Module,
    device: Union[str, torch.device],
    block_indices: Sequence[int],
    num_batches: int = 20,
    token_mode: str = "all",
    use_amp: bool = False,
) -> Dict[int, torch.Tensor]:
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")

    device = torch.device(device)
    previous_training_state = model.training
    model.eval()
    collector = GradientActivationImportanceCollector(
        model=model,
        block_indices=block_indices,
        token_mode=token_mode,
    )

    amp_enabled = bool(use_amp and device.type == "cuda")

    try:
        for batch_index, batch in enumerate(data_loader):
            if batch_index >= num_batches:
                break

            images, labels = _extract_images_labels(batch)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = images.detach().requires_grad_(True)

            model.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            loss.backward()
    finally:
        collector.close()
        model.train(previous_training_state)
        model.zero_grad(set_to_none=True)

    return collector.normalized_scores()


def select_topk_per_block(
    importance_scores: Dict[int, torch.Tensor],
    rank_per_block: int,
) -> Dict[int, torch.Tensor]:
    if rank_per_block <= 0:
        raise ValueError("rank_per_block must be positive")

    selected = {}
    for block_index, score in importance_scores.items():
        if rank_per_block > score.numel():
            raise ValueError(
                f"rank_per_block={rank_per_block} exceeds hidden size "
                f"{score.numel()} in block {block_index}"
            )
        indices = torch.topk(
            score,
            k=rank_per_block,
            largest=True,
            sorted=False,
        ).indices
        selected[block_index] = indices.sort().values.cpu()

    return selected


def select_with_global_budget(
    importance_scores: Dict[int, torch.Tensor],
    total_budget: int,
    min_rank_per_block: int = 1,
    max_rank_per_block: int = 128,
    normalize_mode: str = "mean",
    eps: float = 1e-8,
) -> Dict[int, torch.Tensor]:
    block_indices = sorted(importance_scores)

    if not block_indices:
        raise ValueError("importance_scores cannot be empty")
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    if min_rank_per_block < 0:
        raise ValueError("min_rank_per_block cannot be negative")
    if max_rank_per_block <= 0:
        raise ValueError("max_rank_per_block must be positive")
    if min_rank_per_block > max_rank_per_block:
        raise ValueError(
            "min_rank_per_block cannot exceed max_rank_per_block"
        )

    normalize_mode = normalize_mode.lower()
    if normalize_mode not in {"mean", "l2", "max", "none"}:
        raise ValueError(
            "normalize_mode must be one of: "
            "'mean', 'l2', 'max', 'none'"
        )

    minimum_budget = min_rank_per_block * len(block_indices)
    maximum_budget = sum(
        min(
            max_rank_per_block,
            importance_scores[block_index].numel(),
        )
        for block_index in block_indices
    )

    if total_budget < minimum_budget:
        raise ValueError(
            f"total_budget={total_budget} is smaller than the required "
            f"minimum budget={minimum_budget}"
        )
    if total_budget > maximum_budget:
        raise ValueError(
            f"total_budget={total_budget} exceeds the maximum budget "
            f"{maximum_budget} under max_rank_per_block="
            f"{max_rank_per_block}"
        )

    normalized_scores: Dict[int, torch.Tensor] = {}

    for block_index in block_indices:
        score = (
            importance_scores[block_index]
            .detach()
            .float()
            .cpu()
            .clamp_min(0)
        )

        if min_rank_per_block > score.numel():
            raise ValueError(
                f"Block {block_index}: min_rank_per_block="
                f"{min_rank_per_block} exceeds hidden size="
                f"{score.numel()}"
            )

        layer_max_rank = min(
            max_rank_per_block,
            score.numel(),
        )
        if min_rank_per_block > layer_max_rank:
            raise ValueError(
                f"Block {block_index}: min_rank_per_block="
                f"{min_rank_per_block} exceeds the allowed maximum "
                f"rank={layer_max_rank}"
            )

        if normalize_mode == "mean":
            denominator = score.mean()
        elif normalize_mode == "l2":
            denominator = torch.sqrt(
                torch.mean(score.pow(2))
            )
        elif normalize_mode == "max":
            denominator = score.max()
        else:
            denominator = torch.tensor(
                1.0,
                dtype=score.dtype,
            )

        normalized_scores[block_index] = (
            score / denominator.clamp_min(eps)
        )

    selected_sets: Dict[int, set[int]] = {
        block_index: set()
        for block_index in block_indices
    }

    for block_index in block_indices:
        score = normalized_scores[block_index]

        if min_rank_per_block > 0:
            initial_indices = torch.topk(
                score,
                k=min_rank_per_block,
                largest=True,
                sorted=False,
            ).indices.tolist()

            selected_sets[block_index].update(
                initial_indices
            )

    remaining_budget = total_budget - minimum_budget
    candidates = []

    for block_index in block_indices:
        score = normalized_scores[block_index]

        for neuron_index, value in enumerate(
            score.tolist()
        ):
            if (
                neuron_index
                not in selected_sets[block_index]
            ):
                candidates.append(
                    (
                        float(value),
                        block_index,
                        neuron_index,
                    )
                )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    allocated = 0

    for _, block_index, neuron_index in candidates:
        if allocated >= remaining_budget:
            break

        layer_max_rank = min(
            max_rank_per_block,
            normalized_scores[block_index].numel(),
        )

        if (
            len(selected_sets[block_index])
            >= layer_max_rank
        ):
            continue

        selected_sets[block_index].add(
            neuron_index
        )
        allocated += 1

    if allocated != remaining_budget:
        raise RuntimeError(
            f"Only allocated {allocated} of "
            f"{remaining_budget} remaining paths. "
            "Increase max_rank_per_block or reduce total_budget."
        )

    selected = {}

    for block_index in block_indices:
        indices = sorted(
            selected_sets[block_index]
        )

        selected[block_index] = torch.tensor(
            indices,
            dtype=torch.long,
        )

    return selected


def inject_paired_neuron_paths(
    model: VisionTransformer,
    selected_indices: Dict[int, torch.Tensor],
    update_selected_bias: bool = False,
    fc1_alpha: Optional[float] = None,
    fc2_alpha: Optional[float] = None,
) -> list[int]:
    injected_blocks = []

    for block_index in sorted(selected_indices):
        indices = selected_indices[block_index]
        block = model.blocks[block_index]
        old_fc1 = block.mlp.fc1
        old_fc2 = block.mlp.fc2

        if not isinstance(old_fc1, nn.Linear):
            raise TypeError(
                f"Block {block_index} fc1 must be nn.Linear before "
                f"injection, got {type(old_fc1).__name__}"
            )
        if not isinstance(old_fc2, nn.Linear):
            raise TypeError(
                f"Block {block_index} fc2 must be nn.Linear before "
                f"injection, got {type(old_fc2).__name__}"
            )
        if old_fc1.out_features != old_fc2.in_features:
            raise ValueError(
                f"Block {block_index} fc1/fc2 hidden dimensions do not match"
            )

        block.mlp.fc1 = PartialRowLinear.from_linear(
            old_fc1,
            selected_indices=indices,
            update_selected_bias=update_selected_bias,
            alpha=fc1_alpha,
        )
        block.mlp.fc2 = PartialColumnLinear.from_linear(
            old_fc2,
            selected_indices=indices,
            alpha=fc2_alpha,
        )
        injected_blocks.append(block_index)

    return injected_blocks


@torch.no_grad()
def sync_all_selected_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (PartialRowLinear, PartialColumnLinear)):
            module.sync_weight()


@torch.no_grad()
def merge_all_selected_paths(model: VisionTransformer) -> None:
    for block in model.blocks:
        if isinstance(block.mlp.fc1, PartialRowLinear):
            block.mlp.fc1 = block.mlp.fc1.to_merged_linear()
        if isinstance(block.mlp.fc2, PartialColumnLinear):
            block.mlp.fc2 = block.mlp.fc2.to_merged_linear()


def freeze_all_except_classifier(model: VisionTransformer) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    if model.head_dist is not None:
        for parameter in model.head_dist.parameters():
            parameter.requires_grad = True


def freeze_except_selected_paths_and_classifier(
    model: VisionTransformer,
    update_selected_bias: bool = False,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for module in model.modules():
        if isinstance(module, PartialRowLinear):
            module.selected_rows.requires_grad = True
            if (
                update_selected_bias
                and module.selected_bias is not None
            ):
                module.selected_bias.requires_grad = True
        elif isinstance(module, PartialColumnLinear):
            module.selected_columns.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    if model.head_dist is not None:
        for parameter in model.head_dist.parameters():
            parameter.requires_grad = True

def get_trainable_parameter_groups(
    model: VisionTransformer,
    path_lr: float = 1e-3,
    classifier_lr: Optional[float] = None,
    weight_decay: float = 0.01,
):
    if classifier_lr is None:
        classifier_lr = path_lr

    fc1_parameters = []
    fc2_parameters = []
    selected_bias_parameters = []
    classifier_parameters = []

    for module in model.modules():
        if isinstance(module, PartialRowLinear):
            fc1_parameters.append(module.selected_rows)
            if (
                module.selected_bias is not None
                and module.selected_bias.requires_grad
            ):
                selected_bias_parameters.append(
                    module.selected_bias
                )
        elif isinstance(module, PartialColumnLinear):
            fc2_parameters.append(module.selected_columns)

    classifier_parameters.extend(model.classifier.parameters())
    if model.head_dist is not None:
        classifier_parameters.extend(model.head_dist.parameters())

    groups = []
    if fc1_parameters:
        groups.append(
            {
                "params": fc1_parameters,
                "lr": path_lr,
                "weight_decay": weight_decay,
                "name": "selected_fc1_rows",
            }
        )
    if fc2_parameters:
        groups.append(
            {
                "params": fc2_parameters,
                "lr": path_lr,
                "weight_decay": weight_decay,
                "name": "selected_fc2_columns",
            }
        )
    if selected_bias_parameters:
        groups.append(
            {
                "params": selected_bias_parameters,
                "lr": path_lr,
                "weight_decay": 0.0,
                "name": "selected_fc1_bias",
            }
        )
    if classifier_parameters:
        groups.append(
            {
                "params": classifier_parameters,
                "lr": classifier_lr,
                "weight_decay": weight_decay,
                "name": "classifier",
            }
        )

    return groups

def build_head_optimizer(
    model: "VIT_GradActPaired",
    lr: float = 1e-3,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.backbone.classifier.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


def build_optimizer(
    model: "VIT_GradActPaired",
    path_lr: float = 1e-3,
    classifier_lr: Optional[float] = None,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    if not model.paths_injected:
        raise RuntimeError(
            "Call model.select_and_inject(...) before building the final optimizer"
        )

    parameter_groups = get_trainable_parameter_groups(
        model.backbone,
        path_lr=path_lr,
        classifier_lr=classifier_lr,
        weight_decay=weight_decay,
    )
    return torch.optim.AdamW(parameter_groups)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def _unwrap_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must contain a state_dict-like dictionary"
        )

    for key in ("model", "state_dict", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    return checkpoint


def _clean_checkpoint_key(key: str) -> str:
    prefixes = ("module.", "model.", "backbone.")

    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True

    return key


def load_pretrained_weights(
    model: nn.Module,
    weights_path: Union[str, Path],
):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}"
        )

    checkpoint = torch.load(
        weights_path,
        map_location="cpu",
    )
    source_state = _unwrap_checkpoint(checkpoint)
    target_state = model.state_dict()

    loaded_state = {}
    skipped_shape = []
    skipped_name = []

    for raw_key, value in source_state.items():
        if not torch.is_tensor(value):
            continue

        key = _clean_checkpoint_key(raw_key)

        if key.startswith("head."):
            key = "classifier." + key[len("head."):]

        if key not in target_state:
            skipped_name.append(raw_key)
            continue

        if target_state[key].shape != value.shape:
            skipped_shape.append(
                (
                    raw_key,
                    tuple(value.shape),
                    tuple(target_state[key].shape),
                )
            )
            continue

        loaded_state[key] = value

    missing, unexpected = model.load_state_dict(
        loaded_state,
        strict=False,
    )

    return {
        "loaded": len(loaded_state),
        "missing": missing,
        "unexpected": unexpected,
        "skipped_shape": skipped_shape,
        "skipped_name_count": len(skipped_name),
    }


def _init_vit_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def vit_base_patch16_224(
    num_classes: int = 1000,
) -> VisionTransformer:
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=None,
        num_classes=num_classes,
    )


class VIT_GradActPaired(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        weights_path: Optional[Union[str, Path]] = (
            r"vit_base_patch16_224.pth"
        ),
        target_blocks: Optional[Union[str, Sequence[int]]] = "all",
    ):
        super().__init__()

        self.backbone = vit_base_patch16_224(
            num_classes=num_classes,
        )

        self.load_report = None
        if weights_path is not None:
            self.load_report = load_pretrained_weights(
                self.backbone,
                weights_path,
            )

        self.target_block_indices = _normalize_target_blocks(
            depth=len(self.backbone.blocks),
            target_blocks=target_blocks,
        )
        self.importance_scores: Dict[int, torch.Tensor] = {}
        self.selected_indices: Dict[int, torch.Tensor] = {}
        self.update_selected_bias = False
        self.paths_injected = False

        freeze_all_except_classifier(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def select_and_inject(
        self,
        data_loader: Iterable,
        criterion: nn.Module,
        device: Union[str, torch.device],
        num_calibration_batches: int = 20,
        rank_per_block: Optional[int] = 64,
        total_budget: Optional[int] = None,
        min_rank_per_block: int = 1,
        max_rank_per_block: int = 128,
        normalize_mode: str = "mean",
        token_mode: str = "cls",
        use_amp: bool = False,
        update_selected_bias: bool = False,
        fc1_alpha: Optional[float] = None,
        fc2_alpha: Optional[float] = None,
    ) -> Dict[str, object]:
        if self.paths_injected:
            raise RuntimeError("Selected neuron paths have already been injected")
        if rank_per_block is None and total_budget is None:
            raise ValueError(
                "Provide rank_per_block or total_budget"
            )
        if rank_per_block is not None and total_budget is not None:
            raise ValueError(
                "Use either rank_per_block or total_budget, not both"
            )

        self.importance_scores = collect_gradient_activation_importance(
            model=self.backbone,
            data_loader=data_loader,
            criterion=criterion,
            device=device,
            block_indices=self.target_block_indices,
            num_batches=num_calibration_batches,
            token_mode=token_mode,
            use_amp=use_amp,
        )

        if total_budget is not None:
            self.selected_indices = select_with_global_budget(
                importance_scores=self.importance_scores,
                total_budget=total_budget,
                min_rank_per_block=min_rank_per_block,
                max_rank_per_block=max_rank_per_block,
                normalize_mode=normalize_mode,
            )
        else:
            self.selected_indices = select_topk_per_block(
                importance_scores=self.importance_scores,
                rank_per_block=int(rank_per_block),
            )

        self.update_selected_bias = bool(update_selected_bias)
        injected_blocks = inject_paired_neuron_paths(
            model=self.backbone,
            selected_indices=self.selected_indices,
            update_selected_bias=self.update_selected_bias,
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
        )
        freeze_except_selected_paths_and_classifier(
            self.backbone,
            update_selected_bias=self.update_selected_bias,
        )
        self.paths_injected = True

        rank_by_block = {
            index: int(indices.numel())
            for index, indices in self.selected_indices.items()
        }
        mean_score_by_block = {
            index: float(self.importance_scores[index].mean())
            for index in self.importance_scores
        }

        return {
            "injected_blocks": injected_blocks,
            "rank_by_block": rank_by_block,
            "mean_score_by_block": mean_score_by_block,
            "update_selected_bias": bool(
                self.update_selected_bias
            ),
            "selected_indices": {
                index: indices.clone()
                for index, indices in self.selected_indices.items()
            },
        }

    def inject_from_indices(
        self,
        selected_indices: Dict[int, torch.Tensor],
        update_selected_bias: bool = False,
        fc1_alpha: Optional[float] = None,
        fc2_alpha: Optional[float] = None,
    ) -> None:
        if self.paths_injected:
            raise RuntimeError("Selected neuron paths have already been injected")

        self.selected_indices = {
            int(index): torch.as_tensor(
                indices,
                dtype=torch.long,
            ).cpu()
            for index, indices in selected_indices.items()
        }
        self.update_selected_bias = bool(update_selected_bias)
        inject_paired_neuron_paths(
            model=self.backbone,
            selected_indices=self.selected_indices,
            update_selected_bias=self.update_selected_bias,
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
        )
        freeze_except_selected_paths_and_classifier(
            self.backbone,
            update_selected_bias=self.update_selected_bias,
        )
        self.paths_injected = True

    @torch.no_grad()
    def sync_selected_weights(self) -> None:
        sync_all_selected_weights(self.backbone)

    @torch.no_grad()
    def merge_for_inference(self) -> None:
        sync_all_selected_weights(self.backbone)
        merge_all_selected_paths(self.backbone)
        self.paths_injected = False

    def print_trainable_parameters(self) -> None:
        total, trainable = count_parameters(self)

        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                print(
                    f"{name}: shape={tuple(parameter.shape)}, "
                    f"numel={parameter.numel():,}"
                )

        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(
            f"Trainable ratio: "
            f"{100.0 * trainable / total:.4f}%"
        )
        print(f"Target blocks: {self.target_block_indices}")
        if self.selected_indices:
            print(
                "Selected ranks: "
                + str(
                    {
                        index: int(indices.numel())
                        for index, indices in self.selected_indices.items()
                    }
                )
            )


VIT_PaCA = VIT_GradActPaired
Evidential_MoE = VIT_GradActPaired


import argparse
import datetime
import json
import os
import random

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def print_and_save(file_path: str, message: str) -> None:
    print(message)
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(message + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_target_blocks(value: str):
    value = value.strip()

    if value.lower() == "all":
        return "all"

    if value.lower().startswith("last"):
        return value.lower()

    if "," in value:
        return [
            int(index.strip())
            for index in value.split(",")
            if index.strip()
        ]

    return [int(value)]


def extract_logits(output: torch.Tensor) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def build_data_loaders(
    dataset_root: str,
    batch_size: int,
    num_workers: int,
):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ]
    )

    calibration_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ]
    )

    train_directory = os.path.join(dataset_root, "train")
    validation_directory = os.path.join(dataset_root, "val")

    train_dataset = datasets.ImageFolder(
        train_directory,
        train_transform,
    )
    calibration_dataset = datasets.ImageFolder(
        train_directory,
        calibration_transform,
    )
    validation_dataset = datasets.ImageFolder(
        validation_directory,
        validation_transform,
    )

    pin_memory = torch.cuda.is_available()
    worker_count = min(
        max(int(num_workers), 0),
        os.cpu_count() or 1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=worker_count,
        pin_memory=pin_memory,
    )

    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=pin_memory,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=pin_memory,
    )

    return (
        train_dataset,
        validation_dataset,
        train_loader,
        calibration_loader,
        validation_loader,
    )


def train_head_warmup_one_epoch(
    model: VIT_GradActPaired,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    warmup_epoch: int,
    warmup_epochs: int,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled,
    )

    progress_bar = tqdm(
        data_loader,
        desc=(
            f"Classifier Warm-up Epoch "
            f"{warmup_epoch}/{warmup_epochs}"
        ),
    )

    for images, labels in progress_bar:
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(
            enabled=amp_enabled,
        ):
            logits = extract_logits(model(images))
            loss = criterion(logits, labels)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.detach().item()
        progress_bar.set_postfix(loss=f"{loss.detach().item():.4f}")

    return running_loss / max(len(data_loader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    description: str,
) -> tuple[float, float]:
    model.eval()

    running_loss = 0.0
    correct = 0

    for images, labels in tqdm(
        data_loader,
        desc=description,
    ):
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = extract_logits(model(images))
        loss = criterion(logits, labels)

        running_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += predictions.eq(labels).sum().item()

    average_loss = running_loss / max(len(data_loader), 1)
    accuracy = correct / max(len(data_loader.dataset), 1)

    return average_loss, accuracy


def train_one_epoch(
    model: VIT_GradActPaired,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled,
    )

    progress_bar = tqdm(
        data_loader,
        desc=f"Train Epoch {epoch}/{total_epochs}",
    )

    for images, labels in progress_bar:
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(
            enabled=amp_enabled,
        ):
            logits = extract_logits(model(images))
            loss = criterion(logits, labels)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.detach().item()
        progress_bar.set_postfix(loss=f"{loss.detach().item():.4f}")

    return running_loss / max(len(data_loader), 1)


def make_serializable_selection_report(
    selection_report: dict,
) -> dict:
    return {
        "injected_blocks": [
            int(index)
            for index in selection_report["injected_blocks"]
        ],
        "rank_by_block": {
            str(index): int(rank)
            for index, rank in selection_report["rank_by_block"].items()
        },
        "mean_score_by_block": {
            str(index): float(score)
            for index, score in selection_report[
                "mean_score_by_block"
            ].items()
        },
        "update_selected_bias": bool(
            selection_report["update_selected_bias"]
        ),
        "selected_indices": {
            str(index): indices.tolist()
            for index, indices in selection_report[
                "selected_indices"
            ].items()
        },
    }


def save_checkpoint(
    model: VIT_GradActPaired,
    checkpoint_path: str,
    epoch: int,
    validation_accuracy: float,
    args: argparse.Namespace,
) -> None:
    model.sync_selected_weights()

    checkpoint = {
        "epoch": int(epoch),
        "validation_accuracy": float(validation_accuracy),
        "model_state_dict": model.state_dict(),
        "selected_indices": {
            int(index): indices.detach().cpu()
            for index, indices in model.selected_indices.items()
        },
        "args": vars(args),
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Medical image classification with configurable classifier "
            "warm-up, optional class-balanced gradient importance, and "
            "gradient-activation paired FFN path adaptation"
        )
    )

    parser.add_argument(
        "--method",
        type=str,
        default="GradActPaired",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="BUSI",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--classifier_lr",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--warmup_lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=r"vit_base_patch16_224.pth",
    )
    parser.add_argument(
        "--target_blocks",
        type=str,
        default="all",
    )
    parser.add_argument(
        "--calibration_batches",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--rank_per_block",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--total_budget",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--min_rank_per_block",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--max_rank_per_block",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--normalize_mode",
        type=str,
        choices=("mean", "l2", "max", "none"),
        default="mean",
    )
    parser.add_argument(
        "--token_mode",
        type=str,
        choices=("all", "cls", "patch"),
        default="cls",
    )
    parser.add_argument(
        "--class_balanced_gradient",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Use inverse-frequency class weights only when computing "
            "the calibration gradients: 1=enabled, 0=disabled"
        ),
    )
    parser.add_argument(
        "--update_selected_bias",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Update fc1 bias entries corresponding to selected hidden "
            "neurons: 1=enabled, 0=disabled"
        ),
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.warmup_epochs <= 0:
        raise ValueError("--warmup_epochs must be positive")
    if args.calibration_batches <= 0:
        raise ValueError("--calibration_batches must be positive")
    if args.total_budget <= 0 and args.rank_per_block <= 0:
        raise ValueError(
            "Set a positive --rank_per_block or --total_budget"
        )

    dataset_root = (
        args.dataset_root
        if args.dataset_root is not None
        else os.path.join("dataset", args.dataset)
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )
    set_seed(args.seed)

    current_time = datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    folder_name = (
        f"{args.method}_{args.dataset}_lr{args.lr}_{current_time}"
    )
    save_directory = os.path.join(
        "run_files_GradActPaired_final",
        args.method,
        args.dataset,
        folder_name,
    )
    os.makedirs(
        save_directory,
        exist_ok=True,
    )

    train_log_path = os.path.join(
        save_directory,
        "train_log.txt",
    )
    checkpoint_best_path = os.path.join(
        save_directory,
        "checkpoint_best.pth",
    )
    checkpoint_last_path = os.path.join(
        save_directory,
        "checkpoint_last.pth",
    )
    selection_path = os.path.join(
        save_directory,
        "selection_report.json",
    )

    with open(
        os.path.join(save_directory, "config.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            vars(args),
            file,
            indent=4,
            ensure_ascii=False,
        )

    print_and_save(
        train_log_path,
        f"Experiment: {args.method} | Dataset: {args.dataset}",
    )
    print_and_save(
        train_log_path,
        f"Device: {device}",
    )
    print_and_save(
        train_log_path,
        f"Dataset root: {dataset_root}",
    )
    print_and_save(
        train_log_path,
        f"Save directory: {save_directory}",
    )
    print_and_save(
        train_log_path,
        f"Warm-up epochs: {args.warmup_epochs}",
    )
    print_and_save(
        train_log_path,
        (
            "Class-balanced calibration gradient: "
            f"{bool(args.class_balanced_gradient)}"
        ),
    )
    print_and_save(
        train_log_path,
        (
            "Update selected fc1 bias: "
            f"{bool(args.update_selected_bias)}"
        ),
    )

    (
        train_dataset,
        validation_dataset,
        train_loader,
        calibration_loader,
        validation_loader,
    ) = build_data_loaders(
        dataset_root=dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if len(train_dataset.classes) != args.num_classes:
        raise ValueError(
            f"ImageFolder found {len(train_dataset.classes)} classes "
            f"{train_dataset.classes}, but --num_classes="
            f"{args.num_classes}"
        )

    if train_dataset.class_to_idx != validation_dataset.class_to_idx:
        raise ValueError(
            "The train and validation class mappings are inconsistent"
        )

    print_and_save(
        train_log_path,
        (
            f"Train samples: {len(train_dataset)} | "
            f"Validation samples: {len(validation_dataset)}"
        ),
    )
    print_and_save(
        train_log_path,
        f"Classes: {train_dataset.classes}",
    )

    target_blocks = parse_target_blocks(
        args.target_blocks
    )

    model = VIT_GradActPaired(
        num_classes=args.num_classes,
        weights_path=args.pretrained_path,
        target_blocks=target_blocks,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    class_counts, class_weights = build_balanced_class_weights(
        labels=train_dataset.targets,
        num_classes=args.num_classes,
    )
    if bool(args.class_balanced_gradient):
        calibration_criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device)
        )
    else:
        calibration_criterion = criterion

    print_and_save(
        train_log_path,
        f"Class counts: {class_counts.tolist()}",
    )
    print_and_save(
        train_log_path,
        (
            "Calibration class weights: "
            f"{class_weights.tolist() if bool(args.class_balanced_gradient) else 'disabled'}"
        ),
    )

    total_parameters, trainable_parameters = count_parameters(
        model
    )
    print_and_save(
        train_log_path,
        (
            "Before warm-up - "
            f"Total parameters: {total_parameters:,} "
            f"({total_parameters / 1e6:.2f}M)"
        ),
    )
    print_and_save(
        train_log_path,
        (
            "Before warm-up - "
            f"Trainable parameters: {trainable_parameters:,} "
            f"({trainable_parameters / 1e6:.2f}M)"
        ),
    )

    head_optimizer = build_head_optimizer(
        model,
        lr=args.warmup_lr,
        weight_decay=args.weight_decay,
    )

    warmup_validation_loss = float("nan")
    warmup_validation_accuracy = 0.0

    for warmup_epoch in range(
        1,
        args.warmup_epochs + 1,
    ):
        warmup_loss = train_head_warmup_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=head_optimizer,
            criterion=criterion,
            device=device,
            warmup_epoch=warmup_epoch,
            warmup_epochs=args.warmup_epochs,
            use_amp=args.use_amp,
        )

        (
            warmup_validation_loss,
            warmup_validation_accuracy,
        ) = evaluate(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            description=(
                f"Validation after Warm-up "
                f"{warmup_epoch}/{args.warmup_epochs}"
            ),
        )

        print_and_save(
            train_log_path,
            (
                f"[Warm-up Epoch {warmup_epoch}/"
                f"{args.warmup_epochs}] "
                f"Train Loss: {warmup_loss:.4f} | "
                f"Validation Loss: "
                f"{warmup_validation_loss:.4f} | "
                f"Validation Accuracy: "
                f"{warmup_validation_accuracy:.4f}"
            ),
        )

    del head_optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    use_global_budget = args.total_budget > 0

    if use_global_budget:
        print_and_save(
            train_log_path,
            (
                "Global budget selection: "
                f"total_budget={args.total_budget}, "
                f"min_rank_per_block={args.min_rank_per_block}, "
                f"max_rank_per_block={args.max_rank_per_block}, "
                f"normalize_mode={args.normalize_mode}, "
                f"token_mode={args.token_mode}"
            ),
        )
    else:
        print_and_save(
            train_log_path,
            (
                "Fixed per-block selection: "
                f"rank_per_block={args.rank_per_block}, "
                f"token_mode={args.token_mode}"
            ),
        )

    selection_report = model.select_and_inject(
        data_loader=calibration_loader,
        criterion=calibration_criterion,
        device=device,
        num_calibration_batches=args.calibration_batches,
        rank_per_block=(
            None
            if use_global_budget
            else args.rank_per_block
        ),
        total_budget=(
            args.total_budget
            if use_global_budget
            else None
        ),
        min_rank_per_block=args.min_rank_per_block,
        max_rank_per_block=args.max_rank_per_block,
        normalize_mode=args.normalize_mode,
        token_mode=args.token_mode,
        use_amp=False,
        update_selected_bias=bool(
            args.update_selected_bias
        ),
    )

    serializable_selection_report = (
        make_serializable_selection_report(
            selection_report
        )
    )

    with open(
        selection_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_selection_report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print_and_save(
        train_log_path,
        (
            "Selected ranks by block: "
            + json.dumps(
                serializable_selection_report[
                    "rank_by_block"
                ],
                ensure_ascii=False,
            )
        ),
    )
    print_and_save(
        train_log_path,
        (
            "Update selected fc1 bias: "
            f"{serializable_selection_report['update_selected_bias']}"
        ),
    )
    print_and_save(
        train_log_path,
        (
            "Mean importance by block: "
            + json.dumps(
                serializable_selection_report[
                    "mean_score_by_block"
                ],
                ensure_ascii=False,
            )
        ),
    )

    total_parameters, trainable_parameters = count_parameters(
        model
    )
    print_and_save(
        train_log_path,
        (
            "After path injection - "
            f"Total parameters: {total_parameters:,} "
            f"({total_parameters / 1e6:.2f}M)"
        ),
    )
    print_and_save(
        train_log_path,
        (
            "After path injection - "
            f"Trainable parameters: {trainable_parameters:,} "
            f"({trainable_parameters / 1e6:.2f}M)"
        ),
    )
    print_and_save(
        train_log_path,
        (
            "Trainable ratio: "
            f"{100.0 * trainable_parameters / total_parameters:.4f}%"
        ),
    )

    final_optimizer = build_optimizer(
        model,
        path_lr=args.lr,
        classifier_lr=args.classifier_lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        final_optimizer,
        mode="max",
        patience=10,
    )

    best_accuracy = 0.0
    early_stopping_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=final_optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            use_amp=args.use_amp,
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            description=f"Validation Epoch {epoch}",
        )

        scheduler.step(validation_accuracy)

        current_lrs = [
            float(group["lr"])
            for group in final_optimizer.param_groups
        ]

        print_and_save(
            train_log_path,
            (
                f"[Main Epoch {epoch:03d}/{args.epochs:03d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Validation Loss: {validation_loss:.4f} | "
                f"Validation Accuracy: {validation_accuracy:.4f} | "
                f"LRs: {current_lrs}"
            ),
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            save_checkpoint(
                model=model,
                checkpoint_path=checkpoint_best_path,
                epoch=epoch,
                validation_accuracy=validation_accuracy,
                args=args,
            )
            print_and_save(
                train_log_path,
                (
                    "Best model updated "
                    f"(Accuracy={best_accuracy:.4f})"
                ),
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if (
            early_stopping_counter
            >= args.early_stopping_patience
        ):
            print_and_save(
                train_log_path,
                "Early stopping triggered.",
            )
            break

    save_checkpoint(
        model=model,
        checkpoint_path=checkpoint_last_path,
        epoch=epoch,
        validation_accuracy=validation_accuracy,
        args=args,
    )

    print_and_save(
        train_log_path,
        "=" * 60,
    )
    print_and_save(
        train_log_path,
        f"Best Accuracy: {best_accuracy:.4f}",
    )
    print_and_save(
        train_log_path,
        f"Best checkpoint: {checkpoint_best_path}",
    )
    print_and_save(
        train_log_path,
        f"Last checkpoint: {checkpoint_last_path}",
    )
    print_and_save(
        train_log_path,
        f"Selection report: {selection_path}",
    )
    print_and_save(
        train_log_path,
        "=" * 60,
    )


if __name__ == "__main__":
    main()

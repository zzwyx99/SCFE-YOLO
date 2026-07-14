from __future__ import annotations

import ast
import contextlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parent
MODEL_CFG_PATH = REPO_ROOT / "ultralytics11" / "cfg" / "models" / "AF-YOLO.yaml"

from ultralytics.nn.modules import Bottleneck, Conv  # noqa: E402


def _autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=8):
        super().__init__()
        self.groups = factor
        if channels // self.groups <= 0:
            raise ValueError(f"EMA expects channels >= groups, got channels={channels}, groups={self.groups}")
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class CPCA_ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super().__init__()
        self.fc1 = nn.Conv2d(input_channels, internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(internal_neurons, input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc2(F.relu(self.fc1(x1), inplace=True)).sigmoid()
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc2(F.relu(self.fc1(x2), inplace=True)).sigmoid()
        x = (x1 + x2).view(-1, self.input_channels, 1, 1)
        return inputs * x


class CPCA(nn.Module):
    def __init__(self, channels, channel_attention_reduce=4):
        super().__init__()
        self.ca = CPCA_ChannelAttention(channels, channels // channel_attention_reduce)
        self.dconv5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dconv1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.dconv7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.dconv1_11 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.dconv11_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        inputs = self.act(self.conv(inputs))
        inputs = self.ca(inputs)

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv7_1(self.dconv1_7(x_init))
        x_2 = self.dconv11_1(self.dconv1_11(x_init))
        x_3 = self.dconv21_1(self.dconv1_21(x_init))
        spatial_att = self.conv(x_init + x_1 + x_2 + x_3)
        out = self.conv(spatial_att * inputs)
        return out


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num, group_num=16, eps=1e-10):
        super().__init__()
        if c_num < group_num:
            raise ValueError(f"GroupBatchnorm2d expects c_num >= group_num, got {c_num} < {group_num}")
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(n, c, h, w)
        return x * self.gamma + self.beta


class SRU(nn.Module):
    def __init__(self, oup_channels, group_num=16, gate_treshold=0.5):
        super().__init__()
        self.gn = GroupBatchnorm2d(oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.gamma / self.gn.gamma.sum()
        reweights = self.sigmoid(gn_x * w_gamma)
        info_mask = reweights >= self.gate_treshold
        noninfo_mask = reweights < self.gate_treshold
        return self.reconstruct(info_mask * x, noninfo_mask * x)

    @staticmethod
    def reconstruct(x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    def __init__(
        self,
        op_channel,
        alpha=0.5,
        squeeze_radio=2,
        group_size=2,
        group_kernel_size=3,
    ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        self.gwc = nn.Conv2d(
            up_channel // squeeze_radio,
            op_channel,
            kernel_size=group_kernel_size,
            stride=1,
            padding=group_kernel_size // 2,
            groups=group_size,
        )
        self.pwc1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        self.pwc2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1, bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        y1 = self.gwc(up) + self.pwc1(up)
        y2 = torch.cat([self.pwc2(low), low], dim=1)
        out = torch.cat([y1, y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(
        self,
        op_channel,
        group_num=16,
        gate_treshold=0.5,
        alpha=0.5,
        squeeze_radio=2,
        group_size=2,
        group_kernel_size=3,
    ):
        super().__init__()
        self.sru = SRU(op_channel, group_num=group_num, gate_treshold=gate_treshold)
        self.cru = CRU(
            op_channel,
            alpha=alpha,
            squeeze_radio=squeeze_radio,
            group_size=group_size,
            group_kernel_size=group_kernel_size,
        )

    def forward(self, x):
        return self.cru(self.sru(x))


class Bottleneck_ScConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv1 = ScConv(c_)
        self.cv2 = ScConv(c2)


class ConvCPCA(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.att = CPCA(c2)

    def forward(self, x):
        return self.att(self.act(self.bn(self.conv(x))))

    def forward_fuse(self, x):
        return self.att(self.act(self.conv(x)))


class SCC2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvCPCA((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_ScConv(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class RFAConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.get_weight = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
            nn.Conv2d(in_channel, in_channel * (kernel_size**2), kernel_size=1, groups=in_channel, bias=False),
        )
        self.generate_feature = nn.Sequential(
            nn.Conv2d(
                in_channel,
                in_channel * (kernel_size**2),
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                groups=in_channel,
                bias=False,
            ),
            nn.BatchNorm2d(in_channel * (kernel_size**2)),
            nn.ReLU(),
        )
        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)

    def forward(self, x):
        b, c = x.shape[:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size**2, h, w).softmax(2)
        feature = self.generate_feature(x).view(b, c, self.kernel_size**2, h, w)
        weighted_data = feature * weighted
        weighted_data = weighted_data.view(b, c, self.kernel_size, self.kernel_size, h, w)
        conv_data = weighted_data.permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * self.kernel_size, w * self.kernel_size)
        return self.conv(conv_data)


class ConvEMAT(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.att = EMA(c2)

    def forward(self, x):
        return self.att(self.act(self.bn(self.conv(x))))

    def forward_fuse(self, x):
        return self.att(self.act(self.conv(x)))


class RFABN(nn.Module):
    def __init__(self, c1, c2, shortcut=True, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = RFAConv(c1, c_, 3, 1)
        self.cv2 = RFAConv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class CRDR(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvEMAT((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(RFABN(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode="bilinear"),
        )

    def forward(self, x):
        return self.upsample(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.downsample = nn.Sequential(Conv(in_channels, out_channels, scale_factor, scale_factor, 0))

    def forward(self, x):
        return self.downsample(x)


class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=(256, 512)):
        super().__init__()
        self.inter_dim = inter_dim
        compress_c = 8
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)
        self.upsample = Upsample(channel[1], channel[0])
        self.downsample = Downsample(channel[0], channel[1])
        self.level = level

    def forward(self, x):
        input1, input2 = x
        if self.level == 0:
            input2 = self.upsample(input2)
        elif self.level == 1:
            input1 = self.downsample(input1)

        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        levels_weight = F.softmax(self.weight_levels(torch.cat((level_1_weight_v, level_2_weight_v), 1)), dim=1)
        fused_out = input1 * levels_weight[:, 0:1, :, :] + input2 * levels_weight[:, 1:2, :, :]
        return self.conv(fused_out)


class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=(64, 128, 256)):
        super().__init__()
        self.inter_dim = inter_dim
        compress_c = 8
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)
        self.level = level
        if self.level == 0:
            self.upsample4x = Upsample(channel[2], channel[0], scale_factor=4)
            self.upsample2x = Upsample(channel[1], channel[0], scale_factor=2)
        elif self.level == 1:
            self.upsample2x1 = Upsample(channel[2], channel[1], scale_factor=2)
            self.downsample2x1 = Downsample(channel[0], channel[1], scale_factor=2)
        elif self.level == 2:
            self.downsample2x = Downsample(channel[1], channel[2], scale_factor=2)
            self.downsample4x = Downsample(channel[0], channel[2], scale_factor=4)

    def forward(self, x):
        input1, input2, input3 = x
        if self.level == 0:
            input2 = self.upsample2x(input2)
            input3 = self.upsample4x(input3)
        elif self.level == 1:
            input3 = self.upsample2x1(input3)
            input1 = self.downsample2x1(input1)
        elif self.level == 2:
            input1 = self.downsample4x(input1)
            input2 = self.downsample2x(input2)

        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)
        levels_weight = F.softmax(
            self.weight_levels(torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)),
            dim=1,
        )
        fused_out = (
            input1 * levels_weight[:, 0:1, :, :]
            + input2 * levels_weight[:, 1:2, :, :]
            + input3 * levels_weight[:, 2:, :, :]
        )
        return self.conv(fused_out)


class ASFF_4(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=(64, 128, 256, 512)):
        super().__init__()
        self.inter_dim = inter_dim
        compress_c = 8
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_4 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)
        self.level = level
        if self.level == 0:
            self.upsample8x = Upsample(channel[3], channel[0], scale_factor=8)
            self.upsample4x = Upsample(channel[2], channel[0], scale_factor=4)
            self.upsample2x = Upsample(channel[1], channel[0], scale_factor=2)
        elif self.level == 1:
            self.upsample4x1 = Upsample(channel[3], channel[1], scale_factor=4)
            self.upsample2x1 = Upsample(channel[2], channel[1], scale_factor=2)
            self.downsample2x1 = Downsample(channel[0], channel[1], scale_factor=2)
        elif self.level == 2:
            self.upsample2x2 = Upsample(channel[3], channel[2], scale_factor=2)
            self.downsample2x2 = Downsample(channel[1], channel[2], scale_factor=2)
            self.downsample4x2 = Downsample(channel[0], channel[2], scale_factor=4)
        elif self.level == 3:
            self.downsample2x3 = Downsample(channel[2], channel[3], scale_factor=2)
            self.downsample4x3 = Downsample(channel[1], channel[3], scale_factor=4)
            self.downsample8x3 = Downsample(channel[0], channel[3], scale_factor=8)

    def forward(self, x):
        input1, input2, input3, input4 = x
        if self.level == 0:
            input2 = self.upsample2x(input2)
            input3 = self.upsample4x(input3)
            input4 = self.upsample8x(input4)
        elif self.level == 1:
            input3 = self.upsample2x1(input3)
            input4 = self.upsample4x1(input4)
            input1 = self.downsample2x1(input1)
        elif self.level == 2:
            input4 = self.upsample2x2(input4)
            input1 = self.downsample4x2(input1)
            input2 = self.downsample2x2(input2)
        elif self.level == 3:
            input1 = self.downsample8x3(input1)
            input2 = self.downsample4x3(input2)
            input3 = self.downsample2x3(input3)

        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)
        level_4_weight_v = self.weight_level_4(input4)
        levels_weight = F.softmax(
            self.weight_levels(torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v, level_4_weight_v), 1)),
            dim=1,
        )
        fused_out = (
            input1 * levels_weight[:, 0:1, :, :]
            + input2 * levels_weight[:, 1:2, :, :]
            + input3 * levels_weight[:, 2:3, :, :]
            + input4 * levels_weight[:, 3:, :, :]
        )
        return self.conv(fused_out)


def _available(tasks, *names):
    return frozenset(getattr(tasks, name) for name in names if hasattr(tasks, name))


def _build_parse_model(tasks):
    def parse_model(d, ch, verbose=True):
        legacy = True
        max_channels = float("inf")
        nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
        reg_max = d.get("reg_max", 16)
        depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
        scale = d.get("scale")
        if scales:
            if not scale:
                scale = next(iter(scales.keys()))
                tasks.LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]

        if act:
            tasks.Conv.default_act = eval(act)
            if verbose:
                tasks.LOGGER.info(f"{tasks.colorstr('activation:')} {act}")

        if verbose:
            tasks.LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

        ch = [ch]
        layers, save, c2 = [], [], ch[-1]
        tasks_globals = tasks.__dict__
        base_modules = _available(
            tasks,
            "Classify",
            "Conv",
            "ConvTranspose",
            "GhostConv",
            "Bottleneck",
            "GhostBottleneck",
            "SPP",
            "SPPF",
            "C2fPSA",
            "C2PSA",
            "DWConv",
            "Focus",
            "BottleneckCSP",
            "C1",
            "C2",
            "C2f",
            "C3k2",
            "RepNCSPELAN4",
            "ELAN1",
            "ADown",
            "AConv",
            "SPPELAN",
            "C2fAttn",
            "C3",
            "C3TR",
            "C3Ghost",
            "DWConvTranspose2d",
            "C3x",
            "RepC3",
            "PSA",
            "SCDown",
            "C2fCIB",
            "A2C2f",
        ) | frozenset({torch.nn.ConvTranspose2d, SCC2f, CRDR})
        repeat_modules = _available(
            tasks,
            "BottleneckCSP",
            "C1",
            "C2",
            "C2f",
            "C3k2",
            "C2fAttn",
            "C3",
            "C3TR",
            "C3Ghost",
            "C3x",
            "RepC3",
            "C2fPSA",
            "C2fCIB",
            "C2PSA",
            "A2C2f",
        ) | frozenset({SCC2f, CRDR})
        detect_modules = _available(
            tasks,
            "Detect",
            "WorldDetect",
            "YOLOEDetect",
            "Segment",
            "Segment26",
            "YOLOESegment",
            "YOLOESegment26",
            "Pose",
            "Pose26",
            "OBB",
            "OBB26",
        )
        segment_modules = _available(tasks, "Segment", "Segment26", "YOLOESegment", "YOLOESegment26")
        legacy_modules = _available(
            tasks,
            "Detect",
            "YOLOEDetect",
            "Segment",
            "Segment26",
            "YOLOESegment",
            "YOLOESegment26",
            "Pose",
            "Pose26",
            "OBB",
            "OBB26",
        )
        hg_modules = _available(tasks, "HGStem", "HGBlock")
        tv_modules = _available(tasks, "TorchVision", "Index")
        c3k2_cls = getattr(tasks, "C3k2", None)
        a2c2f_cls = getattr(tasks, "A2C2f", None)
        c2fcib_cls = getattr(tasks, "C2fCIB", None)
        aifi_cls = getattr(tasks, "AIFI", None)
        resnet_layer_cls = getattr(tasks, "ResNetLayer", None)
        concat_cls = getattr(tasks, "Concat", None)
        image_pooling_attn_cls = getattr(tasks, "ImagePoolingAttn", None)
        rtdetr_decoder_cls = getattr(tasks, "RTDETRDecoder", None)
        cblinear_cls = getattr(tasks, "CBLinear", None)
        cbfuse_cls = getattr(tasks, "CBFuse", None)
        v10detect_cls = getattr(tasks, "v10Detect", None)
        c2fattn_cls = getattr(tasks, "C2fAttn", None)
        hgb_cls = getattr(tasks, "HGBlock", None)

        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
            m = (
                getattr(torch.nn, m[3:])
                if "nn." in m
                else getattr(__import__("torchvision").ops, m[16:])
                if "torchvision.ops." in m
                else tasks_globals[m]
            )
            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(ValueError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

            n = n_ = max(round(n * depth), 1) if n > 1 else n
            if m in base_modules:
                c1, c2 = ch[f], args[0]
                if c2 != nc:
                    c2 = tasks.make_divisible(min(c2, max_channels) * width, 8)
                if c2fattn_cls is not None and m is c2fattn_cls:
                    args[1] = tasks.make_divisible(min(args[1], max_channels // 2) * width, 8)
                    args[2] = int(
                        max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                    )

                args = [c1, c2, *args[1:]]
                if m in repeat_modules:
                    args.insert(2, n)
                    n = 1
                if c3k2_cls is not None and m is c3k2_cls:
                    legacy = False
                    if scale in "mlx":
                        args[3] = True
                if a2c2f_cls is not None and m is a2c2f_cls:
                    legacy = False
                    if scale in "lx":
                        args.extend((True, 1.2))
                if c2fcib_cls is not None and m is c2fcib_cls:
                    legacy = False
            elif aifi_cls is not None and m is aifi_cls:
                args = [ch[f], *args]
            elif m in hg_modules:
                c1, cm, c2 = ch[f], args[0], args[1]
                args = [c1, cm, c2, *args[2:]]
                if hgb_cls is not None and m is hgb_cls:
                    args.insert(4, n)
                    n = 1
            elif resnet_layer_cls is not None and m is resnet_layer_cls:
                c2 = args[1] if args[3] else args[1] * 4
            elif m is torch.nn.BatchNorm2d:
                args = [ch[f]]
            elif concat_cls is not None and m is concat_cls:
                c2 = sum(ch[x] for x in f)
            elif m is ASFF_2:
                c2 = ch[f[0]] if args[0] == 0 else ch[f[-1]]
                args = [c2, args[0], [ch[x] for x in f]]
            elif m is ASFF_3:
                if args[0] == 0:
                    c2 = ch[f[0]]
                elif args[0] == 1:
                    c2 = ch[f[1]]
                else:
                    c2 = ch[f[-1]]
                args = [c2, args[0], [ch[x] for x in f]]
            elif m is ASFF_4:
                if args[0] == 0:
                    c2 = ch[f[0]]
                elif args[0] == 1:
                    c2 = ch[f[1]]
                elif args[0] == 2:
                    c2 = ch[f[2]]
                else:
                    c2 = ch[f[-1]]
                args = [c2, args[0], [ch[x] for x in f]]
            elif m in detect_modules:
                args.extend([reg_max, end2end, [ch[x] for x in f]])
                if m in segment_modules:
                    args[2] = tasks.make_divisible(min(args[2], max_channels) * width, 8)
                if m in legacy_modules:
                    m.legacy = legacy
            elif v10detect_cls is not None and m is v10detect_cls:
                args.append([ch[x] for x in f])
            elif image_pooling_attn_cls is not None and m is image_pooling_attn_cls:
                args.insert(1, [ch[x] for x in f])
            elif rtdetr_decoder_cls is not None and m is rtdetr_decoder_cls:
                args.insert(1, [ch[x] for x in f])
            elif cblinear_cls is not None and m is cblinear_cls:
                c2 = args[0]
                c1 = ch[f]
                args = [c1, c2, *args[1:]]
            elif cbfuse_cls is not None and m is cbfuse_cls:
                c2 = ch[f[-1]]
            elif m in tv_modules:
                c2 = args[0]
                c1 = ch[f]
                args = [*args[1:]]
            else:
                c2 = ch[f]

            m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace("__main__.", "")
            m_.np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t
            if verbose:
                tasks.LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return torch.nn.Sequential(*layers), sorted(save)

    return parse_model


def _build_yaml_model_load(tasks):
    original_yaml_model_load = tasks.yaml_model_load

    def yaml_model_load(path):
        d = original_yaml_model_load(path)
        if d.get("scale"):
            return d

        yaml_path = Path(d.get("yaml_file", path))
        if yaml_path.is_file():
            with yaml_path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
            if isinstance(raw, dict) and raw.get("scale"):
                d["scale"] = raw["scale"]
        return d

    return yaml_model_load


def register_custom_modules():
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    tasks.ASFF_2 = ASFF_2
    tasks.ASFF_3 = ASFF_3
    tasks.ASFF_4 = ASFF_4
    tasks.CPCA = CPCA
    tasks.CRDR = CRDR
    tasks.EMA = EMA
    tasks.RFAConv = RFAConv
    tasks.SCC2f = SCC2f

    modules.ASFF_2 = ASFF_2
    modules.ASFF_3 = ASFF_3
    modules.ASFF_4 = ASFF_4
    modules.CPCA = CPCA
    modules.CRDR = CRDR
    modules.EMA = EMA
    modules.RFAConv = RFAConv
    modules.SCC2f = SCC2f

    if not getattr(tasks, "_af_yolo_custom_yaml_model_load", False):
        tasks.yaml_model_load = _build_yaml_model_load(tasks)
        tasks._af_yolo_custom_yaml_model_load = True
    if not getattr(tasks, "_af_yolo_custom_parse_model", False):
        tasks.parse_model = _build_parse_model(tasks)
        tasks._af_yolo_custom_parse_model = True
    return tasks


__all__ = (
    "ASFF_2",
    "ASFF_3",
    "ASFF_4",
    "CPCA",
    "CRDR",
    "EMA",
    "MODEL_CFG_PATH",
    "RFAConv",
    "SCC2f",
    "register_custom_modules",
)

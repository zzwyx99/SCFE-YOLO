import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = ((k - 1) // 2) * d
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ScaleAwareEdgeMixer(nn.Module):
    """A small-object oriented residual block.

    The block mixes three complementary cues:
    1) local textures via depthwise 5x5,
    2) wider sparse context via dilated depthwise 3x3,
    3) high-frequency edge residuals via low-pass subtraction.

    A learned branch router and a spatial confidence gate then fuse these cues.
    The module preserves channel count so it can be inserted into YOLO YAMLs
    without modifying Ultralytics' parser internals.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        hidden = max(channels // reduction, 16)

        self.local_branch = ConvBNAct(channels, channels, k=5, g=channels)
        self.context_branch = ConvBNAct(channels, channels, k=3, g=channels, d=2)
        self.edge_reduce = ConvBNAct(channels, channels, k=1, act=False)
        self.edge_refine = ConvBNAct(channels, channels, k=3, g=channels)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 3, 1, bias=True),
        )
        self.spatial_gate = nn.Sequential(
            ConvBNAct(3, hidden, k=3),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            ConvBNAct(channels, channels, k=1),
            ConvBNAct(channels, channels, k=3, g=channels, act=False),
        )
        nn.init.zeros_(self.project[1].bn.weight)

    def forward(self, x):
        local_feat = self.local_branch(x)
        context_feat = self.context_branch(x)

        low_pass = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        edge_feat = self.edge_refine(self.edge_reduce(x - low_pass))

        branch_logits = self.router(x)
        branch_weights = torch.softmax(branch_logits, dim=1)
        mixed = (
            local_feat * branch_weights[:, 0:1]
            + context_feat * branch_weights[:, 1:2]
            + edge_feat * branch_weights[:, 2:3]
        )

        gate_input = torch.cat(
            [
                mixed.mean(dim=1, keepdim=True),
                edge_feat.abs().mean(dim=1, keepdim=True),
                x.mean(dim=1, keepdim=True),
            ],
            dim=1,
        )
        spatial_weight = self.spatial_gate(gate_input)
        fused = self.project(mixed * spatial_weight)
        return x + fused


class ScaleConsistencyCoupledSAEM(nn.Module):
    """Scale-aware edge mixer with an internal cross-scale consistency path.

    Compared with SAEM, this block explicitly builds a second "scale view"
    by compressing the feature map, extracting complementary cues, and then
    projecting it back to the original resolution. The native-view and
    compressed-view responses are fused through a consistency gate so the
    output is encouraged to remain stable under scale perturbation.
    """

    def __init__(self, channels, reduction=4, scale_factor=0.5):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.scale_factor = scale_factor
        hidden = max(channels // reduction, 16)

        self.local_branch = ConvBNAct(channels, channels, k=5, g=channels)
        self.context_branch = ConvBNAct(channels, channels, k=3, g=channels, d=2)
        self.edge_reduce = ConvBNAct(channels, channels, k=1, act=False)
        self.edge_refine = ConvBNAct(channels, channels, k=3, g=channels)

        self.scale_down = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.scale_local_branch = ConvBNAct(channels, channels, k=3, g=channels)
        self.scale_context_branch = ConvBNAct(channels, channels, k=3, g=channels, d=2)
        self.scale_project = ConvBNAct(channels, channels, k=1)

        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 4, 1, bias=True),
        )
        self.spatial_gate = nn.Sequential(
            ConvBNAct(4, hidden, k=3),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.consistency_gate = nn.Sequential(
            ConvBNAct(channels * 2, hidden, k=1),
            ConvBNAct(hidden, hidden, k=3),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            ConvBNAct(channels, channels, k=1),
            ConvBNAct(channels, channels, k=3, g=channels, act=False),
        )
        nn.init.zeros_(self.project[1].bn.weight)

    def _edge_path(self, x):
        low_pass = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return self.edge_refine(self.edge_reduce(x - low_pass))

    def _compressed_scale_path(self, x):
        h, w = x.shape[-2:]
        x_small = self.scale_down(x)
        scale_local = self.scale_local_branch(x_small)
        scale_context = self.scale_context_branch(x_small)
        scale_mix = self.scale_project(scale_local + scale_context)
        return F.interpolate(scale_mix, size=(h, w), mode="bilinear", align_corners=False)

    def forward(self, x):
        local_feat = self.local_branch(x)
        context_feat = self.context_branch(x)
        edge_feat = self._edge_path(x)
        scale_feat = self._compressed_scale_path(x)

        branch_logits = self.router(x)
        branch_weights = torch.softmax(branch_logits, dim=1)
        native_mix = (
            local_feat * branch_weights[:, 0:1]
            + context_feat * branch_weights[:, 1:2]
            + edge_feat * branch_weights[:, 2:3]
        )
        scale_weight = branch_weights[:, 3:4]
        consistency_weight = self.consistency_gate(torch.cat([native_mix, scale_feat], dim=1))
        aligned_scale = scale_feat * consistency_weight
        mixed = native_mix * (1.0 - scale_weight) + aligned_scale * scale_weight

        gate_input = torch.cat(
            [
                mixed.mean(dim=1, keepdim=True),
                edge_feat.abs().mean(dim=1, keepdim=True),
                scale_feat.mean(dim=1, keepdim=True),
                (native_mix - aligned_scale).abs().mean(dim=1, keepdim=True),
            ],
            dim=1,
        )
        spatial_weight = self.spatial_gate(gate_input)
        fused = self.project(mixed * spatial_weight)
        return x + fused


class _ScaleConsistencyCoupledSAEMAblationBase(nn.Module):
    """Shared building blocks for staged SCC-SAEM ablations."""

    def __init__(
        self,
        channels,
        reduction=4,
        scale_factor=0.5,
        router_branches=0,
        use_scale_path=False,
        use_consistency_gate=False,
    ):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.scale_factor = scale_factor
        self.router_branches = router_branches
        hidden = max(channels // reduction, 16)

        self.local_branch = ConvBNAct(channels, channels, k=5, g=channels)
        self.context_branch = ConvBNAct(channels, channels, k=3, g=channels, d=2)
        self.edge_reduce = ConvBNAct(channels, channels, k=1, act=False)
        self.edge_refine = ConvBNAct(channels, channels, k=3, g=channels)

        self.branch_router = None
        if router_branches > 0:
            self.branch_router = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, 1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, router_branches, 1, bias=True),
            )

        self.scale_down = None
        self.scale_local_branch = None
        self.scale_context_branch = None
        self.scale_project = None
        if use_scale_path:
            self.scale_down = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.scale_local_branch = ConvBNAct(channels, channels, k=3, g=channels)
            self.scale_context_branch = ConvBNAct(channels, channels, k=3, g=channels, d=2)
            self.scale_project = ConvBNAct(channels, channels, k=1)

        self.consistency_gate = None
        if use_consistency_gate:
            self.consistency_gate = nn.Sequential(
                ConvBNAct(channels * 2, hidden, k=1),
                ConvBNAct(hidden, hidden, k=3),
                nn.Conv2d(hidden, channels, 1, bias=True),
                nn.Sigmoid(),
            )

    def _edge_path(self, x):
        low_pass = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return self.edge_refine(self.edge_reduce(x - low_pass))

    def _compressed_scale_path(self, x):
        h, w = x.shape[-2:]
        x_small = self.scale_down(x)
        scale_local = self.scale_local_branch(x_small)
        scale_context = self.scale_context_branch(x_small)
        scale_mix = self.scale_project(scale_local + scale_context)
        return F.interpolate(scale_mix, size=(h, w), mode="bilinear", align_corners=False)

    def _branch_weights(self, x):
        branch_logits = self.branch_router(x)
        return torch.softmax(branch_logits, dim=1)

    def _native_reference(self, x, branch_weights=None):
        local_feat = self.local_branch(x)
        context_feat = self.context_branch(x)
        edge_feat = self._edge_path(x)
        if branch_weights is not None:
            native_mix = (
                local_feat * branch_weights[:, 0:1]
                + context_feat * branch_weights[:, 1:2]
                + edge_feat * branch_weights[:, 2:3]
            )
        else:
            native_mix = (local_feat + context_feat + edge_feat) / 3.0
        return native_mix, edge_feat


class ScaleConsistencyCoupledSAEMNSR(_ScaleConsistencyCoupledSAEMAblationBase):
    """Ablation with native-scale reference construction only."""

    def __init__(self, channels, reduction=4, scale_factor=0.5):
        super().__init__(
            channels,
            reduction=reduction,
            scale_factor=scale_factor,
        )

    def forward(self, x):
        native_mix, _ = self._native_reference(x)
        return x + native_mix


class ScaleConsistencyCoupledSAEMNSRBR(_ScaleConsistencyCoupledSAEMAblationBase):
    """Ablation with native-scale reference construction and branch router."""

    def __init__(self, channels, reduction=4, scale_factor=0.5):
        super().__init__(
            channels,
            reduction=reduction,
            scale_factor=scale_factor,
            router_branches=3,
        )

    def forward(self, x):
        branch_weights = self._branch_weights(x)
        native_mix, _ = self._native_reference(x, branch_weights=branch_weights)
        return x + native_mix


class ScaleConsistencyCoupledSAEMNSRBRCSP(_ScaleConsistencyCoupledSAEMAblationBase):
    """Ablation that adds compressed-scale perturbation with one-shot 4-way BR."""

    def __init__(self, channels, reduction=4, scale_factor=0.5):
        super().__init__(
            channels,
            reduction=reduction,
            scale_factor=scale_factor,
            router_branches=4,
            use_scale_path=True,
        )

    def forward(self, x):
        branch_weights = self._branch_weights(x)
        native_mix, _ = self._native_reference(x, branch_weights=branch_weights)
        scale_feat = self._compressed_scale_path(x)
        mixed = native_mix + scale_feat * branch_weights[:, 3:4]
        return x + mixed


class ScaleConsistencyCoupledSAEMNSRBRCSPCGA(_ScaleConsistencyCoupledSAEMAblationBase):
    """Ablation that adds consistency-gated alignment on top of NSR+BR+CSP."""

    def __init__(self, channels, reduction=4, scale_factor=0.5):
        super().__init__(
            channels,
            reduction=reduction,
            scale_factor=scale_factor,
            router_branches=4,
            use_scale_path=True,
            use_consistency_gate=True,
        )

    def forward(self, x):
        branch_weights = self._branch_weights(x)
        native_mix, _ = self._native_reference(x, branch_weights=branch_weights)
        scale_feat = self._compressed_scale_path(x)
        consistency_weight = self.consistency_gate(torch.cat([native_mix, scale_feat], dim=1))
        aligned_scale = scale_feat * consistency_weight
        mixed = native_mix + aligned_scale * branch_weights[:, 3:4]
        return x + mixed


class ScaleConsistencyCoupledSAEMNoBranchRouter(nn.Module):
    """SCC-SAEM ablation without the learned branch router.

    Local, context, edge, and compressed-scale responses are fused with fixed
    equal weights. The consistency gate and spatial gate are retained to isolate
    the contribution of adaptive branch routing.
    """

    def __init__(self, channels, reduction=4, scale_factor=0.5):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.scale_factor = scale_factor
        hidden = max(channels // reduction, 16)

        self.local_branch = ConvBNAct(channels, channels, k=5, g=channels)
        self.context_branch = ConvBNAct(channels, channels, k=3, g=channels, d=2)
        self.edge_reduce = ConvBNAct(channels, channels, k=1, act=False)
        self.edge_refine = ConvBNAct(channels, channels, k=3, g=channels)

        self.scale_down = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.scale_local_branch = ConvBNAct(channels, channels, k=3, g=channels)
        self.scale_context_branch = ConvBNAct(channels, channels, k=3, g=channels, d=2)
        self.scale_project = ConvBNAct(channels, channels, k=1)

        self.spatial_gate = nn.Sequential(
            ConvBNAct(4, hidden, k=3),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.consistency_gate = nn.Sequential(
            ConvBNAct(channels * 2, hidden, k=1),
            ConvBNAct(hidden, hidden, k=3),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            ConvBNAct(channels, channels, k=1),
            ConvBNAct(channels, channels, k=3, g=channels, act=False),
        )
        nn.init.zeros_(self.project[1].bn.weight)

    def _edge_path(self, x):
        low_pass = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return self.edge_refine(self.edge_reduce(x - low_pass))

    def _compressed_scale_path(self, x):
        h, w = x.shape[-2:]
        x_small = self.scale_down(x)
        scale_local = self.scale_local_branch(x_small)
        scale_context = self.scale_context_branch(x_small)
        scale_mix = self.scale_project(scale_local + scale_context)
        return F.interpolate(scale_mix, size=(h, w), mode="bilinear", align_corners=False)

    def forward(self, x):
        local_feat = self.local_branch(x)
        context_feat = self.context_branch(x)
        edge_feat = self._edge_path(x)
        scale_feat = self._compressed_scale_path(x)

        native_mix = (local_feat + context_feat + edge_feat) / 3.0
        consistency_weight = self.consistency_gate(torch.cat([native_mix, scale_feat], dim=1))
        aligned_scale = scale_feat * consistency_weight
        mixed = (native_mix + aligned_scale) * 0.5

        gate_input = torch.cat(
            [
                mixed.mean(dim=1, keepdim=True),
                edge_feat.abs().mean(dim=1, keepdim=True),
                scale_feat.mean(dim=1, keepdim=True),
                (native_mix - aligned_scale).abs().mean(dim=1, keepdim=True),
            ],
            dim=1,
        )
        spatial_weight = self.spatial_gate(gate_input)
        fused = self.project(mixed * spatial_weight)
        return x + fused


class ScaleConsistencyCoupledSAEMNoSpatialGate(nn.Module):
    """SCC-SAEM ablation without the spatial confidence gate.

    The learned branch router and consistency gate are retained, while the final
    projection receives the mixed response directly.
    """

    def __init__(self, channels, reduction=4, scale_factor=0.5):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.scale_factor = scale_factor
        hidden = max(channels // reduction, 16)

        self.local_branch = ConvBNAct(channels, channels, k=5, g=channels)
        self.context_branch = ConvBNAct(channels, channels, k=3, g=channels, d=2)
        self.edge_reduce = ConvBNAct(channels, channels, k=1, act=False)
        self.edge_refine = ConvBNAct(channels, channels, k=3, g=channels)

        self.scale_down = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.scale_local_branch = ConvBNAct(channels, channels, k=3, g=channels)
        self.scale_context_branch = ConvBNAct(channels, channels, k=3, g=channels, d=2)
        self.scale_project = ConvBNAct(channels, channels, k=1)

        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 4, 1, bias=True),
        )
        self.consistency_gate = nn.Sequential(
            ConvBNAct(channels * 2, hidden, k=1),
            ConvBNAct(hidden, hidden, k=3),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            ConvBNAct(channels, channels, k=1),
            ConvBNAct(channels, channels, k=3, g=channels, act=False),
        )
        nn.init.zeros_(self.project[1].bn.weight)

    def _edge_path(self, x):
        low_pass = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return self.edge_refine(self.edge_reduce(x - low_pass))

    def _compressed_scale_path(self, x):
        h, w = x.shape[-2:]
        x_small = self.scale_down(x)
        scale_local = self.scale_local_branch(x_small)
        scale_context = self.scale_context_branch(x_small)
        scale_mix = self.scale_project(scale_local + scale_context)
        return F.interpolate(scale_mix, size=(h, w), mode="bilinear", align_corners=False)

    def forward(self, x):
        local_feat = self.local_branch(x)
        context_feat = self.context_branch(x)
        edge_feat = self._edge_path(x)
        scale_feat = self._compressed_scale_path(x)

        branch_logits = self.router(x)
        branch_weights = torch.softmax(branch_logits, dim=1)
        native_mix = (
            local_feat * branch_weights[:, 0:1]
            + context_feat * branch_weights[:, 1:2]
            + edge_feat * branch_weights[:, 2:3]
        )
        scale_weight = branch_weights[:, 3:4]
        consistency_weight = self.consistency_gate(torch.cat([native_mix, scale_feat], dim=1))
        aligned_scale = scale_feat * consistency_weight
        mixed = native_mix * (1.0 - scale_weight) + aligned_scale * scale_weight

        fused = self.project(mixed)
        return x + fused


class ScaleOnlyCoupledSAEM(nn.Module):
    """Ablation block that keeps the raw native branch but removes native enhancement.

    Compared with SCC-SAEM, this variant preserves the compressed-scale branch
    and the branch router, but the native branch no longer uses local/context/
    edge feature enhancement. Instead, the raw input feature x directly
    participates in cross-scale consistency coupling. This isolates the effect
    of native-branch enhancement from the scale-consistency mechanism itself.
    """

    def __init__(self, channels, reduction=4, scale_factor=0.5):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.scale_factor = scale_factor
        hidden = max(channels // reduction, 16)

        self.scale_down = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.scale_local_branch = ConvBNAct(channels, channels, k=3, g=channels)
        self.scale_context_branch = ConvBNAct(channels, channels, k=3, g=channels, d=2)
        self.scale_project = ConvBNAct(channels, channels, k=1)

        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, 1, bias=True),
        )
        self.consistency_gate = nn.Sequential(
            ConvBNAct(channels * 2, hidden, k=1),
            ConvBNAct(hidden, hidden, k=3),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            ConvBNAct(4, hidden, k=3),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            ConvBNAct(channels, channels, k=1),
            ConvBNAct(channels, channels, k=3, g=channels, act=False),
        )
        nn.init.zeros_(self.project[1].bn.weight)

    def _compressed_scale_path(self, x):
        h, w = x.shape[-2:]
        x_small = self.scale_down(x)
        scale_local = self.scale_local_branch(x_small)
        scale_context = self.scale_context_branch(x_small)
        scale_mix = self.scale_project(scale_local + scale_context)
        return F.interpolate(scale_mix, size=(h, w), mode="bilinear", align_corners=False)

    def forward(self, x):
        scale_feat = self._compressed_scale_path(x)
        branch_logits = self.router(x)
        branch_weights = torch.softmax(branch_logits, dim=1)
        native_weight = branch_weights[:, 0:1]
        scale_weight = branch_weights[:, 1:2]
        native_mix = x * native_weight
        consistency_weight = self.consistency_gate(torch.cat([native_mix, scale_feat], dim=1))
        aligned_scale = scale_feat * consistency_weight
        mixed = native_mix * (1.0 - scale_weight)  + aligned_scale * scale_weight

        gate_input = torch.cat(
            [
                mixed.mean(dim=1, keepdim=True),
                x.mean(dim=1, keepdim=True),
                scale_feat.mean(dim=1, keepdim=True),
                (x - aligned_scale).abs().mean(dim=1, keepdim=True),
            ],
            dim=1,
        )
        spatial_weight = self.spatial_gate(gate_input)
        fused = self.project(mixed * spatial_weight)
        return x + fused


def register_custom_modules():
    import ultralytics.nn.tasks as tasks

    tasks.ScaleAwareEdgeMixer = ScaleAwareEdgeMixer
    tasks.ScaleConsistencyCoupledSAEM = ScaleConsistencyCoupledSAEM
    tasks.ScaleConsistencyCoupledSAEMNSR = ScaleConsistencyCoupledSAEMNSR
    tasks.ScaleConsistencyCoupledSAEMNSRBR = ScaleConsistencyCoupledSAEMNSRBR
    tasks.ScaleConsistencyCoupledSAEMNSRBRCSP = ScaleConsistencyCoupledSAEMNSRBRCSP
    tasks.ScaleConsistencyCoupledSAEMNSRBRCSPCGA = ScaleConsistencyCoupledSAEMNSRBRCSPCGA
    tasks.ScaleConsistencyCoupledSAEMNoBranchRouter = ScaleConsistencyCoupledSAEMNoBranchRouter
    tasks.ScaleConsistencyCoupledSAEMNoSpatialGate = ScaleConsistencyCoupledSAEMNoSpatialGate
    tasks.ScaleOnlyCoupledSAEM = ScaleOnlyCoupledSAEM

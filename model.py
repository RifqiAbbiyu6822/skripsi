import torch
import torch.nn as nn

class SEBlock1D(nn.Module):

    def __init__(self, channels: int, reduction: int=16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(channels, reduced), nn.ReLU(inplace=True), nn.Linear(reduced, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class LayerNorm1d(nn.Module):

    def __init__(self, num_channels: int, eps: float=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x

class DropPath(nn.Module):

    def __init__(self, drop_prob: float=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class ConvNeXt1DBlock(nn.Module):

    def __init__(self, dim: int, drop_path: float=0.0, layer_scale_init: float=1e-06, se_reduction: int=16, kernel_size: int=7):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-06)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.se = SEBlock1D(dim, reduction=se_reduction)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)
        x = self.se(x)
        x = inp + self.drop_path(x)
        return x

class ConvNeXt1DSE(nn.Module):

    def __init__(self, in_channels: int=12, num_classes: int=5, depths=(3, 3, 9, 3), dims=(64, 128, 256, 512), drop_path_rate: float=0.1, se_reduction: int=16, layer_scale_init: float=1e-06, stem_kernel: int=15, stem_stride: int=4):
        super().__init__()
        assert len(depths) == len(dims) == 4, 'depths & dims harus punya 4 elemen'
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv1d(in_channels, dims[0], kernel_size=stem_kernel, stride=stem_stride, padding=stem_kernel // 2), LayerNorm1d(dims[0], eps=1e-06))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample = nn.Sequential(LayerNorm1d(dims[i], eps=1e-06), nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample)
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[ConvNeXt1DBlock(dims[i], drop_path=dp_rates[cur + j], layer_scale_init=layer_scale_init, se_reduction=se_reduction) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-06)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean(-1)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x

def build_model(variant: str='tiny', in_channels: int=12, num_classes: int=5, drop_path_rate: float=0.1, se_reduction: int=16) -> ConvNeXt1DSE:
    presets = {'nano': dict(depths=(2, 2, 4, 2), dims=(32, 64, 128, 256)), 'tiny': dict(depths=(3, 3, 9, 3), dims=(64, 128, 256, 512)), 'small': dict(depths=(3, 3, 27, 3), dims=(96, 192, 384, 768))}
    if variant not in presets:
        raise ValueError(f"variant '{variant}' tidak dikenal, pilih dari {list(presets)}")
    cfg = presets[variant]
    return ConvNeXt1DSE(in_channels=in_channels, num_classes=num_classes, depths=cfg['depths'], dims=cfg['dims'], drop_path_rate=drop_path_rate, se_reduction=se_reduction)

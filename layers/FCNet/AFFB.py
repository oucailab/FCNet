from timm.layers import DropPath
import torch
import torch.nn as nn
import torch.fft


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AdaptiveFourierFrequencyFilter(nn.Module):
    def __init__(self, dim, h, w, t):
        super(AdaptiveFourierFrequencyFilter, self).__init__()
        self.h = h
        self.w = w
        self.t = t
        self.dim = dim
        self.complex_weight = nn.Parameter(torch.randn(t, h, w // 2 + 1, dim, 2) * 0.02)

    def forward(self, x):
        B, N, D = x.shape

        # Reshape input tensor
        x = x.reshape(B // self.t, self.t, self.h, self.w, D)

        # Apply 2D FFT
        x_fft = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")

        # Convert weight to complex tensor
        weight = torch.view_as_complex(self.complex_weight)

        # Apply filter in frequency domain
        x_fft = x_fft * weight

        # Apply inverse 2D FFT
        x = torch.fft.irfft2(x_fft, s=(self.h, self.w), dim=(2, 3), norm="ortho")

        # Reshape back to original shape
        x = x.reshape(B, N, D)

        return x


class AFFBlock(nn.Module):
    def __init__(
        self,
        t,
        dim,
        h,
        w,
        mlp_ratio,
        drop,
        drop_path,
        act_layer,
        norm_layer=nn.LayerNorm,
    ):
        super(AFFBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AdaptiveFourierFrequencyFilter(dim, h, w, t)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.filter(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

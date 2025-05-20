from functools import partial
import torch
from torch import nn
from utils.FrequencyLoss import FrequencyLoss
from layers.FCNet.HFEB import Block
from layers.FCNet.AFFB import AFFBlock


class PatchEmbed(nn.Module):
    def __init__(
        self, img_size, patch_embed_size, input_dim, embed_dim, norm_layer=None
    ):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_embed_size = patch_embed_size
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.patch = nn.Conv2d(
            input_dim,
            embed_dim,
            kernel_size=patch_embed_size,
            stride=patch_embed_size,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        # Ensure the input image size matches the expected size
        assert H == self.img_size[0] and W == self.img_size[1]

        # Apply patch embedding and flatten spatial dimensions
        x = self.patch(x).flatten(2).transpose(1, 2)

        # Apply normalization if specified
        x = self.norm(x)

        return x


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        upsampling=False,
        act_norm=False,
    ):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(
                *[
                    nn.Conv2d(
                        in_channels,
                        out_channels * 4,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.PixelShuffle(2),
                ]
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size=3,
        downsampling=False,
        upsampling=False,
        act_norm=True,
    ):
        super(ConvSC, self).__init__()
        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(
            C_in,
            C_out,
            kernel_size=kernel_size,
            stride=stride,
            upsampling=upsampling,
            padding=padding,
            act_norm=act_norm,
        )

    def forward(self, x):
        y = self.conv(x)
        return y


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse:
        return list(reversed(samplings[:N]))
    else:
        return samplings[:N]


class Encoder(nn.Module):

    def __init__(self, C_in, C_hid, N_S, spatio_kernel):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0]),
            *[
                ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s)
                for s in samplings[1:]
            ]
        )

    def forward(self, x):  # B*T, C, H, W
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):

    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[
                ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s)
                for s in samplings[:-1]
            ],
            ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1])
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class HFEB(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mlp_ratio=8.0,
        drop=0.0,
        drop_path=0.0,
    ):
        super(HFEB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = Block(
            in_channels,
            kernel_size=21,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            act_layer=nn.GELU,
        )

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class HFEBNet(nn.Module):
    def __init__(
        self,
        channel_in,
        channel_hid,
        N2,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.1,
    ):
        super(HFEBNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)
        ]

        # downsample
        enc_layers = [
            HFEB(
                channel_in,
                channel_hid,
                mlp_ratio,
                drop,
                drop_path=dpr[0],
            )
        ]
        # middle layers
        for i in range(1, N2 - 1):
            enc_layers.append(
                HFEB(
                    channel_hid,
                    channel_hid,
                    mlp_ratio,
                    drop,
                    drop_path=dpr[i],
                )
            )
        # upsample
        enc_layers.append(
            HFEB(
                channel_hid,
                channel_in,
                mlp_ratio,
                drop,
                drop_path=drop_path,
            )
        )
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y


class AFFBNet(nn.Module):
    def __init__(
        self,
        T,
        img_size,
        patch_size,
        patch_embed_size,
        depth=1,
        mlp_ratio=4.0,
        uniform_drop=False,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        dropcls=0.0,
    ):
        super(AFFBNet, self).__init__()
        self.embed_dim = patch_embed_size[0] * patch_embed_size[1]
        H, W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=(H, W),
            patch_embed_size=patch_embed_size,
            input_dim=1,
            embed_dim=self.embed_dim,
        )

        self.h = patch_embed_size[0]
        self.w = patch_embed_size[1]

        self.num_patches_h = patch_size[0] * H // patch_embed_size[0]
        self.num_patches_w = patch_size[1] * W // patch_embed_size[1]

        self.num_patches = self.num_patches_h * self.num_patches_w

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if uniform_drop:
            dpr = [drop_path_rate for _ in range(depth)]
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                AFFBlock(
                    t=T,
                    dim=self.embed_dim,
                    h=self.num_patches_h,
                    w=self.num_patches_w,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(self.embed_dim)

        num_upsamples = torch.log2(torch.tensor(patch_embed_size[0])).item()

        if not num_upsamples.is_integer():
            raise ValueError("patch_embed_size must be a power of 2.")

        layers = []
        in_channels = self.embed_dim
        num_upsamples = int(num_upsamples)

        for i in range(num_upsamples):
            out_channels = 2 ** (2 * (num_upsamples - 1 - i))
            layers.append(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2)
                )
            )
            layers.append(nn.Softsign())
            in_channels = out_channels

        self.progressive_upsampler = nn.Sequential(*layers)

        if dropcls > 0:
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

    def forward_features(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T * C, 1, H, W)
        x = self.patch_embed(x)
        x = x.reshape(B * T, self.num_patches, self.embed_dim)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x).transpose(1, 2)
        x = torch.reshape(
            x, [B * T, self.embed_dim, self.num_patches_h, self.num_patches_w]
        )
        return x

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.progressive_upsampler(x)
        x = x.reshape(B, T, C, H, W)
        return x


class FCNet(nn.Module):
    def __init__(
        self,
        T,
        C,
        img_size,
        patch_size,
        patch_embed_size,
        hid_S=32,
        hid_T=256,
        N_S=4,
        N_T=4,
        spatio_kernel_enc=(3, 3),
        spatio_kernel_dec=(3, 3),
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        dropcls=0.0,
    ):
        super(FCNet, self).__init__()
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec)

        self.hid = HFEBNet(
            T * hid_S,
            hid_T,
            N_T,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            drop_path=drop_path_rate,
        )

        self.AFFB_module = AFFBNet(
            T=T,
            img_size=img_size,
            patch_size=patch_size,
            patch_embed_size=patch_embed_size,
            depth=N_T,
            mlp_ratio=mlp_ratio,
            uniform_drop=False,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=None,
            dropcls=dropcls,
        )

        self.criterion = nn.HuberLoss()
        self.FrequencyLoss = FrequencyLoss()

    def forward(self, input_x, targets):
        assert len(input_x.shape) == 5

        B, T, C, H, W = input_x.shape
        x = input_x.view(B * T, C, H, W)
        AFFB_out = self.AFFB_module(input_x)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B * T, C_, H_, W_)

        HFEB_out = self.dec(hid, skip)
        HFEB_out = HFEB_out.reshape(B, T, C, H, W)

        pred = HFEB_out + AFFB_out
        pred = torch.clamp(pred, 0, 1)

        loss = self.criterion(pred, targets) + 0.05 * self.FrequencyLoss(pred, targets)

        return pred, loss

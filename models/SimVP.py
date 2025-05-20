import torch
from torch import nn
from layers.SimVP.MSC import Multi_Scale_Conv_Block


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
        act_inplace=True,
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
        self.act = nn.SiLU(inplace=act_inplace)

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
        act_inplace=True,
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
            act_inplace=act_inplace,
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
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            ConvSC(
                C_in,
                C_hid,
                spatio_kernel,
                downsampling=samplings[0],
                act_inplace=act_inplace,
            ),
            *[
                ConvSC(
                    C_hid, C_hid, spatio_kernel, downsampling=s, act_inplace=act_inplace
                )
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
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[
                ConvSC(
                    C_hid, C_hid, spatio_kernel, upsampling=s, act_inplace=act_inplace
                )
                for s in samplings[:-1]
            ],
            ConvSC(
                C_hid,
                C_hid,
                spatio_kernel,
                upsampling=samplings[-1],
                act_inplace=act_inplace,
            )
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3, 5, 7, 11], groups=8):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        enc_layers = [
            Multi_Scale_Conv_Block(
                channel_in,
                channel_hid // 2,
                channel_hid,
                incep_ker=incep_ker,
                groups=groups,
            )
        ]
        for i in range(1, N2 - 1):
            enc_layers.append(
                Multi_Scale_Conv_Block(
                    channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    incep_ker=incep_ker,
                    groups=groups,
                )
            )
        enc_layers.append(
            Multi_Scale_Conv_Block(
                channel_hid,
                channel_hid // 2,
                channel_hid,
                incep_ker=incep_ker,
                groups=groups,
            )
        )
        dec_layers = [
            Multi_Scale_Conv_Block(
                channel_hid,
                channel_hid // 2,
                channel_hid,
                incep_ker=incep_ker,
                groups=groups,
            )
        ]
        for i in range(1, N2 - 1):
            dec_layers.append(
                Multi_Scale_Conv_Block(
                    2 * channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    incep_ker=incep_ker,
                    groups=groups,
                )
            )
        dec_layers.append(
            Multi_Scale_Conv_Block(
                2 * channel_hid,
                channel_hid // 2,
                channel_in,
                incep_ker=incep_ker,
                groups=groups,
            )
        )

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2 - 1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


class SimVP(nn.Module):
    def __init__(
        self,
        T,
        C,
        hid_S=16,
        hid_T=256,
        N_S=4,
        N_T=4,
        spatio_kernel_enc=3,
        spatio_kernel_dec=3,
        act_inplace=False,
    ):
        super(SimVP, self).__init__()
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        self.hid = MidIncepNet(T * hid_S, hid_T, N_T)

        self.criterion = nn.HuberLoss()

    def forward(self, input_x, targets):
        assert len(input_x.shape) == 5

        B, T, C, H, W = input_x.shape
        x = input_x.view(B * T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B * T, C_, H_, W_)

        next_frames = self.dec(hid, skip)
        next_frames = next_frames.reshape(B, T, C, H, W)
        next_frames = torch.clamp(next_frames, 0, 1)

        loss = self.criterion(next_frames, targets)

        return next_frames, loss

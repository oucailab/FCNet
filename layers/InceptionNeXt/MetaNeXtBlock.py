from timm.models.inception_next import MetaNeXtBlock


class MetaNeXtSubBlock(MetaNeXtBlock):
    """A block of ConvNeXt."""

    def __init__(self, dim, dilation=1, mlp_ratio=4.0, drop_path=0.0):
        super().__init__(
            dim=dim,
            dilation=dilation,
            mlp_ratio=mlp_ratio,
            ls_init_value=1e-6,
            drop_path=drop_path,
        )

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x

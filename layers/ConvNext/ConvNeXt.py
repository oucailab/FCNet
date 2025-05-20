from timm.models.convnext import ConvNeXtBlock


class ConvNeXtSubBlock(ConvNeXtBlock):
    """A block of ConvNeXt."""

    def __init__(self, dim, mlp_ratio=4.0, use_grn=False, drop_path=0.0):
        super().__init__(
            dim,
            mlp_ratio=mlp_ratio,
            use_grn=use_grn,
            drop_path=drop_path,
            ls_init_value=1e-6,
            conv_mlp=True,
        )

    def forward(self, x):
        x = x + self.drop_path(
            self.gamma.reshape(1, -1, 1, 1) * self.mlp(self.norm(self.conv_dw(x)))
        )
        return x

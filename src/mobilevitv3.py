"""
MobileViTV3:  MOBILE-FRIENDLY VISION TRANSFORMER WITH SIMPLE AND EFFECTIVE FUSION OF LOCAL, GLOBAL AND INPUT FEATURES
    - https://arxiv.org/pdf/2209.15159.pdf
    - https://github.com/jaiwei98/mobile-vit-pytorch
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers


def conv_layer(
    dim,
    kernel_size=3,
    stride=1,
    padding=0,
    groups=1,
    use_bias=False,
    norm=True,
    act=True,
    name='conv_layer'
):
    out_layer = [
        layers.ZeroPadding2D(padding, name=f'{name}/pad')
    ]
    if dim == groups:
        out_layer.append(
            layers.DepthwiseConv2D(
                kernel_size, stride, 'valid', use_bias=use_bias,
                name=f'{name}/dwconv'
            )
        )
    else:
        out_layer.append(
            layers.Conv2D(
                dim, kernel_size, stride, 'valid',
                groups=groups, use_bias=use_bias,
                name=f'{name}/conv'
            )
        )
    if norm:
        out_layer.append(layers.BatchNormalization(name=f'{name}/bn'))
    if act:
        out_layer.append(layers.Activation('swish', name=f'{name}/swish'))

    return Sequential(out_layer, name=name)


class InvertedResidual(layers.Layer):

    def __init__(
        self,
        dim,
        stride,
        expand_ratio,
        name='inverted_residual'
    ):
        super().__init__(name=name)
        self.dim = dim
        self.stride = stride
        self.expand_ratio = expand_ratio

    def build(self, input_shape):
        self.block = Sequential(name=f'{self.name}/block')
        if self.expand_ratio != 1:
            hidden_dim = int(round(input_shape[-1] * self.expand_ratio))
            self.block.add(
                conv_layer(
                    hidden_dim, 1, 1, 0,
                    name=f'{self.name}/exp1x1_layer'
                )
            )
        self.block.add(
            conv_layer(
                hidden_dim, 3, self.stride, 1,
                groups=hidden_dim, name='conv3x3_layer'
            )
        )
        self.block.add(
            conv_layer(self.dim, 1, 1, 0, act=False, name='conv1x1_layer')
        )

        self.use_res_connect = (self.stride == 1) and (
            input_shape[-1] == self.dim)

    def call(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class Attention(layers.Layer):

    def __init__(
        self,
        embed_dim,
        heads=4,
        head_dim=8,
        attn_dropout=0,
        name='attn'
    ):
        super().__init__(name=name)
        self.qkv = layers.Dense(3*embed_dim, name=f'{self.name}/qkv')
        self.softmax = layers.Softmax(name=f'{self.name}/softmax')
        self.attn_dropout = layers.Dropout(
            attn_dropout, name=f'{self.name}/drop'
        )
        self.proj = layers.Dense(embed_dim, name=f'{self.name}/proj')
        self.embed_dim = embed_dim
        self.num_heads = heads
        self.scale = head_dim ** -0.5

    def call(self, x):
        B, N, _ = x.shape
        qkv = tf.reshape(
            self.qkv(x),
            [B, N, 3, self.num_heads, -1]
        )
        qkv = tf.transpose(qkv, [0, 3, 2, 1, 4])
        # [B, nH, 3, N, C] --> [B, nH, N, C] x 3
        q, k, v = tf.unstack(qkv, axis=2)

        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)  # [B, nH, N, N]
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        out = tf.matmul(attn, v)  # [B, nH, N, C]
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, N, -1])
        return self.proj(out)


class TransformerEncoder(layers.Layer):

    def __init__(
        self,
        embed_dim,
        ffn_latent_dim,
        heads=8,
        head_dim=8,
        dropout=0,
        attn_dropout=0,
        name='encoder'
    ):
        super().__init__(name=name)
        self.pre_norm_mhsa = Sequential(
            [
                layers.LayerNormalization(
                    epsilon=1e-5,
                    name=f'{self.name}/pre_norm_mhsa/ln'
                ),
                Attention(
                    embed_dim,
                    heads,
                    head_dim,
                    attn_dropout,
                    name=f'{self.name}/pre_norm_mhsa/attn'
                ),
                layers.Dropout(dropout, name=f'{self.name}/pre_norm_mhsa/drop')
            ],
            name=f'{self.name}/pre_norm_mhsa'
        )
        self.pre_norm_ffn = Sequential(
            [
                layers.LayerNormalization(
                    epsilon=1e-5,
                    name=f'{self.name}/pre_norm_ffn/ln'
                ),
                layers.Dense(
                    ffn_latent_dim,
                    name=f'{self.name}/pre_norm_ffn/proj'
                ),
                layers.Activation(
                    'swish',
                    name=f'{self.name}/pre_norm_ffn/swish'
                ),
                layers.Dropout(
                    dropout,
                    name=f'{self.name}/pre_norm_ffn/dropout1'
                ),
                layers.Dense(
                    embed_dim,
                    name=f'{self.name}/pre_norm_ffn/expand'
                ),
                layers.Dropout(
                    dropout,
                    name=f'{self.name}/pre_norm_ffn/dropout2'
                ),
            ]
        )

    def call(self, x):
        # Multi-head attention
        x = x + self.pre_norm_mhsa(x)
        # Feed Forward network
        x = x + self.pre_norm_ffn(x)
        return x


class V1Block(layers.Layer):
    """MobileViTV3-V1 block for XXS, XS, S"""

    def __init__(
        self,
        attn_dim,
        ffn_multiplier,
        heads,
        head_dim,
        attn_blocks: int,
        patch_size: list[int, int],
        name='block'
    ):
        super().__init__(name=name)
        self.attn_dim = attn_dim
        assert isinstance(patch_size, list), \
            "patch_size is sequential argument! not int!"
        self.patch_w, self.patch_h = patch_size

        # global representation
        ffn_dims = [int((ffn_multiplier * attn_dim) // 16 * 16)] * attn_blocks
        blocks = [
            TransformerEncoder(
                attn_dim, ffn_dim, heads,
                head_dim, name=f'{self.name}/encoder{i}'
            )
            for i, ffn_dim in enumerate(ffn_dims)
        ]
        blocks.append(
            layers.LayerNormalization(epsilon=1e-5, name=f'{self.name}/norm')
        )
        self.global_rep = Sequential(blocks, name=f'{self.name}/global_rep')

    def build(self, input_shape):
        in_channel = input_shape[-1]
        # local representation
        self.local_rep = Sequential(
            [
                conv_layer(
                    in_channel, 3, 1, 1, groups=in_channel,
                    name=f'{self.name}/local_rep/dwconv3x3'
                ),
                conv_layer(
                    self.attn_dim, 1, 1, 0, norm=False, act=False,
                    name=f'{self.name}/local_rep/conv1x1_wo_norm_act'
                )
            ],
            name=f'{self.name}/local_rep'
        )
        # make output be same shape of input
        self.proj = conv_layer(in_channel, 1, 1, name=f'{self.name}/proj')
        self.fusion = conv_layer(in_channel, 1, 1, name=f'{self.name}/fusion')

    def call(self, x):
        B, H, W, _ = x.shape
        x_residual = x
        x_local = self.local_rep(x)
        x, n_Ph, n_Pw, interpolate = self.unfold_official_style(x_local)
        x = self.global_rep(x)
        x = self.fold_official_style(x, B, n_Ph, n_Pw, (H, W), interpolate)
        x = self.proj(x)
        x = self.fusion(
            layers.Concatenate()([x_local, x])
        )
        x = layers.Add()([x, x_residual])
        return x

    def get_new_shpae(self, orig_h, orig_w):
        # dtype = orig_w.dtype
        new_h = tf.math.ceil(orig_h / self.patch_h)
        new_h *= tf.cast(self.patch_h, new_h.dtype)

        new_w = tf.math.ceil(orig_w / self.patch_w)
        new_w *= tf.cast(self.patch_w, new_w.dtype)
        return tf.cast(new_h, tf.int32), tf.cast(new_w, tf.int32)

    def unfold_official_style(self, x):
        """
        This original idea is from MobileViT.
        https://github.com/godhj93/MobileViT/blob/main/utils/nets/MobileViT.py#L204

        (B, H, W, C) -> (B*patch_area, n_patches, C)
        """
        B, H, W, C = x.shape
        new_h, new_w = self.get_new_shpae(H, W)

        interpolate = False
        if new_w != W or new_h != H:
            x = tf.image.resize(x, (new_h, new_w), 'bilinear')
            interpolate = True

        n_Ph = new_h // self.patch_h  # n patches of height
        n_Pw = new_w // self.patch_w  # n patches of width
        N = n_Ph * n_Pw  # n patches
        P = self.patch_h * self.patch_w

        patches = tf.reshape(
            x,
            [B, n_Ph, self.patch_h, n_Pw, self.patch_w, -1]
        )
        patches = tf.transpose(patches, [0, 2, 4, 1, 3, 5])
        patches = tf.reshape(patches, [B*P, N, -1])
        return patches, n_Ph, n_Pw, interpolate

    def unfold(self, x):
        """
        This original idea is from MobileViT.
        https://github.com/godhj93/MobileViT/blob/main/utils/nets/MobileViT.py#L204

        (B, H, W, C) -> (B*patch_area, n_patches, C)
        """
        B, H, W, C = x.shape
        # new_h, new_w = self.get_new_shpae(H, W)

        patches = tf.space_to_batch_nd(
            x,
            block_shape=[self.patch_h, self.patch_w],
            paddings=[[0, 0], [0, 0]]
        )  # (B*pH*pW, H // pH, W // pW, C)

        # (B*pH*pW, H//pH * W//pW, C)
        # equal to (B*P, N, C)
        # P: patch area, N: n patches
        patches = tf.reshape(
            patches,
            [
                B*self.patch_h*self.patch_w, -1, C
            ]
        )
        return patches

    def fold(self, patches, n_PH, n_PW):
        BP, N, C = patches.shape
        patches = tf.reshape(
            patches,
            [
                BP, n_PH, n_PW, C
            ]
        )
        x = tf.batch_to_space(
            patches,
            [self.patch_h, self.patch_w],
            [[0, 0], [0, 0]]
        )
        return x

    def fold_official_style(
        self,
        patches,
        B, n_Ph, n_Pw,
        original_hw,
        interpolate: bool
    ):
        BP, N, C = patches.shape
        P = BP // B
        # [BP, N, C] --> [B, P, N, C]
        x = tf.reshape(
            patches,
            [
                B,
                P,  # self.patch_h * self.patch_w
                N,
                C
            ]
        )
        x = tf.reshape(
            x,
            [B*self.patch_h, self.patch_w, n_Ph, n_Pw*C]
        )
        x = tf.transpose(
            x,
            [0, 2, 1, 3]
        )
        x = tf.reshape(
            x,
            [B, self.patch_h*n_Ph, self.patch_w*n_Pw, C]
        )
        if interpolate:
            x = tf.image.resize(x, original_hw, 'bilinear')
        return x


class MobileViTV3(Model):

    def __init__(
        self,
        num_classes=1000,
        patch_size=[2, 2],
        mv2_exp_mult=4,
        ffn_multiplier=2,
        last_layer_exp_factor=4,
        channels=[16, 32, 48, 96, 160, 160],
        attn_dim=[96, 120, 144],
        include_top=False,
        name='mobilevit'
    ):
        super().__init__(name=name)
        self.include_top = include_top

        self.stem = conv_layer(channels[0], 3, 2, name=f'{self.name}/stem')
        self.stage1 = InvertedResidual(
            channels[1], stride=1, expand_ratio=mv2_exp_mult,
            name=f'{self.name}/stage1'
        )
        self.stage2 = Sequential(
            [
                InvertedResidual(
                    channels[2], stride=2, expand_ratio=mv2_exp_mult,
                    name=f'{self.name}/stage2/mv2_block1'
                ),
                InvertedResidual(
                    channels[2], stride=1, expand_ratio=mv2_exp_mult,
                    name=f'{self.name}/stage2/mv2_block2'
                ),
                InvertedResidual(
                    channels[2], stride=1, expand_ratio=mv2_exp_mult,
                    name=f'{self.name}/stage2/mv2_block3'
                )
            ],
            name=f'{self.name}/stage2'
        )
        self.stage3 = Sequential(
            [
                InvertedResidual(
                    channels[3], stride=2, expand_ratio=mv2_exp_mult,
                    name=f'{self.name}/stage3/mv2_block1'
                ),
                V1Block(
                    attn_dim[0],
                    ffn_multiplier,
                    heads=4,
                    head_dim=8,
                    attn_blocks=2,
                    patch_size=patch_size,
                    name=f'{self.name}/stage3/v1_block1'
                )
            ],
            name=f'{self.name}/stage3'
        )
        self.stage4 = Sequential(
            [
                InvertedResidual(
                    channels[4], stride=2, expand_ratio=mv2_exp_mult,
                    name=f'{self.name}/stage4/mv2_block1'
                ),
                V1Block(
                    attn_dim[1],
                    ffn_multiplier,
                    heads=4,
                    head_dim=8,
                    attn_blocks=4,
                    patch_size=patch_size,
                    name=f'{self.name}/stage4/v1_block1'
                )
            ],
            name=f'{self.name}/stage4'
        )
        self.stage5 = Sequential(
            [
                InvertedResidual(
                    channels[5], stride=2, expand_ratio=mv2_exp_mult,
                    name=f'{self.name}/stage5/mv2_block1'
                ),
                V1Block(
                    attn_dim[2],
                    ffn_multiplier,
                    heads=4,
                    head_dim=8,
                    attn_blocks=3,
                    patch_size=patch_size,
                    name=f'{self.name}/stage5/v1_block1'
                )
            ],
            name=f'{self.name}/stage5'
        )

        self.expansion = conv_layer(
            channels[-1] * last_layer_exp_factor, 1, 1,
            name=f'{self.name}/head/expansion'
        )
        if self.include_top:
            self.gap = layers.GlobalAveragePooling2D(
                name=f'{self.name}/head/gap')
            self.classifier = layers.Dense(
                num_classes,
                name=f'{self.name}/head/classifier'
            )

    def call(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.expansion(x)
        if self.include_top:
            x = self.gap(x)
            x = self.classifier(x)
        return x


HPARAMS = {
    'xx_small': {
        'mv2_exp_mult': 2,
        'ffn_multiplier': 2,
        'last_layer_exp_factor': 4,
        'channels': [16, 16, 24, 64, 80, 128],
        'attn_dim': [64, 80, 96]
    },
    'x_small': {
        'mv2_exp_mult': 4,
        'ffn_multiplier': 2,
        'last_layer_exp_factor': 4,
        'channels': [16, 32, 48, 96, 160, 160],
        'attn_dim': [96, 120, 144]
    },
    'small': {
        'mv2_exp_mult': 4,
        'ffn_multiplier': 2,
        'last_layer_exp_factor': 3,
        'channels': [16, 32, 64, 128, 256, 320],
        'attn_dim': [144, 192, 240]
    }
}


def MobileViTV3XXSmall(
    input_shape: list[int, int, int],
    include_top: bool = False,
    name: str = 'mobilevitv3-xxs'
):
    # TODO - model initialization하고 return하도록 추가?!
    return MobileViTV3(
        **HPARAMS['xx_small'],
        include_top=include_top,
        name=name
    )


def MobileViTV3XSmall(
    input_shape: list[int, int, int],
    include_top: bool = False,
    name: str = 'mobilevitv3-xs'
):
    # TODO - model initialization하고 return하도록 추가?!
    return MobileViTV3(
        **HPARAMS['x_small'],
        include_top=include_top,
        name=name
    )


def MobileViTV3Small(
    input_shape: list[int, int, int],
    include_top: bool = False,
    name: str = 'mobilevitv3-small'
):
    # TODO - model initialization하고 return하도록 추가?!
    return MobileViTV3(
        **HPARAMS['small'],
        include_top=include_top,
        name=name
    )

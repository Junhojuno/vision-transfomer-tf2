"""
EfficientFormerV2: Rethinking Vision Transformers for MobileNet Size and Speed
    - https://arxiv.org/abs/2212.08059
    - https://github.com/snap-research/EfficientFormer/blob/main/models/efficientformer_v2.py

"""
from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras import initializers

from src.common import StochasticDepth


EfficientFormer_width = {
    'L': [40, 80, 192, 384],  # 26m 83.3% 6attn
    'S2': [32, 64, 144, 288],  # 12m 81.6% 4attn dp0.02
    'S1': [32, 48, 120, 224],  # 6.1m 79.0
    'S0': [32, 48, 96, 176],  # 75.0 75.7
}

EfficientFormer_depth = {
    'L': [5, 5, 15, 10],  # 26m 83.3%
    'S2': [4, 4, 12, 8],  # 12m
    'S1': [3, 3, 9, 6],  # 79.0
    'S0': [2, 2, 6, 4],  # 75.7
}

# 26m
expansion_ratios_L = {
    '0': [4, 4, 4, 4, 4],
    '1': [4, 4, 4, 4, 4],
    '2': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4],
}

# 12m
expansion_ratios_S2 = {
    '0': [4, 4, 4, 4],
    '1': [4, 4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 3, 3, 3, 3, 4, 4],
}

# 6.1m
expansion_ratios_S1 = {
    '0': [4, 4, 4],
    '1': [4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 4, 4, 4],
    '3': [4, 4, 3, 3, 4, 4],
}

# 3.5m
expansion_ratios_S0 = {
    '0': [4, 4],
    '1': [4, 4],
    '2': [4, 3, 3, 3, 4, 4],
    '3': [4, 3, 3, 4],
}


def stem(
    dim: int,
    act_layer: str = 'relu',
    name: str = 'stem'
):

    return Sequential(
        [
            layers.Conv2D(dim // 2, 3, 2, padding='same'),
            layers.BatchNormalization(),
            layers.Activation(act_layer),
            layers.Conv2D(dim, 3, 2, padding='same'),
            layers.BatchNormalization(),
            layers.Activation(act_layer),
        ],
        name=name
    )


class Attention4D(layers.Layer):

    def __init__(
        self,
        dim: int = 384,
        key_dim: int = 32,
        num_heads: int = 8,
        attn_ratio: int = 4,
        resolution: int = 7,
        act_layer: str = 'relu',
        stride: Optional[int] = None,
        name: str = 'att_4d'
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads

        if stride is not None:
            self.resolution = tf.cast(
                tf.math.ceil(resolution / stride),
                tf.int32
            )
            self.stride_conv = Sequential(
                [
                    layers.DepthwiseConv2D(3, stride, 'same'),
                    layers.BatchNormalization(),
                ],
                name=f'{self.name}/stride_conv'
            )
            self.upsample = layers.UpSampling2D(
                size=(stride, stride),
                interpolation='bilinear',
                name=f'{self.name}/upsample'
            )
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None

        self.N = int(self.resolution ** 2)
        self.N2 = self.N
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        # h = self.dh + self.nh_kd * 2
        self.q = Sequential(
            [
                layers.Conv2D(self.num_heads * self.key_dim, 1, 1, 'valid'),
                layers.BatchNormalization(),
            ],
            name=f'{self.name}/q'
        )
        self.k = Sequential(
            [
                layers.Conv2D(self.num_heads * self.key_dim, 1, 1, 'valid'),
                layers.BatchNormalization(),
            ],
            name=f'{self.name}/k'
        )
        self.v = Sequential(
            [
                layers.Conv2D(self.num_heads * self.d, 1, 1, 'valid'),
                layers.BatchNormalization(),
            ],
            name=f'{self.name}/v'
        )
        self.v_local = Sequential(
            [
                layers.DepthwiseConv2D(3, 1, 'same'),
                layers.BatchNormalization(),
            ],
            name=f'{self.name}/v_local'
        )
        self.talking_head1 = layers.Conv2D(
            self.num_heads, 1, 1, 'valid',
            name=f'{self.name}/talking_head1'
        )
        self.talking_head2 = layers.Conv2D(
            self.num_heads, 1, 1, 'valid',
            name=f'{self.name}/talking_head2')
        self.proj = Sequential(
            [
                layers.Activation(act_layer),
                layers.Conv2D(dim, 1, 1, 'valid'),
                layers.BatchNormalization(),
            ],
            name=f'{self.name}/proj'
        )
        self.attention_biases = self.add_weight(
            'attention_biases',
            shape=[self.num_heads, self.N],
            dtype=self.dtype,
            initializer=initializers.Zeros(),
            trainable=True
        )
        self.attention_bias_idxs = \
            self.set_attention_bias_idxs(self.resolution)

    def set_attention_bias_idxs(self, resolution) -> tf.Tensor:
        """return relative positional information
        https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientformer_v2.py#L88

        Return:
            tf.Tensor: (resolution^2, resolution^2)
        """
        pos = tf.stack(tf.meshgrid(tf.range(resolution), tf.range(resolution)))
        pos = tf.reshape(pos, [2, -1])
        rel_pos = tf.math.abs(
            tf.expand_dims(pos, 2) - tf.expand_dims(pos, 1)
        )
        rel_pos = (rel_pos[0] * resolution) + rel_pos[1]
        # (resolution**2 * resolution**2,)
        return tf.reshape(
            tf.cast(rel_pos, tf.int32),
            [-1]
        )

    def add_attention_bias(self, x):
        attention_biases = tf.gather(
            tf.transpose(self.attention_biases),
            self.attention_bias_idxs
        )
        attention_biases = tf.reshape(
            tf.transpose(attention_biases),
            [self.num_heads, self.resolution ** 2, self.resolution ** 2]
        )
        return x + attention_biases

    def call(self, x, training=None):
        # N == (r) ** 2
        B = x.shape[0]
        if self.stride_conv is not None:
            x = self.stride_conv(x)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        v_local = self.v_local(v)

        q = tf.transpose(
            tf.reshape(q, [B, self.N, self.num_heads, -1]),
            [0, 2, 1, 3]
        )  # (B, nH, N, C)

        k = tf.transpose(
            tf.reshape(k, [B, self.N, self.num_heads, -1]),
            [0, 2, 3, 1]
        )  # (B, nH, C, N)

        v = tf.transpose(
            tf.reshape(v, [B, self.N, self.num_heads, -1]),
            [0, 2, 1, 3]
        )  # (B, nH, N, C')

        attn = (q @ k) * self.scale  # (B, nH, N, N)
        attn = self.add_attention_bias(attn)
        attn = tf.transpose(attn, [0, 2, 3, 1])  # (B, N, N, nH)

        attn = self.talking_head1(attn)
        attn = layers.Softmax(axis=2)(attn)
        attn = self.talking_head2(attn)
        attn = tf.transpose(attn, [0, 3, 1, 2])  # (B, nH, N, N)

        x = attn @ v  # (B, nH, N, C')
        x = tf.reshape(
            tf.transpose(x, [0, 2, 1, 3]),
            [B, self.resolution, self.resolution, self.dh],
            name=f'{self.name}/reshape'
        )  # (B, H, W, dh)
        out = x + v_local

        if self.upsample is not None:
            out = self.upsample(out)

        return self.proj(out)


class Mlp(layers.Layer):
    """shape의 변화는 없음"""

    def __init__(
        self,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: str = 'gelu',
        drop: float = 0.,
        mid_conv: bool = False,
        name: str = 'mlp'
    ):
        super().__init__(name=name)
        self.mid_conv = mid_conv
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.act = layers.Activation(act_layer, name=f'{self.name}/act')
        self.drop = layers.Dropout(drop, name=f'{self.name}/drop')

        if self.mid_conv:
            self.mid = layers.DepthwiseConv2D(
                3, 1, 'same',
                name=f'{self.name}/mid/d_conv'
            )
            self.mid_norm = layers.BatchNormalization(
                name=f'{self.name}/mid/bn'
            )

        self.norm1 = layers.BatchNormalization(name=f'{self.name}/bn1')
        self.norm2 = layers.BatchNormalization(name=f'{self.name}/bn2')

    def build(self, input_shape):
        out_features = self.out_features or input_shape[-1]
        hidden_features = self.hidden_features or input_shape[-1]
        self.fc1 = layers.Conv2D(
            hidden_features, 1, 1, 'valid',
            name=f'{self.name}/fc1'
        )
        self.fc2 = layers.Conv2D(
            out_features, 1, 1, 'valid',
            name=f'{self.name}/fc2'
        )

    def call(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)

        if self.mid_conv:
            x_mid = self.mid(x)
            x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.norm2(x)

        x = self.drop(x)
        return x


class AttnFFN(layers.Layer):

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.,
        act_layer: str = 'relu',
        drop: float = 0.,
        drop_path: float = 0.,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        resolution: int = 7,
        stride: Optional[int] = None,
        name: str = 'attn_ffn'
    ):
        super().__init__(name=name)
        self.token_mixer = Attention4D(
            dim=dim,
            resolution=resolution,
            act_layer=act_layer,
            stride=stride,
            name=f'{self.name}/token_mixer'
        )
        self.mlp = Mlp(
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            mid_conv=True,
            name=f'{self.name}/mlp'
        )
        self.drop_path = \
            StochasticDepth(drop_path) if drop_path > 0. \
            else layers.Activation('linear')

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = self.add_weight(
                'layer_scale_1',
                [1, 1, dim],
                dtype=self.dtype,
                initializer=initializers.Constant(layer_scale_init_value),
                trainable=True
            )
            self.layer_scale_2 = self.add_weight(
                'layer_scale_2',
                [1, 1, dim],
                dtype=self.dtype,
                initializer=initializers.Constant(layer_scale_init_value),
                trainable=True
            )

    def call(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(x))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


class FFN(layers.Layer):

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.,
        act_layer: str = 'gelu',
        drop: float = 0.,
        drop_path: float = 0.,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        name: str = 'ffn'
    ):
        super().__init__(name=name)
        self.mlp = Mlp(
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            mid_conv=True
        )

        self.drop_path = \
            StochasticDepth(drop_path) if drop_path > 0. \
            else layers.Activation('linear')
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = self.add_weight(
                'layer_scale_2',
                [1, 1, dim],
                dtype=self.dtype,
                initializer=initializers.Constant(layer_scale_init_value),
                trainable=True
            )

    def call(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x


def eformer_block(
    dim: int,
    index: int,
    layers: list[int],
    mlp_ratio: float = 4.,
    act_layer: str = 'gelu',
    drop_rate: float = 0.,
    drop_path_rate: float = 0.,
    use_layer_scale: bool = True,
    layer_scale_init_value: float = 1e-5,
    vit_num: int = 1,
    resolution: int = 7,
    e_ratios: Optional[float] = None,
    name: str = 'eformer_block'
):
    prefix = name
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = \
            drop_path_rate \
            * (block_idx + sum(layers[:index])) \
            / (sum(layers) - 1)
        mlp_ratio = e_ratios[str(index)][block_idx]

        if index >= 2 and (block_idx > layers[index] - 1 - vit_num):
            stride = 2 if index == 2 else None
            blocks.append(
                AttnFFN(
                    dim,
                    mlp_ratio,
                    act_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    resolution=resolution,
                    stride=stride,
                    name=f'{prefix}/attn_block{block_idx}'
                )
            )
        else:
            blocks.append(
                FFN(
                    dim,
                    mlp_ratio,
                    act_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    name=f'{prefix}/block{block_idx}'
                )
            )
    return Sequential(blocks, name=name)


def hard_sigmoid(x):
    return layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)


class HardSwish(layers.Layer):

    def __init__(
        self,
        name='hard_swish'
    ):
        super().__init__(name=name)

    def call(self, x):
        return layers.Multiply()([x, hard_sigmoid(x)])


class LGQuery(layers.Layer):
    """Local-Global Query"""

    def __init__(
        self,
        out_dim: int,
        name: str = 'lgq'
    ):
        super().__init__(name=name)
        prefix = name

        self.pool = layers.AveragePooling2D(
            (1, 1), 2, 'valid', name=f'{prefix}/pool'
        )
        self.local = layers.DepthwiseConv2D(
            3, 2, 'same', name=f'{prefix}/local'
        )
        self.proj = Sequential(
            [
                layers.Conv2D(out_dim, 1, 1, 'valid'),
                layers.BatchNormalization()
            ],
            name=f'{prefix}/proj'
        )

    def call(self, x):
        local_q = self.local(x)
        pool_q = self.pool(x)
        q = local_q + pool_q
        q = self.proj(q)
        return q


class Attention4DDownsample(layers.Layer):

    def __init__(
        self,
        dim: int = 384,
        key_dim: int = 16,
        num_heads: int = 8,
        attn_ratio: int = 4,
        resolution: int = 7,
        out_dim: Optional[int] = None,
        act_layer: Optional[str] = None,
        name: str = 'attn_down'
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads

        self.resolution = resolution

        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + self.nh_kd * 2

        if out_dim is not None:
            self.out_dim = out_dim
        else:
            self.out_dim = dim
        self.resolution2 = tf.cast(
            tf.math.ceil(self.resolution / 2),
            tf.int32
        )

        self.q = LGQuery(
            self.num_heads * self.key_dim,
            name=f'{self.name}/query'
        )

        self.N = int(self.resolution ** 2)
        self.N2 = int(self.resolution2 ** 2)

        self.k = Sequential(
            [
                layers.Conv2D(
                    self.num_heads * self.key_dim, 1, 1, 'valid',
                    name=f'{self.name}/key/conv'
                ),
                layers.BatchNormalization(name=f'{self.name}/key/bn')
            ],
            name=f'{self.name}/key'
        )
        self.v = Sequential(
            [
                layers.Conv2D(
                    self.num_heads * self.d, 1, 1, 'valid',
                    name=f'{self.name}/value/conv'
                ),
                layers.BatchNormalization(
                    name=f'{self.name}/value/bn'
                )
            ],
            name=f'{self.name}/value'
        )
        self.v_local = Sequential(
            [
                layers.DepthwiseConv2D(
                    3, 2, 'same',
                    name=f'{self.name}/v_local/d_conv'
                ),
                layers.BatchNormalization(
                    name=f'{self.name}/v_local/bn'
                )
            ]
        )
        self.proj = Sequential(
            [
                layers.Activation(
                    act_layer,
                    name=f'{self.name}/proj/{act_layer}'
                ),
                layers.Conv2D(
                    self.out_dim, 1, 1, 'valid',
                    name=f'{self.name}/proj/conv'
                ),
                layers.BatchNormalization(name=f'{self.name}/proj/bn')
            ],
            name=f'{self.name}/proj'
        )
        self.attention_biases = self.add_weight(
            'attention_biases',
            shape=[self.num_heads, self.N],
            dtype=self.dtype,
            initializer=initializers.Zeros(),
            trainable=True
        )
        self.attention_bias_idxs = \
            self.set_attention_bias_idxs(self.resolution)

    def set_attention_bias_idxs(self, resolution) -> tf.Tensor:
        """return relative positional information
        https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientformer_v2.py#L88

        Return:
            tf.Tensor: ((resolution/2)**2 * resolution**2,)
        """
        k_pos = tf.stack(
            tf.meshgrid(
                tf.range(resolution), tf.range(resolution)
            )
        )
        k_pos = tf.reshape(k_pos, [2, -1])

        q_pos = tf.stack(
            tf.meshgrid(
                tf.range(0, resolution, 2),
                tf.range(0, resolution, 2)
            )
        )
        q_pos = tf.reshape(q_pos, [2, -1])

        rel_pos = tf.math.abs(
            tf.expand_dims(q_pos, 2) - tf.expand_dims(k_pos, 1)
        )
        rel_pos = (rel_pos[0] * resolution) + rel_pos[1]
        # (N2 * N,)
        return tf.reshape(
            tf.cast(rel_pos, tf.int32),
            [-1]
        )

    def add_attention_bias(self, x):
        attention_biases = tf.gather(
            tf.transpose(self.attention_biases),
            self.attention_bias_idxs
        )  # (N2 * N, nH)
        attention_biases = tf.reshape(
            tf.transpose(attention_biases),
            [self.num_heads, self.N2, self.N]
        )  # (nH, N2, N)
        return x + attention_biases

    def call(self, x):
        # N == (r) ** 2
        # N2 = (r / 2) ** 2
        B, C, H, W = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        v_local = self.v_local(v)  # (B, H, W, C')

        q = tf.transpose(
            tf.reshape(q, [B, self.N2, self.num_heads, -1]),
            [0, 2, 1, 3]
        )  # (B, nH, N2, C)
        k = tf.transpose(
            tf.reshape(k, [B, self.N, self.num_heads, -1]),
            [0, 2, 3, 1]
        )  # (B, nH, C, N)
        v = tf.transpose(
            tf.reshape(v, [B, self.N, self.num_heads, -1]),
            [0, 2, 1, 3]
        )  # (B, nH, N, C)

        # (B, nH, N2, N)
        attn = (q @ k) * self.scale
        attn = self.add_attention_bias(attn)
        attn = layers.Softmax(-1, name=f'{self.name}/softmax')(attn)

        # (B, nH, N2, C) -> (B, N2, nH, C) -> (B, H/2, W/2, C')
        out = attn @ v
        out = tf.reshape(
            tf.transpose(out, [0, 2, 1, 3]),
            [B, self.resolution2, self.resolution2, self.dh],
            name=f'{self.name}/reshape'
        )
        out += v_local
        return self.proj(out)


class Embedding(layers.Layer):

    def __init__(
        self,
        patch_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: layers.Layer = layers.BatchNormalization,
        light: bool = False,
        asub: bool = False,
        resolution: Optional[int] = None,
        act_layer: str = 'relu',
        name: str = 'embeddiing'
    ):
        super().__init__(name=name)
        self.light = light
        self.asub = asub
        if self.light:
            self.new_proj = Sequential(
                [
                    layers.DepthwiseConv2D(
                        3, 2, 'same', name=f'{self.name}/new_proj/d_conv'),
                    layers.BatchNormalization(
                        name=f'{self.name}/new_proj/bn1'
                    ),
                    HardSwish(name=f'{self.name}/new_proj/hard_swish'),
                    layers.Conv2D(
                        embed_dim, 1, 1, 'valid',
                        name=f'{self.name}/new_proj/conv'
                    ),
                    layers.BatchNormalization(
                        name=f'{self.name}/new_proj/bn2'
                    ),
                ],
                name=f'{self.name}/new_proj'
            )
            self.skip = Sequential(
                [
                    layers.Conv2D(
                        embed_dim, 1, 2, 'valid',
                        name=f'{self.name}/skip/conv'
                    ),
                    layers.BatchNormalization(
                        name=f'{self.name}/skip/bn'
                    )
                ],
                name=f'{self.name}/skip'
            )
        elif self.asub:
            self.attn = Attention4DDownsample(
                dim=in_chans,
                out_dim=embed_dim,
                resolution=resolution,
                act_layer=act_layer,
                name=f'{self.name}/attn_down'
            )
            self.conv = Sequential(
                [
                    layers.ZeroPadding2D(
                        padding,
                        name=f'{self.name}/conv/pad'
                    ),
                    layers.Conv2D(
                        embed_dim, patch_size, stride, 'valid',
                        name=f'{self.name}/conv/conv'
                    )
                ],
                name=f'{self.name}/conv'
            )
            self.bn = \
                norm_layer(name=f'{self.name}/bn') if norm_layer \
                else layers.Activation('linear', name=f'{self.name}/identity')
        else:
            self.proj = Sequential(
                [
                    layers.ZeroPadding2D(
                        padding, name=f'{self.name}/conv/pad'
                    ),
                    layers.Conv2D(
                        embed_dim, patch_size, stride,
                        'valid', name=f'{self.name}/conv/conv'
                    )
                ],
                name=f'{self.name}/conv'
            )
            self.norm = \
                norm_layer(name=f'{self.name}/norm') if norm_layer \
                else layers.Activation('linear', name=f'{self.name}/identity')

    def call(self, x):
        if self.light:
            out = self.new_proj(x) + self.skip(x)
        elif self.asub:
            out_conv = self.conv(x)
            out_conv = self.bn(out_conv)
            out = self.attn(x) + out_conv
        else:
            x = self.proj(x)
            out = self.norm(x)
        return out


class EfficientFormerV2(Model):

    def __init__(
        self,
        n_layers: list[int],
        embed_dims: list[int],
        downsamples: list[bool],
        mlp_ratios: int = 4,
        norm_layer: layers.Layer = layers.BatchNormalization,
        act_layer: str = 'gelu',
        num_classes: int = 1000,
        down_patch_size: int = 3,
        down_stride: int = 2,
        down_pad: int = 1,
        drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        fork_feat: bool = False,
        init_cfg: Optional[dict] = None,
        pretrained: Optional[bool] = None,
        vit_num: int = 0,
        distillation: bool = True,
        resolution: int = 224,
        e_ratios: dict = expansion_ratios_L,
        include_top: bool = False,
        name: str = 'efficientformerv2'
    ):
        super().__init__(name=name)
        self.dist = distillation

        self.patch_embed = stem(embed_dims[0], act_layer, name='stem')

        network = []
        for i in range(len(n_layers)):
            network.append(
                eformer_block(
                    embed_dims[i],
                    i,
                    n_layers,
                    mlp_ratios,
                    act_layer,
                    drop_rate,
                    drop_path_rate,
                    use_layer_scale,
                    layer_scale_init_value,
                    resolution=tf.math.ceil(resolution / (2 ** (i + 2))),
                    vit_num=vit_num,
                    e_ratios=e_ratios,
                    name=f'{self.name}stage{i}'
                )
            )
            if i >= len(n_layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                asub = (i >= 2)
                network.append(
                    Embedding(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                        resolution=tf.math.ceil(resolution / (2 ** (i + 2))),
                        asub=asub,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        name=f'{self.name}/stage{i}/downsample'
                    )
                )
        self.network = network

        # Classifier head
        self.norm = norm_layer(name=f'{self.name}/classifier/norm')
        self.gap = layers.GlobalAveragePooling2D(
            name=f'{self.name}/classifier/gap'
        )
        self.head = \
            layers.Dense(
                num_classes,
                name=f'{self.name}/classifier/dense'
            ) \
            if num_classes > 0 \
            else layers.Activation('linear')

        if self.dist:
            self.dist_head = layers.Dense(
                num_classes,
                name=f'{self.name}/classifier/distill/dense'
            ) \
                if num_classes > 0 \
                else layers.Activation('linear')
        self.include_top = include_top

    def call(self, x, training=None):
        if training is None:
            training = backend.learning_phase()

        x = self.patch_embed(x)
        for block in self.network:
            x = block(x)

        if not self.include_top:
            return x

        x = self.norm(x)
        if self.dist:
            cls_out = \
                self.head(self.gap(x)), self.dist_head(self.gap(x))
            if not training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(self.gap(x))

        return cls_out

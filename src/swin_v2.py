"""
# NOTE - 초안 완성
"""
from __future__ import annotations

from functools import partial

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras import initializers

from src.common import StochasticDepth


class PatchEmbed(layers.Layer):

    def __init__(
        self,
        patch_size=4,
        embed_dim=96,
        norm_layer=None,
        name='patch_embed'
    ):
        super().__init__(name=name)

        prefix = name
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.patch_size = patch_size

        self.proj = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            name=f'{prefix}/proj'
        )
        if norm_layer is not None:
            self.norm = norm_layer(name=f'{prefix}/norm')
        else:
            self.norm = None

    def call(self, x):
        H, W = x.shape[1:3]
        if (W % self.patch_size[1] != 0) or (H % self.patch_size[0] != 0):
            x = layers.ZeroPadding2D(
                (
                    (0, self.patch_size[0] - H % self.patch_size[0]),
                    (0, self.patch_size[1] - W % self.patch_size[1])
                ),
                name=f'{self.name}/pad'
            )(x)
        x = self.proj(x)
        if self.norm is not None:
            h, w = x.shape[1:3]
            x = layers.Reshape(
                [h*w, -1],
                name=f'{self.name}/reshape1'
            )(x)
            x = self.norm(x)
            x = layers.Reshape(
                [h, w, -1],
                name=f'{self.name}/reshape2'
            )(x)
        return x


class PatchMerging(layers.Layer):

    def __init__(
        self,
        dim,
        norm_layer=layers.LayerNormalization,
        name='patch_merge'
    ):
        super().__init__(name=name)

        prefix = name

        self.reduction = layers.Dense(
            2 * dim,
            use_bias=False,
            name=f'{prefix}/reduction'
        )
        self.norm = norm_layer(name=f'{prefix}/norm')

    def call(self, x):
        B, H, W, C = x.shape
        # assert N == H * W, "input feature has wrong size"

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = layers.ZeroPadding2D(
                ((0, H % 2), (0, W % 2)),
                name=f'{self.name}/pad'
            )(x)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = layers.Concatenate(
            axis=-1,
            name=f'{self.name}/concat'
        )([x0, x1, x2, x3])  # B H/2 W/2 4C
        h, w = x.shape[1:3]
        x = layers.Reshape(
            [-1, 4 * C],
            name=f'{self.name}/reshape1'
        )(x)
        x = self.reduction(x)
        x = self.norm(x)
        x = layers.Reshape(
            [h, w, -1],
            name=f'{self.name}/reshape2'
        )(x)
        return x


class WindowAttention(layers.Layer):
    """
    #TODO - 전체 점검 필요(날 것의 코드 변환 상태)
    """

    def __init__(
        self,
        window_size: list[int],
        num_heads: int,
        attn_drop=0.,
        proj_drop=0.,
        rpe_hidden_dim=512,
        pretrain_window_size: int = -1,
        name='window_attention'
    ):
        super().__init__(name=name)
        self.window_size = window_size
        self.num_heads = num_heads
        self.pretrain_window_size = pretrain_window_size
        # self.name = name

        self.logit_scale = self.add_weight(
            'logit_scale',
            shape=[num_heads, 1, 1],
            initializer=initializers.Constant(tf.math.log(10.)),
            trainable=True,
            dtype=self.dtype
        )

        # mlp to generate table of relative position bias
        self.rpe_mlp = Sequential(
            [
                layers.Dense(rpe_hidden_dim, use_bias=True),
                layers.ReLU(),
                layers.Dense(num_heads, use_bias=False),
            ]
        )
        # self.coord_table = self.get_relative_coords_table()
        # self.pos_index = self.get_relative_index()

        self.attn_drop = layers.Dropout(attn_drop)
        self.proj_drop = layers.Dropout(proj_drop)

    def build(self, input_shape):
        in_channels = input_shape[-1]

        self.qkv = layers.Dense(
            in_channels * 3,
            use_bias=False,
            name=f'{self.name}/qkv'
        )
        self.q_bias = self.add_weight(
            'q_bias',
            shape=[in_channels],
            initializer='zeros',
            trainable=True,
            dtype=self.dtype
        )
        self.v_bias = self.add_weight(
            'v_bias',
            shape=[in_channels],
            initializer='zeros',
            trainable=True,
            dtype=self.dtype
        )
        self.proj = layers.Dense(in_channels, name=f'{self.name}/proj')

    def get_relative_index(self):
        """
        get pair-wise relative position index for each token inside the window
        """
        coords_h = tf.range(self.window_size[0])
        coords_w = tf.range(self.window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w), 0)  # 2, Wh, Ww
        coords_flatten = tf.reshape(coords, [2, -1])  # 2, Wh * Ww

        # (2, Wh*Ww, Wh*Ww)
        relative_coords = \
            tf.expand_dims(coords_flatten, 2) - \
            tf.expand_dims(coords_flatten, 1)
        # (Wh*Ww, Wh*Ww, 2)
        relative_coords = tf.transpose(relative_coords, [1, 2, 0])
        relative_coords_y = relative_coords[..., 0] + (self.window_size[0] - 1)
        relative_coords_x = relative_coords[..., 1] + (self.window_size[1] - 1)
        relative_coords_y *= 2 * self.window_size[0] - 1
        relative_coords = tf.stack([relative_coords_y, relative_coords_x], -1)
        # (Wh*Ww, Wh*Ww)
        relative_position_index = tf.math.reduce_sum(relative_coords, -1)
        return tf.reshape(relative_position_index, [-1])

    def get_relative_coords_table(self):
        # get relative_coords_table
        coords_h = tf.range(
            -(self.window_size[0] - 1),
            self.window_size[0],
            dtype=tf.float32
        )
        coords_w = tf.range(
            -(self.window_size[1] - 1),
            self.window_size[1],
            dtype=tf.float32
        )

        # 1, 2*Wh-1, 2*Ww-1, 2
        coords_table = tf.stack(
            tf.meshgrid(coords_h, coords_w),
            axis=-1
        )
        # coords_table = tf.transpose(relative_coords_table, [1, 2, 0])
        coords_table = tf.expand_dims(coords_table, 0)

        coords_table = tf.stack(
            [
                coords_table[..., 0] / (self.pretrain_window_size - 1),
                coords_table[..., 1] / (self.pretrain_window_size - 1)
            ],
            axis=-1
        )

        # normalize to -8, 8
        scale = 8. / tf.convert_to_tensor(
            [self.pretrain_window_size - 1],
            dtype=coords_table.dtype
        )
        coords_table *= scale
        return tf.sign(coords_table) \
            * tf.math.log1p(tf.abs(coords_table)) \
            / tf.math.log(8.0)

    def call(self, x, mask=None):
        # x, mask = inp
        Bw, N, C = x.shape
        qkv_bias = layers.Concatenate(
            axis=0,
            name=f'{self.name}/qkv_bias_concat'
        )(
            [
                self.q_bias,
                tf.zeros_like(self.v_bias, dtype=self.v_bias.dtype),
                self.v_bias
            ]
        )
        qkv = self.qkv(x)
        qkv = tf.nn.bias_add(qkv, qkv_bias)
        qkv = layers.Reshape([N, 3, self.num_heads, -1])(qkv)
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = tf.unstack(qkv)  # q: (B, n_heads, N, C)

        scale = tf.math.minimum(
            self.logit_scale,
            tf.math.log(1. / .01)
        )
        scale = tf.math.exp(scale)
        q = tf.math.l2_normalize(q, axis=-1, epsilon=1e-12)
        k = tf.math.l2_normalize(k, axis=-1, epsilon=1e-12)
        attn = tf.matmul(q, k, transpose_b=True) * scale

        # relative_position_bias_table: 2*Wh-1 * 2*Ww-1, nH
        bias_table = self.rpe_mlp(self.get_relative_coords_table())
        bias_table = tf.reshape(bias_table, [-1, self.num_heads])
        position_bias = tf.gather(
            bias_table,
            self.get_relative_index()
        )
        position_bias = tf.reshape(
            position_bias,
            [
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1
            ]
        )
        position_bias = tf.transpose(
            position_bias,
            [2, 0, 1]
        )  # nH, Wh*Ww, Wh*Ww
        position_bias = tf.expand_dims(
            16 * tf.math.sigmoid(position_bias),
            axis=0
        )
        attn = attn + position_bias

        if mask is not None:
            nW = mask.shape[0]  # number of windows
            # (nW, 1, N, N)
            mask = tf.expand_dims(mask, 1)
            # (1, nW, 1, N, N)
            mask = tf.expand_dims(mask, 0)
            attn = tf.reshape(
                attn,
                (Bw // nW, nW, self.num_heads, N, N)
            )
            attn += mask
            attn = tf.reshape(
                attn,
                (-1, self.num_heads, N, N)
            )

        attn = layers.Softmax(name=f'{self.name}/softmax')(attn)
        # attn = attn.type_as(x)
        attn = self.attn_drop(attn)

        x = tf.transpose(
            tf.matmul(attn, v),
            [0, 2, 1, 3]
        )
        x = tf.reshape(x, [Bw, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(layers.Layer):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer='gelu',
        drop_rate=0.,
        norm_layer=None,
        name='mlp'
    ):
        super().__init__(name=name)
        prefix = name
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.act_layer = act_layer
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = layers.Dense(
            hidden_features,
            name=f'{prefix}/fc1'
        )
        self.act = layers.Activation(
            act_layer,
            name=f'{prefix}/{act_layer}'
        )
        self.fc2 = layers.Dense(
            out_features,
            name=f'{prefix}/fc2'
        )
        self.drop1 = layers.Dropout(drop_rate, name=f'{prefix}/dropout1')
        self.drop2 = layers.Dropout(drop_rate, name=f'{prefix}/dropout2')
        if norm_layer is not None:
            self.norm = norm_layer(name=f'{prefix}/norm')
        else:
            self.norm = None

    def call(self, x):
        x = self.fc1(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def window_partition(x, window_size, name):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    prefix = name
    H, W, C = x.shape[1:]
    x = layers.Reshape(
        (
            H // window_size,
            window_size,
            W // window_size,
            window_size,
            C
        ),
        name=f'{prefix}/reshape'
    )(x)
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5], name=f'{prefix}/transpose')
    x = tf.reshape(x, (-1, window_size, window_size, C))
    return x


def window_reverse(windows, window_size, height, width):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (height * width / window_size / window_size))
    x = tf.reshape(
        windows,
        [
            B,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            -1
        ]
    )
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, height, width, -1])
    return x


class SwinTransformerBlockPost(layers.Layer):
    """Swin Transformer Block."""

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.,
        attn_drop=0.,
        drop_rate=0.,
        drop_path=0.,
        use_mlp_norm=False,
        endnorm=False,
        act_layer='gelu',
        norm_layer=layers.LayerNormalization,
        rpe_hidden_dim=512,
        pretrain_window_size=-1,
        name='swinV2_block'
    ):
        super().__init__(name=name)
        prefix = name

        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = norm_layer(name=f'{prefix}/norm1')
        self.attn = WindowAttention(
            window_size=[window_size, window_size],
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop_rate,
            rpe_hidden_dim=rpe_hidden_dim,
            pretrain_window_size=pretrain_window_size,
            name=f'{prefix}/wmsa'
        )
        self.drop_path1 = \
            StochasticDepth(drop_path, name=f'{prefix}/drop_path1') \
            if drop_path > 0. \
            else None
        self.drop_path2 = \
            StochasticDepth(drop_path, name=f'{prefix}/drop_path2') \
            if drop_path > 0. \
            else None
        self.norm2 = norm_layer(name=f'{prefix}/norm2')
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop_rate=drop_rate,
            norm_layer=norm_layer if use_mlp_norm else None,
            name=f'{prefix}/mlp'
        )
        if endnorm:
            self.enorm = norm_layer(dim, name=f'{prefix}/end_norm')
        else:
            self.enorm = None

    def call(self, x, mask_matrix):
        B, H, W, C = x.shape

        shortcut = x

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = layers.ZeroPadding2D(
                ((pad_t, pad_b), (pad_l, pad_r)),
                name=f'{self.name}/pad'
            )(x)
        Hp, Wp = x.shape[1:3]

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x,
                (-self.shift_size, -self.shift_size),
                axis=(1, 2),
                name=f'{self.name}/shift'
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows: (nW*B, window_size, window_size, C)
        x_windows = window_partition(
            shifted_x,
            self.window_size,
            name=f'{self.name}/partition'
        )
        x_windows = tf.reshape(
            x_windows,
            [-1, self.window_size*self.window_size, C]
        )

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = tf.reshape(
            attn_windows,
            [-1, self.window_size, self.window_size, C]
        )

        # merge windows
        # come back to regular feature
        # (nW*B, window_size, window_size, C) -> (B, h, w, c)
        shifted_x = window_reverse(
            attn_windows,
            self.window_size,
            Hp, Wp
        )

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x,
                (self.shift_size, self.shift_size),
                axis=(1, 2),
                name=f'{self.name}/un-shift'
            )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        # x = layers.Reshape([H * W, C])(x)

        # FFN
        x = self.norm1(x)
        if self.drop_path1 is not None:
            x = self.drop_path1(x)
        x = layers.Add(name=f'{self.name}/add1')([shortcut, x])
        shortcut = x

        x = self.mlp(x)
        x = self.norm2(x)
        if self.drop_path2 is not None:
            x = self.drop_path2(x)
        x = layers.Add(name=f'{self.name}/add2')([shortcut, x])

        if self.enorm is not None:
            x = self.enorm(x)

        # x = layers.Reshape([H, W, C])(x)
        return x


def get_attention_mask(mask_shape: list[int, int], window_size=7):
    shift_size = window_size // 2
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None)
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None)
    )
    grid_yx = tf.stack(
        tf.meshgrid(tf.range(mask_shape[0]), tf.range(mask_shape[1])),
        axis=-1
    )
    mask = tf.zeros([1, mask_shape[0], mask_shape[1], 1], dtype=tf.int32)
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            indice = grid_yx[h, w, :]
            indice = tf.reshape(indice, [-1, 2])
            indices = tf.concat(
                [
                    tf.zeros([indice.shape[0], 1], dtype=tf.int32),
                    indice,
                    tf.zeros([indice.shape[0], 1], dtype=tf.int32),
                ],
                axis=-1
            )  # (N, 4)
            updates = tf.ones(indices.shape[0], dtype=tf.int32) * cnt
            mask = tf.tensor_scatter_nd_update(
                mask,
                indices,
                updates
            )
            cnt += 1
    return mask


class BasicLayer(layers.Layer):
    """A basic Swin Transformer layer for one stage."""

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.,
        drop_rate=0.,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=layers.LayerNormalization,
        downsample=None,
        endnorm_interval=-1,
        use_mlp_norm=False,
        use_shift=True,
        rpe_hidden_dim=512,
        pretrain_window_size=-1,
        name='basic_layer'
    ):
        super().__init__(name=name)
        prefix = name

        self.window_size = window_size

        self.blocks = [
            SwinTransformerBlockPost(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) or (
                    not use_shift) else window_size // 2,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                drop_rate=drop_rate,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                use_mlp_norm=use_mlp_norm,
                endnorm=((i + 1) % endnorm_interval == 0) and (
                    endnorm_interval > 0),
                norm_layer=norm_layer,
                rpe_hidden_dim=rpe_hidden_dim,
                pretrain_window_size=pretrain_window_size,
                name=f'{prefix}/block{i}'
            ) for i in range(depth)
        ]

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                dim, norm_layer, name=f'{prefix}/downsample')
        else:
            self.downsample = None

    def call(self, x):
        H, W = x.shape[1:3]

        Hp = int(tf.math.ceil(H / self.window_size)) * self.window_size
        Wp = int(tf.math.ceil(W / self.window_size)) * self.window_size

        # calculate attention mask for SW-MSA
        img_mask = get_attention_mask([Hp, Wp], self.window_size)

        # nW, window_size, window_size, 1
        mask_windows = window_partition(
            img_mask, self.window_size, name=f'{self.name}/mask_partition')
        mask_windows = tf.reshape(
            mask_windows,
            [-1, self.window_size * self.window_size]
        )

        # nW, window_size**2, window_size**2
        attn_mask = tf.expand_dims(mask_windows, 1) - \
            tf.expand_dims(mask_windows, 2)
        attn_mask = tf.where(
            attn_mask == 0,
            0.,
            -100.
        )
        for block in self.blocks:
            x = block(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x


class SwinV2TransformerRPE2FC(Model):

    def __init__(
        self,
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-6),
        patch_norm=True,
        use_checkpoint=False,
        endnorm_interval=-1,
        use_mlp_norm_layers=[],
        rpe_hidden_dim=512,
        use_shift=True,
        pretrain_window_size=[-1, -1, -1, -1],
        name: str = 'backbone_swin_v2'
    ):
        super().__init__(name=name)
        self.num_layers = len(depths)

        prefix = name

        if isinstance(window_size, int):
            window_size = [window_size] * self.num_layers
        if isinstance(use_shift, bool):
            use_shift = [use_shift] * self.num_layers
        if isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint] * self.num_layers

        self.patch_embed = PatchEmbed(
            patch_size,
            embed_dim,
            norm_layer=norm_layer if patch_norm else None,
            name=f'{prefix}/patch_embed'
        )
        self.pos_drop = layers.Dropout(drop_rate, name=f'{prefix}/pos_drop')

        # stochastic depth
        dpr = [
            x for x in tf.linspace(0., drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        downsample_layer = PatchMerging
        self.stages = []
        for i_layer in range(self.num_layers):
            cur_dim = int(embed_dim * 2 ** i_layer)
            if i_layer <= self.num_layers - 2:
                cur_downsample_layer = downsample_layer
            else:
                cur_downsample_layer = None
            self.stages.append(
                BasicLayer(
                    dim=cur_dim,
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size[i_layer],
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:i_layer]):sum(depths[:i_layer + 1])
                    ],
                    norm_layer=norm_layer,
                    downsample=cur_downsample_layer,
                    endnorm_interval=endnorm_interval,
                    use_mlp_norm=(i_layer in use_mlp_norm_layers),
                    use_shift=use_shift[i_layer],
                    rpe_hidden_dim=rpe_hidden_dim,
                    pretrain_window_size=pretrain_window_size[i_layer],
                    name=f'{prefix}/stage{i_layer}'
                )
            )

        # add a norm layer for each output
        # actually, last stage is only needed
        self.norm = norm_layer(name=f'{prefix}/last_norm')

        # NOTE - freeze stages 하는 부분
        # 확인한 다음 추가

        # ---

    def call(self, x):
        x = self.patch_embed(x)

        # Wh, Ww = x.shape[1:3]
        # # x = layers.Reshape(
        # #     [Wh, Ww, -1],
        # #     name=f'{self.name}/reshape'
        # # )(x)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            x_out, x = self.stages[i](x)
            if i == (self.num_layers - 1):
                x_out = self.norm(x_out)
            # outs.append(x_out)
        # return outs
        return x_out

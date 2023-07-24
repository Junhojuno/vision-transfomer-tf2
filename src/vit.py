"""
https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
"""
from typing import List

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Input


def patch_embedding(
    x: tf.Tensor,
    patch_size: int = 16,
    emb_dims: int = 512
):
    # after convolution,
    # all patches are changed to
    # 1D embedding vectors
    x = layers.Conv2D(
        emb_dims,
        kernel_size=patch_size,
        strides=patch_size,
        padding='valid',
        name='patch/conv'
    )(x)
    # n_patches = x.shape[1] * x.shape[2]
    x = layers.Reshape(
        [x.shape[1] * x.shape[2], x.shape[3]],
        name='patch/reshape'
    )(x)
    return x


def prepend_class_token(x):
    cls_token = tf.Variable(
        tf.zeros([1, 1, x.shape[-1]]),
        trainable=True,
        name='cls_token'
    )
    return layers.Concatenate(axis=1)([cls_token, x])


def add_position_embedding(
    x,
    name='positional_encoding'
):
    pos_emb_shape = (1, x.shape[1], x.shape[2])
    pos_emb = tf.Variable(
        tf.random.normal(pos_emb_shape, stddev=0.02),
        trainable=True,
        name=f'{name}/embedding'
    )
    return layers.Add(name=f'{name}/add')([x, pos_emb])


def multi_head_attention(
    x: tf.Tensor,
    n_heads: int,
    dropout_rate: float,
    name: str
):
    """multi-head self-attention"""
    def extract_qkv(
        x,
        n_heads: int,
        head_dims: int,
    ):
        """patch embedding -> [query, key, value]"""
        prefix = f'{name}/qkv'
        QKV = layers.Dense(
            3 * n_heads * head_dims,
            use_bias=False,
            name=f'{prefix}/dense'
        )(x)  # (B, N+1, 3*heads*dims)
        QKV = tf.transpose(QKV, [0, 2, 1])

        QKV = layers.Reshape(
            [3*n_heads, head_dims, -1],
            name=f'{prefix}/reshape'
        )(QKV)  # (B, 3*heads, dims, N+1)
        QKV = tf.transpose(QKV, [0, 1, 3, 2])  # (B, 3*heads, N+1, dims)

        return tf.split(
            QKV,
            num_or_size_splits=3,
            axis=1
        )  # (B, heads, N+1, dims)

    n_patches, emb_dims = x.shape[1], x.shape[2]
    head_dims = emb_dims // n_heads
    Q, K, V = extract_qkv(x, n_heads, head_dims)

    # get weight matrix
    W = layers.Activation(
        'softmax',
        name=f'{name}/softmax'
    )(
        tf.matmul(Q, K, transpose_b=True) * (n_heads ** 0.5)
    )  # (B, heads, N+1, N+1)
    W = layers.Dropout(
        dropout_rate,
        name=f'{name}/dropout1'
    )(W)
    out = tf.matmul(W, V)  # (B, heads, N+1, dims)
    out = tf.transpose(out, [0, 2, 1, 3])  # (B, N+1, heads, dims)
    # (B, N+1, heads*dims)
    out = layers.Reshape(
        [n_patches, -1],
        name=f'{name}/reshape'
    )(out)

    # projection to input embedding dimension
    out = layers.Dense(emb_dims, name=f'{name}/dense')(out)
    out = layers.Dropout(dropout_rate, name=f'{name}/dropout2')(out)
    return out


def encoder_1d_block(
    inp,
    mlp_dim: int,
    n_heads: int,
    dropout_rate: float = 0.1,
    attention_dropout_rate: float = 0.1,
    name: str = 'encoder_block'
):
    def mlp_block(x):
        out_dim = x.shape[-1]
        prefix = f'{name}/mlp'
        x = layers.Dense(mlp_dim, name=f'{prefix}/dense1')(x)
        x = layers.Activation('gelu', name=f'{prefix}/gelu')(x)
        x = layers.Dropout(dropout_rate, name=f'{prefix}/dropout1')(x)
        x = layers.Dense(out_dim, name=f'{prefix}/dense2')(x)
        x = layers.Dropout(dropout_rate, name=f'{prefix}/dropout2')(x)
        return x

    x = layers.LayerNormalization(epsilon=1e-06, name=f'{name}/norm')(inp)
    x = multi_head_attention(
        x,
        n_heads,
        attention_dropout_rate,
        name=f'{name}/att'
    )
    x = layers.Dropout(dropout_rate, name=f'{name}/dropout')(x)
    x = layers.Add(name=f'{name}/add')([x, inp])

    # MLP block
    y = layers.LayerNormalization(epsilon=1e-06, name=f'{name}/mlp/norm')(x)
    y = mlp_block(y)

    return x + y


def encoder(
    x,
    n_blocks: int,
    mlp_dim: int,
    n_heads: int,
    dropout_rate: float = 0.1,
    attention_dropout_rate: float = 0.1,
    add_pos_embed: bool = True,
    name='encoder'
):
    if add_pos_embed:
        x = add_position_embedding(x, name=f'{name}/pos_enc')
        x = layers.Dropout(dropout_rate, name=f'{name}/dropout')(x)

    # Input Encoder
    for idx in range(n_blocks):
        prefix = f'{name}/block_{idx}'
        x = encoder_1d_block(
            x,
            mlp_dim,
            n_heads,
            dropout_rate,
            attention_dropout_rate,
            name=prefix
        )
    return layers.LayerNormalization(
        epsilon=1e-06,
        name=f'{name}/norm'
    )(x)


def ViT(
    input_shape: List[int],
    patch_size: int,
    patch_emb_dim: int,
    n_blocks: int,
    mlp_dim: int,
    n_heads: int,
    dropout_rate: float,
    attention_dropout_rate: float,
    num_classes: int,
    name: str = 'vision_transformer'
):
    inp = Input(input_shape, name='vit_input')
    x = patch_embedding(inp, patch_size, patch_emb_dim)
    x = prepend_class_token(x)
    x = encoder(
        x,
        n_blocks,
        mlp_dim,
        n_heads,
        dropout_rate,
        attention_dropout_rate
    )
    x = x[:, 0, :]
    x = layers.Dense(num_classes, name='classifier')(x)
    return Model(inp, x, name=name)

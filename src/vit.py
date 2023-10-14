"""
#NOTE - Vanila Vision Transformer
references
    - https://arxiv.org/abs/2010.11929
    - https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Input


def patch_embedding(
    x: tf.Tensor,
    patch_size: int = 16,
    emb_dims: int = 512
) -> tf.Tensor:
    """convert image into patches with emb_dims each
    (B, H, W, 3) -> (B, N, emb_dims)

    Args:
        x (tf.Tensor): input tensor
        patch_size (int, optional): single patch's size. Defaults to 16.
        emb_dims (int, optional): embedding dims per patch. Defaults to 512.

    Returns:
        tf.Tensor: N patches with dims==emb_dims; (B, N, emb_dims)
    """
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


def prepend_class_token(x: tf.Tensor) -> tf.Tensor:
    """Add class token to the head of patches

    Args:
        x (tf.Tensor): patch embeddings; (B, N, emb_dims)

    Returns:
        tf.Tensor: patch embeddings with class token; (B, N + 1, emb_dims)
    """
    cls_token = tf.Variable(
        tf.zeros([1, 1, x.shape[-1]]),
        trainable=True,
        name='cls_token'
    )
    return layers.Concatenate(axis=1)([cls_token, x])


def add_position_embedding(
    x: tf.Tensor,
    name: str = 'positional_encoding'
) -> tf.Tensor:
    """Add positional embedding to patch embeddings

    Args:
        x (tf.Tensor): patch embeddings; (B, N + 1, emb_dims)
        name (str, optional): layer name to be named.
            Defaults to 'positional_encoding'.

    Returns:
        tf.Tensor: patch embeddings to which position embeddings are added
    """
    prefix = name
    pos_emb_shape = (1, x.shape[1], x.shape[2])
    pos_emb = tf.Variable(
        tf.random.normal(pos_emb_shape, stddev=0.02),
        trainable=True,
        name=f'{prefix}/embedding'
    )
    return layers.Add(name=f'{prefix}/add')([x, pos_emb])


def multi_head_attention(
    x: tf.Tensor,
    n_heads: int,
    dropout_rate: float,
    name: str
) -> tf.Tensor:
    """This is Multi-Head Attention
        - 1. 입력 patch embeddings으로부터 (query, key, value) 추출
        - 2. weight matrix 추출(weights = query @ key)
        - 3. weight matrix를 value에 곱하기
        - 4. 입력 patch embeddings와 동일한 shape으로 맞춰주고, FC layer 적용

    Args:
        x (tf.Tensor): patch embeddings; (B, N+1, emb_dims)
        n_heads (int): the number of heads. emb_dims를 나눌 개수
        dropout_rate (float): dropout rate
        name (str): name

    Returns:
        tf.Tensor: patch embeddings; (B, N+1, emb_dims)
    """
    prefix = name

    def extract_qkv(
        x: tf.Tensor,
        n_heads: int,
        head_dims: int,
        name: str
    ) -> list[tf.Tensor, tf.Tensor, tf.Tensor]:
        """patch embedding -> [query, key, value]

        Args:
            x (tf.Tensor): patch embeddings; (B, N+1, emb_dims)
            n_heads (int): the number of heads
            head_dims (int): dimensions that each head has
                head_dims = (emb_dims) // n_heads
            name (str): name

        Returns:
            list[tf.Tensor, tf.Tensor, tf.Tensor]: [query key value]
        """
        prefix = f'{name}/extract_qkv'
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
        )  # list of (B, heads, N+1, dims)

    n_patches, emb_dims = x.shape[1], x.shape[2]
    head_dims = emb_dims // n_heads
    Q, K, V = extract_qkv(x, n_heads, head_dims, prefix)

    # NOTE - get weight matrix
    W = layers.Activation(
        'softmax',
        name=f'{prefix}/weight/softmax'
    )(
        tf.matmul(Q, K, transpose_b=True,
                  name=f'{prefix}/weight/matmul') * (n_heads ** 0.5)
    )  # (B, heads, N+1, N+1)
    W = layers.Dropout(
        dropout_rate,
        name=f'{prefix}/weight/dropout'
    )(W)

    out = tf.matmul(W, V, name=f'{prefix}/matmul')  # (B, heads, N+1, dims)
    # (B, N+1, heads, dims)
    out = tf.transpose(
        out,
        [0, 2, 1, 3],
        name=f'{prefix}/transpose'
    )
    out = layers.Reshape(
        [n_patches, -1],
        name=f'{prefix}/reshape'
    )(out)  # (B, N+1, heads*dims)

    # projection to input embedding dimension
    out = layers.Dense(emb_dims, name=f'{name}/prejection/dense')(out)
    out = layers.Dropout(dropout_rate, name=f'{name}/prejection/dropout')(out)
    return out


def encoder_1d_block(
    inp,
    mlp_dim: int,
    n_heads: int,
    dropout_rate: float = 0.1,
    attention_dropout_rate: float = 0.1,
    name: str = 'encoder_block'
) -> tf.Tensor:
    prefix = name

    def mlp_block(x: tf.Tensor, name: str = 'mlp_block'):
        """feature map의 shape에 변화 없다.

        (B, N+1, emb_dims) -> (B, N+1, emb_dims)
        """
        prefix = name

        out_dim = x.shape[-1]
        x = layers.Dense(mlp_dim, name=f'{prefix}/dense1')(x)
        x = layers.Activation('gelu', name=f'{prefix}/gelu')(x)
        x = layers.Dropout(dropout_rate, name=f'{prefix}/dropout1')(x)
        x = layers.Dense(out_dim, name=f'{prefix}/dense2')(x)
        x = layers.Dropout(dropout_rate, name=f'{prefix}/dropout2')(x)
        return x

    x = layers.LayerNormalization(epsilon=1e-06, name=f'{prefix}/norm')(inp)
    x = multi_head_attention(
        x,
        n_heads,
        attention_dropout_rate,
        name=f'{prefix}/mha'
    )
    x = layers.Dropout(dropout_rate, name=f'{prefix}/dropout')(x)
    x = layers.Add(name=f'{prefix}/add')([x, inp])

    # MLP block
    y = layers.LayerNormalization(epsilon=1e-06, name=f'{prefix}/mlp/norm')(x)
    y = mlp_block(y, name=f'{prefix}/mlp')

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
    prefix = name
    if add_pos_embed:
        x = add_position_embedding(x, name=f'{prefix}/positional')
        x = layers.Dropout(dropout_rate, name=f'{prefix}/dropout')(x)

    # Input Encoder
    for idx in range(n_blocks):
        x = encoder_1d_block(
            x,
            mlp_dim,
            n_heads,
            dropout_rate,
            attention_dropout_rate,
            name=f'{prefix}/block_{idx}'
        )
    return layers.LayerNormalization(epsilon=1e-06,
                                     name=f'{prefix}/norm')(x)


def ViT(
    input_shape: list[int],
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

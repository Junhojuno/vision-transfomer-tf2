import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import (
    Layer, Conv2D, Reshape, Concatenate, 
    Add, Dropout, LayerNormalization,
    Dense, Softmax, MultiHeadAttention
)


class AddPositionEmbs(Layer):
    """input embeddings + position embeddings"""

    def __init__(self, seq_len, emb_dims) -> None:
        super(AddPositionEmbs, self).__init__()
        self.pos_embedding = self.add_weight(
            'pos_embedding',
            (1, seq_len, emb_dims),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )

    def call(self, x):
        return x + tf.cast(self.pos_embedding, x.dtype)


class TokenLayer(Layer):
    """prepend cls token to patches"""
    
    def __init__(self, emb_dims):
        super(TokenLayer, self).__init__()
        self.cls = self.add_weight(
            'cls_token',
            [1, 1, emb_dims],
            initializer='zeros'
        )
        self.concat = Concatenate(axis=1)

    def call(self, x):
        cls_token = tf.cast(self.cls, x.dtype)
        cls_token = cls_token + tf.zeros_like(x[:, 0, :]) # batch 단위로 복사
        return self.concat([cls_token, x])


class GELU(Layer):
    
    def __init__(self, approximate: bool = True):
        super().__init__()
        self.approximate = approximate

    def call(self, inputs):
        return tf.nn.gelu(inputs, approximate=self.approximate)

    def get_config(self):
        config = {"approximate": self.approximate}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape


class MLP(Layer):
    """MLP in Encoder-block"""

    def __init__(self, dims, dropout_rate):
        super().__init__()
        self.dims = dims
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.mlp = Sequential([
            Dense(self.dims),
            GELU(),
            Dropout(self.dropout_rate),
            Dense(input_shape[-1]),
            Dropout(self.dropout_rate)
        ])

    def call(self, inputs):
        return self.mlp(inputs)


class MultiheadSelfAttention(Layer):
    """Multi-head Self-Attention"""

    def __init__(
        self,
        n_heads: int,
        head_dims: int,
        dropout_rate: float,
    ):
        super(MultiheadSelfAttention, self).__init__()
        self._n_heads = n_heads
        self._dropout_rate = dropout_rate
        
        self.parallel_matmul = Dense(3 * self._n_heads * head_dims, use_bias=False)
        self.softmax = Softmax()
        self.dropout = Dropout(dropout_rate)
        self.reshape_1 = Reshape([3*n_heads, head_dims, -1])
        self.scale = self._n_heads ** 0.5

    def build(self, input_shape):
        self.reshape_2 = Reshape([input_shape[1], -1])
        self.proj_layer = Sequential([
            Dense(input_shape[2]),
            Dropout(self._dropout_rate)
        ])
    
    def call(self, inputs):
        """return same shape of input(B, N+1, dims)"""
        Q, K, V = self._cal_qkv(inputs)
        W = Softmax()(tf.matmul(Q, K, transpose_b=True) * self.scale) # (B, heads, N+1, N+1)
        W = self.dropout(W)
        out = tf.matmul(W, V) # (B, heads, N+1, dims)
        out = tf.transpose(out, [0, 2, 1, 3]) # (B, N+1, heads, dims)
        out = self.reshape_2(out)
        return self.proj_layer(out)

    def _cal_qkv(self, x):
        QKV = self.parallel_matmul(x) # (B, N+1, 3*heads*dims)
        QKV = tf.transpose(QKV, [0, 2, 1])
        QKV = self.reshape_1(QKV) # (B, 3*heads, dims, N+1)
        QKV = tf.transpose(QKV, [0, 1, 3, 2]) # (B, 3*heads, N+1, dims)
        return tf.split(QKV, num_or_size_splits=3, axis=1) # (B, heads, N+1, dims)


class EncoderBlock(Layer):
    """1 cycle in Encoder"""

    def __init__(
        self, 
        n_heads: int, 
        head_dims: int,
        att_dropout_rate: float,
        mlp_dropout_rate: float,
        mlp_dims: int,
        norm_epsilon: float = 1e-6
    ):
        super().__init__()
        self._n_heads = n_heads
        self._head_dims = head_dims
        self._att_dropout_rate = att_dropout_rate
        self._mlp_dropout_rate = mlp_dropout_rate
        self._mlp_dims = mlp_dims
        self._norm_epsilon = norm_epsilon
        
        self.layer_norm_1 = LayerNormalization(epsilon=self._norm_epsilon)
        self.msa = MultiheadSelfAttention(self._n_heads, self._head_dims, self._att_dropout_rate)
        self.add = Add()
        self.layer_norm_2 = LayerNormalization(epsilon=self._norm_epsilon)
        self.mlp = MLP(mlp_dims, self._mlp_dropout_rate)
        
    def call(self, inputs):
        x = self.layer_norm_1(inputs)
        x = self.msa(x)
        residual = self.add([inputs, x])
        x = self.layer_norm_2(residual)
        x = self.mlp(x)
        return x + residual


class Encoder(Layer):
    """Transformer Encoder"""

    def __init__(
        self,
        num_layers: int,
        n_heads: int,
        head_dims: int,
        att_dropout_rate: float,
        mlp_dropout_rate: float,
        mlp_dims: int,
        norm_epsilon: float
    ):
        super(Encoder, self).__init__()
        self._num_layers = num_layers
        self._n_heads = n_heads
        self._head_dims = head_dims
        self._att_dropout_rate = att_dropout_rate
        self._mlp_dropout_rate = mlp_dropout_rate
        self._mlp_dims = mlp_dims
        self._norm_epsilon = norm_epsilon

        self._encoder_layers = []
        for _ in range(self._num_layers):
            encoder_layer = EncoderBlock(
                self._n_heads,
                self._head_dims,
                self._att_dropout_rate,
                self._mlp_dropout_rate,
                self._mlp_dims,
                self._norm_epsilon
            )
            self._encoder_layers.append(encoder_layer)
        self._norm = LayerNormalization(epsilon=norm_epsilon)

    def call(self, inputs):
        for encoder_layer in self._encoder_layers:
            x = encoder_layer(inputs)
        x = self._norm(x)
        return x

    def get_config(self):
        config = {
            'num_layers': self._num_layers,
            'n_heads': self._n_heads,
            'head_dims': self._head_dims,
            'attention_dropout_rate': self._attention_dropout_rate,
            'mlp_dropout_rate': self._mlp_dropout_rate,
            'mlp_dims': self._mlp_dims,
            'norm_epsilon': self._norm_epsilon
        }
        base_config = super().get_config()
        return base_config.update(config)


class ViT(Model):
    """implementation of Vision Transformer

    call method가 없는 구조
    """
    def __init__(
        self,
        input_shape,
        patch_size,
        emb_dims,
        kernel_regularizer=None,
        kernel_initializer='lecun_normal',
        use_classifier=True,
        dropout_rate=0.1,
        att_dropout_rate=0.1,
        mlp_dropout_rate=0.1,
        n_encode_layers=1,
        n_heads=12,
        head_dims=64,
        mlp_dims=1,
        norm_epsilon=1e-6,
        num_classes=1000
    ):
        inputs = Input(input_shape)
        x = Conv2D(
            emb_dims,
            patch_size,
            patch_size,
            'valid',
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer
        )(inputs)
        seq_len = (input_shape[0] // patch_size) * (input_shape[1] // patch_size) # n_patches
        x = Reshape([seq_len, emb_dims])(x)
        if use_classifier:
            x = TokenLayer(emb_dims)(x)
        x = AddPositionEmbs(seq_len, emb_dims)(x)
        x = Dropout(dropout_rate)(x)

        x = Encoder(
            n_encode_layers,
            n_heads,
            head_dims,
            att_dropout_rate,
            mlp_dropout_rate,
            mlp_dims,
            norm_epsilon
        )(x)
        x = x[:, 0]
        if use_classifier:
            x = Dense(num_classes)(x)
        super(ViT, self).__init__(inputs=inputs, outputs=x)

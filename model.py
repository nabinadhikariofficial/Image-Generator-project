import tensorflow as tf
from tensorflow.keras import layers
from config import IMG_SIZE, CHANNELS

# --- U-Net Building Blocks ---

class TimestepEmbeddingLayer(layers.Layer):
    """Custom Keras layer for sinusoidal positional embeddings."""
    def __init__(self, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim

    def call(self, timesteps):
        half_dim = self.embedding_dim // 2
        exponent = tf.range(half_dim, dtype=tf.float32) / (half_dim - 1)
        exponent = 1e-4 ** exponent
        embedding = tf.cast(timesteps, tf.float32)[:, None] * exponent[None, :]
        embedding = tf.concat([tf.sin(embedding), tf.cos(embedding)], axis=-1)
        return embedding

class ResnetBlock(layers.Layer):
    """A ResNet block with time embedding injection."""
    def __init__(self, units, groups=8):
        super().__init__()
        self.units = units
        self.conv1 = layers.Conv2D(units, kernel_size=3, padding="same")
        self.norm1 = layers.GroupNormalization(groups=groups)
        self.act1 = layers.Activation("swish")
        self.conv2 = layers.Conv2D(units, kernel_size=3, padding="same")
        self.norm2 = layers.GroupNormalization(groups=groups)
        self.act2 = layers.Activation("swish")
        self.time_mlp = layers.Dense(units)
        self.residual_conv = None

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.residual_conv = layers.Conv2D(self.units, 1, padding="same")

    def call(self, x, time_emb):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        time_emb = self.time_mlp(time_emb)
        x = x + time_emb[:, None, None, :] # Inject time embedding
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        if self.residual_conv:
            residual = self.residual_conv(residual)
        return x + residual

class SelfAttention(layers.Layer):
    """A self-attention block."""
    def __init__(self, units, groups=8):
        super().__init__()
        self.norm = layers.GroupNormalization(groups=groups)
        self.query = layers.Conv2D(units, kernel_size=1)
        self.key = layers.Conv2D(units, kernel_size=1)
        self.value = layers.Conv2D(units, kernel_size=1)
        self.proj = layers.Conv2D(units, kernel_size=1)
        self.scale = units ** -0.5

    def call(self, x):
        b, h, w, c = x.shape
        residual = x
        x = self.norm(x)
        q, k, v = self.query(x), self.key(x), self.value(x)
        q = layers.Reshape((h * w, c))(q)
        k = layers.Reshape((h * w, c))(k)
        v = layers.Reshape((h * w, c))(v)
        attn_scores = tf.linalg.einsum("bic,bjc->bij", q, k) * self.scale
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        attn_output = tf.linalg.einsum("bij,bjc->bic", attn_weights, v)
        attn_output = layers.Reshape((h, w, c))(attn_output)
        return residual + self.proj(attn_output)

# --- U-Net Model Assembly ---

def build_unet(embed_dim=64):
    """Builds the U-Net architecture."""
    image_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS), name="image_input")
    time_input = layers.Input(shape=(), dtype=tf.int64, name="time_input")

    time_emb = TimestepEmbeddingLayer(embed_dim)(time_input)
    time_emb = layers.Dense(embed_dim * 4, activation="swish")(time_emb)
    time_emb = layers.Dense(embed_dim * 4)(time_emb)

    # Encoder Path
    x = layers.Conv2D(embed_dim, kernel_size=3, padding="same")(image_input)
    skips = [x]
    down_dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
    for i, dim in enumerate(down_dims):
        x = ResnetBlock(dim)(x, time_emb)
        x = ResnetBlock(dim)(x, time_emb)
        if i in [1, 2]: x = SelfAttention(dim)(x)
        skips.append(x)
        if i != len(down_dims) - 1:
            x = layers.Conv2D(dim, kernel_size=4, strides=2, padding="same")(x)

    # Bottleneck
    x = ResnetBlock(down_dims[-1])(x, time_emb)
    x = SelfAttention(down_dims[-1])(x)
    x = ResnetBlock(down_dims[-1])(x, time_emb)

    # Decoder Path
    up_dims = reversed(down_dims)
    for i, dim in enumerate(up_dims):
        x = layers.Concatenate()([x, skips.pop()])
        x = ResnetBlock(dim)(x, time_emb)
        x = ResnetBlock(dim)(x, time_emb)
        if i in [1, 2]: x = SelfAttention(dim)(x)
        if i != len(down_dims) - 1:
            next_dim = list(reversed(down_dims))[i + 1]
            x = layers.Conv2DTranspose(next_dim, kernel_size=4, strides=2, padding="same")(x)

    x = layers.Concatenate()([x, skips.pop()])
    x = ResnetBlock(embed_dim)(x, time_emb)
    output_noise = layers.Conv2D(CHANNELS, kernel_size=1, padding="same")(x)
    
    return tf.keras.Model([image_input, time_input], output_noise)
import tensorflow as tf
import numpy as np
from config import TIMESTEPS

def cosine_beta_schedule(timesteps, s=0.008):
    """Generates a cosine variance schedule."""
    steps = timesteps + 1
    x = tf.linspace(0.0, timesteps, steps)
    s_const = tf.constant(s, dtype=tf.float32)
    pi_tensor = tf.constant(np.pi, dtype=tf.float32)
    half_pi = pi_tensor * 0.5
    alphas_cumprod = tf.cos(((x / timesteps) + s_const) / (1.0 + s_const) * half_pi) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return tf.clip_by_value(betas, 0.0001, 0.9999)

# Pre-calculate diffusion constants.
beta = cosine_beta_schedule(TIMESTEPS)
alpha = 1.0 - beta
alpha_bar = tf.math.cumprod(alpha)

def forward_noise(x0, t):
    """
    Applies the forward diffusion process to a batch of images.
    
    This function takes an initial image `x0` and a timestep `t`, and returns
    the noised image `x_t` and the noise that was added.
    """
    noise = tf.random.normal(shape=tf.shape(x0))
    # These tensors are pre-calculated and gathered at timestep t.
    # The dimensions are expanded to match the image batch shape for broadcasting.
    sqrt_alpha_bar_t = tf.gather(tf.sqrt(alpha_bar), t)[:, None, None, None]
    sqrt_one_minus_alpha_bar_t = tf.gather(tf.sqrt(1.0 - alpha_bar), t)[:, None, None, None]

    noisy_image = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
    return noisy_image, noise
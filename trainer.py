import tensorflow as tf
import matplotlib.pyplot as plt
import os
from config import *
from model import build_unet
from diffusion import forward_noise, alpha, alpha_bar, beta

def setup_training():
    """Initializes the model, optimizer, and loss function."""
    unet_model = build_unet()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        decay_steps=STEPS_PER_EPOCH * 10,
        decay_rate=0.95
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = tf.keras.losses.MeanSquaredError()
    return unet_model, optimizer, loss_fn

@tf.function
def train_step(batch_images, unet_model, optimizer, loss_fn):
    """Performs a single gradient descent step."""
    t = tf.random.uniform(shape=[tf.shape(batch_images)[0]], minval=0, maxval=TIMESTEPS, dtype=tf.int32)
    noisy_images, noise = forward_noise(batch_images, t)
    
    with tf.GradientTape() as tape:
        predicted_noise = unet_model([noisy_images, t], training=True)
        loss = loss_fn(noise, predicted_noise)
        
    gradients = tape.gradient(loss, unet_model.trainable_variables)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, CLIP_NORM)
    optimizer.apply_gradients(zip(clipped_gradients, unet_model.trainable_variables))
    return loss

def sample(model, initial_noise=None):
    """
    Generates an image by reversing the diffusion process (denoising).
    Starts from noise and iteratively refines it into an image.
    """
    if initial_noise is None:
        x = tf.random.normal((1, IMG_SIZE, IMG_SIZE, CHANNELS))
    else:
        x = initial_noise

    # Reverse iteration from T-1 down to 1
    for t in reversed(range(1, TIMESTEPS)):
        timesteps = tf.constant([t])
        predicted_noise = model([x, timesteps], training=False)
        
        # Denoising step formula
        alpha_t = tf.gather(alpha, t)
        alpha_bar_t = tf.gather(alpha_bar, t)
        term1 = 1 / tf.sqrt(alpha_t)
        term2 = (1 - alpha_t) / tf.sqrt(1 - alpha_bar_t)
        x = term1 * (x - term2 * predicted_noise)
        
        # Add a small amount of noise back in, except for the last step
        if t > 1:
            beta_t = tf.gather(beta, t)
            noise = tf.random.normal(shape=tf.shape(x))
            x += tf.sqrt(beta_t) * noise
            
    # Denormalize the image from [-1, 1] to [0, 255]
    x = (x + 1) * 127.5
    x = tf.clip_by_value(x, 0, 255)
    return tf.cast(x, tf.uint8)
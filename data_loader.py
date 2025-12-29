import tensorflow as tf
import tensorflow_datasets as tfds
from config import IMG_SIZE

def load_and_preprocess_dataset():
    """
    Loads the Oxford-IIIT Pet dataset, maps the preprocessing function,
    and returns the configured tf.data.Dataset object.
    """
    def preprocess_image(data):
        """Applies resizing and normalization to a single image."""
        image = data['image']
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE), antialias=True)
        # Normalize pixel values from [0, 255] to the [-1, 1] range.
        image = (image / 127.5) - 1.0
        return image

    ds = tfds.load('oxford_iiit_pet', split='train', as_supervised=False)
    train_ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    return train_ds
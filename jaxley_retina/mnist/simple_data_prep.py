import tensorflow as tf
import tensorflow_datasets as tfds

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

def randomize_contrast(image, label, contrast_range):
    """Randomly adjust contrast of image."""
    contrast_factor = tf.random.uniform(
        shape=[], minval=contrast_range[0], maxval=contrast_range[1]
        )
    image = tf.image.adjust_contrast(image, contrast_factor)
    return image, label

def randomize_luminance(image, label, lum_range):
    """Randomly adjust luminance of image."""
    luminance_factor = tf.random.uniform(
        shape=[], minval=lum_range[0], maxval=lum_range[1]
        )
    image = tf.image.adjust_brightness(image, luminance_factor)
    return image, label

def remap_labels(image, label, digits):
    """Convert the digit labels to ordinal class labels."""
    new_label = tf.where(tf.equal(label, digits))
    new_label = tf.squeeze(new_label)
    return image, new_label

def build_train_loader(
        batch_size, 
        digits=[0, 1], 
        contrast_range=(0., 1.), 
        lum_range=(0., 1.), 
        splits=["train", "test"],
        data_dir=None
        ) -> list:
    """Load the mnist data with contrast and luminance distortions."""
    mnist_data = tfds.load(
        'mnist',
        split=splits,
        as_supervised=True,
        shuffle_files=True,
        with_info=False,
        data_dir=data_dir
        )

    data_loaded = []
    for ds in mnist_data:
        if digits != "all":
            def filter_digits(image, label):
                return tf.reduce_any(tf.equal(label, digits))
            ds = ds.filter(filter_digits)
            
            # Remap labels to sequential indices (0, 1, ...) after filtering
            ds = ds.map(
                lambda img, lbl: remap_labels(img, lbl, digits),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        ds = ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(
            lambda ims, labs: randomize_contrast(ims, labs, contrast_range), 
            num_parallel_calls=tf.data.AUTOTUNE
            )
        ds = ds.map(
            lambda ims, labs: randomize_luminance(ims, labs, lum_range),
            num_parallel_calls=tf.data.AUTOTUNE
            )
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        data_loaded.append(iter(tfds.as_numpy(ds)))

    return data_loaded
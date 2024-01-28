
import os
import time
import numpy as np
import cv2
from glob import glob
from matplotlib import pyplot
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

IMG_H = 64
IMG_W = 64
IMG_C = 3  ## Change this to 1 for grayscale.

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img)
    img = tf.cast(img, tf.float32)
    img = (img - 127.5) / 127.5
    return img

def tf_dataset(images_path, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(images_path)
    ds = ds.shuffle(buffer_size=1000).map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def build_generator(latent_dim):
    noise = L.Input((latent_dim), name="noise_input")

    ## 1.
    x = L.Dense(256)(noise)
    x = L.LeakyReLU(0.2)(x)

    ## 2.
    x = L.Dense(1024)(x)
    x = L.LeakyReLU(0.2)(x)

    ## 2.
    x = L.Dense(4096)(x)
    x = L.LeakyReLU(0.2)(x)

    ## 4.
    x = L.Dense(IMG_H * IMG_W * IMG_C)(x)
    x = L.LeakyReLU(0.2)(x)

    ## 5.
    x = L.Reshape((IMG_H, IMG_W, IMG_C))(x)

    fake_output = L.Activation("tanh")(x)

    return Model(noise, fake_output, name="generator")


def build_discriminator():
    inputs = L.Input((IMG_H, IMG_W, IMG_C), name="disc_input")

    ## 1.
    x = L.Flatten()(inputs)

    ## 2.
    x = L.Dense(4096)(x)
    x = L.LeakyReLU(0.2)(x)
    x = L.Dropout(0.3)(x)

    ## 3.
    x = L.Dense(1024)(x)
    x = L.LeakyReLU(0.2)(x)
    x = L.Dropout(0.3)(x)

    ## 4.
    x = L.Dense(256)(x)
    x = L.LeakyReLU(0.2)(x)
    x = L.Dropout(0.3)(x)

    ## 5.
    x = L.Dense(1)(x)

    return Model(inputs, x, name="discriminator")

@tf.function
def train_step(real_images, latent_dim, generator, discriminator, g_opt, d_opt):
    batch_size = tf.shape(real_images)[0]
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)

    ## Discriminator
    noise = tf.random.normal([batch_size, latent_dim])
    for _ in range(2):
        with tf.GradientTape() as dtape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(real_images, training=True)
            fake_output = discriminator(generated_images, training=True)

            d_real_loss = bce_loss(tf.ones_like(real_output), real_output)
            d_fake_loss = bce_loss(tf.zeros_like(fake_output), fake_output)
            d_loss = d_real_loss + d_fake_loss

            d_grad = dtape.gradient(d_loss, discriminator.trainable_variables)
            d_opt.apply_gradients(zip(d_grad, discriminator.trainable_variables))

    with tf.GradientTape() as gtape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)

        g_loss = bce_loss(tf.ones_like(fake_output), fake_output)

        g_grad = gtape.gradient(g_loss, generator.trainable_variables)
        g_opt.apply_gradients(zip(g_grad, generator.trainable_variables))

    return d_loss, g_loss

def save_plot(examples, epoch, n):
    n = int(n)
    examples = (examples + 1) / 2.0
    examples = examples * 255
    file_name = f"samples/generated_plot_epoch-{epoch+1}.png"

    cat_image = None
    for i in range(n):
        start_idx = i*n
        end_idx = (i+1)*n

        image_list = examples[start_idx:end_idx]
        if i == 0:
            cat_image = np.concatenate(image_list, axis=1)
        else:
            tmp = np.concatenate(image_list, axis=1)
            cat_image = np.concatenate([cat_image, tmp], axis=0)

    cv2.imwrite(file_name, cat_image)


if __name__ == "__main__":
    """ Hyperparameters """
    batch_size = 128
    latent_dim = 64
    num_epochs = 1000
    n_samples = 100

    """ Images """
    images_path = glob("/media/nikhil/BACKUP/ML_DATASET/Anime Faces/*.png")
    print(f"Images: {len(images_path)}")

    """ Folders """
    create_dir("samples")
    create_dir("saved_model")

    """ Model """
    g_model = build_generator(latent_dim)
    d_model = build_discriminator()

    g_model.summary()
    d_model.summary()

    """ Training """
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    images_dataset = tf_dataset(images_path, batch_size)
    seed = np.random.normal(size=(n_samples, latent_dim))

    for epoch in range(num_epochs):
        start = time.time()

        d_loss = 0.0
        g_loss = 0.0
        for image_batch in images_dataset:
            d_batch_loss, g_batch_loss = train_step(image_batch, latent_dim, g_model, d_model, g_optimizer, d_optimizer)
            d_loss += d_batch_loss
            g_loss += g_batch_loss

        d_loss = d_loss/len(images_dataset)
        g_loss = g_loss/len(images_dataset)

        g_model.save("saved_model/g_model.h5")
        d_model.save("saved_model/d_model.h5")

        examples = g_model.predict(seed, verbose=0)
        save_plot(examples, epoch, np.sqrt(n_samples))

        time_taken = time.time() - start
        print(f"[{epoch+1:1.0f}/{num_epochs}] {time_taken:2.2f}s - d_loss: {d_loss:1.4f} - g_loss: {g_loss:1.4f}")


    ##


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
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

IMG_H = 128
IMG_W = 128
IMG_C = 3
STRIDE = 32

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img)
    img = tf.image.resize(img, [IMG_H, IMG_W])
    img = tf.cast(img, tf.float32)
    img = (img - 127.5) / 127.5
    return img, label

def tf_dataset(images_path, images_label, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((images_path, images_label))
    ds = ds.shuffle(buffer_size=1000).map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def conv_block(x, num_filters, kernel_size=4, padding="same", strides=1, activation=True):
    x = L.Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
        padding=padding,
        strides=strides,
        use_bias=False
    )(x)

    if activation:
        x = L.LeakyReLU(alpha=0.2)(x)
        x = L.Dropout(0.3)(x)
    return x

def deconv_block(x, num_filters, kernel_size=5, padding="same", strides=4):
    x = L.Conv2DTranspose(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
        use_bias=False
    )(x)
    x = L.BatchNormalization()(x)
    x = L.LeakyReLU(alpha=0.2)(x)
    return x

def build_generator(latent_dim, embed_dim, num_classes):
    noise = L.Input((latent_dim), name="noise_input")
    x = L.Dense(IMG_H//STRIDE * IMG_W//STRIDE * 256)(noise)
    x = L.BatchNormalization()(x)
    x = L.LeakyReLU(0.2)(x)
    x = L.Reshape((IMG_H//STRIDE, IMG_W//STRIDE, 256))(x)

    label = L.Input((1), name="class_label")
    l = L.Embedding(num_classes, embed_dim)(label)
    l = L.Dense(IMG_H//STRIDE * IMG_W//STRIDE * 256)(l)
    l = L.BatchNormalization()(l)
    l = L.LeakyReLU(0.2)(l)
    l = L.Reshape((IMG_H//STRIDE, IMG_W//STRIDE, 256))(l)

    x = L.Concatenate()([x, l])

    x = deconv_block(x, 512, kernel_size=5, strides=2)
    x = deconv_block(x, 256, kernel_size=5, strides=2)
    x = deconv_block(x, 128, kernel_size=5, strides=2)
    x = deconv_block(x, 64, kernel_size=5, strides=2)
    x = deconv_block(x, 32, kernel_size=5, strides=2)

    x = conv_block(
        x,
        num_filters=3,
        kernel_size=5,
        strides=1,
        activation=False
    )

    fake_output = L.Activation("tanh")(x)

    return Model([noise, label], fake_output, name="generator")

def build_discriminator(embed_dim, num_classes):
    image = L.Input((IMG_H, IMG_W, IMG_C), name="disc_input")

    label = L.Input((1), name="class_label")
    l = L.Embedding(num_classes, embed_dim)(label)
    l = L.Dense(IMG_H * IMG_W * 1)(l)
    l = L.BatchNormalization()(l)
    l = L.LeakyReLU(0.2)(l)
    l = L.Reshape((IMG_H, IMG_W, 1))(l)

    x = L.Concatenate()([image, l])

    x = conv_block(x, 32, kernel_size=5, strides=2)
    x = conv_block(x, 64, kernel_size=5, strides=2)
    x = conv_block(x, 128, kernel_size=5, strides=2)
    x = conv_block(x, 256, kernel_size=5, strides=2)
    x = conv_block(x, 512, kernel_size=5, strides=2)

    x = L.Flatten()(x)
    x = L.Dense(1)(x)

    return Model([image, label], x, name="discriminator")

@tf.function
def train_step(real_images, real_labels, latent_dim, num_classes, generator, discriminator, g_opt, d_opt):
    batch_size = tf.shape(real_images)[0]
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)

    ## Discriminator
    noise = tf.random.normal([batch_size, latent_dim])

    for _ in range(3):
        with tf.GradientTape() as dtape:
            generated_images = generator([noise, real_labels], training=True)

            real_output = discriminator([real_images, real_labels], training=True)
            fake_output = discriminator([generated_images, real_labels], training=True)

            d_real_loss = bce_loss(tf.ones_like(real_output), real_output)
            d_fake_loss = bce_loss(tf.zeros_like(fake_output), fake_output)
            d_loss = d_real_loss + d_fake_loss

            d_grad = dtape.gradient(d_loss, discriminator.trainable_variables)
            d_opt.apply_gradients(zip(d_grad, discriminator.trainable_variables))

    with tf.GradientTape() as gtape:
        generated_images = generator([noise, real_labels], training=True)

        fake_output = discriminator([generated_images, real_labels], training=True)

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

    cat_image = cv2.cvtColor(cat_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(file_name, cat_image)


if __name__ == "__main__":
    """ Hyperparameters """
    batch_size = 128
    latent_dim = 128
    embed_dim = 128
    num_classes = 2
    num_epochs = 1000
    n_samples = 100
    path = "/media/nikhil/BACKUP/ML_DATASET/male_vs_female_1/"

    """ Images """
    images_path = glob(f"{path}/*/*.png")

    """ Labels """
    labels_list = os.listdir(f"{path}")
    print(f"Number of labels: {len(labels_list)}")
    print(f"Labels: {labels_list}")

    images_label = []
    for path in images_path:
        name = path.split("/")[-2]
        index = labels_list.index(name)
        images_label.append(index)

    print(f"Images: {len(images_path)} - Labels: {len(images_label)}")

    """ Folders """
    create_dir("samples")
    create_dir("saved_model")

    """ Model """
    g_model = build_generator(latent_dim, embed_dim, num_classes)
    d_model = build_discriminator(embed_dim, num_classes)

    # g_model.load_weights("saved_model/g_model.h5")
    # d_model.load_weights("saved_model/d_model.h5")

    g_model.summary()
    d_model.summary()

    """ Training """
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    images_dataset = tf_dataset(images_path, images_label, batch_size)
    seed = np.random.normal(size=(n_samples, latent_dim))

    seed_class_label = [0]*5 + [1]*5
    seed_label = []
    for item in seed_class_label:
        seed_label += [item] * int(np.sqrt(n_samples))

    seed_label = np.array(seed_label)

    for epoch in range(num_epochs):
        start = time.time()

        d_loss = 0.0
        g_loss = 0.0
        for image_batch, label_batch in images_dataset:
            d_batch_loss, g_batch_loss = train_step(image_batch, label_batch, latent_dim, num_classes, g_model, d_model, g_optimizer, d_optimizer)
            d_loss += d_batch_loss
            g_loss += g_batch_loss

        d_loss = d_loss/len(images_dataset)
        g_loss = g_loss/len(images_dataset)

        g_model.save("saved_model/g_model.keras")
        d_model.save("saved_model/d_model.keras")

        examples = g_model.predict([seed, seed_label], verbose=0)
        save_plot(examples, epoch, np.sqrt(n_samples))

        time_taken = time.time() - start
        print(f"[{epoch+1:1.0f}/{num_epochs}] {time_taken:2.2f}s - d_loss: {d_loss:1.4f} - g_loss: {g_loss:1.4f}")


    ##

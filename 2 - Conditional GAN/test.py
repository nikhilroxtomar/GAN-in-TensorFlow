
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from matplotlib import pyplot

def save_plot(examples, n):
    n = int(n)
    examples = (examples + 1) / 2.0
    examples = examples * 255
    file_name = f"fake_sample.png"

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
    n_samples = 3*3
    latent_dim = 128
    embed_dim = 128
    num_classes = 3

    model = load_model("saved_model/g_model.h5")

    latent_points = np.random.normal(size=(n_samples, latent_dim))

    seed_class_label = [0, 1, 2]
    seed_label = []
    for item in seed_class_label:
        seed_label += [item] * int(np.sqrt(n_samples))

    seed_label = np.array(seed_label)

    examples = model.predict([latent_points, seed_label])
    save_plot(examples, np.sqrt(n_samples))

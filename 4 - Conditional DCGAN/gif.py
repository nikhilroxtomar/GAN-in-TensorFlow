
import numpy as np
import cv2
import imageio
from glob import glob
from tqdm import tqdm

if __name__ == "__main__":
	fps = 10
	with imageio.get_writer("anime.gif", mode="I", duration=(1000 * 1/fps)) as writer:
		for i in tqdm(range(1, 1000)):
			path = f"samples/generated_plot_epoch-{i}.png"
			image = cv2.imread(path, cv2.IMREAD_COLOR)
			writer.append_data(image)

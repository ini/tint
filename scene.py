import numpy as np
import tensorflow as tf


NUM_CATEGORIES = 365
CATEGORIES_PATH = "photos/train/train_categories.txt"

def category(path):
	with open(CATEGORIES_PATH) as file:
		args = file.read().split()
		path_filename = path.split('/')[-1]
		i = args.index(path_filename)
		return int(args[i + 1])

def scene_info(paths):
	try:
		np_scene_info = np.array([np.eye(NUM_CATEGORIES)[category(path)] for path in paths])
	except:
		import resnet.resnet_model
		np_scene_info = np.array([resnet.resnet_model.get_scene_info(path) for path in paths])
	return tf.convert_to_tensor(np_scene_info, dtype=tf.float32)

import torch
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


def visualise_batch_images(images, labels):
    """
    """
    if len(images) < 0:
        return

    if not isinstance(images[0], np.ndarray):
        raise ValueError('Each image passed in `images` argument must be np.ndarray.',
                        'Found %s instead' %type(images[0]))

    if not isinstance(labels[0], (int, float)):
        raise ValueError('Each label passed in `labels` argument must be array-like of int or float.',
                        'Found %s instead' %type(labels[0]))

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 6))
    for idx in np.arange(16):
        ax = fig.add_subplot(2, 16/2, idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        ax.set_title('digit ' + str(labels[idx]), fontsize=16)  

    return fig, ax


def visualise_image_pixels(img, figsize=(12,12)):
    """
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    elif isinstance(img, Image.Image):
        img = np.array(img)
    
    if not isinstance(img, np.ndarray):
        raise ValueError('Image passed in `img` argument must be either np.ndarray or PIL.Image.',
                        'Found %s instead' %type(img))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5

    for x in range(width):
        for y in range(height):
            val = round(img[x][y],2) if img[x][y] !=0 else 0
            ax.annotate(str(val), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y] < thresh else 'black')

    ax.set_title('Kannada Digit in detail: label %d' % labels[1].item());
    return fig, ax


def make_img_tile(imgs, path, epoch, aspect_ratio=1.0, 
			 tile_shape=None, border=1, border_color=0):
    """
    """
	if imgs.ndim != 3 and imgs.ndim != 4:
		raise ValueError('imgs has wrong number of dimensions.')
	
	n_imgs = imgs.shape[0]
	tile_shape = None

	# Grid shape
	img_shape = np.array(imgs.shape[1:3])
	
	if tile_shape is None:
		img_aspect_ratio = img_shape[1] / float(img_shape[0])
		aspect_ratio *= img_aspect_ratio
		tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
		tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
		grid_shape = np.array((tile_height, tile_width))
	else:
		assert len(tile_shape) == 2
		grid_shape = np.array(tile_shape)

	# Tile image shape
	tile_img_shape = np.array(imgs.shape[1:])
	tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

	# Assemble tile image
	tile_img = np.empty(tile_img_shape)
	tile_img[:] = border_color

	for i in range(grid_shape[0]):
		for j in range(grid_shape[1]):
			img_idx = j + i*grid_shape[1]

			# No more images - stop filling out the grid.
            if img_idx >= n_imgs:
				break

			# Convert 1~1 to 0~1
			img = (imgs[img_idx] + 1)/2.0# * 255.0

			yoff = (img_shape[0] + border) * i
			xoff = (img_shape[1] + border) * j
			tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img 

	img_tile = Image.fromarray(np.uint8(tile_img * 255) , 'L')
	
    if path is not None:
        path_name = path + "/iteration_%03d"%(epoch)+".jpg"
        img_tile.save(path_name)

    return img_tile

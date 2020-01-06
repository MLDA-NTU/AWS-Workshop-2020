import numpy as npy
from PIL import Image

import torch
import torch.nn.functional as F


def target_to_onehot(idx, num_classes=10):
	assert torch.max(idx).item() < num_classes

	onehot = torch.zeros(idx.size(0), num_classes)
	onehot = onehot.scatter(1, idx.long(), 1)

	return onehot


def calculate_vae_loss(x, x_reconstructed, z_mean, z_logvar):
	# reconstruction loss
	recloss = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')
	# kl divergence loss
	divloss = - 0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

	return recloss, divloss


def img_tile(imgs, path, epoch, aspect_ratio=1.0, 
			 tile_shape=None, border=1, border_color=0):
	if imgs.ndim != 3 and imgs.ndim != 4:
		raise ValueError('imgs has wrong number of dimensions.')
	
	n_imgs = imgs.shape[0]
	tile_shape = None

	# Grid shape
	img_shape = npy.array(imgs.shape[1:3])
	
	if tile_shape is None:
		img_aspect_ratio = img_shape[1] / float(img_shape[0])
		aspect_ratio *= img_aspect_ratio
		tile_height = int(npy.ceil(npy.sqrt(n_imgs * aspect_ratio)))
		tile_width = int(npy.ceil(npy.sqrt(n_imgs / aspect_ratio)))
		grid_shape = npy.array((tile_height, tile_width))
	else:
		assert len(tile_shape) == 2
		grid_shape = npy.array(tile_shape)

	# Tile image shape
	tile_img_shape = npy.array(imgs.shape[1:])
	tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

	# Assemble tile image
	tile_img = npy.empty(tile_img_shape)
	tile_img[:] = border_color

	for i in range(grid_shape[0]):
		for j in range(grid_shape[1]):
			img_idx = j + i*grid_shape[1]
			if img_idx >= n_imgs:
				# No more images - stop filling out the grid.
				break

			#-1~1 to 0~1
			img = (imgs[img_idx] + 1)/2.0# * 255.0

			yoff = (img_shape[0] + border) * i
			xoff = (img_shape[1] + border) * j
			tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img 

	path_name = path + "/iteration_%03d"%(epoch)+".jpg"

	img = Image.fromarray(npy.uint8(tile_img * 255) , 'L')
	img.save(path_name)

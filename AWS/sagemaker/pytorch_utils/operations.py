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

	return recloss + divloss

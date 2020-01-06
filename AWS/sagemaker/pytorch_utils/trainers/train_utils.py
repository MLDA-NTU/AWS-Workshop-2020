import numpy as npy
from PIL import Image

import torch
import torch.distributed as dist
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    Arguments
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def distributed_average_gradients(model):
    """
    """
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

import os, sys
import logging

import torch
from pytorch_utils.trainers.train_utils import (
    accuracy, distributed_average_gradients
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))


def test_helper(test_loader, model, criterion, epoch,
                device='cpu'):
    """
    """
    # set to validation mode
    model.eval()

    test_loss = 0.0  # record testing loss
    test_top1 = 0.0
    test_top5 = 0.0
    for batch_idx, (data, target) in enumerate(test_loader, start=1):
        # convert tensor for current runtime device
        data, target = data.to(device), target.to(device)

        # generate image x
        out = model(data)

        # calculate loss and optimise network params
        loss = criterion(out, target)

        # calculate testing accuracy for top1 and top5
        top1, top5 = accuracy(out, target, topk=(1,5))

        # update test loss
        test_top1 += top1.item()
        test_top5 += top5.item()
        test_loss += loss.item()

    # display validation/testing result
    test_loss /= len(test_loader.dataset)  # average loss over all images
    test_top1 /= len(test_loader)
    test_top5 /= len(test_loader)

    logger.info('Test Summary Epoch: {:03d} | '
          'Average Top1 Acc: {:.2f}  | Average Top5 Acc: {:.2f} | Loss: {:.4f}'
          .format(epoch, test_top1, test_top5, test_loss))

    return test_loss, test_top1, test_top5


def train_helper(train_loader, model, optimizer, criterion, epoch,
                distributed=False, device='cpu', log_interval=25):
    """
    """
    # set to training mode
    model.train()

    # training result to record
    train_loss = 0.0
    train_top1 = 0.0
    train_top5 = 0.0

    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        # convert tensor for current runtime device
        data, target = data.to(device), target.to(device)

        # reset optimiser gradient to zero
        optimizer.zero_grad()

        # feed forward
        out = model(data)

        # calculate loss and optimise network params
        loss = criterion(out, target)
        loss.backward()

        if distributed and not torch.cuda.is_available():
            # average gradients manually when using multi-machine with cpu device
            distributed_average_gradients(model)

        # optimize weight to account for loss/gradient
        optimizer.step()

        # calculate training accuracy for top1 and top5
        top1, top5 = accuracy(out, target, topk=(1,5))

        # update result records
        train_top1 += top1.item()
        train_top5 += top5.item()
        train_loss += loss.item()

        # logging loss output to stdout
        if batch_idx % log_interval == 0:
            logger.info('Train Epoch: {:03d} [{:05d}/{:05d} ({:2.0f}%)] | '
                  'Top1 Acc: {:4.1f} \t| Top5 Acc: {:4.1f} \t| Loss: {:.4f}'
                  .format(epoch, batch_idx * len(data), len(train_loader.sampler),
                      100 * batch_idx / len(train_loader),
                      top1, top5, loss.item()))

    # display training result
    train_loss /= len(train_loader.dataset)
    train_top1 /= len(train_loader) # average loss over mini-batches
    train_top5 /= len(train_loader) # average loss over mini-batches

    logger.info('Training Summary Epoch: {:03d} | '
          'Average Top1 Acc: {:.2f}  | Average Top5 Acc: {:.2f} | Loss: {:.4f}'
          .format(epoch, train_top1, train_top5, train_loss))

    return train_loss, train_top1, train_top5

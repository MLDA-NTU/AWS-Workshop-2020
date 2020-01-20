import os, sys
import json, logging, argparse

import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import datasets, transforms

from pytorch_utils.models import CNN, CVAE
from pytorch_utils.datasets import KannadaDataset
from pytorch_utils.trainers.train_utils import distributed_average_gradients
from pytorch_utils.trainers.classifier_trainers import train_helper, test_helper


IMGSIZE = 28

# Seeding torch.Random
torch.manual_seed(42)
if torch.cuda.is_available:
    torch.cuda.manual_seed(42)

# Prepare logging system
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))


# -------------------------
# DATA HELPER
# -------------------------

def get_data_loaders(batch_size, data_dir, distributed=False, **kwargs):
    logger.info('- Preparing data loaders ...')

    # Load Image data from csv. Idk why but that's their format
    train_images = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None).values
    val_images = pd.read_csv(os.path.join(data_dir, 'validation.csv'), header=None).values
    test_images = pd.read_csv(os.path.join(data_dir, 'test.csv'), header=None).values

    # Load Labels, must be a list of indices (int)
    train_labels = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'), header=None).values.squeeze().tolist()
    val_labels = pd.read_csv(os.path.join(data_dir, 'validation_labels.csv'), header=None).values.squeeze().tolist()
    test_labels = pd.read_csv(os.path.join(data_dir, 'test_labels.csv'), header=None).values.squeeze().tolist()

    # Training data augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(IMGSIZE),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
        transforms.ToTensor()  # convert to torch.Tensor, then divide pixels by 255
    ])

    # Test/Validation data convert to compatible datatype
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()  # convert to torch.Tensor, then divide pixels by 255
    ])

    # Construct dataset
    train_dataset = KannadaDataset(train_images, train_labels, transforms=train_transform)
    val_dataset = KannadaDataset(val_images, val_labels, transforms=test_transform)
    test_dataset = KannadaDataset(test_images, test_labels, transforms=test_transform)

    # training set loader
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), 
                            sampler=train_sampler, **kwargs)

    # validation set loader
    val_loader = DataLoader(test_dataset, batch_size, shuffle=False, **kwargs)

    # testing set loader
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, **kwargs)

    logger.debug("- Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    logger.debug("- Processes {}/{} ({:.0f}%) of test data".format(
        len(test_loader.sampler), len(test_loader.dataset),
        100. * len(test_loader.sampler) / len(test_loader.dataset)
    ))

    return train_loader, val_loader, test_loader


# -------------------------
# MAIN
# -------------------------

def train_session(args):
    logger.info('Begin training session.')
    logger.info('Status:')
    logger.info('-' * 50)

    distributed = len(args.hosts) > 1 and args.backend is not None
    use_cuda = (args.num_gpus > 0) and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    logger.info('- Distributed training - %s' % distributed)
    logger.debug('- Using Device: %s' % device)
    logger.debug('- Number of gpus available - %s (use_cuda: %s)' % (args.num_gpus, use_cuda))    

    # Setup for distributed computing
    if distributed:
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)

        logger.info('- Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))

    # Prepare data loaders
    train_loader, val_loader, test_loader = get_data_loaders(args.batch_size, args.data_dir, distributed, **kwargs)

    # Initialise network module
    model = CNN(num_classes=args.num_classes).to(device)

    if distributed and use_cuda:
        # multi-threaded on distributed server
        model = torch.nn.parallel.DistributedDataParallel(model).to(device)
    else:
        # multi-threaded on single server
        model = torch.nn.parallel.DataParallel(model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)

    # -------------------------
    # TRAINING SESSION
    # -------------------------

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        # run model on trainings set
        train_helper(train_loader, model, optimizer, criterion, epoch,
                      device=device, distributed=distributed, log_interval=100)
        # run model on validation
        _, test_top1, _ = test_helper(val_loader, model, criterion, epoch, device=device)

        # save trained model to a checkpoint
        if (test_top1 > best_acc):
            best_acc = test_top1
            save_model(model, args.model_dir)

    # finally test model on test set
    logger.info('*'* 50 + '\n' + 'Final, testing on Test set:')
    test_helper(test_loader, model, criterion, epoch, device=device)


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(CVAE(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        num_classes=args.num_classes).to(device))

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))

    return model.to(device)


def save_model(model, model_dir):
    logger.info("- Saving the model.")
    path = os.path.join(model_dir, 'best_kannada_model.pth')
    # recommended way to save model from 
    # http://pytorch.org/docs/master/notes/serialization.html
    if hasattr(model, 'module'):  # check if nn.DataParallel
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)
    logger.info("- Saved model to file: %s" % path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr-step', type=int, default=5, metavar='LR_STEP',
                        help='number of epoch before learning rate is decayed (default: 5)')
    parser.add_argument('--lr-decay', type=float, default=0.1, metavar='LR_DECAY',
                        help='learning rate decay multiplier (default: 0.1)')
    parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                        help='Number of classes in dataset (default: 10)')
    parser.add_argument('--input-dim', type=int, default=784, metavar='N',
                        help='Input dimension of flattened image (default: 784)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--plot-dir', type=str, default=None)
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    args = parser.parse_args()

    # Check args
    # validate_args(args)

    # Begin Training
    train_session(args)

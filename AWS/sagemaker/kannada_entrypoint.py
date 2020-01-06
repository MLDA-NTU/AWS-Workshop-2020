import os, sys
import json, logging, argparse

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import datasets, transforms

from pytorch_utils.models import CNN, CVAE
from pytorch_utils.datasets import KannadaDataSet
from pytorch_utils.trainers.train_utils import distributed_average_gradients
from pytorch_utils.trainers.classifier_trainers import train_helper, test_helper

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
    logger.info('Preparing data loaders')

    # Load Data
    train_images = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_images = pd.read(os.path.join(data_dir, 'validation.csv'))
    test_images = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    # Load Labels
    train_labels = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))
    val_labels = pd.read_csv(os.path.join(data_dir, 'validation_labels.csv'))
    test_labels = pd.read_csv(os.path.join(data_dir, 'test_labels.csv'))

    # Reset Index
    train_images.reset_index(drop=True, inplace=True)
    train_labels.reset_index(drop=True, inplace=True)

    val_images.reset_index(drop=True, inplace=True)
    val_labels.reset_index(drop=True, inplace=True)

    test_images.reset_index(drop=True, inplace=True)
    test_labels.reset_index(drop=True, inplace=True)

    train_transform = transforms.Compose([
        transforms.RandomAffine(10),
        transforms.ToTensor()
    ])

    # Construct dataset
    train_dataset = KannadaDataSet(train_images, train_labels, transforms=train_transform)
    val_dataset = KannadaDataSet(val_images, val_labels, transforms=transforms.ToTensor())
    test_dataset = KannadaDataSet(test_images, test_labels, transforms=transforms.ToTensor())

    # training set loader
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), 
                            sampler=train_sampler, **kwargs)

    # validation set loader
    val_loader = DataLoader(test_dataset, batch_size, shuffle=False, **kwargs)

    # testing set loader
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, **kwargs)

    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    logger.debug("Processes {}/{} ({:.0f}%) of test data".format(
        len(test_loader.sampler), len(test_loader.dataset),
        100. * len(test_loader.sampler) / len(test_loader.dataset)
    ))

    return train_loader, val_loader, test_loader


# -------------------------
# MAIN
# -------------------------

def train_session(args):
    logger.info('Begin training session')

    distributed = len(args.hosts) > 1 and args.backend is not None
    use_cuda = args.num_gpus > 0
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    logger.info('Distributed training - %s' % distributed)
    logger.debug('Number of gpus available - %s (%s)' % (args.num_gpus, torch.cuda.is_available))    

    # Setup for distributed computing
    if distributed:
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)

        logger.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))

    # Prepare data loaders
    train_loader, val_loader, test_loader = get_data_loaders(args.batch_size, args.data_dir, distributed, **kwargs)

    # Initialise network module
    model = CNN(num_classes=args.num_classes)

    if distributed and use_cuda:
        # multi-threaded on distributed server
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # multi-threaded on single server
        model = torch.nn.parallel.DataParallel(model)

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
        if test_top1.item() > best_acc:
            best_acc = test_top1.item()
            save_model(model, args.model_dir)

    # finally test model on test set
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
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'best_kannada_model.pth')
    # recommended way to save model from 
    # http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)

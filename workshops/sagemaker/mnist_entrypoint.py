import os, sys
import json, logging, argparse

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import datasets, transforms

from torch_mnist.model import CVAE
from torch_mnist.utils_vae import (
    target_to_onehot, calculate_vae_loss,
    img_tile
)

torch.manual_seed(42)
if torch.cuda.is_available:
    torch.cuda.manual_seed(42)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))


# -------------------------
# TRAINING HELPER
# -------------------------

def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def _train_helper(train_loader, vae_model, optimizer, epoch, 
                device='cpu', distributed=True, log_interval=25):
    # set to training mode
    vae_model.train()

    train_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader, start=1):
        # convert target to one-hot encoding
        y = target_to_onehot(y.view(-1, 1))

        # convert tensor for current runtime device
        x, y = x.view(-1, 28 * 28).to(device), y.to(device)

        # reset optimiser gradient to zero
        optimizer.zero_grad()

        # generate image x
        x_reconstructed, z_mu, z_logvar = vae_model(x, y)

        # calculate loss and optimise network params
        recloss, divloss = calculate_vae_loss(x, x_reconstructed, z_mu, z_logvar)
        loss = recloss + divloss
        loss.backward()

        if distributed and not torch.cuda.is_available:
            # average gradients manually for multi-machine with cpu device
            _average_gradients(vae_model)
        
        optimizer.step()

        # update records
        train_loss += loss.item()

        # logging loss output to stdout
        if batch_idx % log_interval == 0:
            logger.info('Train Epoch: {:03d} [{:05d}/{:05d} ({:2.0f}%)] Loss: {:.4f} {:.4f}'.format(
                epoch, batch_idx * len(x), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), recloss/ x.size(0),
                divloss / x.size(0)))

    train_loss /= len(train_loader.dataset)


def _test_helper(test_loader, vae_model, epoch, device='cpu', plot_path=None):
    # set to training mode
    vae_model.eval()

    test_recloss = 0.0
    test_divloss = 0.0
    for batch_idx, (x, y) in enumerate(test_loader, start=1):
        # convert target to one-hot encoding
        y = target_to_onehot(y.view(-1, 1))

        # convert tensor for current runtime device
        x, y = x.view(-1, 28 * 28).to(device), y.to(device)

        # generate image x
        x_reconstructed, z_mu, z_logvar = vae_model(x, y)

        # calculate loss and optimise network params
        recloss, divloss = calculate_vae_loss(x, x_reconstructed, z_mu, z_logvar)

        # update records
        test_recloss += recloss.item()
        test_divloss += divloss.item()

        if batch_idx == 1 and plot_path is not None:
            imgs = x_reconstructed[:64].view(-1, 28, 28)
            imgs = imgs.cpu().detach().numpy()
            img_tile(imgs, plot_path, epoch, aspect_ratio=1.0, tile_shape=None, border=1, border_color=0)

    test_recloss /= len(test_loader.dataset)
    test_divloss /= len(test_loader.dataset)

    logger.info('Test set: Average Reconstruction loss: {:.4f}' 
                ' | Average Divergence Loss: {:.4f}\n'.format(test_recloss, test_divloss))


# -------------------------
# DATA HELPER
# -------------------------

def _get_train_loader(batch_size, data_dir, distributed=False, **kwargs):
    logger.info('Preparing training dataloader')

    transform = transforms.Compose([
        transforms.RandomAffine(10),
        transforms.ToTensor()
    ])
    dataset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    train_sampler = DistributedSampler(dataset) if distributed else None
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=(train_sampler is None), 
                              sampler=train_sampler, **kwargs)

    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    return train_loader


def _get_test_loader(batch_size, data_dir, device='cpu', **kwargs):
    logger.info('Preparing testing dataloader')

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(data_dir, train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)

    logger.debug("Processes {}/{} ({:.0f}%) of test data".format(
        len(test_loader.sampler), len(test_loader.dataset),
        100. * len(test_loader.sampler) / len(test_loader.dataset)
    ))

    return test_loader


# -------------------------
# MAIN
# -------------------------

def train_session(args):
    distributed = len(args.hosts) > 1 and args.backend is not None
    use_cuda = args.num_gpus > 0
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    logger.info('Distributed training - %s' % distributed)
    logger.debug('Number of gpus available - %s (%s)' % (args.num_gpus, torch.cuda.is_available))
    logger.info('Begin training session')

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
    train_loader = _get_train_loader(args.batch_size, args.data_dir, distributed, **kwargs)
    test_loader = _get_test_loader(args.test_batch_size, args.data_dir, **kwargs)

    # Initialise network module
    model = CVAE(input_dim=(28 * 28), latent_dim=args.latent_dim, 
                 num_classes=args.num_classes).to(device)
    
    if distributed and use_cuda:
        # multi-threaded on distributed server
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # multi-threaded on single server
        model = torch.nn.parallel.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, 
                                          gamma=args.lr_decay)

    # -------------------------
    # TRAINING SESSION
    # -------------------------

    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        # training
        _train_helper(train_loader, model, optimizer, epoch, 
                      device=device, distributed=distributed)
        # validation
        _test_helper(test_loader, model, epoch, device=device)

    # save trained model to a checkpoint 
    save_model(model, args.model_dir)


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
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


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
    parser.add_argument('--latent-dim', type=int, default=75, metavar='N',
                        help='Latent z dimension in the bottleneck layer (default: 75)')
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

    # Begin Training
    train_session(parser.parse_args())

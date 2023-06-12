from model import model
from torchvision.datasets import CIFAR10, LSUN, CelebA
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import diffusion
import argparse
from PIL import Image
from utils import *
import torchvision.transforms as trans
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
import sys
import os

# you can download 50k images from here
# https://drive.google.com/file/d/1IMDjxG2ELX9E8fJSeusPSFOS5nh66GHX/view?usp=sharing
LSUN_DIR = '' #path to directory with images
EPOCH = 500
BATCH_SIZE = 128
# Hyperparameters
SIZE = 128
LR = 3e-4

def load_dataset(dataset_name):
    transform = trans.Compose([trans.Resize((SIZE, SIZE)),
                               trans.ToTensor(),
                               Obscure(SIZE)])
    if dataset_name == 'cifar10':
        _cifar10 = CIFAR10(root='data', train=True, transform=transform, download=True)
        return _cifar10
    elif dataset_name == 'lsun':
        _lsun  = torchvision.datasets.ImageFolder(root=LSUN_DIR, transform=transform)
        return _lsun
    else:
        sys.exit("Dataset not implemented")


def _mp_fn(index, lr, dataset):
    torch.set_default_tensor_type('torch.FloatTensor')
    dev = xm.xla_device()
    print(dev)
    model = WRAPPED_MODEL.to(dev)
    loaded_dataset=SERIAL_EXEC.run(lambda: load_dataset(dataset))
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        loaded_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    _trainDataLoader = DataLoader(
        loaded_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=8,
        drop_last=True)

    
    _diffusion = diffusion.Diffusion(device=dev)
    _diffusion.train_xla(model, EPOCH, _trainDataLoader, lr)

def main_gpu(dev, n, dataset, modelpath):
    diff = diffusion.Diffusion(device=dev)

    if n == 0:
        m = model(6, 3)
        if os.path.exists(modelpath):
            m.load_state_dict(torch.load(modelpath))

        datas = load_dataset(dataset)
        _train = DataLoader(datas, batch_size=128, shuffle=True)
        diff.train_gpu(m, 500, _train, lr)

    if n != 0:
        m = model(6, 3)
        m.load_state_dict(torch.load(modelpath))
        m.eval()
        diff.gen(m, n, dataset=load_dataset(dataset))


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-mp", "--modelpath", type=str, required=True,
                    help="path to output trained model")
    ap.add_argument("-n", "--number", type=int, required=True,
                    help="number of pictures to generate. If n is 0 then model will be train")
    ap.add_argument("-d", "--dataset", type=str, default='cifar10',
                    help="Dataset to load.")
    ap.add_argument('-m', '--mode', type=str, default='gpu', help='On which device model will train')

    args = vars(ap.parse_args())

    

    SERIAL_EXEC = xmp.MpSerialExecutor()

    WRAPPED_MODEL=xmp.MpModelWrapper(model(6, 3))
    
    lr=LR*xm.xrt_world_size()

    if args['mode'] == 'tpu':
        print("TRENING")
        xmp.spawn(_mp_fn,
                  args=(lr,args['dataset']),
                  nprocs=8,
                  start_method='fork')
    else:
        print(f'TRENING ON: {args["mode"]}')
        main_gpu(args['mode'], args['number'], args['dataset'], args['modelpath'])


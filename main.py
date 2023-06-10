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



def load_dataset():
    transform = trans.Compose([trans.Resize((SIZE, SIZE)),
                               trans.ToTensor(),
                               Obscure(SIZE)])

    _cifar10 = CIFAR10(root='data', train=True, transform=transform, download=True)

    # Trzeba dopisać kolejne datasety

    return _cifar10

def _mp_fn(index, lr):
    
    EPOCH = 500
    BATCH_SIZE = 128

    torch.set_default_tensor_type('torch.FloatTensor')
    dev = xm.xla_device()
    print(dev)
    model=WRAPPED_MODEL.to(dev)
    _cifar10=SERIAL_EXEC.run(load_dataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        _cifar10,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    _trainDataLoader = DataLoader(
        _cifar10, 
        batch_size=BATCH_SIZE,
        sampler=train_sampler, 
        num_workers=8,
        drop_last=True)
    
    _diffusion = diffusion.Diffusion(device=dev)
    _diffusion.train_xla(model, EPOCH, _trainDataLoader, lr)


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, required=True,
                    help="path to output trained model")
    ap.add_argument("-n", "--number", type=int, required=True,
                    help="number of pictures to generate. If n is 0 then model will be train")

    args = vars(ap.parse_args())

    # Hyperparameters
    SIZE = 128
    LR = 3e-4
    

    SERIAL_EXEC = xmp.MpSerialExecutor()

    WRAPPED_MODEL=xmp.MpModelWrapper(model(6, 3))
    
    lr=LR*xm.xrt_world_size()
    
    if args['number'] == 0:
        print("TRENING")
        xmp.spawn(_mp_fn, 
                  args=(lr,),
                  nprocs=8,
                  start_method='fork')
    else:
        #sampling behaves strangely on tpu's so we leave it for local tests
        print('GEN')

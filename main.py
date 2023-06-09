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
from torch import optim

import sys
import os

# you can download 50k images from here
# https://drive.google.com/file/d/1IMDjxG2ELX9E8fJSeusPSFOS5nh66GHX/view?usp=sharing
LSUN_DIR = './lsun' #path to directory with images, this directory should exists after running setup.py
EPOCH = 50
BATCH_SIZE = 8

# Hyperparameters
SIZE = 128
LR = 3e-4

def load_dataset(dataset_name,transform_train=True):
    transform = trans.Compose([trans.Resize((SIZE, SIZE)),
                               trans.ToTensor(),
                               Obscure(SIZE,train=transform_train)])
    if dataset_name == 'cifar10':
        _cifar10 = CIFAR10(root='data', train=True, transform=transform, download=True)
        return _cifar10
    elif dataset_name == 'lsun':
        _lsun  = torchvision.datasets.ImageFolder(root=LSUN_DIR, transform=transform)
        return _lsun
    else:
        sys.exit("Dataset not implemented")


def _mp_fn(index, lr, dataset, optimizer):
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

    
    _diffusion = diffusion.Diffusion(optimizer=optimizer,device=dev,schedule='cosine')
    _diffusion.train_xla(model, EPOCH, _trainDataLoader, lr)

def main_gpu(dev, n, dataset, modelpath,optimizer):
    m = model(6, 3).to(dev)
    optimizer = optim.AdamW(model.parameters(),LR)
    diff = diffusion.Diffusion(optimizer=optimizer,device=dev)

    if n == 0:
        best_model = find_best_model(args['modelpath'],'ckpt', 'pt')
        if best_model:
            checkpoint = torch.load(best_model)
            m.load_state_dict(checkpoint['model_state_dict'])
        datas = load_dataset(dataset)
        _train = DataLoader(datas, batch_size=BATCH_SIZE, shuffle=True)
        diff.train_gpu(m, 500, _train, LR)

    if n != 0:
        m.load_state_dict(torch.load(modelpath)['model_state_dict'])
        m.eval()
        original,broken,restored=diff.gen(m, n, dataset=load_dataset(dataset,transform_train=False))
        save_input(original,os.path.join("pictures", "original.jpg"))
        save_input(broken,os.path.join("pictures", "broken.jpg"))
        save_images(restored,os.path.join("pictures", "restored.jpg"))


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-mp", "--modelpath", type=str, required=True,
                    help="path to output trained model")
    ap.add_argument("-n", "--number", type=int, required=True,
                    help="number of pictures to generate. If n is 0 then model will be train")
    ap.add_argument("-d", "--dataset", type=str, default='cifar10',
                    help="Dataset to load.")
    ap.add_argument('-m', '--mode', type=str, default='cuda', help='On which device model will train')

    args = vars(ap.parse_args())

    



    if args['mode'] == 'tpu':

        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla.utils.utils as xu

        SERIAL_EXEC = xmp.MpSerialExecutor()

     
        model = model(6,3)
        lr=LR*xm.xrt_world_size()

        optimizer = optim.AdamW(model.parameters(),lr)


        best_model = find_best_model(args['modelpath'],'ckpt', 'pt')
        if best_model:
            checkpoint = torch.load(best_model)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("model loaded")
        

        WRAPPED_MODEL=xmp.MpModelWrapper(model)
        

        print("TRENING")
        xmp.spawn(_mp_fn,
                  args=(lr,args['dataset'], optimizer),
                  nprocs=8,
                  start_method='fork')
    else:
        print(f'TRENING ON: {args["mode"]}')
        main_gpu(args['mode'], args['number'], args['dataset'], args['modelpath'])


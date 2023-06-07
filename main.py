import diffusion
from model import model
import argparse
from torchvision.datasets import CIFAR10, LSUN, CelebA
from torch.utils.data import DataLoader
import torchvision.transforms as trans
from torchvision.utils import make_grid
from PIL import Image
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu

def save_images(images, path):
    grid = make_grid(images)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    print(path)
    im.save(path)

def load_dataset():
    _cifar10 = CIFAR10(root='data', train=True, transform=transform, download=True)
    return _cifar10

def _mp_fn(index, epochs, data,lr):
    torch.set_default_tensor_type('torch.FloatTensor')
    device = xm.xla_device()
    model=WRAPPED_MODEL.to(device)
    #pytanie co z tym czy to
    #gdyby diffusion nie dostawało modelu ani device można by zainicjować wcześniej, a tu tywoływać już funkcję z ospowiednimi
    _diffusion = diffusion.Diffusion(model,device=device)
    
    _diffusion.train_xla(epochs, device, data, lr)


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, required=True,
                    help="path to output trained model")
    ap.add_argument("-n", "--number", type=int, required=True,
                    help="number of pictures to generate. If n is 0 then model will be train")

    args = vars(ap.parse_args())

    # Hyperparameters
    
    SIZE = 128
    EPOCH = 500
    BATCH_SIZE = 12
    LR = 3e-4

    SERIAL_EXEC = xmp.MpSerialExecutor()

    WRAPPED_MODEL=xmp.MpModelWrapper(model(3, 3))
    
    lr=LR*xm.xrt_world_size()

    # Loading datasets
    transform = trans.Compose([trans.Resize((SIZE, SIZE)),
                               trans.ToTensor()])
    
    _cifar10=SERIAL_EXEC.run(load_dataset)
    #_lsun = LSUN(root='data', transform=transform)
    #_celeba = CelebA(root='data', split='train', transform=transform, download=True)

    # Add when lsun and celba are added/download
    #_train_sets = torch.utils.data.ConcatDataset([_cifar10, _lsun, _celeba])
    #_train_loader = DataLoader(dataset=_train_sets, BATCH_SIZE, shuffle=True)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    # _cifar10,
    # num_replicas=xm.xrt_world_size(),
    # rank=xm.get_ordinal(),
    # shuffle=True)
    _trainDataLoader = DataLoader(_cifar10, BATCH_SIZE, shuffle=True)




    

    if args['number'] == 0:
        print("TRENING")
        xmp.spawn(_mp_fn, args=(),
          start_method='fork')
        _diffusion.train_xla( EPOCH, DEVICE, _trainDataLoader, lr)
    else:
        #trzeba będzie dodać znowu wczytywanie modelu w tym wypadku
        print('GEN')
        pictures = _diffusion.gen(WRAPPED_MODEL, args['number'])
        save_images(pictures, f'output\{args["number"]}.jpg')
        #output\GeneratedPics\
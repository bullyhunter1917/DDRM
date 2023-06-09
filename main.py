import diffusion
from model import model
import argparse
from torchvision.datasets import CIFAR10, LSUN, CelebA
from torch.utils.data import DataLoader
import torchvision.transforms as trans
from torchvision.utils import make_grid
from PIL import Image
from utils import Obscure


def save_images(images, path):
    grid = make_grid(images)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    print(path)
    im.save(path)

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, required=True,
                    help="path to output trained model")
    ap.add_argument("-n", "--number", type=int, required=True,
                    help="number of pictures to generate. If n is 0 then model will be train")

    args = vars(ap.parse_args())

    # Hyperparameters
    DEVICE = 'cuda'
    SIZE = 128
    EPOCH = 500
    BATCH_SIZE = 12
    LR = 3e-4

    # Loading datasets
    transform = trans.Compose([trans.Resize((SIZE, SIZE)),
                               trans.ToTensor(),
                               Obscure(SIZE)])

    _cifar10 = CIFAR10(root='data', train=True, transform=transform, download=True)
    #_lsun = LSUN(root='data', transform=transform)
    #_celeba = CelebA(root='data', split='train', transform=transform, download=True)

    # Add when lsun and celba are added/download
    #_train_sets = torch.utils.data.ConcatDataset([_cifar10, _lsun, _celeba])
    #_train_loader = DataLoader(dataset=_train_sets, BATCH_SIZE, shuffle=True)

    _trainDataLoader = DataLoader(_cifar10, BATCH_SIZE, shuffle=True)

    _model = model(3, 3).to(DEVICE)
    _diffusion = diffusion.Diffusion()

    if args['number'] == 0:
        print("TRENING")
        _diffusion.train(_model, EPOCH, DEVICE, _trainDataLoader, LR)
    else:
        print('GEN')
        pictures = _diffusion.gen(_model, args['number'])
        save_images(pictures, f'output\{args["number"]}.jpg')
        #output\GeneratedPics\

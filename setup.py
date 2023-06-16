#!/usr/bin/env python3


import gdown
import os
import tarfile

# check if models directory exists, if not create it
# check if lsun dataset is downloaded if not pull it from my drive and unpack
# finally set up TPU


PATH = './'
LSUN = 'lsun'
DEVICE = 'TPU'

#directory with preprocessed by us to 128x128 300 k subset of lsun bedroom dataset
URL = "https://drive.google.com/u/0/uc?id=147Mok0tiLTA43WmfBMXnzL0bNOX1mSVh&export=download"
OUTPUT = "lsun.tar.xz"

if __name__ == '__main__':

    #check if models directory exists
    print("Checking models directory...")
    if not os.path.isdir(PATH + 'models'):
        os.mkdir(PATH + 'models')

    #check if lsun directory exists
    print("Checking lsun...")
    if not os.path.isdir(PATH + LSUN):
        if not os.path.isfile(PATH + LSUN + '.tar.xz'):
            print("Downloadind lsun...")
            gdown.download(URL, OUTPUT, quiet=False)
        print("Unpacking lsun...")
        with tarfile.open(PATH + LSUN + '.tar.xz') as f:
            f.extractall(PATH)

    if DEVICE == 'TPU':
        #setting up tpu
        print("Setting up TPU...")
        os.system("export XRT_TPU_CONFIG='localservice;0;localhost:51011'")

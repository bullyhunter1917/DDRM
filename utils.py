import random
import torch
import torchvision
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import re
import glob
import os

def save_input(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray((ndarr * 255).astype(np.uint8))
    print(path)
    im.save(path)

#so that we don't break it in other places
def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    print(path)
    im.save(path)


def valid_pixel_range(T):
  T = (T.clamp(-1,1) + 1)/2
  T = (T * 255).type(torch.uint8) #valid pixel range 
  return T

def get_imgs(sample_size,dataset):
  (x,_ ) =  next(iter(DataLoader(dataset, sample_size, shuffle=True)))
  return x

class Obscure(object):
    """
    Add extra 3 channels to image that will represent image
    obscured with gray rectangles
    """
    def __init__(self, img_size, train, rectangles=14, max_rect=0.2):
        self.img_size = img_size
        self.train = train
        self.max_rect_area = int(max_rect * self.img_size)
        self.rectangles = rectangles


    def __call__(self, img):
        if self.train:
          obscured_noise = self.random_obscure(img.clone())
          return torch.concat((img,obscured_noise), dim=0)
        #for testing create both grayscaled and noisified rectangles
        else:  
          obscured_noise, obscured_gray  = self.random_obscure(torch.concat((img.clone(),img.clone()),dim=0))
          return torch.concat((img, obscured_noise, obscured_gray), dim=0)

    def obscure_image_grey(self,x, x_s,x_e,y_s,y_e):
      if x_e < x_s:
        x_e,x_s = x_s,x_e
      if y_e < y_s:
        y_e,y_s = y_s,y_e
      x[:, y_s:y_e, x_s:x_e] = .5
      return x

    def obscure_image_noise(self,x, x_s,x_e,y_s,y_e):
      if x_e < x_s:
        x_e,x_s = x_s,x_e
      if y_e < y_s:
        y_e,y_s = y_s,y_e
      x[:, y_s:y_e, x_s:x_e] = torch.randn_like(x[:, y_s:y_e, x_s:x_e])

      return x

    def random_obscure(self,x):
      if not self.train:
         x_grey = x[3:]
         x = x [:3]
      cnt = random.randint(int(self.rectangles * 0.8) , self.rectangles)
      for i in range(cnt):
        #try to place rectangles evenly
        x_e,y_e = random.sample(range(0, self.img_size), 2)
        x_s = random.randint(x_e-self.max_rect_area, x_e)
        y_s = random.randint(y_e-self.max_rect_area,y_e)
        x = self.obscure_image_noise(x, x_s,x_e,y_s,y_e)
        if not self.train:
          x_grey = self.obscure_image_grey(x_grey, x_s,x_e,y_s,y_e)
      if self.train:
          return x
      else:
          return x, x_grey 


def extract_numbers(string):
    pattern = r"(\d+)\.pt$"
    match = re.search(pattern, string)
    if match:
        number = match.group(1)
        return int(number)
    else:
        return None


def find_best_model(modelpath,filename, extension):
    current_directory = os.getcwd()
    current_directory +='/'
    current_directory += modelpath
    file_pattern = f"{filename}*.{extension}"

    matching_files = glob.glob(os.path.join(current_directory, file_pattern))
    if matching_files == []:
        return None

    nrs = [extract_numbers(s) for s in matching_files]
    nr = max(nrs)
    print(f"found {nr}")
    return(current_directory + f'/{filename}{nr}.{extension}')



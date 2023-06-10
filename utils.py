import random
import torch
import torchvision
from torch.utils.data import DataLoader
from PIL import Image

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
    def __init__(self, img_size, rectangles=4, max_rect=0.4):
        self.img_size = img_size
        self.max_rect_area = int(max_rect * self.img_size)
        self.rectangles = rectangles

    def __call__(self, img):
        obscured = self.random_obscure(img.clone())
        return torch.concat((img,obscured), dim=0)

    def obscure_image(self,x, x_s,x_e,y_s,y_e):
      if x_e < x_s:
        x_e,x_s = x_s,x_e
      if y_e < y_s:
        y_e,y_s = y_s,y_e
      x[:, y_s:y_e, x_s:x_e] = .5
      return x

    def random_obscure(self,x):
      cnt = random.randint(int(self.rectangles * 0.6) , self.rectangles)
      for i in range(cnt):
        #try to place rectangles evenly
        x_e,y_e = random.sample(range(0, self.img_size), 2)
        x_s = random.randint(x_e-self.max_rect_area, x_e - (self.max_rect_area // 3))
        y_s = random.randint(y_e-self.max_rect_area,y_e - (self.max_rect_are // 3))
        x = self.obscure_image(x, x_s,x_e,y_s,y_e)
      return x


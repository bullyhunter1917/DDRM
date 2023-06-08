import torchvision
from PIL import Image
import random

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    print(path)
    im.save(path)

class Obscure_images:
  def __init__(self, X, rectangles=10, max_rect=0.3):
    self.imgs = X.clone()
    self.img_count = self.imgs.shape[0]
    self.img_size = self.imgs.shape[2]
    self.max_rect = int(max_rect * self.img_size)
    self.rectangles = rectangles
    self.obscure_imgs_tensor()
  
  def obscure_image(self,x, x_s,x_e,y_s,y_e):
    if x_e < x_s:
      x_e,x_s = x_s,x_e
    if y_e < y_s:
      y_e,y_s = y_s,y_e
    x[:, y_s:y_e, x_s:x_e] = .5
    return x

  def random_obscure(self,x):
    cnt = 10
    for i in range(cnt):
      x_e,y_e = random.sample(range(0, self.img_size), 2)
      x_s = random.randint(x_e-self.max_rect, x_e)
      y_s = random.randint(y_e-self.max_rect,y_e)
      x = self.obscure_image(x, x_s,x_e,y_s,y_e)
    return x
  
  def obscure_imgs_tensor(self):
    img_cnt = imgs.shape[0]
    for i in range(self.img_count):
      self.imgs[i] = self.random_obscure(self.imgs[i])

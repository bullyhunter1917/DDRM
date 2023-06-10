import os
import math
import time
import torch
from tqdm import tqdm
from torch import optim
from torch import nn
from utils import *
from torch.utils.tensorboard import SummaryWriter
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
#import torch_xla.utils.serialization as xser

BATCH_SIZE = 128

class Diffusion:
    def __init__(self, schedule='linear', steps=1000, img_size=128, device='cuda'):
        self.schedule = schedule
        self.steps = steps
        self.img_size = img_size
        self.device = device

        self.beta = self.make_noise_schedule()
        self.alpha = 1-self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def make_noise_schedule(self):
        ''' 
        As stated in paper "Improved denoising diffusion probabilistic models"
        linear schedule works well for high resolution but is suboptimal for res
        such as 64x64 or 32x32
        '''
        if self.schedule == 'linear':
            scale = 1000 / self.steps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, self.steps, device=self.device)
        
        elif self.schedule == 'cosine':
            max_beta = 0.999
            betas = []
            fn = fn = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

            for i in range(self.steps):
                t1 = i / self.steps
                t2 = (i + 1) / self.steps
                betas.append(min(1 - fn(t2) / fn(t1), max_beta))

            return torch.as_tensor(betas, device=self.device)

        else:
            raise NotImplementedError(f"unknown beta schedule: {self.schedule}")

    def noise_images(self, x, t):
        '''
        Noisfied image x at timesteps t
        '''
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None,None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1-self.alpha_hat[t])[:, None, None, None]
        #random noise
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps , eps
    
    def sample_timesteps(self,n):
        return torch.randint(low=1, high=self.steps, size=(n,))
    
    def sample(self, model, n, images):
        assert(len(images) == n)
        print(f"Sampling {n} new images....")
        model.eval()

        # see Algorithm 2 Sampling from "Denoising Diffusion Probabilistic Models"
        with torch.no_grad():
            x = torch.randn(n,3,self.img_size, self.img_size, device=self.device)
            
            for i in tqdm(reversed(range(1, self.steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                 # with each step get new restored image concat it with obscured and pass to model
                predicted_noise = model(torch.concat((x,images),dim=1), t) 

                alpha = self.alpha[t][:, None,None,None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                #remove little bit of noise
                x = 1 / torch.sqrt(alpha) * (x -  ((1-alpha) / (torch.sqrt(1-alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        model.train()
        return valid_pixel_range(x)

    def trainfunc(self, model, loader, optimizer):
        tracker = xm.RateTracker()
        for i, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            t = self.sample_timesteps(x.shape[0]).to(self.device)
            x_t, epsilon = self.noise_images(x[:,:3], t)
            x_t = torch.concat((x_t, x[:,3:]), dim=1)

            predicted_epsilon = model(x_t, t, self.device)
            loss = self.lossfunc(epsilon, predicted_epsilon)
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(BATCH_SIZE)
            if i % 10 == 0:
                print(f'[xla:{xm.get_ordinal()}] ,cat: |replace with "y" for cat|, Loss={loss.item()} Rate={tracker.rate()} GlobalRate={tracker.global_rate()} Time={time.asctime()}')
#                                                                   {y}


    def train_xla(self, model, epochs, data,lr):
        print("trenuje")
        self.lossfunc = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(),lr)
        for epoch in range(epochs):   
            para_loader = pl.ParallelLoader(data, [self.device])
            self.trainfunc(model, para_loader.per_device_loader(self.device), optimizer)
            
            xm.master_print(f'Finished training epoch {epoch}')
            
            if epoch%10==0:
                print(epoch)
                xm.save(model.state_dict(),os.path.join("models", f"ckpt{epoch}.pt"))
                
                #sampling behaves strangely on tpu's so we leave it for local tests
                #sampled_images = self.sample(model,10)
                #save_images(sampled_images, os.path.join("results", f"{epoch}.jpg"))

    
    def gen(self, model, size, dataset):
        imgs = get_imgs(size, dataset)
        pictures_obscured = imgs[:,3:]
        pictures_original = imgs[:,:3]
        pictures_restored = self.sample(model, size, pictures_obscured)
        return pictures_original, pictures_obscured, pictures_restored


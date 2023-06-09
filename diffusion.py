import os
import math
import torch
from tqdm import tqdm
from torch import optim
from torch import nn
from utils import *
from torch.utils.tensorboard import SummaryWriter

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

    def noise_images(self, x_, t):
        '''
        Noisfied image x at timesteps t
        '''
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None,None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1-self.alpha_hat[t])[:, None, None, None]
        #random noise
        eps = torch.randn_like(x)
        return torch.concat((sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, x_obs), dim=1) , eps
    
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

                x = 1 / torch.sqrt(alpha) * (x -  ((1-alpha) / (torch.sqrt(1-alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        model.train()
        return valid_pixel_range(x)
    
    def train(self, model, epochs, device, data,lr):
        optimizer=optim.AdamW(model.parameters(),lr)
        lossfunc=nn.MSELoss()
        l = len(data)
        logger = SummaryWriter("trainingrun1")
        for epoch in range(epochs):
            pbar = tqdm(data)
            for j, (x, _) in enumerate(pbar):
                # Those are 6 channels images now, first three Channels are RGB of original later three are from obscured image
                images = x.to(device)
                t = self.sample_timesteps(x.shape[0]).to(device)
                #add noise only to original images
                x_t, epsilon = self.noise_images(images[:,:3],t)    
                x_t = torch.concat((x_t, images[:,3:]), dim=1)

                predicted_epsilon = model(x_t,t)
                loss = lossfunc(epsilon,predicted_epsilon)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logger.add_scalar("MSE", loss.item(), global_step=epoch * l + j)
            
            NR_OF_SAMPLES = 1
            smpl_images = next(iter(data))[:NR_OF_SAMPLES]                           # gets random images
            restored_images = self.sample(model, NR_OF_SAMPLES, smpl_imgs[:,3:])     # pass obscured this will return restored
            save_images(smpl_images[:,:3], os.path.join("results", f"{epoch}.jpg"))  # original images, without obscuration
            save_images(restored_images, os.path.join("results", f"{epoch}.jpg"))    # restored using diffusion 
            torch.save(model.state_dict(), os.path.join("models", f"ckpt.pt"))

        
    def gen(self, model, size, dataset):
        imgs = get_imgs(size, dataset).to(self.device)
        pictures_obscured = imgs[:,3:]
        pictures_original = imgs[:,:3]
        pictures_restored = self.sample(model, size, pictures_obscured)
        #also put images to valid range
        pictures_obscured = valid_pixel_range(pictures_obscured)
        pictures_original = valid_pixel_range(pictures_original)

        return pictures_original, pictures_obscured, pictures_restored



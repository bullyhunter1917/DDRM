import math
import torch
from tqdm import tqdm
from torch import optim
from torch import nn

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
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat([t]))[:, None, None,None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1-self.alpha_hat([t]))[:, None, None, None]
        #random noise
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps
    
    def sample_timesteps(self,n):
        return torch.randint(low=1, high=self.steps, size=(n,))
    
    def sample(self, model, n):
        print(f"Sampling {n} new images....")
        model.eval()

        # see Algorithm 2 Sampling from "Denoising Diffusion Probabilistic Models"
        with torch.no_grad():
            x = torch.randn(n,3,self.img_size, self.img_size, device=self.device)

            for i in tqdm(reversed(range(1, self.steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)

                alpha = self.alpha[t][:, None,None,None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x -  ((1-alpha) / (torch.sqrt(1-alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        model.train()
        x = (x.clamp(-1,1) + 1)/2
        x = (x * 255).type(torch.uint8)
        return x
    
    def train(self, model, epochs, device, images,lr):
        optimizer=optim.AdamW(model.parameters(),lr)
        lossfunc=nn.MSEloss()
        
        for epoch in epochs:
            pbar = tqdm(images)
            for j, (x, _) in enumerate(pbar):
                images = x.to(device)
                t = self.sample_timesteps(x.shape[0]).to(device)
                x_t, epsilon = self.noise_images()
                predicted_epsilon = model(x_t,t)
                loss = lossfunc(epsilon,predicted_epsilon)
                optimizer.zero_grad()
                lossfunc.backward()
                optimizer.step()


    # add here saving model and pictures every n epochs

    def gen(self, model, size):
        pictures = self.sample(model, size)
        return pictures


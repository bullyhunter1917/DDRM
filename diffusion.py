import os
import math
import time
import torch
from tqdm import tqdm
from torch import optim
from torch import nn
from utils import *
from torch.utils.tensorboard import SummaryWriter
# import torch_xla
# import torch_xla.core.xla_model as xm
# import torch_xla.distributed.parallel_loader as pl
# import torch_xla.distributed.xla_multiprocessing as xmp



from main import BATCH_SIZE

class Diffusion:
    def __init__(self, optimizer, schedule='linear', steps=1000, img_size=128, device='cuda'):
        self.schedule = schedule
        self.steps = steps
        self.img_size = img_size
        self.device = device
        self.optimizer = optimizer
        
        self.beta = self.make_noise_schedule()
        self.alpha = 1-self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alpha_hat_prev = torch.cat( ( torch.Tensor([1.]).to(self.device), self.alpha_hat[:-1]))

            

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
            #restored image at timestep t
            y_t = torch.randn(n,3,self.img_size, self.img_size, device=self.device)
            
            for i in tqdm(reversed(range(1, self.steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                
                alpha = self.alpha[t][:, None,None,None]
                alpha_hat = self.alpha_hat[t][:, None, None, None] #gamma at timestep t
                alpha_hat_prev = self.alpha_hat_prev[t][:, None, None, None] #gamma at timestep t
                beta = self.beta[t][:, None, None, None] # beta is 1-alpha
                
                noise_level =  alpha_hat 
                
                predicted_noise = model(torch.concat((y_t,images),dim=1), t,self.device) 

                #predict start from noise                
                y_0_approx = torch.sqrt(1 / alpha_hat) * y_t - torch.sqrt(1/alpha_hat - 1) * predicted_noise

                y_0_approx.clip_(-1.,1.)

                model_mean = beta * torch.sqrt(alpha_hat_prev) / (1 - alpha_hat) * y_0_approx + (1 - alpha_hat_prev) * torch.sqrt(alpha) / (1 - alpha_hat) * y_t
                model_log_variance_clipped = torch.log(torch.max(beta * (1-alpha_hat_prev) / (1-alpha_hat) , torch.tensor([1e-20]).to(self.device) ))

                noise = torch.randn_like(y_t) if i > 1 else torch.zeros_like(y_t)

                y_t = model_mean + noise * (0.5 * model_log_variance_clipped).exp()

        model.train()
        return valid_pixel_range(x)


    def train_gpu(self, model, epochs, data, lr):
        
        print('Trenuje na gpu')
        lossfunc = nn.MSELoss()
        for epoch in range(epochs):
            pbar = tqdm(data)
            for j, (x, _) in enumerate(pbar):
                images = x.to(self.device)
                t = self.sample_timesteps(x.shape[0]).to(self.device)
                
                x_t, epsilon = self.noise_images(images, t)
                predicted_epsilon = model(x_t, t,self.device)
                loss = lossfunc(epsilon, predicted_epsilon)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            sampled_images = self.sample(model, 1)
            save_images(sampled_images, os.path.join("results", f"{epoch}.jpg"))
            torch.save(model.state_dict(), os.path.join("models", f"ckpt.pt"))


    def train_xla(self, model, epochs, data, lr):
        import torch_xla.utils.utils as xu
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp

        def trainfunc(model, loader, optimizer):
            tracker = xm.RateTracker()
            for i, (x, y) in enumerate(loader):
                optimizer.zero_grad()
                t = self.sample_timesteps(x.shape[0]).to(self.device)
                x_t, epsilon = self.noise_images(x[:, :3], t)
                x_t = torch.concat((x_t, x[:, 3:]), dim=1)

                predicted_epsilon = model(x_t, t, self.device)
                loss = self.lossfunc(epsilon, predicted_epsilon)
                loss.backward()
                self.loss = loss
                xm.optimizer_step(optimizer)
                tracker.add(BATCH_SIZE)
                if i % 10 == 0:
                    print(
                        f'[xla:{xm.get_ordinal()}] , Loss={loss.item()} Rate={tracker.rate()} GlobalRate={tracker.global_rate()} Time={time.asctime()}')

        #                                                                   {y}

        print("Trenuje na tpu")
        self.lossfunc = nn.MSELoss()
        for epoch in range(epochs):
            para_loader = pl.ParallelLoader(data, [self.device])
            trainfunc(model, para_loader.per_device_loader(self.device), self.optimizer)

            xm.master_print(f'Finished training epoch {epoch}')

            #epochs will take longer so we will save model state in all iterations
            print(epoch)
            xm.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            }, os.path.join("models", f"ckpt{epoch}.pt"))

            #sampling behaves strangely on tpu's so we leave it for local tests
            #sampled_images = self.sample(model,10)
            #save_images(sampled_images, os.path.join("results", f"{epoch}.jpg"))


    # this function will be called when testing, it will use 9 channels 
    # obscured_noise will be used as input to model
    # but to presend results we will use image with grayscales_mask
    def gen(self, model, size, dataset):
        imgs = get_imgs(size, dataset).to(self.device)
        pictures_obscured_gray = imgs[:,6:]
        pictures_obscured_noise = imgs[:,3:6]
        pictures_original = imgs[:,:3]
        pictures_restored = self.sample(model, size, pictures_obscured_noise)
        return pictures_original, pictures_obscured_gray, pictures_restored

import torch
import torch.nn as nn
import torchvision

# Neural network model

class model(nn.Module):
    def __init__(self, input_channels, output_channels, time_dim=1024, device="cuda"):
        super().__init__()

        self.time_dim = time_dim
        self.device = device

        self.input = DoubleConv(3, 64) # res 256 channels 64

        self.down1 = Down(64, 128) # res 128 channels 128
        self.down2 = Down(128, 256) # res 64 channels 256
        self.down3 = Down(256, 512) # res 32 channels 512

        self.d = nn.AvgPool2d(kernel_size=2) # res 16
        self.conv_mid_1 = DoubleConv(512, 1024)
        self.sa = SelfAttention(1024, 16)
        self.conv_mid_2 = DoubleConv(1024, 512)

        self.up1 = Up(1024, 256) # channels 1024
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)

        self.end = nn.Conv2d(kernel_size=1, in_channels=64, out_channels=output_channels)
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )

        pos_enc_a = torch.sin(t.repeat(1, channels//2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels//2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)

        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.input(x)

        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)

        x5 = self.d(x4)
        x5 = self.conv_mid_1(x5)
        x5 = self.sa(x5)
        x5 = self.conv_mid_2(x5)

        x = self.up1(x5, x4, t)
        x = self.up2(x, x3, t)
        x = self.up3(x, x2, t)
        x = self.up4(x, x1, t)

        x = self.end(x)

        return x

class DoubleConv(nn.Module):
    def __init__(self, inChannels, outChannels, midChannels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not midChannels:
            midChannels = outChannels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=midChannels, kernel_size=(3, 3), padding=1, bias=False),
            nn.GroupNorm(1, midChannels),
            nn.GELU(),
            nn.Conv2d(in_channels=midChannels, out_channels=outChannels, kernel_size=(3, 3), padding=1, bias=False),
            nn.GroupNorm(1, outChannels),
        )

    def forward(self, x):
        if self.residual:
            return nn.functional.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, input_channels, output_channels, emb_dim=1024):
        super().__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(input_channels, output_channels, residual=True),
            DoubleConv(output_channels, output_channels)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, output_channels),
        )

    def forward(self, x, t):
        x = self.down(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + emb

class Up(nn.Module):
    def __init__(self, input_channels, output_channels, emb_dim=1024):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(input_channels, input_channels//2),
            DoubleConv(input_channels//2, output_channels)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, output_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + emb

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size

        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value

        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

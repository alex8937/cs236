from typing import Tuple
import gin
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import ignite.distributed as idist


@gin.configurable
class Discriminator(ImageBaseModel):
    def __init__(self, in_channels:int=3, n_layers:int=4, n_channels:int=64, num_epochs:int=100, steps_per_epoch:int=100000, lr:float=4e-06, **kwargs):
        super().__init__(**kwargs)

        self.n_layers = n_layers
        self.window_size = 4
        for _ in range(n_layers):
            self.window_size = self.window_size * 2

        blocks = [
            nn.Conv2d(in_channels=in_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        n = n_channels
        for _ in range(n_layers):
            blocks += [
                nn.Conv2d(in_channels=n, out_channels=n*2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(n*2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            n *= 2

        blocks += [
            nn.Conv2d(in_channels=n, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)
        ]
        self.blocks = nn.Sequential(*blocks)
        self.independent = True

        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.optimizer = idist.auto_optim(self.optimizer, clip_grad_norm)
        self.scheduler = CosineSchedulerWithWarmup(num_warmup_steps=steps_per_epoch//2).build(
            self.optimizer, num_epochs=num_epochs, steps_per_epoch=steps_per_epoch)

    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        return self.blocks(inputs)

    def train(self, originals:torch.Tensor, reconstructions:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_real = self(originals)
        logits_fake = self(reconstructions)

        # GAN Hinge Loss Geometric GAN
        loss_real = torch.mean(F.relu(1.0 - logits_real))
        loss_fake = torch.mean(F.relu(1.0 + logits_fake))
        loss = 0.5 * loss_real + 0.5 * loss_fake

        return self.train_step(loss), logits_to_probs(logits_real).mean().detach(), logits_to_probs(logits_fake).mean().detach()

    def __repr__(self):
        return f"{self.__class__.__name__}(n_layers={self.n_layers}, window_size={self.window_size})"

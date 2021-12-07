import timeit
import logging
from typing import Optional, List, Any, Dict, Tuple
import math
import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import ignite.distributed as idist

import clip

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        channels = (in_channels + out_channels) // 2
        self.blocks = nn.ModuleList()
        self.blocks.append(Normalize(in_channels))
        self.blocks.append(Swish())
        self.blocks.append(nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1))
        self.blocks.append(Normalize(channels))
        self.blocks.append(Swish())
        self.blocks.append(nn.Dropout(dropout))
        self.blocks.append(nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=3, padding=1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        states = inputs
        for block in self.blocks:
            states = block(states)
        return inputs + states


class DownSampleBlock(nn.Module):
    def __init__(self, channels: int, **kwargs):
        super().__init__(**kwargs)
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = nn.functional.pad(states, (0, 1, 0, 1))
        states = self.conv(states)
        return states


class UpSampleBlock(nn.Module):
    def __init__(self, channels: int, **kwargs):
        super().__init__(**kwargs)
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = nn.functional.interpolate(states, scale_factor=2.0, mode="nearest")
        states = self \
            .conv(states)
        return states


class Generator(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, nscales: int, nres: int, nattn: int, in_channels: int,
                 channels: int, ch_mult: Optional[Tuple[float]], gradient_checkpointing: bool = False,
                 gmapping: bool = False, w_dim:int = None, **kwargs):
        super().__init__(**kwargs)
        self.gmapping = gmapping
        if self.gmapping:
            if w_dim is None:
                w_dim = latent_dim
            self.GMapping = GMapping(z_dim=latent_dim, w_dim=w_dim)
            latent_dim = w_dim

        proj_dim = output_dim >> nscales
        # output_dim needs to be power of 2
        assert output_dim % proj_dim == 0, f"output_dim % proj_dim is expected to be 0, but output_dim={output_dim}, proj_dim={proj_dim}"
        self.proj = nn.Linear(latent_dim, in_channels * proj_dim * proj_dim)
        self.in_channels = in_channels
        self.proj_dim = proj_dim

        self.gradient_checkpointing = gradient_checkpointing
        if ch_mult is None:
            ch_mult = (1,) * nscales
        elif len(ch_mult) < nscales:
            ch_mult = ch_mult + (ch_mult[-1],) * (nscales - len(ch_mult))

        self.blocks = nn.ModuleList()
        self.blocks.append(self.proj)
        self.blocks.append(nn.Unflatten(1, (in_channels, proj_dim, proj_dim)))

        in_ch = int(channels * ch_mult[-1])
        self.blocks.append(nn.Conv2d(in_channels=in_channels, out_channels=in_ch, kernel_size=3, padding=1))
        self.blocks.append(ResidualBlock(in_ch, in_ch))

        for _ in range(nattn):
            self.blocks.append(Attention(in_ch))
            self.blocks.append(ResidualBlock(in_ch, in_ch))

        ch_mult = tuple(reversed(ch_mult))
        for i in reversed(range(nscales)):
            out_ch = int(channels * ch_mult[i])
            for _ in range(nres):
                self.blocks.append(ResidualBlock(in_ch, out_ch))
                in_ch = out_ch
            self.blocks.append(UpSampleBlock(in_ch))

        self.blocks.append(nn.Sequential(
            Normalize(in_ch),
            Swish(),
            nn.Conv2d(in_channels=in_ch, out_channels=3, kernel_size=3, padding=1)
        ))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if self.gmapping:
            states = self.GMapping(states)
        for ix, block in enumerate(self.blocks):
            # print(states.shape)
            # checkpoint = self.gradient_checkpointing and self.training and states.shape[-1] >= 64 and ix < len(self.blocks)-1
            # states = idist.maybe_checkpoint(block, states, active=checkpoint)
            states = block(states)
        return states


@gin.configurable
class CLIPGAN(ImageBaseModel):
    def __init__(self, latent_dim=512, rand_dim=0, output_dim=256, nres: int = 2, nattn: int = 0, channels: int = 64,
                 per_weight: float = 1.0, dsc_weight: float = 0.1,
                 gan_start: float = 0.5, dsc_layers: int = 3, dsc_channels: int = 32,
                 no_clip: bool = False,
                 clip_loss: bool = False,
                 gmapping: bool = False, w_dim=128,
                 num_epochs: int = 100, steps_per_epoch: int = 100000, lr: float = 4e-3,
                 ch_mult: Optional[Tuple[float]] = None, gradient_checkpointing: bool = False, **kwargs):
        super().__init__(**kwargs)
        nscales = int(math.log2(output_dim)) - 4
        self.latent_dim = latent_dim
        self.rand_dim = rand_dim
        self.no_clip = no_clip
        self.decoder = Generator(latent_dim + self.rand_dim, output_dim, nscales, nres, nattn, in_channels=channels,
                                 channels=channels,
                                 ch_mult=ch_mult, gradient_checkpointing=gradient_checkpointing, gmapping=gmapping, w_dim=w_dim)

        self.per_weight = per_weight
        self.dsc_weight = dsc_weight
        self.gan_steps = int(gan_start * steps_per_epoch)
        self.step = 0
        self.state_vars.append("step")
        self.step_time = None
        self.gradient_checkpointing = gradient_checkpointing

        self.perceptual_loss = PerceptualSimilarityLoss()
        self.discriminator = Discriminator(n_layers=dsc_layers, n_channels=dsc_channels, num_epochs=num_epochs,
                                           steps_per_epoch=steps_per_epoch, lr=lr)
        self.clip_loss = clip_loss
        if self.clip_loss:
            self.clip, _ = clip.load("ViT-B/16", device='cpu')

    #
    # forward
    #
    def forward(self, inputs: torch.Tensor, latents: torch.Tensor, targets: Optional[torch.Tensor] = None,
                labels: Optional[List[str]] = None) -> Output:
        originals = targets if targets is not None else inputs

        batch_size = latents.shape[0]
        if self.rand_dim > 0:
            rand_latent = torch.randn(batch_size, self.rand_dim).to(inputs.device)
            latents = torch.cat([latents, rand_latent], dim=-1)

        if self.no_clip:
            latents = torch.randn(batch_size, self.latent_dim).to(inputs.device)

        reconstructions = self.decoder(latents)
        # perceptual loss
        lpips_loss = self.perceptual_loss(originals, reconstructions).mean()

        # shortcut when not training
        if not self.training:
            kwargs = {
                'lpips_loss': lpips_loss,
                'result': reconstructions,
            }
            return Output(loss=lpips_loss, **kwargs)

        # train discriminator
        dsc_loss, real_prob, fake_prob = self.discriminator.train(originals.detach(), reconstructions.detach())
        # recognition loss
        rec_loss = torch.nn.functional.mse_loss(originals, reconstructions)
        z_rec_loss = torch.tensor(0.0, device=self.device)
        if self.clip_loss:
            z_reconstructions = self.clip.encode_image(reconstructions)
            z_rec_loss = torch.nn.functional.mse_loss(latents, z_reconstructions)

        # GAN loss (after some number of steps)
        zero = torch.tensor([0.0], device=self.device)
        gan_loss = zero
        gan_weight = zero
        gan_loss = 1.0 - logits_to_probs(self.discriminator(reconstructions)).mean()
        # gan_loss = torch.log(logits_to_probs(self.discriminator(inputs)).mean()) + torch.log(
        #     1.0 - logits_to_probs(self.discriminator(reconstructions)).mean())
        gan_weight = max(0.0, min((self.step - self.gan_steps) / self.gan_steps, 1.0)) * self.dsc_weight
        gan_weight = torch.tensor(gan_weight, device=self.device)
        if self.training and gan_weight > 0.0:
            try:
                gan_weight = gan_weight * self.calculate_adaptive_weight(z_rec_loss + rec_loss, gan_loss)
            except RuntimeError as e:
                logger.error(f"RuntimeError {e}")

        # increment step count
        row_count = inputs.shape[0] * idist.get_data_parallelism()
        rows_per_sec = 0
        self.step += row_count
        if self.step_time is not None:
            rows_per_sec = row_count / (timeit.default_timer() - self.step_time)
        self.step_time = timeit.default_timer()
        rows_per_sec = torch.tensor([rows_per_sec], device=self.device)

        # total loss
        # loss = rec_loss + self.per_weight * lpips_loss # + gan_weight * gan_loss
        loss = z_rec_loss + rec_loss + self.per_weight * lpips_loss + gan_weight * gan_loss

        # results
        kwargs = {
            'result': reconstructions,
            'z_rec_loss': z_rec_loss,
            'rec_loss': rec_loss,
            'lpips_loss': lpips_loss,
            'gan_loss': gan_loss,
            'gan_weight': gan_weight,
            'dsc_loss': dsc_loss,
            'real_prob': real_prob,
            'fake_prob': fake_prob,
            'rows_per_sec': rows_per_sec,
        }
        return Output(loss=loss, **kwargs)

    def calculate_adaptive_weight(self, rec_loss, gan_loss):
        last_layer = self.decoder.blocks[-1][-1].weight
        rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True, allow_unused=True)[0]
        gan_grads = torch.autograd.grad(gan_loss, last_layer, retain_graph=True, allow_unused=True)[0]

        dsc_weight = torch.norm(rec_grads) / (torch.norm(gan_grads) + 1e-4)
        dsc_weight = torch.clamp(dsc_weight, 0.0, 1e4).detach()
        return dsc_weight

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.rand_dim > 0:
                batch_size = latent.shape[0]
                rand_latent = torch.randn(batch_size, self.rand_dim).to(latent.device)
                latent = torch.cat([latent, rand_latent], dim=-1)

            if self.no_clip:
                latent = torch.randn(batch_size, self.latent_dim).to(inputs.device)
            # quantized = self.quantizer.resolve(latent)
            # quantized = self.dec_conv(quantized)
            return self.decoder(latent)

from typing import List
import torch
from torch_diffusion.model.difussion_model import DiffusionModule
from dataclasses import dataclass
import numpy as np
import torchvision.transforms as transforms
from PIL.Image import Image


@dataclass
class InferenceConfig:
    checkpoint_file: str
    features: int
    height: int
    width: int
    timesteps: int = 500
    beta1: float = 1e-4
    beta2: float = 0.02


class Inference:
    _model: DiffusionModule
    _config: InferenceConfig
    _device: torch.device
    _ab_t: torch.Tensor
    _b_t: torch.Tensor
    _a_t: torch.Tensor

    def __init__(self, config: InferenceConfig) -> None:
        self._config = config
        self._model = DiffusionModule.load_from_checkpoint(config.checkpoint_file)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else torch.device("cpu")
        )
        self.init_noise_schedule()

    def infer(
        self,
    ):
        samples, intermediate = self.sample_ddpm(1)
        self.save_sample(samples)
        self.save_gif(intermediate)

    def init_noise_schedule(self):
        self._b_t = (self._config.beta2 - self._config.beta1) * torch.linspace(
            0, 1, self._config.timesteps + 1, device=self._device
        ) + self._config.beta1
        self._a_t = 1 - self._b_t
        self._ab_t = torch.cumsum(self._a_t.log(), dim=0).exp()
        self._ab_t[0] = 1

    def denoise_ddim(self, x, t, t_prev, pred_noise):
        ab = self._ab_t[t]
        ab_prev = self._ab_t[t_prev]

        x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
        dir_xt = (1 - ab_prev).sqrt() * pred_noise

        return x0_pred + dir_xt

    @staticmethod
    def unorm(x):
        # unity norm. results in range of [0,1]
        # assume x (h,w,3)
        xmax = x.max((0, 1))
        xmin = x.min((0, 1))
        return (x - xmin) / (xmax - xmin)

    @staticmethod
    def norm_all(store, n_t, n_s):
        # runs unity norm on all timesteps of all samples
        nstore = np.zeros_like(store)
        for t in range(n_t):
            for s in range(n_s):
                nstore[t, s] = Inference.unorm(store[t, s])
        return nstore

    @staticmethod
    def norm_torch(x_all):
        # runs unity norm on all timesteps of all samples
        # input is (n_samples, 3,h,w), the torch image format
        x = x_all.cpu().numpy()
        xmax = x.max((2, 3))
        xmin = x.min((2, 3))
        xmax = np.expand_dims(xmax, (2, 3))
        xmin = np.expand_dims(xmin, (2, 3))
        nstore = (x - xmin) / (xmax - xmin)
        return torch.from_numpy(nstore)

    # helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
    def denoise_add_noise(self, x, t, pred_noise, z=None):
        if z is None:
            z = torch.randn_like(x)
        noise = self._b_t.sqrt()[t] * z
        mean = (
            x - pred_noise * ((1 - self._a_t[t]) / (1 - self._ab_t[t]).sqrt())
        ) / self._a_t[t].sqrt()
        return mean + noise

    @torch.no_grad()
    def sample_ddpm(self, n_sample, save_rate=20):
        # x_T ~ N(0, 1), sample initial noise
        samples = torch.randn(n_sample, 3, self._config.height, self._config.width).to(
            self._device
        )

        # array to keep track of generated steps for plotting
        intermediate = []
        for i in range(self._config.timesteps, 0, -1):
            print(f"sampling timestep {i:3d}", end="\r")

            # reshape time tensor
            t = torch.tensor([i / self._config.timesteps])[:, None, None, None].to(
                self._device
            )

            # sample some random noise to inject back in. For i = 1, don't add back in noise
            z = torch.randn_like(samples) if i > 1 else 0

            eps = self._model(samples, t)  # predict noise e_(x_t,t)
            samples = self.denoise_add_noise(samples, i, eps, z)
            if i % save_rate == 0 or i == self._config.timesteps or i < 8:
                intermediate.append(samples.detach().cpu().numpy())

        intermediate = np.stack(intermediate)
        return samples, intermediate

    def _undo_normalize(self, image):
        image = (image * 0.5) + 0.5
        return image

    def _to_pil(self, image) -> Image:
        first_image = self._undo_normalize(image)
        first_image_pil = transforms.ToPILImage()(first_image)
        return first_image_pil

    def save_gif(self, intermediate: List[torch.Tensor]):
        pils: List[Image] = list(map(self._to_pil, intermediate))
        pils[0].save(
            "output.gif",
            save_all=True,
            append_images=pils[1:],
            duration=300,  # duration between frames in milliseconds
            loop=0,
        )

    def save_sample(self, sample: torch.Tensor) -> None:
        self._to_pil(sample).save("output.png")

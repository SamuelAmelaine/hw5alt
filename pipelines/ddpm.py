from typing import List, Optional, Tuple, Union
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import randn_tensor


class DDPMPipeline:
    def __init__(self, unet, scheduler, vae=None, class_embedder=None):
        self.unet = unet
        self.scheduler = scheduler

        # NOTE: this is for latent DDPM
        self.vae = None
        if vae is not None:
            self.vae = vae

        # NOTE: this is for CFG
        if class_embedder is not None:
            self.class_embedder = class_embedder

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(
                image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 1000,
        classes: Optional[Union[int, List[int]]] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        device=None,
    ):
        image_shape = (batch_size, self.unet.input_ch,
                       self.unet.input_size, self.unet.input_size)
        if device is None:
            device = next(self.unet.parameters()).device

        # Handle CFG
        if classes is not None:
            if isinstance(classes, int):
                classes = [classes] * batch_size
            classes = torch.tensor(classes, device=device)
            class_embeds = self.class_embedder(classes)
            # Double the batch for classifier-free guidance
            image = randn_tensor(
                (2 * batch_size, *image_shape[1:]), 
                generator=generator, 
                device=device
            )
        else:
            image = randn_tensor(image_shape, generator=generator, device=device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device)

        # Denoising loop
        for t in self.progress_bar(self.scheduler.timesteps):
            # Expand for classifier free guidance
            if guidance_scale is not None:
                model_input = torch.cat([image] * 2)
                timestep_input = torch.cat([t.unsqueeze(0)] * 2)
                model_output = self.unet(model_input, timestep_input, class_embeds)
                noise_pred_uncond, noise_pred_cond = model_output.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.unet(image, t, None)

            # Compute previous image: x_t -> x_t-1
            image = self.scheduler.step(noise_pred, t, image, generator=generator)

        # Handle VAE decoding if using latent diffusion
        if self.vae is not None:
            image = 1 / 0.18215 * image
            image = self.vae.decode(image)

        # Return final image, re-scale to [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return self.numpy_to_pil(image)

"""Adapter for manipulating encoder / rosinality StyleGAN2 latents in SeFa."""

import sys
from pathlib import Path

import torch
import torch.nn as nn


PSP_MODEL_NAME = 'psp_lenna_ffhq1024'
E4E_MODEL_NAME = 'e4e_ffhq_encode'
PSP_REPO_DIR = Path(__file__).resolve().parents[1] / 'pixel2style2pixel'
PSP_CHECKPOINT = PSP_REPO_DIR / 'pretrained_models' / 'psp_ffhq_encode.pt'
E4E_CHECKPOINT = Path(__file__).resolve().parent / 'checkpoints' / 'e4e_ffhq_encode.pth'

ENCODER_MODEL_CONFIGS = {
    PSP_MODEL_NAME: PSP_CHECKPOINT,
    E4E_MODEL_NAME: E4E_CHECKPOINT,
}


class EncoderSefaGenerator(nn.Module):
    """Wraps a pSp/e4e StyleGAN2 decoder with the methods SeFa expects."""

    def __init__(self, checkpoint_path=PSP_CHECKPOINT, model_name=PSP_MODEL_NAME):
        super().__init__()
        self.is_psp_sefa_generator = True
        self.model_name = model_name
        self.z_space_dim = 512
        self.w_space_dim = 512
        self.num_layers = 18
        self.checkpoint_path = str(checkpoint_path)

        Generator = self._load_psp_generator_class()

        self.decoder = Generator(1024, 512, 8)
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        if 'state_dict' not in checkpoint:
            raise KeyError(f'Checkpoint `{self.checkpoint_path}` has no `state_dict`.')
        state_dict = {
            key[len('decoder.') :]: value
            for key, value in checkpoint['state_dict'].items()
            if key.startswith('decoder.')
        }
        if not state_dict:
            raise KeyError(f'Checkpoint `{self.checkpoint_path}` has no decoder weights.')
        self.decoder.load_state_dict(state_dict, strict=True)

    def synthesis(self, code):
        """Synthesizes an image from a W+ latent code with shape [N, 18, 512]."""
        image, _ = self.decoder(
            [code],
            input_is_latent=True,
            randomize_noise=False,
            return_latents=False,
        )
        return {'image': image}

    def forward(self, code):
        """Keeps the PGGAN-style call path available, though UI uses synthesis."""
        return self.synthesis(code)

    def mapping(self, z):
        """Maps Z to W if the UI needs to sample after the saved Lenna code."""
        return {'w': self.decoder.get_latent(z)}

    def truncation(self, w, trunc_psi=1.0, trunc_layers=None):
        """Repeats one W code into W+; pSp Lenna itself is loaded from .npy."""
        del trunc_psi, trunc_layers
        return w.unsqueeze(1).repeat(1, self.num_layers, 1)

    def sefa_weights(self, layers):
        """Returns style-modulation affine weights used for SeFa directions."""
        modules = [self.decoder.conv1.conv, self.decoder.to_rgb1.conv]
        for conv1, conv2, to_rgb in zip(
            self.decoder.convs[::2],
            self.decoder.convs[1::2],
            self.decoder.to_rgbs,
        ):
            modules.extend([conv1.conv, conv2.conv, to_rgb.conv])
        return [modules[idx].modulation.weight.T for idx in layers]

    @staticmethod
    def _load_psp_generator_class():
        """Imports pSp's Generator despite SeFa also having a ``models`` package."""
        old_path = list(sys.path)
        old_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == 'models' or name.startswith('models.')
        }
        for name in list(old_modules):
            del sys.modules[name]
        sys.path.insert(0, str(PSP_REPO_DIR))
        try:
            from models.stylegan2.model import Generator
        finally:
            for name in list(sys.modules):
                if name == 'models' or name.startswith('models.'):
                    del sys.modules[name]
            sys.modules.update(old_modules)
            sys.path[:] = old_path
        return Generator


def is_psp_model(model_name):
    """Checks whether a model name is handled by this adapter."""
    return model_name in ENCODER_MODEL_CONFIGS


def build_encoder_sefa_generator(model_name):
    """Builds the registered encoder-decoder adapter for SeFa."""
    if model_name not in ENCODER_MODEL_CONFIGS:
        raise KeyError(f'Unknown encoder model `{model_name}`.')
    return EncoderSefaGenerator(ENCODER_MODEL_CONFIGS[model_name], model_name)


PSpSefaGenerator = EncoderSefaGenerator

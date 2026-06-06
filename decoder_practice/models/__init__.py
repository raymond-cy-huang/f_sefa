# python3.7
"""Minimal generator registry for decoding latent codes into images."""

from .model_zoo import MODEL_ZOO
from .pggan_generator import PGGANGenerator
from .stylegan_generator import StyleGANGenerator

__all__ = [
    'MODEL_ZOO', 'PGGANGenerator', 'StyleGANGenerator',
    'build_generator', 'parse_gan_type'
]

_GAN_TYPES_ALLOWED = ['pggan', 'stylegan']


def build_generator(gan_type, resolution, **kwargs):
    """Builds the generator architecture needed by a checkpoint.

    Args:
        gan_type: GAN family used by the checkpoint.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the generator.

    Raises:
        ValueError: If the `gan_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """
    if gan_type not in _GAN_TYPES_ALLOWED:
        raise ValueError(f'Invalid GAN type: `{gan_type}`!\n'
                         f'Types allowed: {_GAN_TYPES_ALLOWED}.')

    if gan_type == 'pggan':
        return PGGANGenerator(resolution, **kwargs)
    if gan_type == 'stylegan':
        return StyleGANGenerator(resolution, **kwargs)
    raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


def parse_gan_type(module):
    """Parses GAN family from the generator instance.

    Args:
        module: The generator to identify.

    Returns:
        A string, indicating the GAN type.

    Raises:
        ValueError: If the GAN type is unknown.
    """
    # decode_latent.py uses this to choose PGGAN.forward vs StyleGAN.synthesis.
    if isinstance(module, PGGANGenerator):
        return 'pggan'
    if isinstance(module, StyleGANGenerator):
        return 'stylegan'
    raise ValueError(f'Unable to parse GAN type from type `{type(module)}`!')

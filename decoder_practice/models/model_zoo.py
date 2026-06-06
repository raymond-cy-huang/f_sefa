# python3.7
"""Local decoder model registry.

Only models that have both a latent file under ``decoder_practice/latent_codes``
and a checkpoint under ``decoder_practice/checkpoints`` are listed here.
"""

# MODEL_ZOO is the bridge from a model name to the generator architecture.
# decode_latent.py removes the URL field before constructing the generator.
MODEL_ZOO = {
    # PGGAN: latent is a Z code and the generator forward pass outputs image.
    'pggan_celebahq1024': dict(
        gan_type='pggan',
        resolution=1024,
        url='local checkpoint: checkpoints/pggan_celebahq1024.pth',
    ),

    # StyleGAN: saved latents are layer-wise W+ codes consumed by synthesis().
    'stylegan_animeface512': dict(
        gan_type='stylegan',
        resolution=512,
        url='local checkpoint: checkpoints/stylegan_animeface512.pth',
    ),
    'stylegan_car512': dict(
        gan_type='stylegan',
        resolution=512,
        url='local checkpoint: checkpoints/stylegan_car512.pth',
    ),
    'stylegan_cat256': dict(
        gan_type='stylegan',
        resolution=256,
        url='local checkpoint: checkpoints/stylegan_cat256.pth',
    ),
}

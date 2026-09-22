import torch
from psp_sefa_adapter import build_encoder_sefa_generator

g = build_encoder_sefa_generator('e4e_ffhq_encode').cuda()
z = torch.randn(1, 512, device='cuda')
print('z', z.shape, 'latent', g.decoder.get_latent(z).shape)
for i, m in enumerate(g.decoder.style):
    z = m(z)
    print(i, type(m).__name__, z.shape)

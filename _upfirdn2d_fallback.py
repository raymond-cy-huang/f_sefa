import torch
import torch.nn.functional as F

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])
    px0, px1, py0, py1 = pad
    b, c, h, w = input.shape
    minor = 1
    x = input.reshape(-1, h, 1, w, 1, minor)
    x = F.pad(x, [0, 0, 0, up - 1, 0, 0, 0, up - 1])
    x = x.reshape(-1, h * up, w * up, minor)
    x = F.pad(x, [0, 0, max(px0, 0), max(px1, 0), max(py0, 0), max(py1, 0)])
    x = x[:, max(-py0, 0):x.shape[1] - max(-py1, 0), max(-px0, 0):x.shape[2] - max(-px1, 0), :]
    kh, kw = kernel.shape
    x = x.permute(0, 3, 1, 2).reshape(-1, 1, x.shape[1], x.shape[2])
    x = F.conv2d(x, torch.flip(kernel, [0, 1]).to(x).view(1, 1, kh, kw))
    x = x.reshape(b, c, x.shape[2], x.shape[3])
    return x[:, :, ::down, ::down]

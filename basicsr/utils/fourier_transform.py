import cv2
import numpy as np
import torch
import torch.fft as fft
import torchvision.transforms as transforms
from PIL import Image


def compute_fft(img, device='cpu'):
    # device = args.device if args.device == "cpu" else int(args.device)
    # device = 'cpu'  # Do not modify
    batch, channel, h, w = img.shape
    lpf = torch.zeros((h, w))
    R = (h + w) // 8
    lpf[w//2-R:w//2+R, h//2-R:h//2+R]
    hpf = 1 - lpf
    hpf, lpf = hpf.to(device), lpf.to(device)

    img = img.to(device)
    # img = transforms.Normalize(2, 0.5)(img)
    fft_img = torch.fft.fftn(img, dim=(1, 2))
    # print('fft size', fft_img.size()) # [1, 3, 224, 224]
    # put low_freq into the center of img
    fft_img = torch.roll(fft_img, (h // 2, w // 2), dims=(1, 2))
    f_low = fft_img * lpf
    f_high = fft_img * hpf
    X_low = torch.abs(torch.fft.ifftn(f_low, dim=(1, 2)))
    X_high = torch.abs(torch.fft.ifftn(f_high, dim=(1, 2)))
    return X_low, X_high


def fft_get_hf(img, device='cpu'):
    batch, channel, h, w = img.shape
    lpf = torch.zeros((h, w))
    R = (h + w) // 8
    lpf[w // 2 - R:w // 2 + R, h // 2 - R:h // 2 + R]
    hpf = 1 - lpf
    hpf, lpf = hpf.to(device), lpf.to(device)

    img = img.to(device)
    # img = transforms.Normalize(2, 0.5)(img)
    fft_img = torch.fft.fftn(img, dim=(2, 3))
    # print('fft size', fft_img.size()) # [1, 3, 224, 224]
    # put low_freq into the center of img
    fft_img = torch.roll(fft_img, (h // 2, w // 2), dims=(2, 3))
    f_high = fft_img * hpf
    X_high = torch.abs(torch.fft.ifftn(f_high, dim=(2, 3)))
    return X_high
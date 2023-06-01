import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.fft as fft
import torchvision.transforms as transforms
from torchvision import utils as vutils
from PIL import Image

def gaussian_filter_high_f(fshift, D):
    # 获取索引矩阵及中心点坐标
    h, w = fshift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))

    # 计算中心距离矩阵
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2

    # 计算变换矩阵
    template = np.exp(- dis_square / (2 * D ** 2))

    return template * fshift

def gaussian_filter_low_f(fshift, D):
    # 获取索引矩阵及中心点坐标
    h, w, c = fshift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))

    # 计算中心距离矩阵
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2

    # 计算变换矩阵
    template = 1 - np.exp(- dis_square / (2 * D ** 2)) # 高斯过滤器

    return template * fshift

def circle_filter_high_f(fshift, radius_ratio):
    """
    过滤掉除了中心区域外的高频信息
    """
    # 1, 生成圆形过滤器, 圆内值1, 其他部分为0的过滤器, 过滤
    template = np.zeros(fshift.shape, np.uint8)
    crow, ccol = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)  # 圆心
    radius = int(radius_ratio * img.shape[0] / 2)
    if len(img.shape) == 3:
        cv2.circle(template, (crow, ccol), radius, (1, 1, 1), -1)
    else:
        cv2.circle(template, (crow, ccol), radius, 1, -1)
    # 2, 过滤掉除了中心区域外的高频信息
    return template * fshift


def circle_filter_low_f(fshift, radius_ratio):
    """
    去除中心区域低频信息
    """
    # 1 生成圆形过滤器, 圆内值0, 其他部分为1的过滤器, 过滤
    filter_img = np.ones(fshift.shape, np.uint8)
    crow, col = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)
    radius = int(radius_ratio * img.shape[0] / 2)
    if len(img.shape) == 3:
        cv2.circle(filter_img, (crow, col), radius, (0, 0, 0), -1)
    else:
        cv2.circle(filter_img, (crow, col), radius, 0, -1)
    # 2 过滤中心低频部分的信息
    return filter_img * fshift


def ifft(fshift):
    """
    傅里叶逆变换
    """
    ishift = np.fft.ifftshift(fshift)  # 把低频部分sift回左上角
    iimg = np.fft.ifftn(ishift)  # 出来的是复数，无法显示
    iimg = np.abs(iimg)  # 返回复数的模
    return iimg

def get_low_high_f(img, radius_ratio, D):
    """
    获取低频和高频部分图像
    """
    # 傅里叶变换
    # np.fft.fftn
    f = np.fft.fftn(img)
    fshift = np.fft.fftshift(f)
    f = fft.fftn(img_tensor, dim=(1, 2))  # Compute the N-dimensional discrete Fourier Transform. 零频率分量位于频谱图像的左上角
    f_shift = fft.fftshift(f)  # 零频率分量会被移到频域图像的中心位置，即低频

    # 获取低频和高频部分
    # hight_parts_fshift = circle_filter_low_f(f_shift.copy(), radius_ratio=radius_ratio)  # 过滤掉中心低频
    # low_parts_fshift = circle_filter_high_f(fshift.copy(), radius_ratio=radius_ratio)
    hight_parts_fshift = gaussian_filter_low_f(f_shift.copy(), D=D)
    low_parts_fshift = gaussian_filter_high_f(f_shift.copy(), D=D)

    low_parts_img = ifft(low_parts_fshift)  # 先sift回来，再反傅里叶变换
    high_parts_img = ifft(hight_parts_fshift)

    # 显示原始图像和高通滤波处理图像
    img_new_low = (low_parts_img - np.amin(low_parts_img)) / (np.amax(low_parts_img) - np.amin(low_parts_img) + 0.00001)
    img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (
                np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)

    # uint8
    img_new_low = np.array(img_new_low * 255, np.uint8)
    img_new_high = np.array(img_new_high * 255, np.uint8)
    return img_new_low, img_new_high


def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img


def compute_fft(img, device='cpu'):
    # device = args.device if args.device == "cpu" else int(args.device)
    # device = 'cpu'  # Do not modify
    channel, h, w = img.shape
    lpf = torch.zeros((h, w))
    R = (h + w) // 8
    for x in range(w):
        for y in range(h):
            if ((x - w / 2) ** 2 + (y - h / 2) ** 2) < R ** 2:
                lpf[y, x] = 1
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


# 频域中使用高斯滤波器能更好的减少振铃效应
if __name__ == '__main__':
    radius_ratio = 0.5  # 圆形过滤器的半径：ratio * w/2
    D = 50              # 高斯过滤器的截止频率：2 5 10 20 50 ，越小越模糊信息越少
    img = cv2.imread('butterfly.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img.shape)
    transf = transforms.ToTensor()
    img_tensor = transf(img)
    # ori = transform_convert(img_tensor, ToTensor_transform)

    # unloader = transforms.ToPILImage()
    # image = img_tensor.cpu().clone()
    # image = image.squeeze(0)
    # image = unloader(image)
    # image.save('ori.png')
    x_low, x_high = compute_fft(img_tensor)
    vutils.save_image(x_high, "phigh.png")
    vutils.save_image(x_low, "plow.png")
    # ToTensor_transform = transforms.Compose([transforms.ToTensor()])
    # low = transform_convert(x_low, ToTensor_transform)
    # high = transform_convert(x_high, ToTensor_transform)
    # cv2.imwrite("high.png", high)
    # plt.imshow(low)
    # plt.savefig('./lenalow.png')
    # plt.imshow(high)
    # plt.savefig('./lenahigh.png')




import torch
import torch.nn as nn
import torch.nn.functional as F
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY
import numbers
from einops import rearrange


class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d1, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_div = 5
        g = inp_channels // self.n_div

        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)

        self.weight[:, 0*g:1*g, 0, 1] = 1.0
        self.weight[:, 1*g:2*g, 1, 2] = 1.0
        self.weight[:, 2*g:3*g, 1, 1] = 1.0
        self.weight[:, 3*g:4*g, 2, 1] = 1.0
        self.weight[:, 4 * g:, 1, 1] = 1.0

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        out = self.conv1x1(y)
        return out


class LFE1(nn.Module):
    def __init__(self, in_channels, out_channels, exp_ratio=4, act_type='relu'):
        super(LFE1, self).__init__()
        self.exp_ratio = exp_ratio
        self.act_type = act_type
        self.conv0 = ShiftConv2d1(in_channels, out_channels*exp_ratio)
        self.conv1 = ShiftConv2d1(out_channels*exp_ratio, out_channels)
        if self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = None

    def forward(self, x):
        y = self.conv0(x)
        y = self.act(y)
        y = self.conv1(y)
        return y


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.bsconv = BSConvU(dim, hidden_features*2, kernel_size=3,  stride=1, padding=1, bias=bias)


        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x1, x2 = self.bsconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = BSConvU(dim, dim*3,3,1,1,1,bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        # qkv = self.qkv_dwconv(self.qkv(x))
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# shift channel attention block
class SCAB(nn.Module):
    def __init__(self, in_channels, out_channels, layernorm_type, ffn_factor=2.66, bias=False):
        super(SCAB, self).__init__()
        self.norm1 = LayerNorm(in_channels, layernorm_type)
        self.lfe = LFE1(in_channels, out_channels)
        # self.ca = ChannelAttention(out_channels, 8)
        self.ca = CCALayer(out_channels, 16)
        self.conv = nn.Conv2d(out_channels,out_channels, 1)
        self.norm2 = LayerNorm(out_channels,layernorm_type)
        self.ffn = FeedForward(out_channels,ffn_factor,bias)

    def forward(self, x):
        x_shortcut = x
        x = self.norm1(x)
        x1 = self.lfe(x)
        x2 = self.ca(x1)
        x1 = x1*x2
        x1 = self.conv(x1)
        y = x1+x_shortcut
        out = y + self.ffn(self.norm2(y))
        return out


# blueprint self-attention block
class BSAB(nn.Module):
    def __init__(self, dim, num_heads, layernorm_type, ffn_factor=2.66, bias=False):
        super(BSAB, self).__init__()
        self.norm1 = LayerNorm(dim, layernorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layernorm_type)
        self.ffn = FeedForward(dim, ffn_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


# hybrid attention transformer block
class HATB(nn.Module):
    def __init__(self, dim, num_heads, layernorm_type, ffn_factor=2.66, bias=False):
        super(HATB, self).__init__()
        self.scab = SCAB(dim, dim, layernorm_type, ffn_factor, bias)
        self.bsab = BSAB(dim, num_heads, layernorm_type, ffn_factor, bias)

    def forward(self, x):
        x = self.scab(x)
        y = self.bsab(x)
        return y


# residual attention transformer group
class RATG(nn.Module):
    def __init__(self, dim, num_heads, layernorm_type, ffn_factor=2.66, bias=False):
        super(RATG, self).__init__()
        group = []
        for _ in range(2):
            group.append(HATB(dim, num_heads,layernorm_type, ffn_factor, bias))
        self.group = nn.Sequential(*group)
        self.endconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.group(x)
        y = self.endconv(y)
        return y+x





@ARCH_REGISTRY.register()
class BHAN(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48, num_blocks=4,
                 heads = [1,2,4,8], ffn_factor = 2.66, bias = False, LayerNorm_type = 'WithBias',
                 upscale=4):
        super(BHAN, self).__init__()
        self.conv1 = nn.Conv2d(inp_channels,dim,3, padding=1)
        body = []
        for i in range(num_blocks):
            h = heads[i] if i < len(heads) else 8
            body.append(RATG(dim, h, LayerNorm_type, ffn_factor, bias))
        self.body = nn.Sequential(*body)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        m_tail = [
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=dim, num_out_ch=out_channels)
        ]


        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.conv1(x)
        y = self.body(x)
        y = self.conv2(y)+x
        out = self.tail(y)
        return out
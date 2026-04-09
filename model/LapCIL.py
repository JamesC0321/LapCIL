import math
from cupy_layers.aggregation_zeropad import LocalConvolution
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GCA_SA import g_spatial,g_channel_Laplacian,g_channel_linear_Laplacian


def reshape_tensor(input_tensor):
    input_tensor = input_tensor.unsqueeze(dim=0)
    input_tensor = input_tensor.transpose(1,2)

    if len(input_tensor.shape) != 3:
        raise ValueError("The shape of the input tensor must be (b, x, x)")

    b, l, x = input_tensor.shape  #


    w = math.ceil(math.sqrt(x))
    target_len = w * w
    padding_needed = target_len - x

    if padding_needed > 0:

        mean_vals = input_tensor.mean(dim=2, keepdim=True)
        padding_tensor = mean_vals.expand(b, l, padding_needed)
        input_tensor = torch.cat([input_tensor, padding_tensor], dim=2)

    output_tensor = input_tensor.view(b, l, w, w)
    return output_tensor

def restore_tensor(input_tensor, original_shape):

    if len(input_tensor.shape) != 4:

        raise ValueError("The shape of the input tensor must be (b, x, w, w)")

    b, l, _, _ = input_tensor.shape


    restored_tensor = input_tensor.view(b, l, -1)[:,:, :original_shape[0]]


    return restored_tensor.transpose(1,2)
class CotLayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CotLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2 * dim, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1,
                                           padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm2d(dim)

        self.radix =2

    def forward(self, x):
        x_cat = x
        k = self.key_embed(x)  # key
        qk = torch.cat([x, k], dim=1)  # key and query
        b, c, qk_hh, qk_ww = qk.size()

        w = self.embed(qk)  # 1x1  weight
        w = w.view(b, 1, -1, self.kernel_size * self.kernel_size, qk_hh, qk_ww)

        x = self.conv1x1(x)
        x = self.local_conv(x, w)
        x = self.bn(x)

        x = F.hardswish(x)



        out_x = g_channel_Laplacian(x)  # dongtai
        out_x = g_spatial(out_x)
        out_k = g_channel_Laplacian(k)  # jingtai
        out_k = g_spatial(out_k)


        out = out_x + out_k
        out = out + x_cat


        return out.contiguous()



class ChannelAttentionClassifier(nn.Module):
    def __init__(self, num_cls, in_features=1024, device='cpu'):

        super(ChannelAttentionClassifier, self).__init__()
        self.classifier = nn.Linear(in_features, num_cls)
        self.device = device
        self.to(device)

    def forward(self, tAA):



        tAA_att = g_channel_linear_Laplacian(tAA)



        tfeat_avg = torch.mean(tAA_att, dim=2)      # [B, C]
        tfeat_max = torch.max(tAA_att, dim=2).values  # [B, C]

        tfeat_tensor = tfeat_avg + tfeat_max         # [B, C]


        logits = self.classifier(tfeat_tensor)       # [B, num_cls]

        return logits

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1,  act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                  drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion


        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        if stride > 1:  # set stride > 1 if good?
            self.avd = nn.AvgPool2d(3, 2, padding=1)
        else:
            self.avd = None

        self.conv2 = CotLayer(width, kernel_size=3)




        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)


        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = g_spatial(x)

        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        if self.avd is not None:
            x = self.avd(x)

        x = self.conv2(x)

        x = self.conv3(x)
        x = g_spatial(x)

        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)


        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual

        x = self.act3(x)

        return x

def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])

def _make_layer(block,inplanes, planes, blocks, stride=1,downsample=None,dilation=1):
    layers = []
    for i in range(0, blocks):
        downsample = None
        if i == 0:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block.expansion, kernel_size=1,
                stride=stride, dilation=dilation, first_dilation=dilation, norm_layer=nn.BatchNorm2d)
            downsample =  downsample_conv(**down_kwargs)
        layers.append(block(inplanes, planes, stride,downsample,dilation))
        inplanes = planes * block.expansion
    return nn.Sequential(*layers)

class Contextual_intences(nn.Module):
    def __init__(self, block, inplanes, planes, blocks, stride=1, downsample=None, dilation=1):
        super(Contextual_intences, self).__init__()
        self.layers = self._make_layer(block, inplanes, planes, blocks, stride, downsample, dilation)  #  Bottleneck 1024 512 1 1 None 1

    def _make_layer(self, block, inplanes, planes, blocks, stride, downsample, dilation):
        layers = []
        for i in range(0, blocks):
            downsample = None
            if i == 0:
                down_kwargs = dict(
                    in_channels=inplanes, out_channels=planes * block.expansion, kernel_size=1,
                    stride=stride, dilation=dilation, first_dilation=dilation, norm_layer=nn.BatchNorm2d)
                downsample =  downsample_conv(**down_kwargs)
            layers.append(block(inplanes, planes, stride, downsample, dilation))
            inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    CoTNet = Contextual_intences(Bottleneck, 1024, 256, 1,).to('cuda')
    print(CoTNet)
    data = torch.randn(2550,1024).to('cuda')
    tfeat_tensor2d = reshape_tensor(data)
    output = CoTNet(tfeat_tensor2d)
    print(output.shape)

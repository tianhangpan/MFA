import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import re
from timm.models.layers import trunc_normal_, DropPath

from model.network import Conv2d


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = None

    def _initialize_weights(self, conv_std=0.01):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=conv_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(cfg, in_channels=3, norm=False, dilation=False, dropout=False):
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                if norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=False)]
                in_channels = v
        return nn.Sequential(*layers)


class CSRNet(Network):
    def __init__(self, in_channels=3, load_weights=False, norm=False, dropout=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.input_feat = [64]
        self.in_channels = in_channels
        self.frontend_feat = [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.input_end = self.make_layers(self.input_feat, in_channels=in_channels)
        self.frontend = self.make_layers(self.frontend_feat, in_channels=64)
        self.backend = self.make_layers(self.backend_feat, in_channels=512, norm=norm, dilation=True, dropout=dropout)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self._initialize_weights()
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            fs = self.frontend.state_dict()
            ms = mod.state_dict()
            for key in fs:
                w_n = re.findall(r'(\d+)\.(\w+)', key)
                ms_key = 'features.' + str(int(w_n[0][0]) + 2) + '.' + w_n[0][1]
                fs[key] = ms[ms_key]
            try:
                self.frontend.load_state_dict(fs)
            except:
                print('amb model frontend1 did not load pretrained data')
        else:
            print("Don't pre-train on ImageNet")

    def forward(self, x):
        x = self.input_end(x)
        x = self.frontend(x)
        x = self.backend(x)
        out = self.output_layer(x)
        # x = F.interpolate(x, scale_factor=8)
        return out, x


class CSRNetFuse(Network):
    def __init__(self):
        super(CSRNetFuse, self).__init__()
        self.r1 = Conv2d(128, 32, kernel_size=3, NL='prelu', same_padding=True)
        self.r2 = Conv2d(128, 16, kernel_size=5, NL='prelu', same_padding=True)
        self.r3 = Conv2d(128, 8, kernel_size=7, NL='prelu', same_padding=True)
        self.rp = nn.Sequential(Conv2d(32+16+8, 16, kernel_size=7, NL='prelu', same_padding=True),
                                Conv2d(16, 8, kernel_size=5, NL='prelu', same_padding=True),
                                Conv2d(8, 1, kernel_size=3, NL='nrelu', same_padding=True))
        self._initialize_weights()

    def forward(self, res_features, img_features):
        concat_features = torch.cat((img_features, res_features), dim=1)
        rm1 = self.r1(concat_features)
        rm2 = self.r2(concat_features)
        rm3 = self.r3(concat_features)
        final_prediction = self.rp(torch.cat((rm1, rm2, rm3), dim=1))
        return final_prediction


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, stride=1)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_image = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input_image + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., mode=1
                 ):
        super().__init__()

        self.mode = mode
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        if self.mode in [1, 2, 4, 6]:
            for i in range(3):
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
                self.downsample_layers.append(downsample_layer)
        elif self.mode in [3]:
            for i in range(3):
                if i in [1]:
                    downsample_layer = nn.Sequential(
                        LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                        nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                    )
                elif i in [0, 2]:
                    downsample_layer = nn.Sequential(
                        LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                        nn.Conv2d(dims[i], dims[i + 1], kernel_size=1, stride=1),
                    )
                self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        if self.mode in [1, 2]:
            x = self.stages[0](x)
            for i in range(1, 4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
        elif self.mode in [3]:
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
        elif self.mode in [4, 6]:
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
        # return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


class ConvNeXtIndoorR(Network):
    def __init__(self, in_channels=3, mode=1):
        super(ConvNeXtIndoorR, self).__init__()
        self.mode = mode
        self.in_channels = in_channels
        if self.mode in [5, 7, 8, 9] and in_channels != 3:
            self.input_layer = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1)
        else:
            self.input_layer = nn.Conv2d(in_channels=in_channels, out_channels=48 if self.mode == 2 else 64,
                                         kernel_size=3, padding=1)
        if self.mode in [1, 2, 3]:
            self.integrated_layer = nn.Conv2d(in_channels=384 if self.mode == 2 else 512, out_channels=128,
                                              kernel_size=1)
        elif self.mode in [4, 5, 8, 9]:
            self.integrated_layer = nn.Sequential(
                LayerNorm(768, eps=1e-6, data_format='channels_first'),
                nn.ConvTranspose2d(in_channels=768, out_channels=192, kernel_size=2, stride=2),
                nn.GELU(),
                LayerNorm(192, eps=1e-6, data_format='channels_first'),
                nn.ConvTranspose2d(in_channels=192, out_channels=48, kernel_size=2, stride=2),
                nn.GELU(),
            )
        elif self.mode in [6]:
            self.integrated_layer = nn.Sequential(
                LayerNorm(512, eps=1e-6, data_format='channels_first'),
                nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=2, stride=2),
                nn.GELU(),
                LayerNorm(128, eps=1e-6, data_format='channels_first'),
                nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=2, stride=2),
                nn.GELU(),
            )
        elif self.mode in [7]:
            self.integrated_layer = nn.Sequential(
                LayerNorm(768, eps=1e-6, data_format='channels_first'),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_channels=768, out_channels=192, kernel_size=3, padding=1),
                nn.GELU(),
                LayerNorm(192, eps=1e-6, data_format='channels_first'),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_channels=192, out_channels=48, kernel_size=3, padding=1),
                nn.GELU(),
            )
        if self.mode in [1, 2, 3]:
            self.output_layer = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)
        elif self.mode in [4, 5, 7, 8, 9]:
            self.output_layer = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=1)
        elif self.mode in [6]:
            self.output_layer = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self._initialize_weights(conv_std=.02)
        self.convnext = None
        if self.mode == 1:
            self.convnext = ConvNeXt(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], mode=1)
        elif self.mode == 2:
            self.convnext = ConvNeXt(depths=[2, 2, 4, 2], dims=[48, 96, 192, 384], mode=2)
        elif self.mode == 3:
            self.convnext = ConvNeXt(in_chans=in_channels, depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], mode=3)
        elif self.mode == 4:
            self.convnext = ConvNeXt(in_chans=in_channels, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], mode=4)
        elif self.mode in [5, 7]:
            self.convnext = ConvNeXt(in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], mode=4,
                                     num_classes=21841)
            url = 'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth'
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
            self.convnext.load_state_dict(checkpoint['model'])
        elif self.mode == 6:
            self.convnext = ConvNeXt(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], mode=6)
        elif self.mode in [8, 9]:
            self.convnext = ConvNeXt(in_chans=3, depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], mode=4,
                                     num_classes=21841)
            if self.mode in [9]:
                url = 'https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth'
                checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
                self.convnext.load_state_dict(checkpoint['model'])

    def forward(self, x):
        if self.mode in [1, 2] or (self.mode in [5, 7, 8, 9] and self.in_channels == 6):
            x = self.input_layer(x)
        x = self.convnext(x)
        x = self.integrated_layer(x)
        out = self.output_layer(x)
        return out, x


class ConvNeXtFuse(Network):
    def __init__(self, mode=1):
        super(ConvNeXtFuse, self).__init__()
        self.mode = mode
        if self.mode in [1, 2]:
            self.fuse_layer = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        elif self.mode in [3]:
            self.r1 = Conv2d(256, 64, kernel_size=3, NL='prelu', same_padding=True)
            self.r2 = Conv2d(256, 32, kernel_size=5, NL='prelu', same_padding=True)
            self.r3 = Conv2d(256, 16, kernel_size=7, NL='prelu', same_padding=True)
            self.rp = nn.Sequential(Conv2d(64 + 32 + 16, 24, kernel_size=7, NL='prelu', same_padding=True),
                                    Conv2d(24, 8, kernel_size=5, NL='prelu', same_padding=True),
                                    Conv2d(8, 1, kernel_size=3, NL='nrelu', same_padding=True))
        elif self.mode in [4, 5, 7, 8, 9]:
            self.norm = LayerNorm(96, eps=1e-6, data_format='channels_first')
            self.r1 = Conv2d(96, 32, kernel_size=3, NL='prelu', same_padding=True)
            self.r2 = Conv2d(96, 16, kernel_size=5, NL='prelu', same_padding=True)
            self.r3 = Conv2d(96, 8, kernel_size=7, NL='prelu', same_padding=True)
            self.rp = nn.Sequential(
                LayerNorm(56, eps=1e-6, data_format='channels_first'),
                Conv2d(56, 28, kernel_size=7, NL='prelu', same_padding=True),
                Conv2d(28, 8, kernel_size=5, NL='prelu', same_padding=True),
                Conv2d(8, 1, kernel_size=3, NL='nrelu', same_padding=True))
        elif self.mode in [6]:
            self.norm = LayerNorm(64, eps=1e-6, data_format='channels_first')
            self.r1 = Conv2d(64, 32, kernel_size=3, NL='prelu', same_padding=True)
            self.r2 = Conv2d(64, 16, kernel_size=5, NL='prelu', same_padding=True)
            self.r3 = Conv2d(64, 8, kernel_size=7, NL='prelu', same_padding=True)
            self.rp = nn.Sequential(
                LayerNorm(56, eps=1e-6, data_format='channels_first'),
                Conv2d(56, 28, kernel_size=7, NL='prelu', same_padding=True),
                Conv2d(28, 8, kernel_size=5, NL='prelu', same_padding=True),
                Conv2d(8, 1, kernel_size=3, NL='nrelu', same_padding=True))
        self._initialize_weights(conv_std=.02)

    def forward(self, img_features, res_features):
        concat_features = torch.cat((img_features, res_features), dim=1)
        final_prediction = None
        if self.mode in [1, 2]:
            final_prediction = self.fuse_layer(concat_features)
        elif self.mode in [3]:
            rm1 = self.r1(concat_features)
            rm2 = self.r2(concat_features)
            rm3 = self.r3(concat_features)
            final_prediction = self.rp(torch.cat((rm1, rm2, rm3), dim=1))
        elif self.mode in [4, 5, 6, 7, 8, 9]:
            concat_features = self.norm(concat_features)
            rm1 = self.r1(concat_features)
            rm2 = self.r2(concat_features)
            rm3 = self.r3(concat_features)
            final_prediction = self.rp(torch.cat((rm1, rm2, rm3), dim=1))
        return final_prediction


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


if __name__ == '__main__':
    model = ConvNeXtIndoorR(in_channels=6, mode=5)
    a = torch.rand([1, 6, 128, 128])
    b, c = model(a)
    print(b.size())
    print(c.size())


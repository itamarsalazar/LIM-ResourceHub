""" Full assembly of the parts to form the complete network """
import functools
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet_Strohm2020(nn.Module):
    """
    This is a regular UNet where "two convolutional layers with strides 2 and 3 were added"
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_Strohm2020, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.downsampling = nn.Sequential(
        #     nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=(2,1), padding=1, bias=False),
        #     # nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=3, padding=1, bias=False),
        # )
        self.inc = DoubleConvLeaky(n_channels, 64)
        self.down1 = DownLeaky(64, 128)
        self.down2 = DownLeaky(128, 256)
        self.down3 = DownLeaky(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownLeaky(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # self.upsample = nn.Upsample(size=(1600, 256), mode='bilinear', align_corners=True)

    def forward(self, x):
        # print(f"x: {x.size()}")
        # x0 = self.downsampling(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class Wang2020UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    # net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    # def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
    def __init__(self, input_nc, output_nc, num_downs=6, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Wang2020UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class Wang2020UnetDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    # net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Wang2020UnetDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)



class UNet_nair2020(nn.Module):
    """
    A unet architecture with one encoder and 2 decoders
    Encoder: VGG13 like architecture
    """
    def __init__(self, n_channels, n_classes):
        # model input: torch.Size([1, 2, 800, 128])
        super(UNet_nair2020, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1d = Up_v2(1024, 256, mid_channels=256)
        self.up2d = Up_v2(512, 128, mid_channels=128)
        self.up3d = Up_v2(256, 64, mid_channels=64)
        self.up4d = Up_v2(128, 64, mid_channels=64)
        self.outcd = OutConv(64, n_classes)

        self.up1sp = Up_v2(1024, 256, mid_channels=256)
        self.up2sp = Up_v2(512, 128, mid_channels=128)
        self.up3sp = Up_v2(256, 64, mid_channels=64)
        self.up4sp = Up_v2(128, 64, mid_channels=64)
        self.outcsp = OutConv(64, n_classes)

    def forward(self, x):
        # model input: torch.Size([1, 2, 800, 128])
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d = self.up1d(x5, x4)
        d = self.up2d(d, x3)
        d = self.up3d(d, x2)
        d = self.up4d(d, x1)
        logits_D = self.outcd(d)

        sp = self.up1sp(x5, x4)
        sp = self.up2sp(sp, x3)
        sp = self.up3sp(sp, x2)
        sp = self.up4sp(sp, x1)
        logits_Sp = self.outcsp(sp)

        return logits_D, logits_Sp

    # def calculate_receptive_field(self, input_size):
    #     rf = 1
    #     for module in self.modules():
    #         if isinstance(module, nn.Conv2d):
    #             rf += (module.kernel_size[0] - 1) * module.dilation[0]
    #             rf += (module.kernel_size[0] - 1) * module.dilation[0]
    #             rf -= module.padding[0]
    #     return rf

class UNet_nair2020_1deco(nn.Module):
    """
    A unet architecture with one encoder and 1 decoders
    Encoder: VGG13 like architecture
    """
    def __init__(self, n_channels, n_classes):
        super(UNet_nair2020_1deco, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1d = Up_v2(1024, 256, mid_channels=256)
        self.up2d = Up_v2(512, 128, mid_channels=128)
        self.up3d = Up_v2(256, 64, mid_channels=64)
        self.up4d = Up_v2(128, 64, mid_channels=64)
        self.outcd = OutConv(64, n_classes)

        self.up1sp = Up_v2(1024, 256, mid_channels=256)
        self.up2sp = Up_v2(512, 128, mid_channels=128)
        self.up3sp = Up_v2(256, 64, mid_channels=64)
        self.up4sp = Up_v2(128, 64, mid_channels=64)
        self.outcsp = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d = self.up1d(x5, x4)
        d = self.up2d(d, x3)
        d = self.up3d(d, x2)
        d = self.up4d(d, x1)
        logits_D = self.outcd(d)

        return logits_D
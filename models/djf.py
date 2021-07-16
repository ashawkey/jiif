# modified from https://github.com/ZQPei/deep_joint_filter

import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class CNN(BaseNetwork):
    def __init__(self, num_conv=3, c_in=1, channel=[96,48,1], kernel_size=[9,1,5], stride=[1,1,1], padding=[2,2,2]):
        super(CNN, self).__init__()

        layers = []
        for i in range(num_conv):
            layers += [nn.Conv2d(c_in if i == 0 else channel[i-1], channel[i], kernel_size[i], stride[i], padding[i], bias=True)]
            if i != num_conv-1:
                layers += [nn.ReLU(inplace=True)]

        self.feature = nn.Sequential(*layers)

        self.init_weights()

    
    def forward(self, x):
        fmap = self.feature(x)
        return fmap


# called DJFR if residual = True
class DJF(BaseNetwork):
    def __init__(self, init_weights=True, residual=True):
        super().__init__()
        self.residual = residual

        self.cnn_t = CNN(c_in=1, channel=[96, 48, 1])
        self.cnn_g = CNN(c_in=3, channel=[96, 48, 1])
        self.cnn_f = CNN(c_in=2, channel=[64, 32, 1])

        if init_weights:
            self.init_weights()


    def forward(self, data):

        image = data['image']
        depth = data['lr']

        fmap1 = self.cnn_t(depth)
        fmap2 = self.cnn_g(image)

        output = self.cnn_f(torch.cat([fmap1, fmap2], dim=1))

        if self.residual:
            output = output + depth

        return output

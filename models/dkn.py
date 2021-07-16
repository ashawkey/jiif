# modified from https://github.com/cvlab-yonsei/dkn

import torch
import torch.nn.functional as F
import torch.nn as nn

def grid_generator(k, r, n):
    grid_x, grid_y = torch.meshgrid([torch.linspace(k//2, k//2+r-1, steps=r),
                                     torch.linspace(k//2, k//2+r-1, steps=r)])
    grid = torch.stack([grid_x,grid_y],2).view(r,r,2)

    return grid.unsqueeze(0).repeat(n,1,1,1).cuda()


class Kernel_DKN(nn.Module):
    def __init__(self, input_channel, kernel_size):
        super(Kernel_DKN, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, 7)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 2, stride=(2,2))
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 2, stride=(2,2))
        self.conv5 = nn.Conv2d(64, 128, 5)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.conv7 = nn.Conv2d(128, 128, 3)
        
        self.conv_weight = nn.Conv2d(128, kernel_size**2, 1)
        self.conv_offset = nn.Conv2d(128, 2*kernel_size**2, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        offset = self.conv_offset(x)
        weight = torch.sigmoid(self.conv_weight(x))
        
        return weight, offset
    
class DKN(nn.Module):
    def __init__(self, kernel_size, filter_size, residual=True):
        super(DKN, self).__init__()
        self.ImageKernel = Kernel_DKN(input_channel=3, kernel_size=kernel_size)
        self.DepthKernel = Kernel_DKN(input_channel=1, kernel_size=kernel_size)
        self.residual = residual
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        
    def forward(self, data):

        image = data['image']
        depth = data['lr']

        ### DKN assumes depth is normalized to [0, 1] and resized to full resolution.
        b, _, h, w = image.shape
        
        weight, offset = self._shift_and_stitch(image, depth)
        
        k = self.filter_size
        r = self.kernel_size
        hw = h*w
        
        # weighted average
        # (b, 2*r**2, h, w) -> (b*hw, r, r, 2)
        offset = offset.permute(0,2,3,1).contiguous().view(b*hw, r,r, 2)
        # (b, r**2, h, w) -> (b*hw, r**2, 1)
        weight = weight.permute(0,2,3,1).contiguous().view(b*hw, r*r, 1)
        
        # (b*hw, r, r, 2)
        grid = grid_generator(k, r, b*hw)

        coord = grid + offset
        coord = (coord / k * 2) -1
        
        # (b, k**2, hw) -> (b*hw, 1, k, k)
        depth_col = F.unfold(depth, k, padding=k//2).permute(0,2,1).contiguous().view(b*hw, 1, k,k)
        
        # (b*hw, 1, k, k), (b*hw, r, r, 2) => (b*hw, 1, r^2)
        depth_sampled = F.grid_sample(depth_col, coord, align_corners=False).view(b*hw, 1, -1)
        
        # (b*w*h, 1, r^2) x (b*w*h, r^2, 1) => (b, 1, h,w)
        out = torch.bmm(depth_sampled, weight).view(b, 1, h,w)

        if self.residual:
            out += depth

        return out
    
    def _infer(self, image, depth):
         
        imkernel, imoffset = self.ImageKernel(image)
        depthkernel, depthoffset = self.DepthKernel(depth)
        
        weight = imkernel * depthkernel
        offset = imoffset * depthoffset
        
        if self.residual:
            weight -= torch.mean(weight, 1).unsqueeze(1).expand_as(weight)
        else:
            weight /= torch.sum(weight, 1).unsqueeze(1).expand_as(weight)            
        
        return weight, offset
        
    def _shift_and_stitch(self, image, depth):
        
        offset = torch.zeros((image.size(0), 2*self.kernel_size**2, image.size(2), image.size(3)),
                             dtype=image.dtype, layout=image.layout, device=image.device)
        weight = torch.zeros((image.size(0), self.kernel_size**2, image.size(2), image.size(3)),
                             dtype=image.dtype, layout=image.layout, device=image.device)

        for i in range(4):
            for j in range(4):

                m = nn.ZeroPad2d((25-j,22+j,25-i,22+i))        
                m = nn.ZeroPad2d((25-j,22+j,25-i,22+i))        

                img_shift = m(image)
                depth_shift = m(depth)

                w, o = self._infer(img_shift, depth_shift)

                weight[:,:,i::4,j::4] = w
                offset[:,:,i::4,j::4] = o
                
        return weight, offset

    
def resample_data(input, s):
    """
        input: torch.floatTensor (N, C, H, W)
        s: int (resample factor)
    """    
    
    assert( not input.size(2)%s and not input.size(3)%s)
    
    if input.size(1) == 3:
        # bgr2gray (same as opencv conversion matrix)
        input = (0.299 * input[:,2] + 0.587 * input[:,1] + 0.114 * input[:,0]).unsqueeze(1)
        
    out = torch.cat([input[:,:,i::s,j::s] for i in range(s) for j in range(s)], dim=1)

    """
        out: torch.floatTensor (N, s**2, H/s, W/s)
    """
    return out


class Kernel_FDKN(nn.Module):
    def __init__(self, input_channel, kernel_size, factor=4):
        super(Kernel_FDKN, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        
        self.conv_weight = nn.Conv2d(128, kernel_size**2*(factor)**2, 1)
        self.conv_offset = nn.Conv2d(128, 2*kernel_size**2*(factor)**2, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6(x))

        offset = self.conv_offset(x)
        weight = torch.sigmoid(self.conv_weight(x))
        
        return weight, offset


class FDKN(nn.Module):
    def __init__(self, kernel_size, filter_size, residual=True):
        super(FDKN, self).__init__()
        self.factor = 4 # resample factor
        self.ImageKernel = Kernel_FDKN(input_channel=16, kernel_size=kernel_size, factor=self.factor)
        self.DepthKernel = Kernel_FDKN(input_channel=16, kernel_size=kernel_size, factor=self.factor)
        self.residual = residual
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        
    def forward(self, data):
        
        image = data['image']
        depth = data['lr']

        ### DKN assumes depth is normalized to [0, 1] and resized to full resolution.
        b, _, h, w = image.shape
        
        re_im = resample_data(image, self.factor)
        re_dp = resample_data(depth, self.factor)
        
        imkernel, imoffset       = self.ImageKernel(re_im)
        depthkernel, depthoffset = self.DepthKernel(re_dp)
        
        weight = imkernel * depthkernel
        offset = imoffset * depthoffset
        
        ps = nn.PixelShuffle(4)
        weight = ps(weight)
        offset = ps(offset)
        
        if self.residual:
            weight -= torch.mean(weight, 1).unsqueeze(1).expand_as(weight)
        else:
            weight /= torch.sum(weight, 1).unsqueeze(1).expand_as(weight)            
            
        b, h, w = image.size(0), image.size(2), image.size(3)
        k = self.filter_size
        r = self.kernel_size
        hw = h*w
        
        # weighted average
        # (b, 2*r**2, h, w) -> (b*hw, r, r, 2)
        offset = offset.permute(0,2,3,1).contiguous().view(b*hw, r,r, 2)
        # (b, r**2, h, w) -> (b*hw, r**2, 1)
        weight = weight.permute(0,2,3,1).contiguous().view(b*hw, r*r, 1)
        
        # (b*hw, r, r, 2)
        grid = grid_generator(k, r, b*hw)
        coord = grid + offset
        coord = (coord / k * 2) -1
        
        # (b, k**2, hw) -> (b*hw, 1, k, k)
        depth_col = F.unfold(depth, k, padding=k//2).permute(0,2,1).contiguous().view(b*hw, 1, k,k)
        
        # (b*hw, 1, k, k), (b*hw, r, r, 2) => (b*hw, 1, r^2)
        depth_sampled = F.grid_sample(depth_col, coord, align_corners=False).view(b*hw, 1, -1)
        
        # (b*w*h, 1, r^2) x (b*w*h, r^2, 1) => (b, 1, h, w)
        out = torch.bmm(depth_sampled, weight).view(b, 1, h,w)

        if self.residual:
            out += depth
        
        return out
    
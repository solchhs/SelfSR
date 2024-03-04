import os, time
import cv2
import torch
import math
import numpy as np

from models import *
from common_utils import *

class Multi_SR:
    def __init__(self, conf, dataloader):
        # Configuration
        self.conf = conf

        # Dataloader
        self.imgs_np = dataloader.imgs_np
        self.base_np = dataloader.base_np
        self.imgs_bicubic_np = dataloader.imgs_bicubic_np
        self.base_bicubic_np = dataloader.base_bicubic_np
        self.imgs_gray_np = dataloader.imgs_gray_np
        self.base_gray_np = dataloader.base_gray_np
        self.shape = dataloader.shape
        if self.conf.reg_mode == 'gt':
            self.imgs_fname = dataloader.imgs_fname
        if self.conf.reg_mode == 'net':
            self.base_bicubic_np = dataloader.base_bicubic_np

        # Hyperparameter
        self.input_depth = 32
        self.LR = 0.01
        
        self.INPUT = 'noise'
        self.pad = 'reflection'
        self.NET_TYPE = 'skip'
        # self.NET_TYPE = 'UNet'
        self.upsample_mode = 'bilinear'
        # self.upsample_mode = 'deconv'

        self.dtype = torch.cuda.FloatTensor
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

        # CFA
        self.conf.CFA_mask = torch.Tensor()

        if self.conf.cfa:
            if self.conf.cfa_type == 'bayer':
                if self.conf.cfa_order == 'RGGB':
                    pass
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        # Blur
        if self.conf.blur:
            if self.conf.blur_type == 'Gaussian':
                self.blur_tensor = torch.nn.Conv2d(in_channels=self.conf.n_channels, out_channels=self.conf.n_channels, kernel_size=self.conf.ksize, padding=int((self.conf.ksize-1)/2), groups=self.conf.n_channels, bias=False)
                self.blur_tensor.weight.data = self.gaussian_kernel(ksize=self.conf.ksize, sigma=self.conf.sigma)
                self.blur_tensor.weight.requires_grad = False
            elif self.conf.blur_type == 'Custom':
                pass
            else:
                raise ValueError(f'{self.conf.blur_type} is not valid blur type.')

        # Regularizer
        if self.conf.regul:
            if self.conf.regul_type == 'tv':
                self.dh = torch.Tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).view(1, 1, 3, 3).type(self.dtype)
                self.dv = torch.Tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).view(1, 1, 3, 3).type(self.dtype)
            elif self.conf.regul_type == 'btv':
                self.dh = torch.Tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).view(1, 1, 3, 3).type(self.dtype)
                self.dv = torch.Tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).view(1, 1, 3, 3).type(self.dtype)
                self.dhv = torch.Tensor([[0, 0, 0], [0, 1, 0], [0, 0, -1]]).view(1, 1, 3, 3).type(self.dtype)
                self.dvh = torch.Tensor([[0, 0, 0], [0, 1, 0], [-1, 0, 0]]).view(1, 1, 3, 3).type(self.dtype)
            else:
                raise NotImplementedError

        # Loss type
        if self.conf.loss_type == 'L1':
            self.loss = torch.nn.L1Loss().type(self.dtype)
        elif self.conf.loss_type == 'L2':
            self.loss = torch.nn.MSELoss().type(self.dtype)
        else:
            raise NotImplementedError

        self.mse = torch.nn.MSELoss().type(self.dtype)

        self.hr_base_loss = None
        self.hr_imgs_loss = [None] * self.conf.nimgs
        self.hr_img_total_loss = None
        self.lr_nets_loss = [None] * self.conf.nimgs

        self.best_hr_loss = 10e8

        self.init_img()
        self.init_reg()
        self.init_net()

    def init_img(self):
        self.base_tensor = np_to_torch(self.base_np).cuda()
        self.base_bicubic_tensor = np_to_torch(self.base_bicubic_np).cuda()
        
        self.imgs_tensor = []
        for img_np in self.imgs_np:
            img_tensor = np_to_torch(img_np).cuda()
            self.imgs_tensor.append(img_tensor)
        
        self.imgs_bicubic_tensor = []
        for img_bicubic_np in self.imgs_bicubic_np:
            img_bicubic_tensor = np_to_torch(img_bicubic_np).cuda()
            self.imgs_bicubic_tensor.append(img_bicubic_tensor)

        # if self.conf.reg_mode == 'net':
        #     self.base_bicubic_tensor = np_to_torch(self.base_bicubic_np).cuda()

    def init_reg(self):
        self.regs = np.zeros((self.conf.nimgs, 2))

        if self.conf.reg_mode == 'optflow':
            if self.conf.optflow_type == 'LK':
                raise NotImplementedError
            elif self.conf.optflow_type == 'FB':
                for idx, img in enumerate(self.imgs_gray_np):
                    # Farneback optical flow estimation
                    flow = cv2.calcOpticalFlowFarneback(prev=img, \
                        next=self.base_gray_np,
                        flow=None,
                        pyr_scale=0.5,
                        levels=3,
                        winsize=21,
                        iterations=3,
                        poly_n=5,
                        poly_sigma=1.1,
                        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

                    # Extract good feature points to track
                    goodFeaturesPts = cv2.goodFeaturesToTrack(image=self.base_gray_np, \
                        maxCorners=max(int(self.base_gray_np.size * 0.01), 10),
                        qualityLevel=0.1,
                        minDistance=1,
                        blockSize=3,
                        useHarrisDetector=True, k=0.04)
                    
                    u, v = 0., 0.
                    for item in goodFeaturesPts.astype(np.int_):
                        u = u + flow[item[0, 0], item[0, 1], 0] * self.conf.sf
                        v = v + flow[item[0, 0], item[0, 1], 1] * self.conf.sf
                    u, v = u / len(goodFeaturesPts), v / len(goodFeaturesPts)

                    self.regs[idx, 0], self.regs[idx, 1] = v, u
            else:
                raise ValueError(f'{self.conf.optflow_type} is not valid optical flow estimation method.')
        elif self.conf.reg_mode == 'net':
            pass
        elif self.conf.reg_mode == 'gt':
            for idx, fname in enumerate(self.imgs_fname):
                reg = fname.split('(')[-1].split(')')[0].split(',')
                self.regs[idx, 0], self.regs[idx, 1] = reg[0], reg[1]
        else:
            raise ValueError(f'{self.conf.reg_mode} is not valid sub-pixel registration type.')

    def init_net(self):
        self.hr_net_input = get_noise(self.input_depth, \
            self.INPUT,
            (self.shape[1] * self.conf.sf, self.shape[2] * self.conf.sf)).type(self.dtype).detach()

        self.hr_net = get_net(self.input_depth, \
            self.NET_TYPE,
            self.pad,
            skip_n33d=128,
            skip_n33u=128,
            skip_n11=4,
            num_scales=5,
            upsample_mode=self.upsample_mode,
            n_channels=self.conf.n_channels).type(self.dtype)

        self.best_hr_result = torch.tensor(self.hr_net(self.hr_net_input))

        self.hr_optimizer = torch.optim.Adam(self.hr_net.parameters(), lr=self.LR)
        
        if self.conf.reg_mode == 'net':
            self.lr_nets = []
            self.lr_optimizers = []
            for i in range(self.conf.nimgs):
                lr_net = get_net(self.conf.n_channels, \
                    self.NET_TYPE,
                    self.pad,
                    skip_n33d=32,
                    skip_n33u=32,
                    skip_n11=4,
                    num_scales=5,
                    upsample_mode=self.upsample_mode,
                    n_channels=self.conf.n_channels).type(self.dtype)
                
                lr_optimizer = torch.optim.Adam(lr_net.parameters(), lr=self.LR)

                self.lr_nets.append(lr_net)
                self.lr_optimizers.append(lr_optimizer)

    def train_img(self):
        self.hr_optimizer.zero_grad()
        hr_net_out = self.hr_net(self.hr_net_input)

        if self.conf.reg_mode == 'net':
            # Base frame
            base_tensor = self.base_tensor.detach()
            base_hr_net_out = self.downsample_tensor(hr_net_out, self.conf.sf)
            self.hr_base_loss = self.loss(base_hr_net_out, base_tensor)

            # LR frames
            for i in range(self.conf.nimgs):
                self.lr_optimizers[i].zero_grad()

                img_tensor = self.imgs_tensor[i].detach()
                lr_net = self.lr_nets[i]

                # M matrix (sub-pixel shift)
                img_hr_net_out = lr_net(hr_net_out)

                # B matrix (blur)
                if self.conf.blur:
                    img_hr_net_out = self.blur_tensor(img_hr_net_out)

                # D matrix (down-sampling)
                img_lr_net_out = self.downsample_tensor(img_hr_net_out, self.conf.sf)
                self.hr_imgs_loss[i] = self.loss(img_lr_net_out, img_tensor)

            if self.conf.regul:
                if self.conf.regul_type == 'tv':
                    self.hr_img_total_loss = sum(self.hr_imgs_loss) + self.hr_base_loss + self.tv_loss(hr_net_out, lamb=self.conf.lamb, p=self.conf.p)
                elif self.conf.regul_type == 'btv':
                    self.hr_img_total_loss = sum(self.hr_imgs_loss) + self.hr_base_loss + self.btv_loss(hr_net_out, lamb=self.conf.lamb, p=self.conf.p)
                else:
                    raise NotImplementedError
            else:
                # Sharpness Loss
                # self.hr_img_total_loss = sum(self.hr_imgs_loss) + self.hr_base_loss + self.sharpness_loss(hr_net_out, lamb=self.conf.lamb, thresh=self.conf.thresh, wsize=self.conf.wsize)
                self.hr_img_total_loss = sum(self.hr_imgs_loss) + self.hr_base_loss
            self.recorder(hr_net_out)
            self.hr_img_total_loss.backward()
            self.hr_optimizer.step()
            
            for i in range(self.conf.nimgs):
                self.lr_optimizers[i].step()

    def train_reg(self, mode='LR'):
        if mode == 'LR':
            for i in range(self.conf.nimgs):
                self.lr_optimizers[i].zero_grad()
                lr_net_out = self.lr_nets[i](self.base_tensor)
                self.lr_nets_loss[i] = self.mse(lr_net_out, self.imgs_tensor[i])

                self.lr_nets_loss[i].backward()
                self.lr_optimizers[i].step()
        elif mode == 'HR':
            for i in range(self.conf.nimgs):
                self.lr_optimizers[i].zero_grad()
                lr_net_out = self.lr_nets[i](self.base_bicubic_tensor)
                lr_net_out = self.downsample_tensor(input_tensor=lr_net_out, sf=self.conf.sf)
                self.lr_nets_loss[i] = self.mse(lr_net_out, self.imgs_tensor[i])

                self.lr_nets_loss[i].backward()
                self.lr_optimizers[i].step()

    def freeze_reg_net(self):
        for i in range(self.conf.nimgs):
            for layer in self.lr_nets[i]:
                for p in layer.parameters():
                    p.requires_grad = False

    def downsample_tensor(self, input_tensor, sf):
        return torch.nn.functional.interpolate(input=input_tensor, scale_factor=1.0/sf, mode='nearest')

    def gaussian_kernel(self, ksize, sigma):
        x_cord = torch.arange(ksize).type(self.dtype)
        x_grid = x_cord.repeat(ksize).view(ksize, ksize)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (ksize - 1)/2.
        var = sigma**2.

        kernel = (1./(2.*math.pi*var)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*var))
        kernel = kernel / torch.sum(kernel)

        kernel = kernel.view(1, 1, ksize, ksize)
        kernel = kernel.repeat(self.conf.n_channels, 1, 1, 1)
        return kernel

    def tv_loss(self, img_tensor, lamb=0.01, p=0.5):
        tv_h = torch.nn.functional.conv2d(input=img_tensor, weight=self.dh)
        tv_v = torch.nn.functional.conv2d(input=img_tensor, weight=self.dv)
        tv = torch.pow(torch.pow(tv_h, 2.) + torch.pow(tv_v, 2.) + 1e-9, p)
        bs_img, c_img, h_img, w_img = tv.size()
        return lamb * tv.sum() / (bs_img * c_img * h_img * w_img)
    
    def btv_loss(self, img_tensor, lamb=0.01, p=0.5):
        if img_tensor.shape[1] == 1:
            tv_h = torch.nn.functional.conv2d(input=img_tensor, weight=self.dh)
            tv_v = torch.nn.functional.conv2d(input=img_tensor, weight=self.dv)
            tv_hv = torch.nn.functional.conv2d(input=img_tensor, weight=self.dhv)
            tv_vh = torch.nn.functional.conv2d(input=img_tensor, weight=self.dvh)
            tv = torch.pow(torch.pow(tv_h, 2.) + torch.pow(tv_v, 2.) + torch.pow(tv_hv, 2.) + torch.pow(tv_vh, 2.) + 1e-9, p)
        else:
            tv = 0
            for i in range(img_tensor.shape[1]):
                tv_h = torch.nn.functional.conv2d(input=img_tensor[:, i:i+1, :, :], weight=self.dh)
                tv_v = torch.nn.functional.conv2d(input=img_tensor[:, i:i+1, :, :], weight=self.dv)
                tv_hv = torch.nn.functional.conv2d(input=img_tensor[:, i:i+1, :, :], weight=self.dhv)
                tv_vh = torch.nn.functional.conv2d(input=img_tensor[:, i:i+1, :, :], weight=self.dvh)
                tv = tv + torch.pow(torch.pow(tv_h, 2.) + torch.pow(tv_v, 2.) + torch.pow(tv_hv, 2.) + torch.pow(tv_vh, 2.) + 1e-9, p)
        bs_img, c_img, h_img, w_img = tv.size()
        return lamb * tv.sum() / (bs_img * c_img * h_img * w_img)

    def sharpness_loss(self, img_tensor, lamb=0.01, thresh=0.1, wsize=3):
        # Sharpness Loss Function
        if (wsize % 2) == 0:
            wsize = wsize - 1

        out = torch.Tensor([0.]).type(self.dtype)

        for i in range(wsize):
            for j in range(wsize):
                if not ((i == wsize//2) and (j == wsize//2)):
                    weight = torch.zeros(wsize, wsize).view(1, 1, wsize, wsize).type(self.dtype)
                    weight[:, :, wsize//2, wsize//2] = 1.
                    weight[:, :, i, j] = -1.

                    # Manhattan Distance
                    dist = (abs(i - wsize//2) + abs(j - wsize//2))

                    # Compute sharpness
                    sharpness = torch.exp(torch.pow(torch.nn.functional.conv2d(input=img_tensor, weight=weight), 2.) * (-0.5 * thresh * thresh))
                    bs_img, c_img, h_img, w_img = sharpness.size()
                    out = out + lamb * sharpness.sum() / (bs_img * c_img * h_img * w_img * dist)

        return out

    def recorder(self, hr_net_out):
        if self.best_hr_loss > self.hr_img_total_loss:
            self.best_hr_result = torch.tensor(hr_net_out)
            self.best_hr_loss = torch.tensor(self.hr_img_total_loss)

    def cubic_coefficient(self, reg):
        c1 = -(1 + reg) * (1 + reg) * (1 + reg) + 5 * (1 + reg) * (1 + reg) - 8 * (1 + reg) + 4
        c2 = reg * reg * reg - 2 * reg * reg + 1
        c3 = (1 - reg) * (1 - reg) * (1 - reg) - 2 * (1 - reg) * (1 - reg) + 1
        c4 = -(2 - reg) * (2 - reg) * (2 - reg) + 5 * (2 - reg) * (2 - reg) - 8 * (2 - reg) + 4
        return c1, c2, c3, c4

    def cubic_interpolation(self, input_tensor, reg, sf=1, eps=1e-8, local=False):
        if local:
            output_tensor = torch.zeros([input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2] // self.conf.sf, input_tensor.shape[3] // self.conf.sf]).cuda()
            input_tensor_unfold = input_tensor.unfold(2, self.conf.block_size * self.conf.sf, self.conf.block_size * self.conf.sf).unfold(3, self.conf.block_size * self.conf.sf, self.conf.block_size * self.conf.sf)

            for i1 in range(input_tensor_unfold.shape[2]):
                for i2 in range(input_tensor_unfold.shape[3]):
                    block = input_tensor_unfold[:, :, i1, i2]
                    output_block = self.cubic_interpolation(block, reg[i1, i2], sf=self.conf.sf)
                    output_tensor[:, :, i1*self.conf.block_size:(i1+1)*self.conf.block_size, i2*self.conf.block_size:(i2+1)*self.conf.block_size] = output_block

            return output_tensor
        else:
            input_tensor2 = input_tensor.view(input_tensor.shape[1], input_tensor.shape[0], input_tensor.shape[2], input_tensor.shape[3])

            reg_h = reg[0].item()
            reg_v = reg[1].item()
            if reg_h < 0:
                reg_h = 0.
            if reg_v < 0:
                reg_v = 0.
            if reg_h > (sf - 1):
                reg_h = sf - eps
            if reg_v > (sf - 1):
                reg_v = sf - eps

            h_d = reg_h - np.floor(reg_h)
            v_d = reg_v - np.floor(reg_v)

            h1, h2, h3, h4 = self.cubic_coefficient(h_d)
            v1, v2, v3, v4 = self.cubic_coefficient(v_d)
            h = torch.Tensor([h1, h2, h3, h4]).view(1, 1, 1, 4).type(torch.FloatTensor).cuda()
            v = torch.Tensor([v1, v2, v3, v4]).view(1, 1, 4, 1).type(torch.FloatTensor).cuda()
            hh = torch.nn.functional.conv2d(input_tensor2, h, padding=(0,3))
            vv = torch.nn.functional.conv2d(hh, v, padding=(3,0))

            output_tensor = vv[:, :, 3+int(np.floor(reg_h))::int(sf), 3+int(np.floor(reg_v))::int(sf)]
        
        return output_tensor.view(output_tensor.shape[1], output_tensor.shape[0], output_tensor.shape[2], output_tensor.shape[3])

    def make_directory(self):
        os.makedirs(self.conf.output_dir, exist_ok=True)
        now = time.localtime()
        self.output_dir = "log\\%04d_%02d_%02d_%02d_%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
        os.makedirs(self.output_dir, exist_ok=True)

    def save_output(self, iter):
        output = self.best_hr_result.detach()
        output = np.clip(torch_to_np(output), 0, 1).transpose(1, 2, 0)
        if self.conf.n_channels == 1:
            output = np.squeeze(output, axis=2)
        else:
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.output_dir, f'output_it{iter:04d}.bmp'), output * 255.0)

    def finish(self):
        filename = self.conf.imgs_dir.split('\\')[-1].split('/')[-1]
        if self.conf.cfa:
            filename = filename + f'_{self.conf.cfa_type}'
        filename = filename + f'_{self.conf.reg_mode}'
        if self.conf.reg_mode == 'optflow':
            filename = filename + f'_{self.conf.optflow_type}'
        if self.conf.blur:
            filename = filename + '_blur'
        if self.conf.regul:
            filename = filename + f'_{self.conf.regul_type}_{self.conf.lamb}'
        filename = filename + f'_x{self.conf.sf}_nimgs{self.conf.nimgs}'
        ext = '.bmp'

        output = self.best_hr_result.detach()
        output = np.clip(torch_to_np(output), 0, 1).transpose(1, 2, 0)
        if self.conf.n_channels == 1:
            output = np.squeeze(output, axis=2)
        else:
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        os.makedirs(self.conf.output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.conf.output_dir, f'{filename}{ext}'), output * 255.0)

        if (self.conf.reg_mode == 'net'):
            for i in range(self.conf.nimgs):
                lr_net = self.lr_nets[i]
                lr_net_out = lr_net(self.best_hr_result)

                output = np.clip(torch_to_np(lr_net_out.detach()), 0, 1).transpose(1, 2, 0)
                if self.conf.n_channels == 1:
                    output = np.squeeze(output, axis=2)
                else:
                    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.output_dir, f'{filename}_HR{i:02d}{ext}'), output * 255.0)

                output = self.downsample_tensor(lr_net_out, self.conf.sf).detach()
                output = np.clip(torch_to_np(output), 0, 1).transpose(1, 2, 0)
                if self.conf.n_channels == 1:
                    output = np.squeeze(output, axis=2)
                else:
                    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                
                lr_net_out = lr_net(self.base_tensor)
                output = np.clip(torch_to_np(lr_net_out), 0, 1).transpose(1, 2, 0)
                cv2.imwrite(os.path.join(self.output_dir, f'{filename}_LR{i:02d}{ext}'), output * 255.0)

        if self.conf.gendata:
            print("here")
            # Data generation mode
            hr_filename = f'gendata\\HR.bmp'
            hr_np = np.expand_dims(cv2.imread(hr_filename, cv2.IMREAD_GRAYSCALE), axis=2).transpose(2, 0, 1).astype(np.float32) / 255.0
            hr_tensor = np_to_torch(hr_np).cuda()

            for i in range(self.conf.nimgs):
                lr_net = self.lr_nets[i]
                lr_net_out = lr_net(hr_tensor)
                blur = self.blur_tensor(lr_net_out)
                downampled = self.downsample_tensor(blur, self.conf.sf).detach()
                output = np.clip(torch_to_np(downampled), 0, 1).transpose(1, 2, 0)
                cv2.imwrite(f'gendata\\LR{i:02d}{ext}', output * 255.0)
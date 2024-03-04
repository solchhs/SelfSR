import argparse
import torch
import os

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None

        # Directory
        self.parser.add_argument('--imgs_dir', type=str, default='input', help='path to input images directory')
        self.parser.add_argument('--output_dir', type=str, default='output', help='path to output image directory')
        self.parser.add_argument('--basename', type=str, default='base.bmp', help='base frame filename')

        # CFA
        self.parser.add_argument('--cfa', action='store_true', help='color filter array on/off')
        self.parser.add_argument('--cfa_type', type=str, default='bayer', help='color filter array type')
        self.parser.add_argument('--cfa_order', type=str, default='RGGB', help='color filter array order')

        # Sub-pixel registration
        self.parser.add_argument('--reg_mode', type=str, default='net', help='sub-pixel registration mode (gt/net/optflow)')
        self.parser.add_argument('--optflow_type', type=str, default='FB', help='optical flow estimation methon(LK(lucas-kanade)/FB(Farneback))')
        self.parser.add_argument('--alpha', type=float, default=1.0, help='weight for sub-pixel registration loss')

        # Blur
        self.parser.add_argument('--blur', action='store_true', help='blur on/off')
        self.parser.add_argument('--blur_type', type=str, default='Gaussian', help='blur type(Gaussian/Custom)')
        self.parser.add_argument('--blur_dir', type=str, default='kernel.mat', help='path to custom blur directory')
        self.parser.add_argument('--ksize', type=int, default=7, help='Gaussian blur size')
        self.parser.add_argument('--sigma', type=float, default=1.0, help='Gaussian blur std')

        # Regularization
        self.parser.add_argument('--loss_type', type=str, default='L2', help='loss function type (L1/L2)')
        self.parser.add_argument('--regul', action='store_true', help='regularization on/off')
        self.parser.add_argument('--regul_type', type=str, default='btv', help='regularization type(tv/btv)')
        self.parser.add_argument('--lamb', type=float, default='0.005', help='lambda for tv loss')
        self.parser.add_argument('--p', type=float, default='0.5', help='order of tv loss')
        self.parser.add_argument('--sharp', action='store_true', help='sharpness regularization on/off')
        self.parser.add_argument('--thresh', type=float, default='0.01', help='threshold for sharpness loss')
        self.parser.add_argument('--wsize', type=int, default='3', help='threshold for sharpness loss')

        # Application
        self.parser.add_argument('--gendata', action='store_true', help='Data generation mode on/off')

        # Parameter
        self.parser.add_argument('--sf', type=int, default=2, help='scale factor')
        self.parser.add_argument('--nimgs', type=int, default=4, help='number of images')
        self.parser.add_argument('--reg_iters', type=int, default=1000, help='number of iterations for registration')
        self.parser.add_argument('--img_iters', type=int, default=1500, help='number of iterations for image reconstruction')
        self.parser.add_argument('--n_channels', type=int, default=1, help='number of channels')

        # GPU
        self.parser.add_argument('--gpu_id', type=int, default=0, help='gpu id number')

    def parse(self, args=None):
        self.conf = self.parser.parse_args(args=args)
        self.set_gpu_device()
        return self.conf
        
    def set_gpu_device(self):
        if os.environ.get('CUDA_VISIBLE_DEVICES', '') == '':
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.conf.gpu_id)
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(self.conf.gpu_id)
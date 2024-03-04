import warnings
warnings.filterwarnings(action='ignore')

import tqdm

from config import Config
from dataloader import Dataloader
from multi_sr import Multi_SR

def train(conf):
    dataloader = Dataloader(conf)
    multi_sr = Multi_SR(conf, dataloader)
    multi_sr.make_directory()
    
    # Registration
    # Backup
    # print('Registration...')
    # pbar = tqdm.tqdm(range(conf.reg_iters), ncols=100)
    # for i in pbar:
    #     multi_sr.train_reg()
    
    print('Registration in LR grid...')
    pbar = tqdm.tqdm(range(conf.reg_iters), ncols=100)
    for i in pbar:
        multi_sr.train_reg(mode='LR')

    print('Registration in HR grid...')
    pbar = tqdm.tqdm(range(conf.reg_iters), ncols=100)
    for i in pbar:
        multi_sr.train_reg(mode='HR')

    # Image Reconstruction
    print('Image Reconstruction...')
    pbar = tqdm.tqdm(range(conf.img_iters), ncols=100)
    
    for i in pbar:
        multi_sr.train_img()

        # if (i % 100 == 0):
        if (i % 100 == 0) or (i < 100):
            multi_sr.save_output(iter=i)

    multi_sr.finish()

def main():
    import argparse
    # Parse the command line arugments
    
    prog = argparse.ArgumentParser()

    # Directory
    prog.add_argument('--imgs-dir', '-i', type=str, default='input', help='path to input images directory')
    prog.add_argument('--output-dir', '-o', type=str, default='output', help='path to output image directory')
    prog.add_argument('--basename', '-b', type=str, default='base.bmp', help='base frame filename')

    # CFA
    prog.add_argument('--cfa', action='store_true', help='color filter array on/off')
    prog.add_argument('--cfa-type', type=str, default='bayer', help='color filter array type')
    prog.add_argument('--cfa-order', type=str, default='RGGB', help='color filter array order')

    # Sub-pixel registration
    prog.add_argument('--reg-mode', type=str, default='net', help='sub-pixel registration mode (gt/net/optflow)')
    prog.add_argument('--optflow-type', type=str, default='FB', help='optical flow estimation methon(LK(lucas-kanade)/FB(Farneback))')
    prog.add_argument('--alpha', type=float, default=1.0, help='weight for sub-pixel registration loss')

    # Blur
    prog.add_argument('--blur', action='store_true', help='blur on/off')
    prog.add_argument('--blur-type', type=str, default='Gaussian', help='blur type(Gaussian/Custom)')
    prog.add_argument('--blur-dir', type=str, default='kernel.mat', help='path to custom blur directory')
    prog.add_argument('--ksize', type=int, default=7, help='Gaussian blur size')
    prog.add_argument('--sigma', type=float, default=1.0, help='Gaussian blur std')

    # Loss
    prog.add_argument('--loss-type', type=str, default='L2', help='loss function type (L1/L2)')
    prog.add_argument('--regul', action='store_true', help='regularization on/off')
    prog.add_argument('--regul-type', type=str, default='btv', help='regularization type(tv/btv)')
    prog.add_argument('--lamb', type=float, default='0.005', help='lambda for tv loss')
    prog.add_argument('--p', type=float, default='0.5', help='order of tv loss')
    prog.add_argument('--sharp', action='store_true', help='sharpness regularization on/off')
    prog.add_argument('--thresh', type=float, default='0.01', help='threshold for sharpness loss')
    prog.add_argument('--wsize', type=int, default='3', help='threshold for sharpness loss')

    # Application
    prog.add_argument('--gendata', action='store_true', help='Data generation mode on/off')

    # Parameter
    prog.add_argument('--sf', type=int, default=2, help='scale factor')
    prog.add_argument('--nimgs', type=int, default=4, help='number of images')
    prog.add_argument('--reg-iters', type=int, default=1000, help='number of iterations for registration')
    prog.add_argument('--img-iters', type=int, default=1500, help='number of iterations for image reconstruction')
    prog.add_argument('--n_channels', type=int, default=1, help='number of channels')

    # GPU
    prog.add_argument('--gpu-id', type=int, default=0, help='gpu id number')

    args = prog.parse_args()    
    conf = Config().parse(create_params(args))
    train(conf)

    prog.exit(0)

def create_params(args):
    assert args.sf * args.sf >= args.nimgs, 'exceeded number of images error.'
    params = ['--imgs_dir', args.imgs_dir,
              '--output_dir', args.output_dir,
              '--basename', args.basename,
              '--cfa_type', args.cfa_type,
              '--cfa_order', args.cfa_order,
              '--reg_mode', args.reg_mode,
              '--optflow_type', args.optflow_type,
              '--alpha', str(args.alpha),
              '--blur_type', args.blur_type,
              '--blur_dir', args.blur_dir,
              '--ksize', str(args.ksize),
              '--sigma', str(args.sigma),
              '--loss_type', args.loss_type,
              '--regul_type', args.regul_type,
              '--lamb', str(args.lamb),
              '--p', str(args.p),
              '--thresh', str(args.thresh),
              '--wsize', str(args.wsize),
              '--sf', str(args.sf),
              '--nimgs', str(args.nimgs),
              '--reg_iters', str(args.reg_iters),
              '--img_iters', str(args.img_iters),
              '--n_channels', str(args.n_channels),
              '--gpu_id', str(args.gpu_id)]
    if args.cfa:
        params.append('--cfa')
    if args.blur:
        params.append('--blur')
    if args.regul:
        params.append('--regul')
    if args.sharp:
        params.append('--sharp')
    if args.gendata:
        params.append('--gendata')
    return params

if __name__ == '__main__':
    main()
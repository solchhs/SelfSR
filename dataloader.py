import os, sys
import glob
import cv2

from random import sample

from common_utils import *

class Dataloader:
    def __init__(self, conf):
        # Configuration
        self.imgs_dir = conf.imgs_dir
        self.basename = conf.basename
        self.cfa = conf.cfa
        self.cfa_type = conf.cfa_type
        self.cfa_order = conf.cfa_order
        self.reg_mode = conf.reg_mode
        self.sf = conf.sf
        self.nimgs = conf.nimgs
        self.n_channels = conf.n_channels

        self.imgidxs = sorted(sample([i for i in range(self.nimgs)], self.nimgs))

        print("Image # :", self.imgidxs)

        self.imgs_np = None
        self.imgs_gray_np = None
        self.base_np = None
        self.base_gray_np = None

        self.load_dataset()

    def load_dataset(self):
        imgs_orig_np = []
        imgs_orig_np_gray = []
        shape = None

        if self.reg_mode == 'gt':
            imgs_fname = []
        if self.reg_mode == 'net':
            imgs_bicubic_np = []

        for idx, fname in enumerate(glob.glob(self.imgs_dir + "/LR_*.bmp")):
            if idx in self.imgidxs:
                if self.n_channels == 1:
                    img_orig_np = np.expand_dims(cv2.imread(fname, cv2.IMREAD_GRAYSCALE), axis=2).transpose(2, 0, 1).astype(np.float32) / 255.0
                    img_bicubic_np = np.expand_dims(cv2.resize(cv2.imread(fname, cv2.IMREAD_GRAYSCALE), dsize=None, fx=self.sf, fy=self.sf), axis=2).transpose(2, 0, 1).astype(np.float32) / 255.0
                    img_orig_np_gray = cv2.imread(fname)
                else:
                    img_orig_np = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
                    img_bicubic_np = cv2.resize(cv2.imread(fname), dsize=None, fx=self.sf, fy=self.sf).transpose(2, 0, 1).astype(np.float32) / 255.0
                    img_orig_np_gray = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2GRAY)

                if shape == None:
                    shape = img_orig_np.shape
                else:
                    assert shape == img_orig_np.shape, "input image size error"

                imgs_orig_np.append(img_orig_np)
                imgs_bicubic_np.append(img_bicubic_np)
                imgs_orig_np_gray.append(img_orig_np_gray)
                
                if self.reg_mode == 'gt':
                    imgs_fname.append(fname)
        
        print('Scale Factor : %s, # of imgs : %s, Shape : %s' % (str(self.sf), str(self.nimgs), f'{shape[1]} x {shape[2]} x {shape[0]}'))

        self.imgs_np = imgs_orig_np
        self.imgs_gray_np = imgs_orig_np_gray
        if self.reg_mode == 'gt':
            self.imgs_fname = imgs_fname
        if self.reg_mode == 'net':
            self.imgs_bicubic_np = imgs_bicubic_np

        base_fname = os.path.join(self.imgs_dir, self.basename)

        if self.n_channels == 1:
            img_orig_np = np.expand_dims(cv2.imread(base_fname, cv2.IMREAD_GRAYSCALE), axis=2).transpose(2, 0, 1).astype(np.float32) / 255.0
            img_bicubic_np = np.expand_dims(cv2.resize(cv2.imread(base_fname, cv2.IMREAD_GRAYSCALE), dsize=None, fx=self.sf, fy=self.sf), axis=2).transpose(2, 0, 1).astype(np.float32) / 255.0
            img_orig_np_gray = cv2.imread(base_fname)
        else:
            img_orig_np = cv2.cvtColor(cv2.imread(base_fname), cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
            img_orig_np_gray = cv2.cvtColor(cv2.imread(base_fname), cv2.COLOR_BGR2GRAY)

        # if self.reg_mode == 'net':
            # img_bicubic_np = cv2.resize(src=cv2.imread(base_fname), dsize=(0,0), fx=self.sf, fy=self.sf, interpolation=cv2.INTER_CUBIC)
            # img_bicubic_np = cv2.cvtColor(img_bicubic_np, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0

        if shape == None:
            shape = img_orig_np.shape
        else:
            assert shape == img_orig_np.shape, "input image size error"

        self.base_np = img_orig_np
        self.base_gray_np = img_orig_np_gray

        if self.reg_mode == 'net':
            self.base_bicubic_np = img_bicubic_np

        self.shape = shape
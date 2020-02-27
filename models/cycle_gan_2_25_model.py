import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable
import numpy as np
# from .dltk_model import parse_dltk_model
import os

class CycleGAN225Model(BaseModel):

    def name(self):
        return 'CycleGAN 2_25 model'

    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--init_D', type=str, default=None, help='initialization for netD')
            parser.add_argument('--init_G', type=str, default=None, help='initialization for netG')
            parser.add_argument('--dltk_CLS', type=str, default=None, help='folder for dltk classification model', required=True)
            parser.add_argument('--lambda_CLS', type=float, default=0.1, help='weight for classification loss (KL-divergence)') 

    def load_single_network(self, net, pth):
        state_dict = torch.load(pth, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        # patch InstanceNorm checkpoints prior to 0.4
        # for key in list(state_dict.keys()):
            # self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.module.load_state_dict(state_dict)
        return net   

    def setup(self, opt):
        super(CycleGAN225Model, self).setup(opt)
        if self.isTrain:
            if not opt.init_D is None:
                print('initializing discriminator network from %s' % opt.init_D)
                self.netD = self.load_single_network(self.netD, opt.init_D).cuda()
            if not opt.init_G is None:
                print('initializing generator network from %s' % opt.init_G)
                self.netG = self.load_single_network(self.netG, opt.init_G).cuda()
            print('initializing classification network from %s' % opt.dltk_CLS)

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.loss_names = ['D', 'G', 'sem']
        self.visual_names = ['real_A', 'fake_B']
        if self.isTrain:
            self.model_names = ['G', 'D', 'CLS']
        else:
            self.model_names = ['G']
        self.AtoB = opt.direction == 'AtoB'
        self.lambda_CLS = opt.lambda_CLS

        input_nc,output_nc = (opt.input_nc,opt.output_nc) if self.AtoB else (opt.output_nc,opt.input_nc)
        self.netG = networks.define_G(input_nc, output_nc, opt.ngf, opt.netG, opt.norm,
                                    not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netG = torch.nn.DataParallel(self.netG)
        if self.isTrain:
            self.netD = networks.define_D(output_nc, opt.ndf, opt.netD,
                                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netD = torch.nn.DataParallel(self.netD)
            # self.netCLS = parse_dltk_model(opt.dltk_CLS)
            self.netCLS = torch.nn.DataParallel(torch.load(os.path.join(opt.dltk_CLS, 'model'))).cuda()

        if self.isTrain:
            self.fake_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCLS = torch.nn.KLDivLoss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):

        self.real_A = input['A' if self.AtoB else 'B'].to(self.device)
        self.real_B = input['B' if self.AtoB else 'B'].to(self.device)
        self.image_paths = input['A_paths' if self.AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)
        if self.isTrain:
            self.pred_real_A = self.netCLS(self.real_A)
            self.pred_fake_B = self.netCLS(self.fake_B)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        fake_B = self.fake_pool.query(self.fake_B)
        self.loss_D = self.backward_D_basic(self.netD, self.real_B, fake_B)

    def backward_G(self):
        self.loss_G = self.criterionGAN(self.netD(self.fake_B), True)
        self.loss_sem = 0.0
        for prA,pfB in zip(self.pred_real_A, self.pred_fake_B):
            self.loss_sem += self.criterionCLS(prA.log(), pfB)
        self.loss_sem /= len(self.pred_real_A)
        self.loss_G = self.loss_G + self.lambda_CLS*self.loss_sem
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netG], True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

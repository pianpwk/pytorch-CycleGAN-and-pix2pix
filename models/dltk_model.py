import os
os.environ['DLTK_BACKEND'] = 'pytorch'
import sys
sys.path.insert(0, '/mnt/ssfs/usr/pian.pawakapan/dltk')

import torch
from torch import nn
from torchvision import models, transforms

from dltk.backend import get_model
from dltk.utils.model_loader import load_taxonomy_grpc
from dltk.core.data.taxonomy import Taxonomy
from dltk.backend import get_augmenter

import yaml
import json

class DLTKModel(nn.Module):

    def __init__(self, folder_pth):
        super(DLTKModel, self).__init__()
        self.model = torch.load(os.path.join(folder_pth, 'model')).cuda()
        self.normalize = self.get_normalization(os.path.join(folder_pth, 'complete_config.json'))

    def get_normalization(self, json_fn): # only supports
        j = json.load(open(json_fn, 'r'))
        j_preproc = json.loads(j['test.preprocessing'])
        j_augment = json.loads(j['test.augmentation'])

        normalize = list(filter(lambda x:x['name']=='normalize', itertools.chain(j_preproc, j_augment)))[0]
        mean = torch.FloatTensor(normalize['params']['mean']).view(1,3,1,1).cuda()
        std = torch.FloatTensor(normalize['params']['std']).view(1,3,1,1).cuda()
        return {'mean': mean, 'std': std, 'max_pixel_value': normalize['params']['max_pixel_value'], 
                'norm': torch.FloatTensor([0.5, 0.5, 0.5]).view(1,3,1,1).cuda()}

    def forward(self, x):
        # first unnormalizes with GAN settings (0.5, 0.5, 0.5) then renormalizes with dltk settings
        # doesn't support other augmentations
        norm = self.normalize['norm']
        x = (x*norm)+norm
        x = (x-self.normalize['mean'])/self.normalize['std']
        x = torch.clamp(x, max=self.normalize['max_pixel_value'])
        
        # input into model
        return self.model(x)

def parse_dltk_model(pth):

    config_pth = os.join(pth, 'model_config.yaml')
    taxonomy_pth = os.join(pth, 'taxonomy.pb')
    model_pth = os.join(pth, 'model.pth')

    model_config = yaml.load(open(config_path, 'r'))['params']
    taxonomy = load_taxonomy_grpc(taxonomy_pth)
    taxonomy = Taxonomy('', taxonomy)

    dltk_model = get_model('classification', taxonomy=taxonomy, model_config=model_config, gpus=[-1])
    torch.manual_seed(0)

    sd = torch.load(model_pth)
    net = dltk_model.model
    net.load_state_dict(sd)
    net = net.cuda()
    net.eval()

    return net

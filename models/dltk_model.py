import os
os.environ['DLTK_BACKEND'] = 'pytorch'

import torch
from torch import nn
from torchvision import models

from dltk.backend import get_model
from dltk.utils.model_loader import load_taxonomy_grpc
from dltk.core.data.taxonomy import Taxonomy

import yaml

def parse_dltk_model(pth):

    config_pth = os.join(pth, 'model_config.yaml')
    taxonomy_pth = os.join(pth, 'taxonomy.pb')
    model_pth = os.join(pth, 'model.pth')

    model_config = yaml.load(open(config_path, 'r'))['params']
    taxonomy = load_taxonomy_grpc(taxonomy_pth)
    taxonomy = Taxonomy('', taxonomy)

    dltk_model = get_model('classification', taxonomy=taxonomy, model_config=model_config, gpus[-1])
    torch.manual_seed(0)

    sd = torch.load(model_pth)
    net = dltk_model.model
    net.load_state_dict(sd)
    net = net.cuda()
    net.eval()

    return net
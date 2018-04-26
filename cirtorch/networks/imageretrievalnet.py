import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

import torchvision

from cirtorch.layers.pooling import MAC, SPoC, GeM, RMAC
from cirtorch.layers.normalization import L2N
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.utils.general import get_data_root

# for some models, we have imported features (convolutions) from caffe because the image retrieval performance is higher for them
FEATURES = {
    'vgg16'         : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-vgg16-features-d369c8e.pth',
    'resnet50'      : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet50-features-ac468af.pth',
    'resnet101'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet101-features-10a101d.pth',
    'resnet152'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet152-features-1011020.pth',
}

POOLING = {
    'mac'  : MAC,
    'spoc' : SPoC,
    'gem'  : GeM,
    'rmac' : RMAC,
}

WHITENING = {
    'alexnet-gem'   : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-whiten-454ad53.pth',
    'vgg16-gem'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-whiten-eaa6695.pth',
    'resnet101-gem' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-whiten-22ab0c1.pth',
}

OUTPUT_DIM = {
    'alexnet'       :  256,
    'vgg11'         :  512,
    'vgg13'         :  512,
    'vgg16'         :  512,
    'vgg19'         :  512,
    'resnet18'      :  512,
    'resnet34'      :  512,
    'resnet50'      : 2048,
    'resnet101'     : 2048,
    'resnet152'     : 2048,
    'densenet121'   : 1024,
    'densenet161'   : 2208,
    'densenet169'   : 1664,
    'densenet201'   : 1920,
    'squeezenet1_0' :  512,
    'squeezenet1_1' :  512,
}


class ImageRetrievalNet(nn.Module):
    
    def __init__(self, features, pool, whiten, meta):
        super(ImageRetrievalNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.pool = pool
        self.whiten = whiten
        self.norm = L2N()
        self.meta = meta
    
    def forward(self, x):
        # features -> pool -> norm
        o = self.norm(self.pool(self.features(x))).squeeze(-1).squeeze(-1)
        # if whiten exist: whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o))
        # permute so that it is Dx1 column vector per image (DxN if many images)
        return o.permute(1,0)

    def __repr__(self):
        tmpstr = super(ImageRetrievalNet, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n' # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     whitening: {}\n'.format(self.meta['whitening'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr


def init_network(model='resnet101', pooling='gem', whitening=False, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], pretrained=True):

    # loading network from torchvision
    if pretrained:
        if model not in FEATURES:
            # initialize with network pretrained on imagenet in pytorch
            net_in = getattr(torchvision.models, model)(pretrained=True)
        else:
            # initialize with random weights, later on we will fill features with custom pretrained network
            net_in = getattr(torchvision.models, model)(pretrained=False)
    else:
        # initialize with random weights
        net_in = getattr(torchvision.models, model)(pretrained=False)

    # initialize features
    # take only convolutions for features,
    # always ends with ReLU to make last activations non-negative
    if model.startswith('alexnet'):
        features = list(net_in.features.children())[:-1]
    elif model.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif model.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif model.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif model.startswith('squeezenet'):
        features = list(net_in.features.children())
    else:
        raise ValueError('Unsupported or unknown model: {}!'.format(model))
    
    # initialize pooling
    pool = POOLING[pooling]()

    # get output dimensionality size
    dim = OUTPUT_DIM[model]

    # initialize whitening
    if whitening:
        w = '{}-{}'.format(model, pooling)
        whiten = nn.Linear(dim, dim, bias=True)
        if w in WHITENING:
            print(">> {}: for '{}' custom computed whitening '{}' is used"
                .format(os.path.basename(__file__), w, os.path.basename(WHITENING[w])))
            whiten_dir = os.path.join(get_data_root(), 'whiten')
            whiten.load_state_dict(model_zoo.load_url(WHITENING[w], model_dir=whiten_dir))
        else:
            print(">> {}: for '{}' there is no whitening computed, random weights are used"
                .format(os.path.basename(__file__), w))
    else:
        whiten = None

    # create meta information to be stored in the network
    meta = {'architecture':model, 'pooling':pooling, 'whitening':whitening, 'outputdim':dim, 'mean':mean, 'std':std}

    # create a generic image retrieval network
    net = ImageRetrievalNet(features, pool, whiten, meta)

    # initialize features with custom pretrained network if needed
    if pretrained and model in FEATURES:
        print(">> {}: for '{}' custom pretrained features '{}' are used"
            .format(os.path.basename(__file__), model, os.path.basename(FEATURES[model])))
        model_dir = os.path.join(get_data_root(), 'networks')
        net.features.load_state_dict(model_zoo.load_url(FEATURES[model], model_dir=model_dir))

    return net


def extract_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=10):
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    # extracting vectors
    vecs = torch.zeros(net.meta['outputdim'], len(images))
    for i, input in enumerate(loader):
        input_var = Variable(input.cuda())

        if len(ms) == 1:
            vecs[:, i] = extract_ss(net, input_var)
        else:
            vecs[:, i] = extract_ms(net, input_var, ms, msp)

        if (i+1) % print_freq == 0 or (i+1) == len(images):
            print('\r>>>> {}/{} done...'.format((i+1), len(images)), end='')
    print('')
    return vecs


def extract_ss(net, input_var):
    return net(input_var).cpu().data.squeeze()


def extract_ms(net, input_var, ms, msp):
    
    v = torch.zeros(net.meta['outputdim'])
    
    for s in ms: 
        if s == 1:
            input_var_t = input_var.clone()
        else:    
            size = (int(input_var.size(-2) * s), int(input_var.size(-1) * s))
            input_var_t = nn.functional.upsample(input_var, size=size, mode='bilinear')
        v += net(input_var_t).pow(msp).cpu().data.squeeze()
        
    v /= len(ms)
    v = v.pow(1./msp)
    v /= v.norm()

    return v
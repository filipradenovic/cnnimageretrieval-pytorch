import os
from os import path
import numpy as np

import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms

from cirtorch.networks.imageretrievalnet import init_network, extract_ms, extract_ss
from cirtorch.datasets.datahelpers import imresize, default_loader
from cirtorch.utils.general import get_data_root
from cirtorch.utils.whiten import whitenapply


TRAINED = {
'rSfM120k-tl-resnet101-gem-w':'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
'retrievalSfM120k-resnet101-gem':'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth'
}


def main():
    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    input_resol = 512; # resolution of input image, will resize to that if larger
    # input_resol = 1024; # resolution of input image, will resize to that if larger
    scales = [1, 1/np.sqrt(2), 1/2] # re-scaling factors for multi-scale extraction

    # sample image    
    img_file = 'sanjuan.jpg'
    if not path.exists(img_file):
        os.system('wget https://raw.githubusercontent.com/gtolias/tma/master/data/input/'+img_file)
    img = default_loader(img_file)
    

    print("use network trained with gem pooling and FC layer")
    state = load_url(TRAINED['rSfM120k-tl-resnet101-gem-w'], model_dir=os.path.join(get_data_root(), 'networks'))
    net = init_network({'architecture':state['meta']['architecture'],'pooling':state['meta']['pooling'],'whitening':state['meta'].get('whitening')})
    net.load_state_dict(state['state_dict'])
    net.eval()
    net.cuda()        
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=state['meta']['mean'], std=state['meta']['std'])])
    # single-scale extraction
    vec = extract_ss(net, transform(imresize(img, input_resol)).unsqueeze(0).cuda())
    vec = vec.data.cpu().numpy()
    print(vec)
    # multi-scale extraction
    vec = extract_ms(net, transform(imresize(img, input_resol)).unsqueeze(0).cuda(), ms = scales, msp = 1.0)
    vec = vec.data.cpu().numpy()
    print(vec)
    print("\n")


    print("use network trained with gem pooling, and apply the learned whitening transformation")
    state = load_url(TRAINED['retrievalSfM120k-resnet101-gem'], model_dir=os.path.join(get_data_root(), 'networks'))
    net = init_network({'architecture':state['meta']['architecture'],'pooling':state['meta']['pooling'],'whitening':state['meta'].get('whitening')})
    net.load_state_dict(state['state_dict'])
    net.eval()
    net.cuda()        
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=state['meta']['mean'], std=state['meta']['std'])])
    
    # single-scale extraction
    vec = extract_ss(net, transform(imresize(img, input_resol)).unsqueeze(0).cuda())
    vec = vec.data.cpu().numpy()
    print(vec)
    whiten_ss = state['meta']['Lw']['retrieval-SfM-120k']['ss']
    vec = whitenapply(vec.reshape(-1,1), whiten_ss['m'], whiten_ss['P']).reshape(-1)
    print(vec)

    # multi-scale extraction
    vec = extract_ms(net, transform(imresize(img, input_resol)).unsqueeze(0).cuda(), ms = scales, msp = net.pool.p.item())
    vec = vec.data.cpu().numpy()
    print(vec)
    whiten_ms = state['meta']['Lw']['retrieval-SfM-120k']['ms']
    vec = whitenapply(vec.reshape(-1,1), whiten_ms['m'], whiten_ms['P']).reshape(-1)
    print(vec)
    print("\n")
    

    print("use pre-trained (on ImageNet) network with appended mac pooling")
    net = init_network({'architecture':'resnet101','pooling':'mac','pretrained':True})
    net.eval()
    net.cuda()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=net.meta['mean'], std=net.meta['std'])])
    
    # single-scale extraction
    vec = extract_ss(net, transform(imresize(img, input_resol)).unsqueeze(0).cuda())
    vec = vec.data.cpu().numpy()
    print(vec)
    # multi-scale extraction
    vec = extract_ms(net, transform(imresize(img, input_resol)).unsqueeze(0).cuda(), ms = scales, msp = 1.0)
    vec = vec.data.cpu().numpy()
    print(vec)
    print("\n")



if __name__ == '__main__':
    main()

# CNN Image Retrieval in PyTorch: Training and evaluating CNNs for Image Retrieval in PyTorch

This is a Python toolbox that implements the training and testing of the approach described in our papers:


**Fine-tuning CNN Image Retrieval with No Human Annotation**,  
Radenović F., Tolias G., Chum O., 
TPAMI 2018 [[arXiv](https://arxiv.org/abs/1711.02512)]

**CNN Image Retrieval Learns from BoW: Unsupervised Fine-Tuning with Hard Examples**,  
Radenović F., Tolias G., Chum O., 
ECCV 2016 [[arXiv](http://arxiv.org/abs/1604.02426)]


<img src="http://cmp.felk.cvut.cz/cnnimageretrieval/img/cnnimageretrieval_network_medium.png" width=\textwidth/>

## What is it?

This code implements:

1. Training (fine-tuning) CNN for image retrieval
1. Learning supervised whitening for CNN image representations
1. Testing CNN image retrieval on Oxford5k and Paris6k datasets

## Prerequisites

In order to run this toolbox you will need:

1. Python3 (tested with Python 3.7.0 on Debian 8.1)
1. PyTorch deep learning framework (tested with version 1.0.0)
1. All the rest (data + networks) is automatically downloaded with our scripts

## Usage and papers implementation

Navigate (```cd```) to the root of the toolbox ```[YOUR_CIRTORCH_ROOT]```.

### Training

Example training script is located in ```YOUR_CIRTORCH_ROOT/cirtorch/examples/train.py```
```
python3 -m cirtorch.examples.train [-h] [--training-dataset DATASET] [--no-val]
                [--test-datasets DATASETS] [--test-whiten DATASET]
                [--test-freq N] [--arch ARCH] [--pool POOL]
                [--local-whitening] [--regional] [--whitening]
                [--not-pretrained] [--loss LOSS] [--loss-margin LM]
                [--image-size N] [--neg-num N] [--query-size N]
                [--pool-size N] [--gpu-id N] [--workers N] [--epochs N]
                [--batch-size N] [--optimizer OPTIMIZER] [--lr LR]
                [--momentum M] [--weight-decay W] [--print-freq N]
                [--resume FILENAME]
                EXPORT_DIR
```

For detailed explanation of the options run:
```
python3 -m cirtorch.examples.train -h
```

For example, to train our best network described in the TPAMI 2018 paper run the following command. 
After each epoch, the fine-tuned network will be tested on the revisited Oxford and Paris benchmarks:
```
python3 -m cirtorch.examples.train YOUR_EXPORT_DIR --gpu-id '0' --training-dataset 'retrieval-SfM-120k' 
            --test-datasets 'roxford5k,rparis6k' --arch 'resnet101' --pool 'gem' --loss 'contrastive' 
            --loss-margin 0.85 --optimizer 'adam' --lr 5e-7 --neg-num 5 --query-size=2000 
            --pool-size=22000 --batch-size 5 --image-size 362
```

Networks can be evaluated with learned whitening after each epoch. To achieve this run the following command. 
Note that this will significantly slow down the entire training procedure, and you can evaluate networks with learned whitening later on using the example test script.

```
python3 -m cirtorch.examples.train YOUR_EXPORT_DIR --gpu-id '0' --training-dataset 'retrieval-SfM-120k' 
            --test-datasets 'roxford5k,rparis6k' --test-whiten 'retrieval-SfM-30k' 
            --arch 'resnet101' --pool 'gem' --loss 'contrastive' --loss-margin 0.85 
            --optimizer 'adam' --lr 5e-7 --neg-num 5 --query-size=2000 --pool-size=22000 
            --batch-size 5 --image-size 362
```

**Note**: Data and networks used for training and testing are automatically downloaded when using the example script.

**Note**: Adjusted (lower) learning rate is set to achieve similar performance as with [MatConvNet](https://github.com/filipradenovic/cnnimageretrieval) and [PyTorch-0.3.0](https://github.com/filipradenovic/cnnimageretrieval-pytorch/tree/v1.0) implementation of the training.

### Testing

Example testing script is located in ```YOUR_CIRTORCH_ROOT/cirtorch/examples/test.py```
```
python3 -m cirtorch.examples.test [-h] (--network-path NETWORK | --network-offtheshelf NETWORK)
               [--datasets DATASETS] [--image-size N]
               [--multiscale MULTISCALE] [--whitening WHITENING] [--gpu-id N]
```

For detailed explanation of the options run:
```
python3 -m cirtorch.examples.test -h
```


#### Our pretrained networks

We provide the pretrained networks trained using the same parameters as in our TPAMI 2018 paper, with precomputed whitening. 
To evaluate them run:
```
python3 -m cirtorch.examples.test --gpu-id '0' --network-path 'retrievalSfM120k-resnet101-gem' 
                --datasets 'oxford5k,paris6k,roxford5k,rparis6k' 
                --whitening 'retrieval-SfM-120k'
                --multiscale '[1, 1/2**(1/2), 1/2]'
```
or
```
python3 -m cirtorch.examples.test --gpu-id '0' --network-path 'retrievalSfM120k-vgg16-gem' 
                --datasets 'oxford5k,paris6k,roxford5k,rparis6k' 
                --whitening 'retrieval-SfM-120k'
                --multiscale '[1, 1/2**(1/2), 1/2]'
```
The table below shows the performance comparison of networks trained with this framework and the networks used in the paper which were trained with our [CNN Image Retrieval in MatConvNet](https://github.com/filipradenovic/cnnimageretrieval):

| Model | Oxford | Paris | ROxf (M) | RPar (M) | ROxf (H) | RPar (H) |
|:------|:------:|:------:|:------:|:------:|:------:|:------:|
| VGG16-GeM (MatConvNet) | 87.9 | 87.7 | 61.9 | 69.3 | 33.7 | 44.3 |
| VGG16-GeM (PyTorch) | 87.3 | 87.8 | 60.9 | 69.3 | 32.9 | 44.2 |
| ResNet101-GeM (MatConvNet) | 87.8 | 92.7 | 64.7 | 77.2 | 38.5 | 56.3 |
| ResNet101-GeM (PyTorch) | 88.2 | 92.5 | 65.4 | 76.7 | 40.1 | 55.2 |


#### Your trained networks

To evaluate your trained network using single scale and without learning whitening:
```
python3 -m cirtorch.examples.test --gpu-id '0' --network-path YOUR_NETWORK_PATH 
                --datasets 'oxford5k,paris6k,roxford5k,rparis6k'
```

To evaluate trained network using multi scale evaluation and with learned whitening as post-processing:
```
python3 -m cirtorch.examples.test --gpu-id '0' --network-path YOUR_NETWORK_PATH 
                --datasets 'oxford5k,paris6k,roxford5k,rparis6k'
                --whitening 'retrieval-SfM-120k' 
                --multiscale '[1, 1/2**(1/2), 1/2]'
```

#### Off-the-shelf networks

Off-the-shelf networks can be evaluated as well, for example:
```
python3 -m cirtorch.examples.test --gpu-id '0' --network-offtheshelf 'resnet101-gem'
                --datasets 'oxford5k,paris6k,roxford5k,rparis6k'
                --whitening 'retrieval-SfM-120k' 
                --multiscale '[1, 1/2**(1/2), 1/2]'
```

**Note**: Data used for testing are automatically downloaded when using the example script.

## Networks with whitening learned end-to-end

Our code can be used to fine-tune networks with whitening added as an FC layer after the pooling.
Whitening FC layer is initialized in a supervised manner using our training data and off-the-shelf features.
When whitening is added in the fine-tuning procedure, we notice that the performance is highest if the images are with a similar high-resolution at train and test time. 
Also, the distribution of pairwise distances changes significantly, so roughly twice larger margin should be used for contrastive loss.
In this scenario, triplet loss performs better. 
To train such a setup you should run the following command (the performance will be evaluated every 5 epochs on `roxford5k` and `rparis6k`):
```
python3 -m cirtorch.examples.train YOUR_EXPORT_DIR --gpu-id '0' --training-dataset 'retrieval-SfM-120k' 
            --loss 'triplet' --loss-margin 0.5 --optimizer 'adam' --lr 5e-7 
            --arch 'resnet101' --pool 'gem' --whitening 
            --neg-num 4 --query-size=2000 --pool-size=20000 
            --batch-size 5 --image-size 1024 --epochs 50 
            --test-datasets 'roxford5k,rparis6k' --test-freq 5 
```

We also provide our end-to-end pre-trained networks, trained both on `retrieval-SfM-120k (rSfM120k)` and [`Google Landmarks 2018 (GL18)`](https://www.kaggle.com/google/google-landmarks-dataset) train datasets.
Whitening is learned during the end-to-end training, so there is no need to compute it as a post-processing step.
For example, to evaluate ResNet101 with GeM and whitening trained on `Google Landmarks 2018 (GL18)` dataset using high-resolution images and a triplet loss, you should run the following script:
```
python3 -m cirtorch.examples.test_e2e --gpu-id '0' --network 'gl18-tl-resnet101-gem-w' 
            --datasets 'roxford5k,rparis6k' --multiscale '[1, 2**(1/2), 1/2**(1/2)]'
```

Multi-scale performance of all available pre-trained networks is given in the following table:

| Model | ROxf (M) | RPar (M) | ROxf (H) | RPar (H) |
|:------|:------:|:------:|:------:|:------:|
| rSfM120k-tl-resnet50-gem-w  | TBD | TBD | TBD | TBD |
| rSfM120k-tl-resnet101-gem-w | TBD | TBD | TBD | TBD |
| rSfM120k-tl-resnet152-gem-w | TBD | TBD | TBD | TBD |
| [gl18-tl-resnet50-gem-w](http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth)  | 63.6 | 78.0 | 40.9 | 57.5 |
| [gl18-tl-resnet101-gem-w](http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth) | 67.3 | 80.6 | 44.3 | 61.5 |
| [gl18-tl-resnet152-gem-w](http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth) | 68.7 | 79.7 | 44.2 | 60.3 |

**Note**: Data, networks, and whitening initialization used for training and testing are automatically downloaded when using the example scripts.


## Related publications

### Training (fine-tuning) convolutional neural networks 
```
@article{RTC18,
 title = {Fine-tuning {CNN} Image Retrieval with No Human Annotation},
 author = {Radenovi{\'c}, F. and Tolias, G. and Chum, O.}
 journal = {TPAMI},
 year = {2018}
}
```
```
@inproceedings{RTC16,
 title = {{CNN} Image Retrieval Learns from {BoW}: Unsupervised Fine-Tuning with Hard Examples},
 author = {Radenovi{\'c}, F. and Tolias, G. and Chum, O.},
 booktitle = {ECCV},
 year = {2016}
}
```

### Revisited benchmarks for Oxford and Paris ('roxford5k' and 'rparis6k')
```
@inproceedings{RITAC18,
 author = {Radenovi{\'c}, F. and Iscen, A. and Tolias, G. and Avrithis, Y. and Chum, O.},
 title = {Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking},
 booktitle = {CVPR},
 year = {2018}
}
```

## Versions

### [master](https://github.com/filipradenovic/cnnimageretrieval-pytorch/tree/master) (development)

- Added mutli-scale performance on `roxford5k` and `rparis6k` for new pre-trained networks with end-to-end whitening, trained on both `retrieval-SfM-120` **\[TO BE DONE\]** and `Google Landmarks 2018` train datasets
- Added a new example test script without post-processing, for networks that are trained in a fully end-to-end manner, with whitening as FC layer learned during training
- Added few things in train example: GeMmp pooling, triplet loss, small trick to handle really large batches
- Added more pre-computed whitening options in imageretrievalnet
- Added triplet loss 
- Added GeM pooling with multiple parameters (one p per channel/dimensionality)
- Added script to enable download on Windows 10 as explained in Issue [#39](https://github.com/filipradenovic/cnnimageretrieval-pytorch/issues/39), courtesy of [SongZRui](https://github.com/SongZRui)

### [v1.1](https://github.com/filipradenovic/cnnimageretrieval-pytorch/tree/v1.1) (12 Jun 2019)

- Migrated code to PyTorch 1.0.0, removed Variable, added torch.no_grad for more speed and less memory at evaluation
- Added rigid grid regional pooling that can be combined with any global pooling method (R-MAC, R-SPoC, R-GeM)
- Added PowerLaw normalization layer
- Added multi-scale testing with any given set of scales, in example test script
- Fix related to precision errors of covariance matrix estimation during whitening learning
- Fixed minor bugs

### [v1.0](https://github.com/filipradenovic/cnnimageretrieval-pytorch/tree/v1.0) (09 Jul 2018)

- First public version
- Compatible with PyTorch 0.3.0

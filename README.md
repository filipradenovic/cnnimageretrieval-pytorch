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

1. Python3 (tested with Python 3.5.3 on Debian 8.1)
1. PyTorch deep learning framework (tested with version 0.3.0.post4)
1. All the rest (data + networks) is automatically downloaded with our scripts

## Usage

Navigate (```cd```) to the root of the toolbox ```[YOUR_CIRTORCH_ROOT]```.

### Training

Example training script is located in ```YOUR_CIRTORCH_ROOT/cirtorch/examples/train.py```
```
python3 -m cirtorch.examples.train.py [-h] [--training-dataset DATASET] [--no-val]
                [--test-datasets DATASETS] [--test-whiten DATASET]
                [--arch ARCH] [--pool POOL] [--whitening] [--not-pretrained]
                [--loss LOSS] [--loss-margin LM] [--image-size N]
                [--neg-num N] [--query-size N] [--pool-size N] [--gpu-id N]
                [--workers N] [--epochs N] [--batch-size N]
                [--optimizer OPTIMIZER] [--lr LR] [--momentum M]
                [--weight-decay W] [--print-freq N] [--resume FILENAME]
                DIR
```

For detailed explanation of the options run:
```
python3 -m cirtorch.examples.train.py -h
```

For example, to train our best network described in the TPAMI 2018 paper run the following command. After each epoch, the fine-tuned network will be tested on the revisited Oxford and Paris benchmarks:
```
python3 -m cirtorch.examples.train YOUR_EXPORT_DIR --gpu-id '0' --training-dataset 'retrieval-SfM-120k' 
            --test-datasets 'roxford5k,rparis6k' --arch 'resnet101' --pool 'gem' --loss 'contrastive' 
            --loss-margin 0.85 --optimizer 'adam' --lr 1e-6 --neg-num 5 --query-size=2000 
            --pool-size=20000 --batch-size 5 --image-size 362
```

Networks can be evaluated with learned whitening after each epoch. To achieve this run the following command. Note that this will significantly slow down the entire training procedure, and you can evaluate networks with learned whitening later on using the example test script.

```
python3 -m cirtorch.examples.train YOUR_EXPORT_DIR --gpu-id '0' --training-dataset 'retrieval-SfM-120k' 
            --test-datasets 'roxford5k,rparis6k' --test-whiten 'retrieval-SfM-30k' 
            --arch 'resnet101' --pool 'gem' --loss 'contrastive' --loss-margin 0.85 
            --optimizer 'adam' --lr 1e-6 --neg-num 5 --query-size=2000 --pool-size=20000 
            --batch-size 5 --image-size 362
```

**Note**: Data and networks used for training and testing are automatically downloaded when using the example script.

### Testing

Example testing script is located in ```YOUR_CIRTORCH_ROOT/cirtorch/examples/test.py```
```
python3 -m cirtorch.examples.test.py [-h] (--network-path NETWORK | --network-offtheshelf NETWORK)
               [--datasets DATASETS] [--image-size N] [--multiscale]
               [--whitening WHITENING] [--gpu-id N]
```

For detailed explanation of the options run:
```
python3 -m cirtorch.examples.test.py -h
```


#### Our pretrained networks

We provide the pretrained networks trained using the same parameters as in our TPAMI 2018 paper, with precomputed whitening. To evaluate them run:
```
python3 -m cirtorch.examples.test --gpu-id '0' --network-path 'retrievalSfM120k-resnet101-gem' 
                --datasets 'oxford5k,paris6k,roxford5k,rparis6k' 
                --whitening 'retrieval-SfM-120k' --multiscale
```
or
```
python3 -m cirtorch.examples.test --gpu-id '0' --network-path 'retrievalSfM120k-vgg16-gem' 
                --datasets 'oxford5k,paris6k,roxford5k,rparis6k' 
                --whitening 'retrieval-SfM-120k' --multiscale
```
Performance comparison with the networks used in the paper, trained with our [CNN Image Retrieval in MatConvNet](https://github.com/filipradenovic/cnnimageretrieval):

| Model | Oxford | Paris | ROxf (M) | RPar (M) | ROxf (H) | RPar (H) |
|:------|:------:|:------:|:------:|:------:|:------:|:------:|
| VGG16-GeM (MatConvNet) | 87.9 | 87.7 | 61.9 | 69.3 | 33.7 | 44.3 |
| VGG16-GeM (PyTorch) | 87.2 | 87.8 | 60.5 | 69.3 | 32.4 | 44.3 |
| ResNet101-GeM (MatConvNet) | 87.8 | 92.7 | 64.7 | 77.2 | 38.5 | 56.3 |
| ResNet101-GeM (PyTorch) | 88.2 | 92.5 | 65.3 | 76.6 | 40.0 | 55.2 |


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
                --whitening 'retrieval-SfM-120k' --multiscale
```

#### Off-the-shelf networks

Off-the-shelf networks can be evaluated as well, for example:
```
python3 -m cirtorch.examples.test --gpu-id '0' --network-offtheshelf 'resnet101-gem'
                --datasets 'oxford5k,paris6k,roxford5k,rparis6k'
                --whitening 'retrieval-SfM-120k' --multiscale
```

**Note**: Data used for testing are automatically downloaded when using the example script.


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
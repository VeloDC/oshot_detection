# Demo implementation of OSHOT: One SHOT unsupervised cross domain detection.

This code is based on [Maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
and uses Pytorch and CUDA.

This readme will guide you through a full run of our method for the Pascal VOC -> AMD benchmarks. 
Configuration files are provided also to perform other experiments.

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Datasets

Create a folder named `datasets` and include VOC2007 and VOC2012 source datasets (download from
[Pascal VOC's website](http://host.robots.ox.ac.uk/pascal/VOC/)).

Download and extract clipart1k, comic2k and watercolor2k from [authors'
website](https://naoto0804.github.io/cross_domain_detection/).

## Performing OSHOT pretraining 

To perform the pretraing using Pascal VOC as source dataset:

```bash
python tools/train_net.py --config-file configs/amd/voc_pretrain.yaml
```

By default training and inference are performed on a single GPU.

The final model will be saved in `VOC_RS_baseline/model_final.pth`. 

## Testing pretrained model

You can test a pretrained model on one of the AMD referring to the correct config-file. For example
for clipart:

```bash
python tools/test_net.py --config-file configs/amd/oshot_clipart_target.yaml --ckpt VOC_RS_baseline/model_final.pth
```

## Performing OSHOT adaptation

To use OSHOT adaptation rocedure and obtain results on one of the AMD please refer to one of the
config files. For example for clipart:

```bash
python tools/oshot_net.py --config-file configs/amd/oshot_clipart_target.yaml --ckpt VOC_RS_baseline/model_final.pth
```

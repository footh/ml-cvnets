# Steps to reproduce best results for TinyImageNet

* Follow "Installation" section below to install required environment and packages
* Place the "tiny-imagenet-200" folder in "image_data" at the same level as the project root folder
* Run the script "setup_val.py" to setup validation data
* Folder with validation data is formed 
       
* When in the required environment enter following command in main repo folder to start training the required resnet model: 
```
(ml-cvnets) C:\ml-cvnets>cvnets-train --common.config-file config/classification/resnet_tiny_depth.yaml --common.results-loc results_resnet > results\resnet_depth50_best.txt
```

* When in the main repo folder and required environment enter following command to start trainng the required mobileViT mode:
```
(ml-cvnets) C:\ml-cvnets>cvnets-train --common.config-file config/classification/mobilevittiny_best.yaml --common.results-loc results_mobilevit > results\mobilevit_best.txt
```

* Once the run is complete the following command can be used to generate summary plots from resnet_depth50_best.txt
```
(ml-cvnets2) C:\ml-cvnets>utils/summary_plots.py "results\\resnet_depth50_best.txt" "results\\mobilevittiny_best.csv"
```


# --------------------------- original repo author readme file starts from here ---------------------
# CVNets: A library for training computer vision networks

This repository contains the source code for training computer vision models. Specifically, it contains the source code of the [MobileViT](https://arxiv.org/abs/2110.02178?context=cs.LG) paper for the following tasks:
   * Image classification on the ImageNet dataset
   * Object detection using [SSD](https://arxiv.org/abs/1512.02325)
   * Semantic segmentation using [Deeplabv3](https://arxiv.org/abs/1706.05587)

***Note***: Any image classification backbone can be used with object detection and semantic segmentation models

Training can be done with two samplers:
   * Standard distributed sampler
   * [Mulit-scale distributed sampler](https://arxiv.org/abs/2110.02178?context=cs.LG)

We recommend to use multi-scale sampler as it improves generalization capability and leads to better performance. See [MobileViT](https://arxiv.org/abs/2110.02178?context=cs.LG) for details.

## Installation

CVNets can be installed in the local python environment using the below command:
``` 
    git clone git@github.com:apple/ml-cvnets.git
    cd ml-cvnets
    pip install -r requirements.txt
    pip install --editable .
```

We recommend to use Python 3.6+ and [PyTorch](https://pytorch.org) (version >= v1.8.0) with `conda` environment. For setting-up python environment with conda, see [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

## Getting Started

   * General instructions for training and evaluation different models are given [here](README-training-and-evaluation.md). 
   * Examples for a training and evaluating a specific model are provided in the [examples](examples) folder. Right now, we support following models.
     * [MobileViT](examples/README-mobilevit.md) 
     * [MobileNetv2](examples/README-mobilenetv2.md) 
     * [ResNet](examples/README-resnet.md)
   * For converting PyTorch models to CoreML, see [README-pytorch-to-coreml.md](README-pytorch-to-coreml.md).

## Citation

If you find our work useful, please cite the following paper:

``` 
@article{mehta2021mobilevit,
  title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
  author={Mehta, Sachin and Rastegari, Mohammad},
  journal={arXiv preprint arXiv:2110.02178},
  year={2021}
}
```

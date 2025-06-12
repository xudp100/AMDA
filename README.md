# AMDA: Adaptive Moment Dual Averaging Algorithm for Faster Optimizing Deep Network Models
## Introduction
This repository implements a ResNet-based image classification system for [CIFAR-10 dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/), featuring an AMDA optimizer for enhanced training performance.

## Project Structure
	classification_cifar10/
	├── Model/
	│ └── Resnet.py
	├── Optimizer/
	│ └── AdaVAM.py
	└── main.py

## Installation
We assume that you're using [Python 3.8+] installed. You need to run the following inside the root directory to install the dependencies:

```bash
pip install torch, numpy
```
This will install the following dependencies:
* [torch](https://pytorch.org/) (the library was tested on version 1.12.1+cu113)
* [numpy](https://numpy.org/) (the library was tested on version 1.26.1)



## Running Experiments
To run the classifier: 

```bash
python classification_cifar10/main.py
```

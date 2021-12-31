# CAE-Transformer
<h4> A Transformer-based Framework for Classification of Lung Nodules </h4>

In this repository, the implementation codes related to the CAE-Transformer model are released.
The detailed structure of the framework is availabele at <a href="https://arxiv.org/abs/2110.08721">ArXiv</a>.
The provided codes are slightly different version of the proposed framework in the sense of number of layers, transformer heads, and hyper parameters. The overal structure of the framework and the implementation is, however, the same.


<img src="https://github.com/ShahinSHH/CAE-Transformer/blob/main/Figs/cae-transformer.png" width="630" height="450"/>

## IMPORTANT
<b>!!! The provided files are not released for reproduction purposes. !!!</b>

The aim is to provide further insight into  the implementation of different parts of the proposed pipeline
for those interested in developing a similar framework.
Note that this implementation is particularly designed to work with a specific in-house dataset, and will not be executed on your system without setting the
required paths and modifying the configuration based on tour dataset structure. In what follows, the functionality of each file in this repository is explained.
You are welcome to adopt this implementation partially or fully for your project or research work.


## Framework
CAE-Transformer is predictive transformer-based framework, developed to predict the invasiveness of Lung Cancer, more specifically <b>Lung Adenocarcinoma (LUAC)</b>.
The CAE-Transformer utilizes a Convolutional Auto-Encoder (CAE) to automatically extract informative features from CT
slices, which are then fed to a modified transformer model to capture global inter-slice relations.
We performed several experiments on an in-house dataset of 114 pathologically proven Sub-Solid Nodules (SSNs) and the obtained results
demonstrate the superiority of the CAE-Transformer over the histogram/radiomics-based models, such as the model proposed in this <a href="https://www.nature.com/articles/s41598-019-42340-5">paper</a>, and also
its DL-based counterparts.

<img src="https://github.com/ShahinSHH/CAE-Transformer/blob/main/Figs/sample-ct.png" width="490" height="260"/>

## Pipeline
The following list outlines the step-by-step process taking place in the training and test steps proposed CAE-Transformer framework.

* <b>Step 1: Lung Region Segmentation</b>

All CT images are passed to a Lung Region Segmentation module to obtain lung areas and discard unimortant component in CT images.
<br>
The segmentation module is adopted from <a href="https://github.com/JoHof/lungmask">here</a> and can be installed using the following line of code:

```
pip install git+https://github.com/JoHof/lungmask
```

Make sure to have torch installed in your system. Otherwise you can't use the lungmask module.
<a href = "https://pytorch.org">https://pytorch.org</a>


* <b>Step 2: Preprocessing</b>

All images are resized from the original size of (512,512) into (256,256) and normalized using the Max-Min normalization function.

* <b>Step 3: Convolutional Auto-Encoder (CAE) and Pre-Training</b>

The extracted lung regions are going through a Convolutional Auto-Encoder (CAE) to provide slice-level feature maps in an unsupervised fashion.






## Requirements

## Citation
If you found this implementation and the related paepr useful in your research, please consider citing:

```
@article{Heidarian2021,
archivePrefix = {arXiv},
arxivId = {2110.08721},
author = {Heidarian, Shahin and Afshar, Parnian and Oikonomou, Anastasia and Plataniotis, Konstantinos N. and Mohammadi, Arash},
eprint = {2110.08721},
month = {oct},
title = {{CAE-Transformer: Transformer-based Model to Predict Invasiveness of Lung Adenocarcinoma Subsolid Nodules from Non-thin Section 3D CT Scans}},
url = {http://arxiv.org/abs/2110.08721},
year = {2021}
}

```

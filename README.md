# CAE-Transformer
<h4> A Transformer-based Framework for Classification of Lung Nodules </h4>

In this repository, the implementation codes related to the CAE-Transformer model are released.
The detailed structure of the framework is availabele at <a href="https://arxiv.org/abs/2110.08721">ArXiv</a>.
The provided codes are slightly different version of the proposed framework in the sense of number of layers, transformer heads, and hyper parameters. The overal structure of the framework and
the implementation is, however, the same.

## IMPORTANT
<b>!!! The provided files are not released for reproduction purposes. !!!</b>

The aim is to provide further insight into  the implementation of different parts of the proposed pipeline
for those interested in developing a similar framework.
Note that this implementation is particularly designed to work with a specific in-house dataset, and will not be executed on your system without setting the
required paths and modifying the configuration based on tour dataset structure. In what follows, the functionality of each file in this repository is explained.
You are welcome to adopt this implementation partially or fully for your project or research work.

<img src="https://github.com/ShahinSHH/COVID-FACT/blob/main/Figures/heatmap1.jpg" width="500" height="350"/>


## Framework
CAE-Transformer is predictive transformer-based framework, developed to predict the invasiveness of Lung Cancer, more specifically <b>Lung Adenocarcinoma (LUAC)</b>.
The CAE-Transformer utilizes a Convolutional Auto-Encoder (CAE) to automatically extract informative features from CT
slices, which are then fed to a modified transformer model to capture global inter-slice relations.
Experimental results on the in-house dataset of 114 pathologically proven Sub-Solid Nodules (SSNs)
demonstrate the superiority of the CAE-Transformer over the histogram/radiomics-based models and
its DL-based counterparts.


## Preprocessing



## Convolutional Auto-Encoder (CAE) and Pre-Training




## Transformer

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

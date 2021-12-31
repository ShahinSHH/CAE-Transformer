# CAE-Transformer
<h4> A Transformer-based Framework for Classification of Lung Nodules </h4>

In this repository, the implementation codes related to the CAE-Transformer model are released.
The detailed structure of the framework is availabele at <a href="https://arxiv.org/abs/2110.08721">ArXiv</a>.
The provided codes are slightly different version of the proposed framework in the sense of number of layers, transformer heads, and hyper parameters. The overal structure of the framework and the implementation is, however, the same.


<img src="https://github.com/ShahinSHH/CAE-Transformer/blob/main/Figs/cae-transformer.png" width="630" height="450"/>

## IMPORTANT
<h4>!!! The provided files are not released for reproduction!!!</h4>

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

* <h3>Step 1: Lung Region Segmentation</h3>

    All CT images are passed to a Lung Region Segmentation module to obtain lung areas and discard unimortant component in CT images.
    <br>
    The following files are used in this step:
     * <i>lung_segmentation_module.py</i> : provides the necessary functions and classes
     * <i>segmentation_main.py</i> : the main file to perform the segmentation and save the outputs

     The segmentation module is adopted from <a href="https://github.com/JoHof/lungmask">here</a> and can be installed using the following line of code:

    ```
    pip install git+https://github.com/JoHof/lungmask
    ```

    Make sure to have torch installed in your system. Otherwise you can't use the lungmask module.
    <a href = "https://pytorch.org">https://pytorch.org</a>


* <h3>Step 2: Preprocessing</h3>

    All images are resized from the original size of (512,512) into (256,256) and normalized using the Max-Min normalization function. The resizing and     normalization functions are available in the <i>utils.py</i> file.

* <h3>Step 3: Convolutional Auto-Encoder (CAE) and Pre-Training</h3>

     The extracted lung regions are then going to a Convolutional Auto-Encoder (CAE) to provide slice-level feature maps in an unsupervised fashion.

     The CAE is first pre-trained on the <a href= "https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI">LIDC-IDRI</a> dataset, and then fined tuned on   the target dataset (our in-house dataset of 114 patients). The following files are used for pre-training, fine-tuning, and saving the outputs of the CAE model:

    * <i>read_lidc_annotations.py</i> : reads the cases and their corresponding annotations in the <a href=     "https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI">LIDC-IDRI</a> dataset and save those slices with the evidence of lung nodule to be used as the pre-training data.
  
     This code is written based on the <b>pylidc</b> library. The official doccumentation of the pylidc library is available at: https://pylidc.github.io/
  
    * <i>CAE.py</i> : provides the functions and classes to implement the Convolutional Auto-Encoder.
    * <i>pretrain_cae.py</i> : the main file for pre-training the model 
    * <i>fine_tune_cae.py</i> : the main file for fine-tuning the model
    * <i>cae_save_outputs-sequential.py</i>: saves the outputs of the CAE model as a sequential data. Each sequence contains the feature maps generated for  slices  with the evidence of nodule in each patient.

     <b>Note:</b> To comply with the input size requirements of the subsequet modules, all sequences are zero-padded to have the equal size of (25,256), in which   25 represents the maximum number of slices for each patient, and 256 is the number of extracted features from each slice.


* <h3>Step 4: Transformer</h3>

  The following codes are used to implement and train the transformer-based classifier:
  * <i>transformers.py/i> : implements the transformer class
  * <i>train_transformer.py</i> : the main file for training the classification model and saving the best model

## Requirements
 
* Tested with tensorflow-gpu 2 and keras-gpu 2.2.4 on NVIDIA's GeForce RTX 3090
* Python 3.7
* PyTorch 1.4.0
* Torch 1.5.1
* PyDicom 1.4.2 --><a href="https://pydicom.github.io/pydicom/stable/tutorials/installation.html">Installation<a/>
* SimpleITK --><a href="https://simpleitk.readthedocs.io/en/v1.1.0/Documentation/docs/source/installation.html">Installation</a>
* lungmask --><a href="https://github.com/JoHof/lungmask">Installation</a>
* pylidc --> <a href="https://github.com/notmatthancock/pylidc">Installation</a>
* OpenCV
* Scikit-learn
* Pandas
* OS
* Numpy
* Matplotlib

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

# DermFoundation_Inference
## Vision foundation models pre-trained on skin lesions can detect oral lesions too!

The repository contains the implementation of the following paper. \
\
Title - **Vision foundation models pre-trained on skin lesions can detect oral lesions too!** \
Authors -  \
DOI - 

## Abstract
While there exist large public image datasets for skin cancer (more common in the global north), there only exist a couple of small public datasets for oral cancer (more common the global south), thus potentially indicating a hidden bias from the point of view of AI fairness and benchmarking machine learning models. While there has been some work using convolutional neural network (CNN) models on these smaller oral cancer datasets, it has not been possible yet to train a state-of-the-art large vision transformer (ViT) based foundation model on oral cancer yet due to data scarcity. For the first time, in this paper, it is demonstrated that a vision transformer model trained on skin lesion images (Google DermFoundation) can be used directly as backbone feature embedders without any retraining or tuning for robust feature representation on oral lesion images. 
\
\
\
![Graphical Abstract](images/Graphical_Abstract.jpg)




# Data
The two datasets used in this paper are available at the following links. Of these, the first dataset is publicly available and hosted on Kaggle while the second dataset has not been made public and has been made available to us for this study by special permission from the authors of this dataset:
1. [Oral Cancer (Lips and Tongue) Images](https://www.kaggle.com/datasets/shivam17299/oral-cancer-lips-and-tongue-images)
2. [Dataset of Annotated Oral Cavity Images for Oral Cancer Detection](https://zenodo.org/records/10664056)



# Getting started

## Installation
To install all requirements execute the following line.
```bash
pip install -r requirements.txt 
```
And then clone the repository as follows. 
```bash
git clone https://github.com/SamarupBhattacharya/DermFoundation_Inference.git
```

## Data

The folder **Data** contains numpy files containing the embeddings generated for the images with the help of Google's Derm Foundation model along with the ground truth labels for the images and CSV files containing additional information for the images as well as for the patients.
1. dataset_2_embeddings.npy - Contains a numpy array of 6144-dimensional embeddings for all the images in the dataset (as well as certain augmentations for the images and have not been utilized for this paper)
2. dataset_2_labels.npy - Contains a numpy array of labels for the images in the dataset (the 748 labels at the end correspond to the augmentations of the images and have not been utilized for this paper)
3. Imagewise_Data.csv - Contains image-level information for the images in the dataset such as Image Name, Category, Clinical Diagnosis, Lesion Annotation Count.
4. Patientwise_Data.csv - Contains patient-level information for the images in the dataset including Patient ID, Age, Gender, Smoking, Chewing_Betel_Quid, Alcohol, Image Count



## Saved Models
The folder **Checkpoints** contains the saved checkpoints for each of the five folds used for cross validation and for each of the three modes explored. In order to replicate the performance claimed in our paper, these checkpoints will find use.
Note: We used Kaggle's P100 GPU for training and evaluating the model, and therefore to replicate our performance, the reader is humbly requested to use the same.

### Training/Testing
To train/test DermFoundation_Inference execute the following.
```bash
python main.py --checkpoint_dir CHECKPOINT_DIRECTORY_PATH --data_dir DATA_DIRECTORY_PATH --mode MODE --inference_only WHETHER_INFERENCE_ONLY_MODE_IS_DESIRED
```
Here:

- checkpoint_dir: Requires the path of the checkpoint directory i.e. path to the directory where the contents of the "Checkpoints" directory have been downloaded and saved
- data_dir:  Requires the path of the checkpoint directory i.e. path to the directory where the contents of the "Data" directory have been downloaded and saved
- mode: Three modes are supported by the training/testing code
  - unimodal: Only the images are used to train/test the MLP Classifier
  - multimodal_all_features: All five columns of the patient-level metadata are used for training/testing the Fusion Model Classifier in conjunction with the embeddings of the images
  - multimodal_best_features: The two most important columns of features from the patient-level metadata(determined with the help of XGBoost) are used for training/testing the Fusion Model Classifier in conjunction with the embeddings of the images
- inference_only: Requires one of the following as input:
  - yes: Model is used for inference only using the checkpoints saved in "Checkpoints" directory
  - no: Model is trained first and then inference is performed


## Notebooks
This directory contains the Kaggle notebooks that were used for model training and evaluation.




# Acknowledgement 



# Citation
```bash

```

```bash

```

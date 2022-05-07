This repository contains code and artifacts to reproduce results obtained by a research paper `(Sharma, Rani & Gupta 2020)` [[1]](#1) which classifies COIVD based on X-Ray images.

# Original research paper
This work [[1]](#1) is done to build efficient deep learning models using the convolutional neural network (CNN) using a very limited set of chest X-ray images to differentiate COVID-19 cases from healthy cases and other types of illnesses. The overall goal of the paper is to train models such that rapid screening of COVID-19 patients is possible in a non-invasive and automated fashion.

In this study, authors have applied following techniques to overcome issues mentioned above:
* To overcome scarcity of x-ray images, authors proposed creating a much larger artificial dataset using smaller original dataset. This artificial dataset was created by applying multiple image augmentation techniques on the original data.
* Careful image selection: Authors propose to use similar type of images (same view, same age group) to train a very targeted model. This study claims that models built with similar type of images perform better than those models that take a wide variety of images

Overall, authors showed that artificially generated x-ray images using image augmentation techniques greatly improved model performance when compared with original smaller set of images.

> **Link to original research paper (Sharma, Rani & Gupta 2020) [[1]](#1)**: https://www.hindawi.com/journals/ijbi/2020/8889023/

> **Link to GitHub repository [[2]](#2) of original research paper**: https://github.com/arunsharma8osdd/covidpred

# Reproduction work (this repository)
This work reproduces the work done in original paper to verify if model trained with augmented images (artificial dataset) outperforms model trained with only original set of images.

> **Video Presentation link**: https://mediaspace.illinois.edu/media/t/1_d2sh1778

For this work, we created 26 other artificial datasets based on origianl set of images. Further, from these dataset we trained following models:
* Model trained with original images only.
* Model trained with images rotated by 120 degrees.
* Model trained with images rotated by 140 degrees.
* Model trained with original images and augmented images combined.

# Dependencies

Following are depencies for this work
*    Python3
*    pytorch
*    Tensorflow v1
*    CloDsa [[6]](#6) and it's dependencies- More about this in Prerequisites section.

# Data download instructions

Data from following three sources is used in this paper.
* [github (Cohen’s covid-chestxray-dataset)](https://github.com/ieee8023/covid-chestxray-dataset) [[3]](#3): This dataset is used to get 'covid 19' and 'non covid 19' images
* [Kaggle NIH dataset](https://www.kaggle.com/nih-chest-xrays/data) [[4]](#4): This dataset is used to get Pneumonia images.
* [National Library of medicine](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html) [[5]](#5): This dataset is used to get normal and TB images.

Manually download this data in `Data/FullSet` directory. Following should be final directory structure.

| Source | Directory Path | Comment |
| --- | --- | --- |
| Kaggle | `Data/FullSet/Kaggle Dataset/Images/` | There will be several directories containing images |
| Github | `Data/FullSet/MontgomerySet/` | There will be two directories `CXR_png` with images and `ClinicalReadings` with metadata |
| National Library of medicine | `Data/FullSet/ieee8023_covid-chestxray-dataset` | All images will be present in this directory | 

Please refer to `Data\FullSet` directory in this repo. This directory contains sample images with correct folder structure.

# Data pre-processing

Following diagram depicts all steps we performed to gather and process data for model training and validation purposes.
![Data_Processing](https://user-images.githubusercontent.com/17690014/163689385-e15138a6-13ea-4c4a-ab1f-4f53c5a8f060.png)

Note: Number in each box represents number of images.

We filtered images from these datasets using same filtering criteria cited by authors in their paper.
Following is the general filtering criteria.
* Age of patient must be 19 years or older.
* Only chest X-ray images.
* X-ray image view must be PA.

After filtering the data, we divided the dataset into two parts using random sampling. We reserved 10 percent of the data for external validation (test set) and remaining 90 percent for model training (training set).
After this, we created 25 new datasets by applying different augmentation techniques on original set of training and test images. Some of the augmentation techniques used are:
* Rotate images by 45, 60, 90, 120, 140 and 160 degrees
* Raise blue, green, red and hue
* Crop images
* Flip images horizontally, vertically and in both directions.
* Introduce blur to the images.
We used [CloDSA](https://github.com/joheras/CLoDSA) [[6]](#6) library for image augmentation.
We further created one more dataset by combining original dataset and all augmented datasets. Thus we had total 27 datasets.

## Prerequisites
Image augmentation library CloDSA must be installed on the machine before proceeding further. 

### Steps to install CloDSA
This library needs a number of packages to be pre-installed before installing clodsa.
* Execute script `\Scripts\CheckPkgs.py` to check if all pre-requisites to install CloDSA are met. This script will list packages that are needed for CloDSA but are missing on machine. Install all missing packages.

```
python '\Scripts\CheckPkgs.py'
```

* Install CloDSA by following [CLoDSA Instructions](https://github.com/joheras/CLoDSA) [[6]](#6)

## How to filter, augment images and create train/ test sets.
Following steps explain how to create datasets for training using code in this repository.

* **Filtering data**: Execute following scripts (present in `Scripts\DataManipulation` directory) in given order to create datasets.
  * **github_select_images_for_reproduction.py** - filters data from github source and copies filtered images to directories `\Data\FinalSet\covid_19` and `\Data\FinalSet\non_covid_19`

```
python '\Scripts\DataManipulation\github_select_images_for_reproduction.py'
```

  * **Kaggle_select_images_for_reproduction.py** - filters data from Kaggle source and copies filterd images to directory `\Data\FinalSet\Pneumonia`

```
python '\Scripts\DataManipulation\Kaggle_select_images_for_reproduction.py'
```

  * **Montgomery_select_images_for_reproduction.py** - filters data from last source and copies filtered images to directories `\Data\FinalSet\Normal` and `\Data\FinalSet\TB`

```
python '\Scripts\DataManipulation\Montgomery_select_images_for_reproduction.py'
```

* **Train-test split**: Execute script `train_test_split.py`. This script will split filtered data for each category into train (90%) and test (10%) set. Following two directoris will be created.
  * `Data\train`
  * `Data\test`

```
python '\Scripts\DataManipulation\train_test_split.py'
```

* **Augmentation**: Execute script `augmentation_main.py` to create artificial datasets by using 25 augmentation techniques on original images. All these datasets will be created in directory `\Data\Augmented`.
  *  This directory will contain 2 directories for each augmentation technique; one for training and other for testing.
  *  This directory will contain two additional directories that will keep combined dataset (i.e. dataset with all 25 augmented ones combined)

```
python '\Scripts\DataManipulation\augmentation_main.py'
```

*  **Manual steps**: Following steps have to be done manually. These are not covered by scripts.
    *   Copy images from origianl train and test datasets (from `Data/train` and `Data/test` folders respectively) to following folders
    *   Copy folder `Data\train` to `Data\Augmented`
    *   Copy images in `Data\train` to `Data\Augmented\train_combined`
    *   Copy folder `Data\test` to `Data\Augmented`
    *   Copy images in `Data\test` to `Data\Augmented\test_combined`

## How to train models
Following steps explain how to train models using scripts present in the repository. Model training scripts are present in directory '\Scripts\ModelTraining'

* **train_generic_original.py**: This script trains model by using original dataset.

```
python '\Scripts\ModelTraining\train_generic_original.py'
```

* **train_augment_generic_120.py**: This script trains model by using images rotated by 120 degrees.

```
python '\Scripts\ModelTraining\train_augment_generic_120.py'
```

* **train_augment_generic_140.py**: This script trains model by using images rotated by 140 degrees.

```
python '\Scripts\ModelTraining\train_augment_generic_140.py'
```

* **train_augment_generic.py**: This script trains model by using combined dataset (original images + all 25 augmented datasets combined).

```
python '\Scripts\ModelTraining\train_augment_generic.py'
```

### Output of model training
Following output files are created.

* **Logfiles**: File is created in the same folder where scripts are present. Name and path of the logfile can be customized in the script.
* **Trained model files**: Trained model and it's associated files are stored in `'Model_{filename}'` directory. Here filename = name of dataset. Please find more information about it from script.

## How to validate/ evaluate models
After model training, we want to validate the models on unseen (test) datasets. Execute script `validate_augment_generic.py` to validate the models.

Please modify variables in the script to modify target model, target dataset and logfile path.

```
python '\Scripts\ModelTraining\validate_augment_generic.py'
```

### Output of validation

* **Logfiles**: Created in the same folder from where script is executed.
* **ValidationResult.csv**: Contains validation result (accuracy for each label). This file is stored in same folder where model was stored.

## Pretrained models

Since models trained by our work were large and could not be stored in the github repository, we stored them in separate storage.

We trained following 4 models per this work.
1. Model_train_original
2. Model_train_rotated_120_degree
3. Model_train_rotated_140_degree
4. Model_train_combined_batch_32

These pre-trained models are stored at following location.
https://www.dropbox.com/sh/n4gnlj1sxf4z23r/AAA3tfh7BbwTgiEk0oYC2z0la?dl=0

# Results

Following table summarized results obtained by our reproduction study.

This table shows performance of two models:
* Model trained and tested with original images only
* Model trained with combined images (original and augmented images together) and tested with original images only.

Authors used recall as metric in their study. We can see that recall has significantly improved in combined model for all labels except non-covid which makes us feel that paper's approach of augmentation did not fully satisfy the expected results. However, F1-score has improved across all labels. This refutes the conclusion made by just observing recall and tells us that the combined model is actually better than original model.

![Markdown_Results](https://user-images.githubusercontent.com/17690014/167269449-15efef09-f4ba-4c23-93ab-59d7ac810d4b.PNG)

# References
*    <a id="1">[1]</a> [Original Paper](https://www.hindawi.com/journals/ijbi/2020/8889023/)
*    <a id="2">[2]</a> [Original code - CovidPred](https://github.com/arunsharma8osdd/covidpred)
*    <a id="3">[3]</a> [Cohen's covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)
*    <a id="4">[4]</a> [Kaggle NIH dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)
*    <a id="5">[5]</a> [National Library of medicine](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html)
*    <a id="6">[6]</a> [CloDSA](https://github.com/joheras/CLoDSA)
*    <a id="7">[7]</a> [Accuracy, precision, recall](https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826)

This repository contains code to reproduce results obtained by a research paper which classifies COIVD based on X-Ray images.

# Original research paper
This work is done to build efficient deep learning models using the convolutional neural network (CNN) using a very limited set of chest X-ray images to differentiate COVID-19 cases from healthy cases and other types of illnesses. The overall goal of the paper is to train models such that rapid screening of COVID-19 patients is possible in a non-invasive and automated fashion.

In this study, authors have applied following techniques to overcome issues mentioned above:
* To overcome scarcity of x-ray images, authors proposed creating a much larger artificial dataset using smaller original dataset. This artificial dataset was created by applying multiple image augmentation techniques on the original data.
* Careful image selection: Authors propose to use similar type of images (same view, same age group) to train a very targeted model. This study claims that models built with similar type of images perform better than those models that take a wide variety of images

Overall, authors showed that artificially generated x-ray images using image augmentation techniques greatly improved model performance when compared with original smaller set of images.

Link to original research paper: https://www.hindawi.com/journals/ijbi/2020/8889023/
Link to GitHub repository of original research paper: https://github.com/arunsharma8osdd/covidpred

# Reproduction work (this repository)
This work reproduces the work done in original paper to verify if model trained with augmented images (artificial dataset) outperforms model trained with only original set of images.

For this work, we created 26 other artificial datasets based on origianl set of images. Further, from these dataset we trained following models:
* Model trained with original images only.
* Model trained with images rotated by 120 degrees.
* Model trained with images rotated by 140 degrees.
* Model trained with original images and augmented images combined.

# Data acquisition and processing
Data from following three sources is used in this paper.
* [github (Cohen’s covid-chestxray-dataset)](https://github.com/ieee8023/covid-chestxray-dataset): This dataset is used to get 'covid 19' and 'non covid 19' images
* [Kaggle NIH dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data): This dataset is used to get Pneumonia images.
* [National Library of medicine](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html): This dataset is used to get normal and TB images.

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
We used [CloDSA](https://github.com/joheras/CLoDSA) library for image augmentation.
We further created one more dataset by combining original dataset and all augmented datasets. Thus we had total 27 datasets.

## How to recreate data
Following steps explain how to create datasets for training using code in this repository.

* Create directory struture same as present in this repository.
* Download data available from aforementioned sources to 'Data\FullSet' directory. Name the folders as below:
  * ieee8023_covid-chestxray-dataset - keep Cohen's data here.
  * Kaggle Dataset - Keep data downloaded from Kaggle here.
  * MontgomerySet - Keep data downloaded from last source here.
* Filtering data: Execute following scripts (present in 'Scripts\DataManipulation' directory) in given order to create datasets.
  * github_select_images_for_reproduction.py - filters data from github source and copies filtered images to directories \Data\FinalSet\covid_19 and \Data\FinalSet\non_covid_19
  * Kaggle_select_images_for_reproduction.py - filters data from Kaggle source and copies filterd images to following directory.
    * \Data\FinalSet\Pneumonia
  * Montgomery_select_images_for_reproduction.py - filters data from last source and copies filtered images to following directories.
    * \Data\FinalSet\Normal and \Data\FinalSet\TB
* Train-test split: Execute script train_test_split.py. This script will split filtered data for each category into train (90%) and test (10%) set. Following two directoris will be created.
  * Data\train
  * Data\test
* Augmentation: Execute script augmentation_main.py to create artificial datasets by using 25 augmentation techniques on original images. All these datasets will be created in directory \Data\Augmented.
  *  This directory will contain 2 directories for each augmentation technique; one for training and other for testing.
  *  This directory will contain two additional directories that will keep combined dataset (i.e. dataset with all 25 augmented ones combined)
 *  Manual steps: Following steps have to be done manually. These are not covered by scripts.
  *  Copy images from origianl train and test datasets (from Data/train and Data/test folders respectively) to following folders
   *  Copy folder Data\train to Data\Augmented
   *  Copy images in Data/train to Data\Augmented\train_combined
   *  Copy folder Data\test to Data\Augmented
   *  Copy images in Data/test to Data\Augmented\test_combined

## Directory Sturcture

### Data
Data directory contains data used for this work. In git repo, this directory contains only few samples.

#### FullSet
This directory contains sample data of from all data sources used in this work. A very small subset of images are included in these folders so that someone else can test the scripts and work on full data set confidently.

On local machine, where full code is executed, please download entire data from given sources in this directory.

#### FinalSet
This directory will contain selected images from images in FullSet directory. Images are selected based on various criterions. Those criterions can be found in Python scripts located in 'Script' directory.

#### Augmented
This directory will contain augmented imaged. This directory is populated by executing script augmentation_main.py.

#### Summary
This directory contains csv files that describe source and statistics of data.

### Scripts
This directory contains all scripts used in this work.

#### DataManipulation
This directory contains scripts that perform data manipulation tasks.
1. github_select_images_for_reproduction.py - selects relevant images from github data source.
2. Kaggle_select_images_for_reproduction.py - selects relevant images from Kaggle data source.
3. Montgomery_select_images_for_reproduction.py - selects relevant images from Montgomery data source.
4. augmentation_main.py - This script performs image augmentation.

Note: After downloading full data in FullSet folder. Data manipulation scripts should be executed in same order listed above.

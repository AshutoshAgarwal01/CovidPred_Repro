# CovidPred_Repro
This repository contains code to reproduce results obtained by a research paper which classifies COIVD based on X-Ray images.

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

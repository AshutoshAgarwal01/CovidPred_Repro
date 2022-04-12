'''
We are filtering images from Kaggle dataset https://www.kaggle.com/nih-chest-xrays/data for the purpose of 
reproducing paper 'https://doi.org/10.1155/2020/8889023'

Images fulfilling following criteria will be copied to the output folder.
1. Age >= 19 and Age <= 99
2. View Position = PA
3. Finding labels = "Pneumonia"
'''

import pandas as pd
import shutil
import os
import math
import numpy as np

pneumonia = "Pneumonia" # rows for pneumonia
x_ray_view = "PA" # View of X-Ray

metadata = "Kaggle_metadata.csv" # Meta info
imageDir = "../../Data/FullSet/Kaggle Dataset/Images" # Directory of images
outputDir = '../../Data/FinalSet/Pneumonia' # Output directory to store selected images

if not os.path.exists(outputDir):  # check if directory already exist
    os.mkdir(outputDir)  # create a directory

metadata_csv = pd.read_csv(metadata)

# loop over the rows of the COVID-19 data frame
for (i, row) in metadata_csv.iterrows():
    if row["View Position"] == x_ray_view and not (math.isnan(row["Patient Age"]) or row["Patient Age"] < 19 or row["Patient Age"] > 99):
        filename = row["Image Index"].split(os.path.sep)[-1]

        if row["Finding Labels"] == pneumonia:
            for i in np.arange(1, 13):
                tempFilePath = os.path.sep.join([imageDir, f"images_{i:03}", "images", filename])
                
                if os.path.exists(tempFilePath):  # check if file exists in this directory.
                    shutil.copy2(tempFilePath, outputDir)
                    break

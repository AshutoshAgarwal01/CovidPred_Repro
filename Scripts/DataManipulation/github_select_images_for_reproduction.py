'''
This code is taken from https://github.com/ieee8023/covid-chestxray-dataset/blob/master/scripts/select_covid_patient_X_ray_images.py
We have done following modifiections in the original code.
1. Added one more filter for age.
2. Modified output folder such that covid-19 images go in folder named covid_19 and other images go in folder named non_covid_19.
'''

'''
This code finds all images of patients of a specified VIRUS and X-Ray view and stores selected image to an OUTPUT directory
+ It uses metadata.csv for searching and retrieving images name
+ Using ./images folder it selects the retrieved images and copies them in output folder
Code can be modified for any combination of selection of images
'''

import pandas as pd
import shutil
import os
import math

# Selecting all combination of 'COVID-19' patients with 'PA' X-Ray view
covid = "Pneumonia/Viral/COVID-19" # rows for covid
non_covid_virus_list = ["Pneumonia/Viral/SARS", "Pneumonia/Fungal/Pneumocystis", "Pneumonia/Bacterial/Streptococcus"]
non_covid = "Pneumonia/Viral/COVID-19" # rows for covid
x_ray_view = "PA" # View of X-Ray

# Skip files without failing that were not found
skipNotFound = True

metadata = "../../Data/FullSet/ieee8023_covid-chestxray-dataset/metadata.csv" # Meta info
imageDir = "../../Data/FullSet/ieee8023_covid-chestxray-dataset/images" # Directory of images
covidOutputDir = '../../Data/FinalSet/covid_19' # Output directory to store selected images
nonCovidOutputDir = '../../Data/FinalSet/non_covid_19' # Output directory to store selected images

if not os.path.exists(covidOutputDir):  # check if directory already exist
    os.mkdir(covidOutputDir)  # create a directory

if not os.path.exists(nonCovidOutputDir):  # check if directory already exist
    os.mkdir(nonCovidOutputDir)  # create a directory

metadata_csv = pd.read_csv(metadata)

# loop over the rows of the COVID-19 data frame
for (i, row) in metadata_csv.iterrows():
    if row["view"] == x_ray_view and not (math.isnan(row["age"]) or row["age"] < 19 or row["age"] > 99):
        filename = row["filename"].split(os.path.sep)[-1]
        filePath = os.path.sep.join([imageDir, filename])

        if not os.path.exists(filePath) and skipNotFound:
            continue

        # If finding was not covid then copy images to non_covid_19 folder.
        if row["finding"] == covid:
            shutil.copy2(filePath, covidOutputDir)
        elif row["finding"] in non_covid_virus_list:
            shutil.copy2(filePath, nonCovidOutputDir)

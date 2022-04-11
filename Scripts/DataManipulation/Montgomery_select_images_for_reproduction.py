'''
We are filtering images from Kaggle dataset https://lhncbc.nlm.nih.gov/publication/pub9931 for the purpose of 
reproducing paper 'https://doi.org/10.1155/2020/8889023'

Images fulfilling following criteria will be copied to the output folder.
1. Age >= 19 and Age <= 99

Images are in folder 'CXR_png'.
Properties like age and findings for each image are present in folder 'ClinicalReadings'. There is one txt file for each image.

Finding labels are categorized as normal or TB. Any image which is not categorized as normal is considered as TB.
'''

import pandas as pd
import shutil
import os
import math

normal = "normal" # Finding = normal

metadata = "montgomery_metadata.csv" # Meta info
imageDir = "../../Data/FullSet/MontgomerySet/CXR_png" # Directory of images
metafileFolder = "../../Data/FullSet/MontgomerySet/ClinicalReadings" # Directory containing all meta files
normalOutputDir = '../../Data/FinalSet/normal' # Output directory to store selected images - nornal finding.
tbOutputDir = '../../Data/FinalSet/TB' # Output directory to store selected images - TB finding.

# Skip files without failing that were not found
skipNotFound = True

'''
Common method that reads all filed in the ClinicalReadings (metadata) folder 
and create a readable csv file which contains information about all images.
'''
def createMetadataFile(outputFilePath, metadataFolderPath):
    dict = {'fileName': [], 'sex': [], 'age': [], 'finding':[]}
    list = []
    for filename in os.listdir(metadataFolderPath):
        tmpList = []
        tmpList.append(filename.replace(".txt", ".png"))
        with open(os.path.join(metadataFolderPath, filename), 'r') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                if count == 0:
                    line = line.replace("Patient's Sex: ", "").replace("\n", "")
                if count == 1:
                    line = line.replace("Patient's Age: ", "").replace("Y\n", "")
                if count == 2:
                    line = 'normal' if line == 'normal\n' else 'TB'
                tmpList.append(line)
                count += 1
                if count == 3:
                    break
        list.append(tmpList)

    df = pd.DataFrame(list, columns = ['filename', 'sex', 'age', 'finding'])
    df.to_csv(outputFilePath)

# Create metadata file.
createMetadataFile(metadata, metafileFolder)

if not os.path.exists(normalOutputDir):  # check if directory already exist
    os.mkdir(normalOutputDir)  # create a directory

if not os.path.exists(tbOutputDir):  # check if directory already exist
    os.mkdir(tbOutputDir)  # create a directory

# Read the metadata file.
metadata_csv = pd.read_csv(metadata)

# loop over all the rows of metadata file.
for (i, row) in metadata_csv.iterrows():
    if not (math.isnan(row["age"]) or row["age"] < 19 or row["age"] > 99):
        filename = row["filename"].split(os.path.sep)[-1]
        filePath = os.path.sep.join([imageDir, filename])

        if not os.path.exists(filePath) and skipNotFound:
            continue

        if row["finding"] == normal:
            shutil.copy2(filePath, normalOutputDir)
        else:
            shutil.copy2(filePath, tbOutputDir)
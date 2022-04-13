'''
This script rotates all images in input folder to 140 degrees. 
Refer https://github.com/joheras/CLoDSA/blob/master/docs/augmentation_techniques.md to learn more about augmentation techniques.

Parameters: 
    inputFolderPath: Path of the folder containing source images. This can be a root folder that contains several folders.
    outputFolderPath: Path of the folder where augmented images should be stored. Hierarcy of input folder will be maintained.
'''

import logging
from contextlib import nullcontext
from clodsa.techniques.techniqueFactory import createTechnique
import cv2
import numpy as np
import os
import time

logging.basicConfig(level=logging.DEBUG, filename="augmentation_logfile.txt", filemode="w+", format="%(asctime)-15s %(levelname)-8s %(message)s")

trainFolderPath = "../../Data/train"
testFolderPath = "../../Data/test"

combinedTrainRootFolderPath = f"../../Data/Augmented/train_combined"
combinedTestRootFolderPath = f"../../Data/Augmented/test_combined"

# check if input directory already exist
# If not, then throw exception.
if not os.path.exists(trainFolderPath):
    raise Exception(f"Invalid input directory path: {trainFolderPath}")

# check if input directory already exist
# If not, then throw exception.
if not os.path.exists(testFolderPath):
    raise Exception(f"Invalid input directory path: {testFolderPath}")

# check if directory to hold combined training data already exist If not, then throw exception.
if not os.path.exists(combinedTrainRootFolderPath):
    os.mkdir(combinedTrainRootFolderPath)

# check if directory to hold combined testing data already exist If not, then throw exception.
if not os.path.exists(combinedTestRootFolderPath):
    os.mkdir(combinedTestRootFolderPath)

'''
Prints message to console and logs it into a logfile as well.
Parameters:
    message: message to be logged.
'''
def print_and_log(message):
    print(message)
    logging.info(message)

'''
Common method that applies given augmentation technique over all images in given folder.
Parameters: 
    inputFolderPath: Path of the folder containing source images. This can be a root folder that contains several folders.
    outputRootFolderPath: Path of the folder where augmented images should be stored. Hierarcy of input folder will be maintained.
    technique: Augmentation technique to be used.
    technique_name: For logging purposes.
    combine_dir_path: Path of the directory where all augmented images are kept together. If null, then data is not added to combined dir.
'''
def performAugmentation(inputFolderPath, outputRootFolderPath, technique, technique_name, combine_dir_path = ""):
    print_and_log("")
    print_and_log("*******************************************************")
    print_and_log(f"Starting technique '{technique_name}'.")
    print_and_log("*******************************************************")

    i = 0
    for root, _, filenames in os.walk(inputFolderPath):
         for filename in filenames:
            inputFilePath = os.path.sep.join([root, filename])
            
            dirname = os.path.basename(root)
            outputFolderPath = os.path.sep.join([outputRootFolderPath, dirname])
            outputFilePath = os.path.sep.join([outputFolderPath, filename])
            
            if not os.path.exists(outputFolderPath):
                os.makedirs(outputFolderPath)
                print_and_log ("")
                print_and_log (f"**** Output folder was not found. We creating one. Path: {outputFolderPath} ****")
            
            # print_and_log(f"output file path: {outputFilePath}")
            img = cv2.imread(inputFilePath)
            img1 = technique.apply(img)
            cv2.imwrite(outputFilePath,img1)

            if (combine_dir_path != ""):
                combine_label_dir_path = os.path.sep.join([combine_dir_path, dirname])
                combine_file_path = os.path.sep.join([combine_label_dir_path, f"{os.path.basename(outputRootFolderPath)}_{filename}"])
                if not os.path.exists(combine_label_dir_path):
                    os.makedirs(combine_label_dir_path)
                    print_and_log ("")
                    print_and_log (f"**** Combined folder was not found. We creating one. Path: {combine_label_dir_path} ****")

                # print_and_log(f"combine_file_path: {combine_file_path}")
                cv2.imwrite(combine_file_path, img1)
            
            i = i + 1
        
    print_and_log (f"Total '{i}' input files processed.")

start = time.time()

# Augmentation 1-6 - Rotate
angles = [45, 60, 90, 120, 140, 160]
for angle in angles:
    outputTrainRootFolderPath = f"../../Data/Augmented/train_rotated_{angle}_degree"
    outputTestRootFolderPath = f"../../Data/Augmented/test_rotated_{angle}_degree"
    technique = createTechnique("rotate", {"angle" : angle})
    performAugmentation(trainFolderPath, outputTrainRootFolderPath, technique, f"Train Rotate {angle} degrees.", combinedTrainRootFolderPath)
    performAugmentation(testFolderPath, outputTestRootFolderPath, technique, f"Test Rotate {angle} degrees.", combinedTestRootFolderPath)    

# Augmentation 7-9 - crop
percentages = [0.5, 0.7, 0.9]
for percentage in percentages:
    outputTrainRootFolderPath = f"../../Data/Augmented/train_crop_{percentage}"
    outputTestRootFolderPath = f"../../Data/Augmented/test_crop_{percentage}"
    technique = createTechnique("crop", {"percentage": percentage,"startFrom": "TOPLEFT"})
    performAugmentation(trainFolderPath, outputTrainRootFolderPath, technique, f"Train Crop {percentage}.", combinedTrainRootFolderPath)
    performAugmentation(testFolderPath, outputTestRootFolderPath, technique, f"Test Crop {percentage}.", combinedTestRootFolderPath)    

# Augmentation 10-12 - flip
directions = {0: "hor", 1: "ver", -1: "both"}
for d in directions:
    dname = directions[d]
    outputTrainRootFolderPath = f"../../Data/Augmented/train_flip_{dname}"
    outputTestRootFolderPath = f"../../Data/Augmented/test_flip_{dname}"
    technique = createTechnique("flip", {"flip": d})
    performAugmentation(trainFolderPath, outputTrainRootFolderPath, technique, f"Train flip {dname}.", combinedTrainRootFolderPath)
    performAugmentation(testFolderPath, outputTestRootFolderPath, technique, f"Test flip {dname}.", combinedTestRootFolderPath)    

# Augmentation 13-16 - raise color
colors = {"raise_blue": "blue", "raise_green": "green", "raise_hue": "hue", "raise_red": "red"}
for c in colors:
    cname = colors[c]
    outputTrainRootFolderPath = f"../../Data/Augmented/train_raise_{cname}"
    outputTestRootFolderPath = f"../../Data/Augmented/test_raise_{cname}"
    technique = createTechnique(c, {"power" : 0.9})
    performAugmentation(trainFolderPath, outputTrainRootFolderPath, technique, f"Train raise {cname}.", combinedTrainRootFolderPath)
    performAugmentation(testFolderPath, outputTestRootFolderPath, technique, f"Test raise {cname}.", combinedTestRootFolderPath)    

# Augmentation 17 - median blur
outputTrainRootFolderPath = "../../Data/Augmented/train_median_blur"
outputTestRootFolderPath = "../../Data/Augmented/test_median_blur"
technique = createTechnique("median_blur", {"kernel" : 5})
performAugmentation(trainFolderPath, outputTrainRootFolderPath, technique, "Train median_blur.", combinedTrainRootFolderPath)
performAugmentation(testFolderPath, outputTestRootFolderPath, technique, "Test median_blur.", combinedTestRootFolderPath)

# Augmentation 18-22 - other parameterless techniques
othertechniques = {
    "change_to_hsv": "change_to_hsv",
    "change_to_lab": "change_to_lab",
    "equalize_histogram": "equalize_histogram",
    "invert": "invert",
    "sharpen": "sharpen"}

for tech in othertechniques:
    outputTrainRootFolderPath = f"../../Data/Augmented/train_{tech}"
    outputTestRootFolderPath = f"../../Data/Augmented/test_{tech}"
    technique = createTechnique(tech, {})
    performAugmentation(trainFolderPath, outputTrainRootFolderPath, technique, f"Train {tech}.", combinedTrainRootFolderPath)
    performAugmentation(testFolderPath, outputTestRootFolderPath, technique, f"Test {tech}.", combinedTestRootFolderPath)

# Augmentation 23  - shearing
outputTrainRootFolderPath = f"../../Data/Augmented/train_shearing"
outputTestRootFolderPath = f"../../Data/Augmented/test_shearing"
technique = createTechnique("shearing", {"a":0.5})
performAugmentation(trainFolderPath, outputTrainRootFolderPath, technique, f"Train shearing.", combinedTrainRootFolderPath)
performAugmentation(testFolderPath, outputTestRootFolderPath, technique, f"Test shearing.", combinedTestRootFolderPath)

# Augmentation 24 - resize
outputTrainRootFolderPath = f"../../Data/Augmented/train_resize"
outputTestRootFolderPath = f"../../Data/Augmented/test_resize"
technique = createTechnique("resize", {"percentage" : 0.9, "method": "INTER_NEAREST"})
performAugmentation(trainFolderPath, outputTrainRootFolderPath, technique, f"Train resize.", combinedTrainRootFolderPath)
performAugmentation(testFolderPath, outputTestRootFolderPath, technique, f"Test resize.", combinedTestRootFolderPath)

# Augmentation 25 - gamma
outputTrainRootFolderPath = f"../../Data/Augmented/train_gamma"
outputTestRootFolderPath = f"../../Data/Augmented/test_gamma"
technique = createTechnique("gamma", {"gamma":1.5})
performAugmentation(trainFolderPath, outputTrainRootFolderPath, technique, f"Train gamma.", combinedTrainRootFolderPath)
performAugmentation(testFolderPath, outputTestRootFolderPath, technique, f"Test gamma.", combinedTestRootFolderPath)

# Calculate execution time
end = time.time()
dur = end-start
print_and_log("")
if dur < 60:
    print_and_log(f"Execution Time: {dur} seconds")
elif dur > 60 and dur < 3600:
    dur = dur/60
    print_and_log(f"Execution Time: {dur} minutes")
else:
    dur = dur/(60*60)
    print_and_log(f"Execution Time: {dur} hours")
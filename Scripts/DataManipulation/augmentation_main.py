'''
This script rotates all images in input folder to 140 degrees. 
Refer https://github.com/joheras/CLoDSA/blob/master/docs/augmentation_techniques.md to learn more about augmentation techniques.

Parameters: 
    inputFolderPath: Path of the folder containing source images. This can be a root folder that contains several folders.
    outputFolderPath: Path of the folder where augmented images should be stored. Hierarcy of input folder will be maintained.
'''

from clodsa.techniques.techniqueFactory import createTechnique
import cv2
import numpy as np
import os

trainFolderPath = "../../Data/train"
testFolderPath = "../../Data/test"

# check if input directory already exist
# If not, then throw exception.
if not os.path.exists(trainFolderPath):
    raise Exception(f"Invalid input directory path: {trainFolderPath}")

# check if input directory already exist
# If not, then throw exception.
if not os.path.exists(testFolderPath):
    raise Exception(f"Invalid input directory path: {testFolderPath}")

'''
Common method that applies given augmentation technique over all images in given folder.
Parameters: 
    inputFolderPath: Path of the folder containing source images. This can be a root folder that contains several folders.
    outputRootFolderPath: Path of the folder where augmented images should be stored. Hierarcy of input folder will be maintained.
    technique: Augmentation technique to be used.
    technique_name: For logging purposes.
'''
def performAugmentation(inputFolderPath, outputRootFolderPath, technique, technique_name):
    print("")
    print("*******************************************************")
    print(f"Starting technique '{technique_name}'.")
    print("*******************************************************")
    for root, _, filenames in os.walk(inputFolderPath):
         for filename in filenames:
            inputFilePath = os.path.sep.join([root, filename])
            
            dirname = os.path.basename(root)
            outputFolderPath = os.path.sep.join([outputRootFolderPath, dirname])
            outputFilePath = os.path.sep.join([outputFolderPath, filename])
            
            if not os.path.exists(outputFolderPath):
                os.makedirs(outputFolderPath)
                print ("")
                print (f"**** Output folder was not found. We creating one. Path: {outputFolderPath} ****")
            
            print(f"output file path: {outputFilePath}")
            img = cv2.imread(inputFilePath)
            img1 = technique.apply(img)
            cv2.imwrite(outputFilePath,img1)
            
# Augmentation 1 - Rotate 140 degrees.
outputTrainRootFolderPath = "../../Data/Augmented/train_rotated_140_degree"
outputTestRootFolderPath = "../../Data/Augmented/test_rotated_140_degree"
technique = createTechnique("rotate", {"angle" : 140})
performAugmentation(trainFolderPath, outputTrainRootFolderPath, technique, "Train Rotate 140 degrees.")
performAugmentation(testFolderPath, outputTestRootFolderPath, technique, "Test Rotate 140 degrees.")

# Augmentation 2 - Rotate 120 degrees.
outputTrainRootFolderPath = "../../Data/Augmented/train_rotated_120_degree"
outputTestRootFolderPath = "../../Data/Augmented/test_rotated_120_degree"
technique = createTechnique("rotate", {"angle" : 120})
performAugmentation(trainFolderPath, outputTrainRootFolderPath, technique, "Train Rotate 120 degrees.")
performAugmentation(testFolderPath, outputTestRootFolderPath, technique, "Test Rotate 120 degrees.")

# Augmentation 3 - Raise blue.
outputTrainRootFolderPath = "../../Data/Augmented/train_raise_blue"
outputTestRootFolderPath = "../../Data/Augmented/test_raise_blue"
technique = createTechnique("raise_blue", {"power" : 0.9})
performAugmentation(trainFolderPath, outputTrainRootFolderPath, technique, "Train Raise blue.")
performAugmentation(testFolderPath, outputTestRootFolderPath, technique, "Test Raise blue.")

# Augmentation 4 - Flip both (horizontal and vertical).
outputTrainRootFolderPath = "../../Data/Augmented/train_flip_both"
outputTestRootFolderPath = "../../Data/Augmented/test_flip_both"
technique = createTechnique("flip",{"flip":-1})
performAugmentation(trainFolderPath, outputTrainRootFolderPath, technique, "Train Flip both")
performAugmentation(testFolderPath, outputTestRootFolderPath, technique, "Test Flip both")
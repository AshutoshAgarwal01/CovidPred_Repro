'''
This script is used to validate a pre-trained model with unseen data.
This script is based on original work done in Git repository https://github.com/arunsharma8osdd/covidpred

We have modified this script to run multiple scenarios in one go. Which makes investigation easier.

Pre-requisites: Following are the pre-requisites that should be fulfilled before executing this script.
    1. Training and testing data should exist in folders given by variable train_p and testfolder respectively.
    2. Pre-trained models should be present in folder given by variable model_p

Following variables should be modified in order to validate specific scenario.
    datasets: Add list of scenarios that you need to validate. Full list of scenarios is given under variable full_list_datasets.

Outputs: Following outputs will be created after executing this script.
    1. Logfile: In the same folder from where this script was executed.
    2. ValidationResult.csv: This file will contain confustion matrix for each scenario (a.k.a. dataset) in it's respective model folder (model_p)

'''

from __future__ import division
from datetime import datetime
import logging
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import os
import cv2
import time
import sys
import pandas as pd

# Ashutosh Code
tf.disable_v2_behavior() 

tstamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
logging.basicConfig(level=logging.DEBUG, filename=f"testing_logfile_{tstamp}.txt", filemode="w+", format="%(asctime)-15s %(levelname)-8s %(message)s")

'''
Prints message to console and logs it into a logfile as well.
Parameters:
    message: message to be logged.
'''
def print_and_log(message):
    print(message)
    logging.info(message)

## All classes directories names

all_varities = ['Normal','covid_19','non_covid_19','Pneumonia','TB']

# Full list of datasets/ scenarios that can be executed.
full_list_datasets = ['train_combined',
'train_change_to_hsv',
'train_change_to_lab',
'train_crop_0.5',
'train_crop_0.7',
'train_crop_0.9',
'train_equalize_histogram',
'train_flip_both',
'train_flip_hor',
'train_flip_ver',
'train_gamma',
'train_invert',
'train_median_blur',
'train_raise_blue',
'train_raise_green',
'train_raise_hue',
'train_raise_red',
'train_resize',
'train_rotated_120_degree',
'train_rotated_140_degree',
'train_rotated_160_degree',
'train_rotated_45_degree',
'train_rotated_60_degree',
'train_rotated_90_degree',
'train_sharpen',
'train_shearing',
'train_original']

datasets = ['train_rotated_120_degree', 'train_rotated_140_degree', 'train_original']

resultFileName = 'ValidationResult.csv'

for dataset_name in datasets:
    result_data = []

    # Training images path (will be used for class index reading)
    train_p = rf'../../Data/Augmented/{dataset_name}'
    testfolder = rf'../../Data/Augmented/test_{dataset_name[6:]}/'

    # Saved model directory path
    model_p = f'c:/temp/Model_{dataset_name}'
    model_p2 = model_p+'/trained_model.meta'

    # Result file path
    resultFilePath = f"{model_p}/{resultFileName}"

    # Let us restore the saved model 
    sess = tf.Session()
    
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph(model_p2)
    
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint(model_p))
    
    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    for vrty in all_varities:
        # Validation set images directory path    
        # test_p = rf'../../Data/Augmented/{testfolder}/'+vrty	
        test_p = testfolder + vrty	
    
        pred_class_arr = []
    
        start = time.time()
    
        try:
            import os
    
            train_path = train_p
    
            if not os.path.exists(train_path):
                print_and_log("No such directory")
                raise Exception
    
             # Path of testing images
    
            dir_path = test_p
    
            if not os.path.exists(dir_path):
                print_and_log("No such directory")
                raise Exception
    	    
            img_count=1;
    
            # Walk though all testing images one by one
            for root, dirs, files in os.walk(dir_path):
                for name in files:
    
                    image_path = name
                    filename = dir_path +'/' +image_path
    
    		        # Image size and number of channels (3 for RGB images)
                    image_size=256
                    num_channels=3
                    images = []
    
                    if os.path.exists(filename):
    
                        # Reading the image using OpenCV
                        image = cv2.imread(filename)
    
                        # Resizing the image to our desired size and preprocessing will be done exactly as done during training
                        image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
                        images.append(image)
                        images = np.array(images, dtype=np.uint8)
                        images = images.astype('float32')
                        images = np.multiply(images, 1.0/255.0) 
    
                        # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
                        x_batch = images.reshape(1, image_size,image_size,num_channels)
    
                        # Now, let's get hold of the op that we can be processed to get the output.
                        # In the original network y_pred is the tensor that is the prediction of the network
                        y_pred = graph.get_tensor_by_name("y_pred:0")
    
                        ## Let's feed the images to the input placeholders
                        x= graph.get_tensor_by_name("x:0") 
                        y_true = graph.get_tensor_by_name("y_true:0") 
                        y_test_images = np.zeros((1, len(os.listdir(train_path)))) 
    
    
                        # Creating the feed_dict that is required to be fed to calculate y_pred 
                        feed_dict_testing = {x: x_batch, y_true: y_test_images}
                        result=sess.run(y_pred, feed_dict=feed_dict_testing)
    
                        # Result is of this format [[probabiliy_of_classA probability_of_classB ....]]
                        #print "ID:"+image_path, "Prob:", result
    
                        # Convert np.array to list
                        a = result[0].tolist()
                        r=0
    
                        # Finding the maximum of all outputs
                        max1 = max(a)
                        index1 = a.index(max1)
                        predicted_class = None
    
                        # Walk through directory to find the label of the predicted output
                        count = 0
                        for root, dirs, files in os.walk(train_path):
                            for name in dirs:
                                if count==index1:
                                    predicted_class = name
                                count+=1
    
                        # If the maximum confidence output is largest of all by a big margin then
                        # print the class 
                        for i in a:
                            if i!=max1:
                                if max1-i<i:
                                    r=1                           
                        if r == 0:
                            pred_class_arr.append(predicted_class)
                            img_count=img_count+1 # Ashutosh - fixed
                        else:
                            pred_class_arr.append(predicted_class)
                            img_count=img_count+1 # Ashutosh - fixed
                    # If file does not exist
                    else:
                        print_and_log("File does not exist")
        except Exception as e:
            print_and_log(f"Exception: {e}")
    
        all_varities = ['Normal','covid_19','non_covid_19','Pneumonia','TB']
    
        d = {}
        for item in pred_class_arr:
            if item in d:
                d[item] = d.get(item)+1
            else:
                d[item] = 1
    
        total_test_images = len(pred_class_arr)
        
        print_and_log(f"\n\nTotal test images ({vrty}): {total_test_images}")
    
        print_and_log("Prediction\tProportion (%)\n")
    
        class_result = [f"{dataset_name[6:]}", total_test_images, vrty]

        for vrrt in all_varities:
            acc = 0
            if vrrt in d:
                vrr = d[vrrt]
                acc = round((vrr*100/total_test_images),2)
                print_and_log (vrrt+": {}%".format(acc)) # Ashutosh - fixed
            else:
                print_and_log(vrrt+": 0%")
            class_result.append(acc)
        result_data.append(class_result)
    
    df = pd.DataFrame(result_data, columns = ['model_name', 'image_count', 'test_label', 'Normal','covid_19','non_covid_19','Pneumonia','TB'])
    df.to_csv(resultFilePath)
    
    # Close the session before validating next scenario/ dataset.
    sess.close()

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

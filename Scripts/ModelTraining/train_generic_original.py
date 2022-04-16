'''
This script is based on original work done in Git repository https://github.com/arunsharma8osdd/covidpred

We have modified this script to train one CNN model for multiple scenarios in one go. Which makes investigation easier.

Pre-requisites: Following are the pre-requisites that should be fulfilled before executing this script.
    1. Training data should exist in folders given by variable train_path.

Following variables should be modified in order to validate specific scenario.
    1. datasets: Add list of scenarios for which you need to train the model. Full list of scenarios is given under variable full_list_datasets.
    2. final_iter: This variable should be calculated according to number of epochs required.
    3. batch_size: desired batch size. 

Outputs: Following outputs will be created after executing this script.
    1. Logfile: In the same folder from where this script was executed.
    2. Trained model: Trained model and other files will be stored in directory indicated in method 'train' 
    2. Plot : Plot of validation and training accuracies per epoch. This plot can be found in the directory where model will be stored.

NOTE: Due to a bug in the script, traning multiple models at once works but their validation does not work when we use validate script.
therefore please use one scenario at a time when using this script. Please close python session and start new session before training new scenario.    

If you need to train multiple scenarios in parallel then make copies of this script and change scenario name in 'datasets' variable accordingly.
'''
import logging
import dataset
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

# Ashutosh Code
tf.disable_v2_behavior() 

tstamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
logging.basicConfig(level=logging.DEBUG, filename=f"training_original_logfile_{tstamp}.txt", filemode="w+", format="%(asctime)-15s %(levelname)-8s %(message)s")

##################################################################################
# Function to create weights.
##################################################################################   
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

##################################################################################
# Function to create biases.
##################################################################################   
def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

##################################################################################
# Function to create a convolutional layer
##################################################################################   
def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):  
    
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])

    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    ## https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    ## We shall be using max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')

    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer

##################################################################################
# Function to create a Flatten Layer
##################################################################################        
def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels] 
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer

##################################################################################
# Function to create a Fully - Connected Layer
##################################################################################
def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    # Let's define trainable weights and biases.
    weights = create_weights(shape = [num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

##################################################################################
# Display all stats for every epoch
##################################################################################
# def show_progress(epoch_no, feed_dict_train, feed_dict_validate, val_loss, total_epochs, session, accuracy):
def show_progress(epoch_no, acc, val_acc, val_loss, total_epochs):
    msg = "Training Epoch {0}/{4} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print_and_log(msg.format(epoch_no, acc, val_acc, val_loss, total_epochs))

##################################################################################
# Training function
##################################################################################
def train(total_iterations, filename, session, saver, data, batch_size, optimizer, cost, accuracy):
    acc_arr = []
    val_acc_arr = []
    val_loss_arr = []
    epoch_arr = []

    epoch_size = int(data.train.num_examples/batch_size) # 576 for combined dataset, 23 for other data.
    total_epochs = int(total_iterations / epoch_size) + 1 # 24
    for i in range(0, total_iterations + 1):
        # data.train.num_examples ==> 369 --> 90% of 410
        # data.valid.num_examples ==> 41 --> 10% of 410
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)
        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        # Processing 23 iterations in one epoch.
        # if i % int(data.train.num_examples/batch_size) == 0: 
        if i % epoch_size == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            
            acc = session.run(accuracy, feed_dict=feed_dict_tr)
            val_acc = session.run(accuracy, feed_dict=feed_dict_val)

            epoch = int(i / epoch_size) + 1 # adding one because first epoch is zero.
            
            # show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss, total_epochs, session, accuracy)
            show_progress(epoch, acc, val_acc, val_loss, total_epochs)
            
            # Update array.
            acc_arr.append(acc)
            val_acc_arr.append(val_acc)
            val_loss_arr.append(val_loss)
            epoch_arr.append(epoch)

            # Save model every 3rd epoch or on last epoch.
            if (epoch == 1 or epoch % 3 == 0 or epoch == total_epochs):
                print_and_log ("Saving model...")
                saver.save(session, f'Model_{filename}/trained_model') ### To save model with the name specified in this line
    
    return acc_arr, val_acc_arr, val_loss_arr, epoch_arr

##################################################################################
# Code to graphically plot the Validation loss and Training, Validation accuracy
##################################################################################

def createPlot(datasetname, title, epoch_array, acc_array, val_acc_array, val_loss_array):
    plt.xlabel("Number of Epoch")  ## X-axis label
    plt.ylabel("Loss/Accuracy")  ## Y-axis label

    ## Graph Title 
    plt.title(f"{title} images based model [24 epochs based]")

    ## X-axis values range
    plt.xlim(0,30)
    plt.plot(epoch_array, acc_array, label='Training Accuracy', color='blue')
    plt.plot(epoch_array, val_acc_array, label='Validation Accuracy', color='green')
    plt.plot(epoch_array, val_loss_array, label='Validation Loss', color='red')

    ## Figure legend position
    plt.legend(loc = 'upper right')

    ## Figure save location and figure name
    plt.savefig(f"Model_{datasetname}/{filename}.png")

##################################################################################
# function to print duration
##################################################################################
def print_duration(duration):
    print_and_log("")
    if duration < 60:
        print_and_log(f"Execution Time: {duration} seconds")
    elif duration > 60 and duration < 3600:
        duration = duration/60
        print_and_log(f"Execution Time: {duration} minutes")
    else:
        duration = duration/(60*60)
        print_and_log(f"Execution Time: {duration} hours")

##################################################################################
# function to print and log
##################################################################################
def print_and_log(message):
    print(message)
    logging.info(message)

##################################################################################
# Initialize common parameters across all data sets
##################################################################################

# Total iterations
# final_iter = 400 # (For 24 Epochs, original dataset used by authors)
final_iter = 529 # (For 24 Epochs, 23 iterations per epoch, batch size 16, repro dataset)

# final_iter = 23 * 288 # (For 24 Epochs, 23 iterations per epoch, batch size 32, combined repro dataset)
# final_iter = 23 * 576 # (For 24 Epochs, 23 iterations per epoch, batch size 16, combined repro dataset)

# count of epochs
epoch_count = 24

# Assign the batch value
# batch_size = 32 # Use it only for combined dataset and final_iter = 23 * 288
batch_size = 16

# Uncomment following for quick testing with smaller dataset.
# final_iter = 50
# batch_size = 8

# 10% of the data will automatically be used for validation
validation_size = 0.1

img_size = 256 ## (Image Size)

num_channels = 3

# Network graph params (No of filters and filter size for 1st, 2nd and 3rd convolutional layer)
filter_size_conv1 = 3 
num_filters_conv1 = 32

filter_size_conv2 = 5
num_filters_conv2 = 64

filter_size_conv3 = 7
num_filters_conv3 = 128
    
fc_layer_size = 256

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

# datasets = ['train_rotated_120_degree', 'train_rotated_140_degree', 'train_original']
datasets = ['train_original']

##################################################################################
# Execute.
##################################################################################

for filename in datasets:
    start = time.time()
    session = tf.compat.v1.Session()
    try:
        print_and_log ("********************************************************")
        print_and_log (f"Starting dataset {filename}")
        print_and_log ("********************************************************")
        print_and_log ("")
    
        # Training images path    
        # Augmented Dataset Path to Train model on 120 degree rotated (augmented) images
        train_path = rf'../../Data/Augmented/{filename}'
    
        # Prepare input data
        if not os.path.exists(train_path):
            print_and_log(f"No such directory {train_path}")
            raise Exception
    
        classes = os.listdir(train_path)
        num_classes = len(classes)    
    
        # We shall load all the training and validation images and labels into memory using openCV and use that during training
        data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
    
        # Display the stats
        print_and_log("Complete reading input data. Will Now print a snippet of it")
        print_and_log("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
        print_and_log("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))
    
        # session = tf.compat.v1.Session()
        x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
    
        ## labels
        y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
        y_true_cls = tf.argmax(y_true, dimension=1)
    
        # Create all the layers
        layer_conv1 = create_convolutional_layer(input = x,
                       num_input_channels = num_channels,
                       conv_filter_size = filter_size_conv1,
                       num_filters = num_filters_conv1)
    
        layer_conv2 = create_convolutional_layer(input = layer_conv1,
                       num_input_channels = num_filters_conv1,
                       conv_filter_size = filter_size_conv2,
                       num_filters = num_filters_conv2)
    
        layer_conv3= create_convolutional_layer(input = layer_conv2,
                       num_input_channels = num_filters_conv2,
                       conv_filter_size = filter_size_conv3,
                       num_filters = num_filters_conv3)
              
        layer_flat = create_flatten_layer(layer_conv3)
    
        layer_fc1 = create_fc_layer(input = layer_flat,
                             num_inputs = layer_flat.get_shape()[1:4].num_elements(),
                             num_outputs = fc_layer_size,
                             use_relu = True)
    
        layer_fc2 = create_fc_layer(input = layer_fc1,
                             num_inputs = fc_layer_size,
                             num_outputs = num_classes,
                             use_relu = False) 
    
        y_pred = tf.nn.softmax(layer_fc2, name = 'y_pred')
    
        y_pred_cls = tf.argmax(y_pred, dimension = 1)
        session.run(tf.global_variables_initializer())

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = layer_fc2, labels = y_true)
        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cost)

        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
        session.run(tf.global_variables_initializer()) 
    
        saver = tf.train.Saver()
    
        print_and_log("")
        acc_arr, val_acc_arr, val_loss_arr, epoch_arr = train(final_iter, filename, session, saver, data, batch_size, optimizer, cost, accuracy)

        createPlot(f'{filename}', filename[6:], epoch_arr, acc_arr, val_acc_arr, val_loss_arr)

    except Exception as e:
        print_and_log(f"Exception: {str(e)}")
    finally:
        end = time.time()
        duration = end-start
        session.close()
        print_duration(duration)
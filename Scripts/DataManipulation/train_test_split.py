import numpy as np
import os
import shutil

'''
Splits original data set into two data sets, training set and testing set.
Parameters:
    data: numpy array containing paths of all files.
    test_fraction: fraction of total images that should be included in test set.
Output:
    training_set: numpy array containing path of all files that are part of training set.
    testing_set: numpy array containing path of all files that are part of testing set.
'''
def train_test_split(data, test_fraction):
    data_count = len(data)
    test_data_count = int(data_count * test_fraction)

    if test_data_count < 1:
        test_data_count = 1

    print(f'Total data found: {data_count}')
    print(f'Testing data count: {test_data_count}')
    print(f'Training data count: {data_count - test_data_count}')

    data_indices = np.arange(data_count)
    np.random.shuffle(data_indices)
    
    train_data_indices = data_indices[test_data_count:]
    test_data_indices = data_indices[:test_data_count]
    
    training_set = data[train_data_indices]
    testing_set = data[test_data_indices]
    return training_set, testing_set

'''
Check if directory for training data already exists. If not, then create one.
Parameter:
    dirpath: Path of directory that needs to be checked.
'''
def create_dir_if_not_present(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath) 


# 10% of the data will automatically be used for testing
test_set_size = 0.1

# Path of the folder where all images are stored.
data_path=r'../../Data/FinalSet'

# Path of the folder where training images will be stored.
train_path=r'../../Data/train'

# Path of the folder where testing images will be stored.
test_path=r'../../Data/test'

# Prepare input data
if not os.path.exists(data_path):
    print("No such directory")
    raise Exception

# check if directory for training and test data already exists.
# If not, then create.
create_dir_if_not_present(train_path)
create_dir_if_not_present(test_path)

classes = os.listdir(data_path)
num_classes = len(classes)

print('Going to scan all images...')
for fields in classes:   
    print(f'Processing images for label {fields}.')

    # Get path of class directory.
    class_dir_path = os.path.join(data_path, fields)

    # Get full paths of all files.
    class_filepaths = np.array([os.path.join(class_dir_path, f) for f in os.listdir(class_dir_path)])
    training_set, testing_set = train_test_split(class_filepaths, test_fraction = test_set_size)

    # Create directories for current class in both train and test directories.
    train_class_path = os.path.join(train_path, fields)
    test_class_path = os.path.join(test_path, fields)
    create_dir_if_not_present(train_class_path)
    create_dir_if_not_present(test_class_path)

    for f in training_set:
        shutil.copy2(f, train_class_path)
    
    for f in testing_set:
        shutil.copy2(f, test_class_path)

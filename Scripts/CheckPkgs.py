'''
Image augmentation library clodsa needs a number of packages to be pre-installed before installing clodsa.
https://github.com/joheras/CLoDSA

This script checks if all those packages are already installed or not.
Packages which are not installed will be displayed in output console.
'''

import importlib.util

# List of packages which are needed to be pre-install before installing clodsa
package_names = ['numpy', 'scipy', 'scikit-image', 'mahotas', 'imutils', 'keras', 'commentjson', 'h5py', 'scikit-learn', 'progressbar2', 'tensorflow', 'imageio', 'cv2', 'future']

for package_name in package_names:
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(package_name +" is not installed")
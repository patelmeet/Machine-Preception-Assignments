# For Opencv 3.1 python 2.7

import cv2
import os
import numpy as np

from sklearn.externals import joblib

# One folder contain classes, each class contains two folder named 'test' and 'train'.
# test folder contains testing images and train folder contains images given for training.

dataset_path = "dataset"			# path of dataset
training_names = os.listdir(dataset_path)	# Name of classes

image_paths = []	# List of all training image's path
image_classes = []	# Class of training images(i.e. 0,1)
class_id = 0		# Starting class id

# Get List of all training image's path and class id.
for training_name in training_names:
    cur_dir = os.path.join(dataset_path, training_name)		# cur_dir = dataset/Horses
    cur_dir = os.path.join(cur_dir,"train")			# cur_dir = dataset/Horses/train
    class_path = [os.path.join(cur_dir, image_name) for image_name in os.listdir(cur_dir)]	# List of all file's path inside cur_dir
    image_paths+=class_path				# add list to image_paths
    image_classes+=[class_id]*len(class_path)		# add class id to image_classes
    class_id+=1

# Create feature extraction and keypoint detector object(SIFT or SURF object)
#Feature_Extractor = cv2.xfeatures2d.SIFT_create()		# For SIFT use this
Feature_Extractor = cv2.xfeatures2d.SURF_create()		# For SURF use this

descriptor_list = []				# List of all descriptors of all images
for image_path in image_paths:
    im = cv2.imread(image_path)					# Read image
    kpts, des = Feature_Extractor.detectAndCompute(im, None)	# Get Keypoints and descriptors
    descriptor_list.append((image_path, des))			# append descriptor to descriptor_list

joblib.dump((descriptor_list,image_classes), "descriptors", compress=3)	# Store descriptors to file




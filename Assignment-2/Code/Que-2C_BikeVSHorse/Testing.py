# For Opencv 3.1 python 2.7

import cv2
import numpy as np
import os

from scipy.cluster.vq import kmeans,vq

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

# Load the Logistic Regression Model, scaler, number of clusters and codebook 
logreg, stdSlr, k, codebook = joblib.load("BagOfWordsModel")

dataset_path = "dataset"			# path of dataset
training_names = os.listdir(dataset_path)	# Name of classes

image_paths = []		# List of all testing image's path
image_classes = []		# Class of testing images(i.e. 0,1)
class_id = 0			# Starting class id

# Get List of all testing image's path and class id.
for training_name in training_names:
    cur_dir = os.path.join(dataset_path, training_name)
    cur_dir = os.path.join(cur_dir,"test")
    class_path = [os.path.join(cur_dir, image_name) for image_name in os.listdir(cur_dir)]
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1
    
# Create feature extraction and keypoint detector object(SIFT or SURF object)
#Feature_Extractor = cv2.xfeatures2d.SIFT_create()		# For SIFT use this
Feature_Extractor = cv2.xfeatures2d.SURF_create()		# For SURF use this

# List of all descriptors of all images
descriptor_list = []

for image_path in image_paths:			
    im = cv2.imread(image_path)					# Read image
    kpts, des = Feature_Extractor.detectAndCompute(im, None)	# Get Keypoints and descriptors
    descriptor_list.append((image_path, des))   		# append descriptor to descriptor_list
    
# Compute Histogram of Features(descriptors) of Testing images
# test_features is total_no_of_images by k matrix. Each row contain histogram of one image.
# To compute histogram, we used scipy.vq() function for each image descriptor.
# code contains code book index for each observation and distance is distance between observation and nearest code
test_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(descriptor_list[i][1],codebook)
    for w in words:
        test_features[i][w] += 1

# Scaling the feature histogram, it is preprocessing step for Logistic Regression
# We used sklearn.preprocessing.StandardScaler() to scale, it's fit() method will generate mean and standard deviation of feature histogram matrix and transform() method will apply transformation accordingly mean and std dev on feature histogram matrix.
test_features = stdSlr.transform(test_features)

# Apply Logistic Regression Inference algorith(predict() method) anf predict output class and store it to list named predictions
predictions =  [i for i in logreg.predict(test_features)]

# Compare Prediction with actual class label, if same then append it to 'same' list
same = [i for i, j in zip(image_classes, predictions) if i == j]

# compute accuracy
accuracy = (float(len(same))/float(len(image_classes)))*100
print accuracy

#print image_paths
#print predictions
#print image_classes

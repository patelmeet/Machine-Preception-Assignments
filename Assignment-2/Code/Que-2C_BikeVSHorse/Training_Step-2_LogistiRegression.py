# For Opencv 3.1 python 2.7

import cv2
import numpy as np

from scipy.cluster.vq import kmeans,vq

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

# Read previously computed descriptors from file
# descriptor_list is list of descriptors and image_classes contains image class ids
# total_no_of_images is total number of image in training set
descriptor_list,image_classes = joblib.load("descriptors")
total_no_of_images = len(descriptor_list)


# Combine all descriptors and make List of all descriptors
descriptors = descriptor_list[0][1]
for image_path, descriptor in descriptor_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))


# K-means clustering on descriptors with K=20 and iterations=10
# descriptors -- MxN matrix ,M is number of descriptors, N is size of a descriptor(64 or 128)
# codebook -- kxN matrix of centroids ,k is number of clusters
# distortion is distortion between observations(descriptors) and centroids
k = 100
codebook, distortion = kmeans(descriptors, k, 20) 

# Compute Histogram of Features(descriptors)
# feature_histogram is total_no_of_images by k matrix. Each row contain histogram of one image.
# To compute histogram, we used scipy.vq() function for each image descriptor.
# code contains code book index for each observation and distance is distance between observation and nearest code
feature_histogram = np.zeros((total_no_of_images, k), "float32")
for i in xrange(total_no_of_images):
    code, distance = vq(descriptor_list[i][1],codebook)
    for w in code:
        feature_histogram[i][w] += 1

# Scaling the feature histogram, it is preprocessing step for Logistic Regression
# We used sklearn.preprocessing.StandardScaler() to scale, it's fit() method will generate mean and standard deviation of feature histogram matrix and transform() method will apply transformation accordingly mean and std dev on feature histogram matrix.
stdSlr = StandardScaler().fit(feature_histogram)
feature_histogram = stdSlr.transform(feature_histogram)

# Apply Logistic Regression and get model using sklearn.linear_model.LogisticRegression()
# First define LogisticRegression and then use fit() to fit the model according to the feature histogram matrix and predefined lables(class id)
logreg = LogisticRegression(C=1e5)
logreg.fit(feature_histogram , np.array(image_classes))

# Store Logistic Regression Model Parameter, scaler, number of clusters and codebook into file
joblib.dump((logreg, stdSlr, k, codebook), "BagOfWordsModel", compress=3)

print "**Learning Algorithm Completed**"



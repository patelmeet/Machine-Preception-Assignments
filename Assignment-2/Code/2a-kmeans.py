import numpy as np
import cv2
import matplotlib.pyplot as plt

color_image = cv2.imread("BSDS300/images/train/299091.jpg",cv2.IMREAD_COLOR)	# Read Color Image

ReshapedImage = color_image.reshape((-1,3))		# Reshape image to get Vector(2D) from 3D color image
ReshapedImage = np.float32(ReshapedImage)		# Convert vector to 32 bit float

# Define Termination Criteria for Kmeans
# Here criteria is combination of maximum iteration = 10 or Epsilon(thresold) = 1.0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
	
K = 10		# value of K(No. of segments)

# Apply Kmeans to vector of 32 bit float numbers with predefined criteria and Initial random centers.
ret,label,center=cv2.kmeans(ReshapedImage,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# convert center to 8 bit integer and then get segmented image as vector using lable.flatten() and then reshape it to get final segmented image.
center = np.uint8(center)
intermediate = center[label.flatten()]
segmented_image = intermediate.reshape((color_image.shape))

edges = cv2.Canny(segmented_image,100,250)	# Apply canny edge detection to detect edges

cv2.imshow("Original Image",color_image)
cv2.imshow("Segmented Image K=10",segmented_image)
cv2.imshow("Edges",edges)

cv2.waitKey(0)
cv2.destroyAllWindows()

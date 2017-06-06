import numpy as np
import cv2
import matplotlib.pyplot as plt

color_image = cv2.imread("BSDS300/images/train/299091.jpg",cv2.IMREAD_COLOR)	# Read Color Image

# Define Termination Criteria for Kmeans
# Here criteria is combination of maximum iteration = 10 or Epsilon(thresold) = 1.0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

# Apply MeanShift to color_image with sp=10, cp=10, macLevel=7 and predefined criteria
segmented_image = cv2.pyrMeanShiftFiltering(color_image,15,25,maxLevel=5,termcrit=criteria)

cv2.imshow("Original Image",color_image)
cv2.imshow("Segmented Image MeanShift",segmented_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


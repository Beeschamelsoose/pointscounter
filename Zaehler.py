import sys
print(sys.executable)

import cv2
import numpy as np
import matplotlib.pyplot as plt



imgpath= "./Pictures/FR0P.jpeg"

img = cv2.imread(imgpath)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

mask_not_white = cv2.inRange(hsv,(0,10,0),(180,255,255))

#Rauschen ist nicht, trotzdem mal drüber laufen lassen
kernel = np.ones((3,3),np.uint8)
mask_clean = cv2.morphologyEx(mask_not_white, cv2.MORPH_OPEN, kernel)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean)




plt.imshow(img)
plt.show()

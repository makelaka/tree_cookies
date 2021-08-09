from tifffile import imread
from skimage import measure, filters
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

file = "Row28_Set2_Y_0750.tif"
image = imread(file)
#BLUR VALUE
image = filters.gaussian(image, sigma=1)
#THRESHOLD VALUE
image = image < 0.4*image.max()
#connected components
labels = measure.label(image)
#Which connected component? -3 indicates third largest
circle = np.argsort(np.bincount(labels.flat))[-4]
largest = labels == circle
#take center of mass of the center component
center = ndimage.measurements.center_of_mass(largest)
#plotting 
plt.figure()
plt.imshow(image,cmap='gray')
plt.scatter(center[1],center[0],c='r',s=5)
plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Replace this with the path to your .npy file
file_path = '/home/qihan/NUS-Playground/tmp/depth/0.npy'

# Load the array
array = np.load(file_path)
array = cv2.convertScaleAbs(array, alpha=(255.0/65535.0))
array= cv2.applyColorMap(array, cv2.COLORMAP_JET)
plt.imshow(array)
# Write the array to a file
#np.savetxt('view', array)
plt.show()
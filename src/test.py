import numpy as np
import os

# Print the current working directory
print(os.getcwd())# Replace 'your_file.npy' with the path to your .npy file
npy_file_path = './tmp/depth/0.npy'

# Load the .npy file
data = np.load(npy_file_path)

# Print the shape of the loaded data
print(data.shape)
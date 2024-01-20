import h5py
import numpy as np
import matplotlib.pyplot as plt

def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('/')[-1] #If you use windows change / with \\
    temp = filename_without_dir.split('_')[:-1]
    print(temp)
    dataset_name = "_".join(temp)
    print(dataset_name)
    return dataset_name

filename_path = "/Users/iacopoermacora/Final Project data min_max_scaling segmented/Intra/train/task_motor_105923_1_segment_43.h5"
# Modify the file (replace this part with your own modification logic)
with h5py.File(filename_path,'r') as f:
    dataset_name = get_dataset_name(filename_path)
    print(dataset_name)
    matrix = f.get(dataset_name)[()]
    print(type(matrix))
    print(matrix.shape)

# Maximum value
max_value = np.max(matrix)

# Minimum value
min_value = np.min(matrix)

# Average value
average_value = np.mean(matrix)

# Print the results
print("matrix_norm:")

print("\nAnalysis:")
print(f"Maximum value: {max_value}")
print(f"Minimum value: {min_value}")
print(f"Average value: {average_value}")

# Value distribution (histogram)
counts, bin_edges = np.histogram(matrix.flatten(), bins=1000)  # Flatten the matrix_norm before creating the histogram

# Plot the histogram
plt.hist(matrix.flatten(), bins=1000, edgecolor='black', alpha=0.7)
plt.title('Value Distribution in matrix_norm')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
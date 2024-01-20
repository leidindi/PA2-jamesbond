import os
import shutil
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import h5py
import mne

def scale_matrix(matrix):
    min_value = np.min(matrix)
    max_value = np.max(matrix)
    scaled_matrix = (matrix - min_value) / (max_value - min_value)
    return scaled_matrix

def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('/')[-1] #If you use windows change / with \\
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

def time_wise_min_max_scaling(matrix, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_matrix = scaler.fit_transform(matrix.T).T  # Transpose for time-wise scaling
    return scaled_matrix

def exclude_artifacts(matrix, percentile_left, percentile_right, artifacts_limit):
    count = 0
    for r in matrix: #247
        if count == artifacts_limit: break
        for elem in r:
            if elem > percentile_right or elem < percentile_left:
                count += 1
                break
    if count == artifacts_limit:
        return True
    else:
        return False

# def time_wise_z_score_scaling(matrix):
#     scaler = StandardScaler()
#     scaled_matrix = scaler.fit_transform(matrix.T).T  # Transpose for time-wise scaling
#     return scaled_matrix
    
def copy_and_modify_folder(original_folder, new_folder):
    # Copy the folder structure
    shutil.copytree(original_folder, new_folder)

    artifacts_excluded = 0
    artifacts_not_excluded = 0

    # Traverse the new folder and modify each file
    for foldername, subfolders, filenames in os.walk(new_folder):

        # Exclude folders starting with a dot
        subfolders[:] = [folder for folder in subfolders if not folder.startswith('.')]
    
        for filename in filenames:
            if not filename.startswith('.'):
                file_path = os.path.join(foldername, filename)
                print(file_path)
    
                # Modify the file (replace this part with your own modification logic)
                with h5py.File(file_path,'r') as f:
                    dataset_name = get_dataset_name(file_path)
                    matrix = f.get(dataset_name)[()]
                    print(type(matrix))
                    print(matrix.shape)
                
                matrix = np.delete(matrix, 236, axis=0)
    
                # Generating a sample signal (replace this with your actual signal)
                # For example, a sine wave with a higher sampling rate
                fs_original = 2034  # Original sampling rate (in Hz)
                t = np.arange(0, 1, 1/fs_original)  # Time array
                #original_signal = np.sin(2 * np.pi * 5 * t)  # Example signal (5 Hz sine wave)
                original_signal = matrix
                
                # Downsampling
                desired_fs = 200  # Desired sampling rate (in Hz)
                factor = fs_original // desired_fs  # Downsampling factor
                fs_downsampled = fs_original / factor  # New sampling rate
                
                # Applying low-pass filter before downsampling
                nyquist = 0.5 * fs_original
                cutoff = 0.9 * desired_fs  # Adjust cutoff frequency as needed
                b, a = signal.butter(8, cutoff / nyquist, 'low')  # Creating a low-pass Butterworth filter
                filtered_signal = signal.filtfilt(b, a, original_signal)
                print(filtered_signal.shape)
                
                downsampled_signal = filtered_signal[:,::factor]  # Downsampling by selecting every 'factor' sample

                # data_to_visualize = scale_matrix(downsampled_signal)
                # #data = np.reshape(matrix, (matrix.shape[0], -1)).T
                # ch_names = ['Ch' + str(i + 1) for i in range(data_to_visualize.shape[0])]
                # ch_types = ['eeg' for i in range(data_to_visualize.shape[0])]
                # sfreq = 2034
                # info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

                # raw = mne.io.RawArray(data_to_visualize, info)

                # print(raw.info)
                # print('Duration:', raw.times[-1], 'seconds')
                # raw.plot()

                # Artifax finder
                # Identify percentiles
                percentile_left = np.percentile(downsampled_signal, 0.05)
                percentile_right = np.percentile(downsampled_signal, 99.95)

                window_size = 160  # 0.8 seconds
                overlap_percentage = 40  # 40% overlap

                # Calculate overlap in samples
                overlap_samples = int(window_size * overlap_percentage / 100)

                # Create segments
                for i in range(0, downsampled_signal.shape[1] - window_size + 1, overlap_samples):
                    segment_data = downsampled_signal[:, i:i + window_size]

                    # Save segment to new H5 file
                    output_filename_prefix = os.path.splitext(filename)[0]
                    output_filename = f"{output_filename_prefix}_segment_{i // overlap_samples}.h5"
                    output_filepath = os.path.join(foldername, output_filename)

                    if not exclude_artifacts(segment_data, percentile_left, percentile_right, 5):
                        artifacts_not_excluded += 1
                        # Apply time-wise min-max scaling
                        scaled_matrix = time_wise_min_max_scaling(segment_data)

                        with h5py.File(output_filepath, 'w') as output_h5file:
                            dataset_name = get_dataset_name(output_filepath)
                            new_dataset_name = dataset_name
                            new_dataset = output_h5file.create_dataset(new_dataset_name, data=scaled_matrix)  # Adjust dataset names if needed
                    else:
                        # data_to_visualize = scale_matrix(segment_data)
                        # #data = np.reshape(matrix, (matrix.shape[0], -1)).T
                        # ch_names = ['Ch' + str(i + 1) for i in range(data_to_visualize.shape[0])]
                        # ch_types = ['eeg' for i in range(data_to_visualize.shape[0])]
                        # sfreq = 2034
                        # info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

                        # raw = mne.io.RawArray(data_to_visualize, info)

                        # print(raw.info)
                        # print('Duration:', raw.times[-1], 'seconds')
                        # raw.plot()
                        artifacts_excluded += 1

                file_path = os.path.join(foldername, filename)

                os.remove(file_path)
    
    print("Segments excluded: ", artifacts_excluded)
    print("Segments not excluded: ", artifacts_not_excluded)
            

if __name__ == "__main__":
    original_folder = "/Users/iacopoermacora/Desktop/Final Project data"
    new_folder = "Final Project data dragon no artifacts"

    copy_and_modify_folder(original_folder, new_folder)
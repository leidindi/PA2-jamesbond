import os
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import h5py
import numpy as np
import re
from torchsummary import summary
import pickle
from tqdm import tqdm

class SelfAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(SelfAttention, self).__init__()

        self.query = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.key = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # Assuming input shape: (batch_size, in_channels, sequence_length)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention_weights = torch.nn.functional.softmax(torch.matmul(query.transpose(1, 2), key), dim=-1)
        output = torch.matmul(value, attention_weights).transpose(1, 2)

        return output
# Assuming you have defined your CNN1D class and other necessary components

# Function to load and preprocess your data (adjust as needed)
def hdf5_to_list(hdf5_files):
    data_list = []
    file_list = []
    def _extract_data(name, obj):
        nonlocal data_list
        if isinstance(obj, h5py.Dataset):
            for x in obj:
                x = list(x)
                data_list.append(x)
    for hdf5_file in hdf5_files:
        with h5py.File(hdf5_file, 'r') as file:
            file.visititems(_extract_data)
        file_list.append(data_list)
        data_list = []

    return np.array(file_list)

class CNN2D(nn.Module):
    # Pad the tensor with zeros to make it 256 in length before you run this model
    #padded_tensor = torch.nn.functional.pad(original_tensor, (0, 8), value=0)

    # Reshape the padded tensor to 16x16
    #reshaped_tensor = padded_tensor.view(16, 16)


    def __init__(self, kernel_size1, kernel_size2, dropout1 = 0.2):
        super(CNN2D, self).__init__()
        self.num_classes = 4
        self.in_channels = 248
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2

        self.cv2d_1 = nn.Conv2d(1,  16 , kernel_size=(self.kernel_size1))
        self.cv2d_2 = nn.Conv2d(16, 32 , kernel_size=(self.kernel_size2))
        self.cv2d_3 = nn.Conv2d(60, 60 , kernel_size=(self.kernel_size2))
        #self.self_attention = SelfAttention(in_channels, hidden_dim)
        self.fc_1 = nn.Linear(60, 30)
        self.dropout1 = nn.Dropout(dropout1)
        self.fc_2 = nn.Linear(30, 15)
        #self.fc_3 = nn.Linear(30, 15)
        #self.dropout2 = nn.Dropout(dropout1)
        self.fc_4 = nn.Linear(15, self.num_classes)

    def forward(self, x):
        #x = torch.permute(x,(0,2,1))
        x = self.cv2d_1(x)
        x = F.max_pool2d(x, 2)
        #x = torch.permute(x,(0,2,1))

        #x = torch.flatten(x,2)
        x = self.cv2d_2(x)
        #x = F.
        #x = F.relu(x)
        x = F.max_pool1d(x, 3)

        #x = torch.flatten(x,2)
        x = self.cv2d_3(x)
        #x = F.
        #x = F.relu(x)
        x = F.max_pool1d(x, x.shape[2])


        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        #x = self.self_attention(x)
        #x = torch.mean(x, dim=-1)  # Global average pooling
        x = F.relu(self.fc_1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc_2(x))
        #x = F.relu(self.fc_3(x))
        #x = self.dropout2(x)
        x = F.softmax(self.fc_4(x),dim=1)
        return x


class CNN1D(nn.Module):
    def __init__(self, kernel_size1, kernel_size2, dropout1 = 0.2):
        super(CNN1D, self).__init__()
        self.num_classes = 4
        self.in_channels = 248
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2

        self.cv1d_1_image = nn.Conv1d(160,   320 , kernel_size=(self.kernel_size1))#, stride = kernel_size1//3)
        self.cv1d_2_image = nn.Conv1d(320,   240 , kernel_size=(self.kernel_size1))#, stride = kernel_size1//3)
        self.cv1d_3_image = nn.Conv1d(240,   160 , kernel_size=(self.kernel_size1))#, stride = kernel_size1//3)


        self.cv1d_4_time = nn.Conv1d(25,                100, kernel_size=(self.kernel_size2))#, stride = kernel_size2//3)
        self.cv1d_5_time = nn.Conv1d(100,                100, kernel_size=(self.kernel_size2))#, stride = kernel_size2//3)
        self.cv1d_6_time = nn.Conv1d(100,                25, kernel_size=(self.kernel_size2))#, stride = kernel_size2//3)

        #self.self_attention = SelfAttention(in_channels, hidden_dim)
        self.fc_1 = nn.Linear(25, 25)
        self.dropout1 = nn.Dropout(dropout1)
        self.fc_2 = nn.Linear(25, 15)
        #self.fc_3 = nn.Linear(30, 15)
        #self.dropout2 = nn.Dropout(dropout1)
        self.fc_4 = nn.Linear(15, self.num_classes)

        self.norm320 = nn.BatchNorm1d(320)
        self.norm240 = nn.BatchNorm1d(240)
        self.norm160 = nn.BatchNorm1d(160)
        self.norm100 = nn.BatchNorm1d(100)
        self.norm25 = nn.BatchNorm1d(25)

    def forward(self, x):

        x = torch.permute(x,(0,2,1))

        x = self.cv1d_1_image(x)
        x = self.norm320(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        x = self.cv1d_2_image(x)
        x = self.norm240(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        x = self.cv1d_3_image(x)
        x = self.norm160(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        x = torch.permute(x,(0,2,1))

        x = self.cv1d_4_time(x)
        x = self.norm100(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 4)

        x = self.cv1d_5_time(x)
        x = self.norm100(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        x = self.cv1d_6_time(x)
        x = self.norm25(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.shape[2])


        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        #x = self.self_attention(x)
        #x = torch.mean(x, dim=-1)  # Global average pooling
        x = F.relu(self.fc_1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc_2(x))
        #x = F.relu(self.fc_3(x))
        #x = self.dropout2(x)
        x = F.softmax(self.fc_4(x),dim=1)
        return x

def find_h5_files(root_folder):
    h5_file_paths = []

    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".h5"):
                if not ("__" in str(foldername).lower()):
                    h5_file_paths.append(os.path.join(foldername, filename))
    return h5_file_paths

def label_enc(labels):

    enc_list = []

    for label in labels:
        label = label.lower()
        label = label.split("\\")[-1]
        # Define the regex pattern for extracting seven consecutive digits
        pattern = r'(\d{6})'

        # Search for the pattern in the input string
        match = re.search(pattern, label)

        if match:
            # If the pattern is found, extract the task name or 'rest'
            task_name_match = re.search(r'(rest|task_.+?)_\d{6}', label)
            if task_name_match:
                label = task_name_match.group(1)
            else:
                raise ValueError
        else:
            raise ValueError

        enc = [0.0]*4
        if label == "task_motor":
            enc[0] = 1.0
        elif label == "rest":
            enc[1] = 1.0
        elif label == "task_story_math":
            enc[2] = 1.0
        elif label == "task_working_memory":
            enc[3] = 1.0

        if np.sum(enc) == 0:
            # the encoder should always return something
            raise ValueError

        enc_list.append(enc)

    return np.array(enc_list)

# Save or use the trained model as needed
if __name__ == "__main__":
    # Replace 'your_root_folder' with the path to the root folder you want to start the search from
    root_folder_path = "C:\\Users\\Ingolfur\\Documents\\GitHub\\PA2-jamesbond\\Final Project data global_min_max_scaling segmented\\"
    file_paths = find_h5_files(root_folder_path)
    random.shuffle(file_paths)

    # Initialize the lists to store paths
    intra_paths = []
    cross_paths = []
    intra_test_paths = []
    cross_test_paths = []

    # Loop through the strings in the list
    for path in file_paths:
        path = path.lower()
        if "intra" in path:
            if "test" in path:
                intra_test_paths.append(path)
            else:
                intra_paths.append(path)
        else:
            if "test" in path:
                cross_test_paths.append(path)
            else:
                cross_paths.append(path)

    segment_length = 160
    num_epochs = 1
    learning_rate = 0.001

    try:
        with open('cnn_intra.pkl', 'rb') as f:
            intra_model = pickle.load(f)
    except FileNotFoundError:
        intra_model = CNN1D( kernel_size=11)
        batch_size = 16
        # Initialize your model, loss function, and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(intra_model.parameters(), lr=learning_rate)
        # Training loop
        for epoch in range(num_epochs):
            file_paths = intra_paths
            print(f'Epoch {epoch} has started running')
            last_inputs = 0
            last_labels = 0
            for path_index in range(0,len(file_paths)//batch_size):
                # Load and preprocess data
                files = file_paths[path_index*batch_size:path_index*batch_size+batch_size]
                segments = hdf5_to_list(files)
                # Convert segments to PyTorch tensor
                segments_tensor = torch.tensor(segments, dtype=torch.float32)

                # Assuming you have corresponding labels for your data
                # Adjust label loading logic accordingly
                labels = label_enc(files)
                # Convert labels to PyTorch tensor
                labels_tensor = torch.tensor(labels, dtype=torch.long)

                # Create DataLoader for batching
                dataset = TensorDataset(segments_tensor, labels_tensor)
                dataloader = DataLoader(dataset, batch_size=batch_size)#, shuffle=True) #already shuffled

                # Training iteration
                for dl_inputs, dl_labels in dataloader:
                    optimizer.zero_grad()

                    # Forward pass
                    dl_outputs = intra_model(dl_inputs)  # Assuming input shape (batch_size, in_channels, sequence_length)

                    # Calculate loss

                    dl_labels = dl_labels.float()
                    loss = criterion(dl_outputs, dl_labels)

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()

                    last_inputs = dl_inputs
                    last_labels = dl_labels

                # Print or log training statistics if needed

                #print(f'Batch {path_index+1}')
            print(f'Epoch {epoch} done')

            # Step 1: Decode one-hot-encoded predictions and true labels

            intra_model.eval()  # Set the model to evaluation mode
            predicted_classes = torch.argmax(intra_model(last_inputs), axis=1)
            true_classes = torch.argmax(last_labels, axis=1)

            # Step 2: Compare predictions to true labels
            correct_predictions = torch.sum(predicted_classes == true_classes).item()

            # Step 3: Calculate accuracy
            total_predictions = len(predicted_classes)
            accuracy = correct_predictions / total_predictions

            intra_model.train()

            print(f'Correct-predictions: {correct_predictions}, total_predictions: {total_predictions}, accuracy: {accuracy}')
        with open('cnn_intra.pkl', 'wb') as f:
            pickle.dump(intra_model, f)

    summary(intra_model,(248,160))
    print("done training intra")
    intra_model.eval()

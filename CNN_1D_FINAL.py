import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import re
import pandas as pd

def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('/')[-1] #If you use windows change / with \\
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

class TrainDataset(Dataset):
    def __init__(self, root_dir, file_extension='.h5'):
        self.root_dir = root_dir
        self.file_extension = file_extension
        self.file_list = [file for file in os.listdir(root_dir) if file.endswith(file_extension)]
        for file in self.file_list:
            if file.startswith('rest_105923_1') or file.startswith('task_motor_105923_2') or file.startswith('task_story_math_105923_3') or file.startswith('task_working_memory_105923_4 '):
                self.file_list.remove(file)
        
        # Extract keys without the "_segment_number" part
        self.keys = list(set(filename.split('_segment')[0] for filename in self.file_list))
        
        # Create a DataFrame with file names and empty lists
        self.df = pd.DataFrame({'FileNames': self.keys, 'Values': [[] for _ in range(len(self.keys))]})
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])

        match = re.search(r'^(.*?_\d+_\d+)_.*$', get_dataset_name(file_path))
        item_key = match.group(1)
        
        with h5py.File(file_path, 'r') as file:
            
            dataset_name = get_dataset_name(file_path)
            matrix = file.get(dataset_name)[()]
            signals = torch.tensor(matrix, dtype=torch.float32)

        if dataset_name.startswith('rest'):
            target = torch.Tensor([1,0,0,0])
        elif dataset_name.startswith('task_motor'):
            target = torch.Tensor([0,1,0,0])
        elif dataset_name.startswith('task_story_math'):
            target = torch.Tensor([0,0,1,0])
        elif dataset_name.startswith('task_working_memory'):
            target = torch.Tensor([0,0,0,1])

        return signals, target, item_key
    
class ValDataset(Dataset):
    def __init__(self, root_dir, file_extension='.h5'):
        self.root_dir = root_dir
        self.file_extension = file_extension
        self.file_list = [file for file in os.listdir(root_dir) if file.endswith(file_extension)]
        for file in self.file_list:
            if not file.startswith('rest_105923_1') or not file.startswith('task_motor_105923_2') or not file.startswith('task_story_math_105923_3') or not file.startswith('task_working_memory_105923_4'):
                self.file_list.remove(file)
        
        # Extract keys without the "_segment_number" part
        self.keys = list(set(filename.split('_segment')[0] for filename in self.file_list))
        

        # Create a DataFrame with file names and empty lists
        self.df = pd.DataFrame({'FileNames': self.keys, 'Values': [[] for _ in range(len(self.keys))]})
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])

        match = re.search(r'^(.*?_\d+_\d+)_.*$', get_dataset_name(file_path))
        item_key = match.group(1)
        
        with h5py.File(file_path, 'r') as file:
            
            dataset_name = get_dataset_name(file_path)
            matrix = file.get(dataset_name)[()]
            signals = torch.tensor(matrix, dtype=torch.float32)

        if dataset_name.startswith('rest'):
            target = torch.Tensor([1,0,0,0])
        elif dataset_name.startswith('task_motor'):
            target = torch.Tensor([0,1,0,0])
        elif dataset_name.startswith('task_story_math'):
            target = torch.Tensor([0,0,1,0])
        elif dataset_name.startswith('task_working_memory'):
            target = torch.Tensor([0,0,0,1])

        return signals, target, item_key
    
class TestDataset(Dataset):
    def __init__(self, root_dir, file_extension='.h5'):
        self.root_dir = root_dir
        self.file_extension = file_extension
        self.file_list = [file for file in os.listdir(root_dir) if file.endswith(file_extension)]

        # Extract keys without the "_segment_number" part
        self.keys = list(set(filename.split('_segment')[0] for filename in self.file_list))
        

        # Create a DataFrame with file names and empty lists
        self.df = pd.DataFrame({'FileNames': self.keys, 'Values': [[] for _ in range(len(self.keys))]})
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])

        match = re.search(r'^(.*?_\d+_\d+)_.*$', get_dataset_name(file_path))
        item_key = match.group(1)
        
        with h5py.File(file_path, 'r') as file:
            
            dataset_name = get_dataset_name(file_path)
            matrix = file.get(dataset_name)[()]
            signals = torch.tensor(matrix, dtype=torch.float32)

        if dataset_name.startswith('rest'):
            target = torch.Tensor([1,0,0,0])
        elif dataset_name.startswith('task_motor'):
            target = torch.Tensor([0,1,0,0])
        elif dataset_name.startswith('task_story_math'):
            target = torch.Tensor([0,0,1,0])
        elif dataset_name.startswith('task_working_memory'):
            target = torch.Tensor([0,0,0,1])

        return signals, target, item_key

# Define your data directories
train_data_dir = '/Users/iacopoermacora/Final Project data dragon no artifacts/Intra/train'
test_data_dir = '/Users/iacopoermacora/Final Project data dragon no artifacts/Intra/test'

# Create instances of the dataset
train_dataset = TrainDataset(train_data_dir)
val_dataset = ValDataset(train_data_dir)
test_dataset = TestDataset(test_data_dir)

batch_size = 32

# Create DataLoader instances
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class MEG_CNN1D(nn.Module):
    def __init__(self, layer_one, layer_two, dropout_rate,):
        super(MEG_CNN1D, self).__init__()
        
        self.layer_one = layer_one
        self.layer_two = layer_two

        # First Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=self.layer_one.in_channels, out_channels=self.layer_one.out_channels, kernel_size=self.layer_one.kernel_size, stride=self.layer_one.stride, padding=self.layer_one.padding)
        self.batch_norm1 = nn.BatchNorm1d(self.layer_one.out_channels)
        self.threshold1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Second Convolutional Layer
        self.conv2 = nn.Conv1d(in_channels=self.layer_two.in_channels, out_channels=self.layer_two.out_channels, kernel_size=self.layer_two.kernel_size, stride=self.layer_two.stride, padding=self.layer_two.padding)
        self.batch_norm2 = nn.BatchNorm1d(self.layer_two.out_channels)
        self.threshold2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully Connected Layer
        self.fc1 = nn.Linear(576, 128)  # Adjust the input size based on your data dimensions Input_size = self.layer
        self.fc2 = nn.Linear(128, 4)  # Adjust the output size based on your task

    def forward(self, x):
        # Input: (batch_size, channels, length)
        
        # First Convolutional Layer
        # x = self.pool1(self.threshold1(self.batch_norm1(self.dropout1(F.relu(self.conv1(x))))))
        #print("Before Conv1", x.shape)
        x = self.conv1(x)
        #print("After Conv1", x.shape)
        x = self.threshold1(x)
        #print("After threshold1", x.shape)
        x = self.batch_norm1(x)
        #print("After batch_norm1", x.shape)
        x = self.dropout1(x)
        #print("After dropout1", x.shape)
        x = self.pool1(x)
        #print("After pool1", x.shape)

        # Second Convolutional Layer
        # x = self.pool2(self.threshold2(self.batch_norm2(self.dropout2(F.relu(self.conv2(x))))))
        x = self.conv2(x)
        #print("After Conv2", x.shape)
        x = self.threshold2(x)
        #print("After threshold2", x.shape)
        x = self.batch_norm2(x)
        #print("After batch_norm2", x.shape)
        x = self.dropout2(x)
        #print("After dropout2", x.shape)
        x = self.pool2(x)
        #print("After pool2", x.shape)

        # Reshape before fully connected layer
        # x = x.view(-1, 16 * 40)  # Adjust the size based on your data dimensions
        x = x.reshape((-1, 16*36))
        #print("After reshape", x.shape)

        # Fully Connected Layers
        x = self.fc1(x)
        #print("After fc1", x.shape)
        x = self.fc2(x)
        #print("After fc2", x.shape)
        x = F.softmax(x, dim=1)
        #print("After softmax", x.shape)

        return x

class Layer_param:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

layer_one = Layer_param(247, 64, 15, 1, 1) # in_channels, out_channels, kernel_size, stride, padding
layer_two = Layer_param(64, 16, 5, 1, 1) # in_channels, out_channels, kernel_size, stride, padding
dropout_rate = 0.5

model = MEG_CNN1D(layer_one, layer_two, dropout_rate)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-6)

# Training
num_epochs = 150
output_size = 4
accuracy_train_history = []
accuracy_val_history = []
accuracy_test_history = []
maj_accuracy_train_history = []
maj_accuracy_val_history = []
maj_accuracy_test_history = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    total_correct_train = 0
    total_samples_train = 0
    class_correct_train = [0] * output_size
    class_total_train = [0] * output_size

    for signals, targets, key_name in tqdm(train_dataloader):
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, torch.argmax(targets, dim=1))
        loss.backward()
        optimizer.step()

        _, predicted_train = torch.max(outputs, 1)
        total_samples_train += targets.size(0)
        total_correct_train += (predicted_train == torch.argmax(targets, dim=1)).sum().item()

        # Calculate class-wise accuracy
        for i in range(output_size):
            class_total_train[i] += (targets[:, i] == 1).sum().item()
            class_correct_train[i] += ((predicted_train == i) & (targets[:, i] == 1)).sum().item()
        
        for idx, file in enumerate(key_name):
                train_dataset.df.loc[train_dataset.df['FileNames'] == file, 'Values'].iloc[0].append(torch.argmax(outputs, dim=1)[idx].item())

    train_accuracy = total_correct_train / total_samples_train
    print(f'Training Accuracy: {train_accuracy}')

    unsegmented_average_accuracy = 0
    counter = 0
    for index, row in train_dataset.df.iterrows():
        counter+=1
        filename = row['FileNames']
        values = row['Values']
        predicted_class = max(set(values), key=values.count)
        if filename.startswith('rest'):
            if predicted_class==0:
                unsegmented_average_accuracy += 1
        elif filename.startswith('task_motor'):
            if predicted_class==1:
                unsegmented_average_accuracy += 1
        elif filename.startswith('task_story_math'):
            if predicted_class==2:
                unsegmented_average_accuracy += 1
        elif filename.startswith('task_working_memory'):
            if predicted_class==3:
                unsegmented_average_accuracy += 1
        train_dataset.df.loc[train_dataset.df['FileNames'] == filename, 'Values'].iloc[0] = []
    
    
    train_dataset.df = pd.DataFrame({'FileNames': train_dataset.keys, 'Values': [[] for _ in range(len(train_dataset.keys))]})
    average_epoch_accuracy = unsegmented_average_accuracy/counter
    print("REAL SHIT IS HERE (TRAIN): ", average_epoch_accuracy)
    #train_dataset.df.loc[train_dataset.df['FileNames'] == key_name, 'Values'].iloc[0].append(train_dataset)

    # Print class-wise accuracy
    for i in range(output_size):
        class_accuracy = class_correct_train[i] / class_total_train[i]
        print(f'Training Class {i} Accuracy: {class_accuracy}  | Total tests: {class_total_train[i]}')

    accuracy_train_history.append(train_accuracy)
    maj_accuracy_train_history.append(average_epoch_accuracy)

    # Validating
    model.eval()
    total_correct_val = 0
    total_samples_val = 0
    class_correct_val = [0] * output_size
    class_total_val = [0] * output_size

    with torch.no_grad():
        for signals, targets, key_name in val_dataloader:

            outputs = model(signals)
            _, predicted_val = torch.max(outputs, 1)
            total_samples_val += targets.size(0)
            total_correct_val += (predicted_val == torch.argmax(targets, dim=1)).sum().item()

            # Calculate class-wise accuracy
            for i in range(output_size):
                class_total_val[i] += (targets[:, i] == 1).sum().item()
                class_correct_val[i] += ((predicted_val == i) & (targets[:, i] == 1)).sum().item()

            for idx, file in enumerate(key_name):
                val_dataset.df.loc[val_dataset.df['FileNames'] == file, 'Values'].iloc[0].append(torch.argmax(outputs, dim=1)[idx].item())

    val_accuracy = total_correct_val / total_samples_val
    print(f'Val Accuracy: {val_accuracy}')

    unsegmented_average_accuracy = 0
    counter = 0
    for index, row in val_dataset.df.iterrows():
        counter+=1
        filename = row['FileNames']
        values = row['Values']
        predicted_class = max(set(values), key=values.count)
        if filename.startswith('rest'):
            if predicted_class==0:
                unsegmented_average_accuracy += 1
        elif filename.startswith('task_motor'):
            if predicted_class==1:
                unsegmented_average_accuracy += 1
        elif filename.startswith('task_story_math'):
            if predicted_class==2:
                unsegmented_average_accuracy += 1
        elif filename.startswith('task_working_memory'):
            if predicted_class==3:
                unsegmented_average_accuracy += 1
        val_dataset.df.loc[val_dataset.df['FileNames'] == filename, 'Values'].iloc[0] = []
    
    
    val_dataset.df = pd.DataFrame({'FileNames': val_dataset.keys, 'Values': [[] for _ in range(len(val_dataset.keys))]})
    average_epoch_accuracy = unsegmented_average_accuracy/counter
    print("REAL SHIT IS HERE (VAL): ", average_epoch_accuracy)
    #val_dataset.df.loc[val_dataset.df['FileNames'] == key_name, 'Values'].iloc[0].append(val_accuracy)

    # Print class-wise accuracy for validation
    for i in range(output_size):
        class_accuracy = class_correct_val[i] / class_total_val[i]
        print(f'Val Class {i} Accuracy: {class_accuracy} | Total tests: {class_total_val[i]}')

    accuracy_val_history.append(val_accuracy)
    maj_accuracy_val_history.append(average_epoch_accuracy)

    # Testing
    model.eval()
    total_correct_test = 0
    total_samples_test = 0
    class_correct_test = [0] * output_size
    class_total_test = [0] * output_size

    with torch.no_grad():
        for signals, targets, key_name in test_dataloader:

            outputs = model(signals)
            _, predicted_test = torch.max(outputs, 1)
            total_samples_test += targets.size(0)
            total_correct_test += (predicted_test == torch.argmax(targets, dim=1)).sum().item()

            # Calculate class-wise accuracy
            for i in range(output_size):
                class_total_test[i] += (targets[:, i] == 1).sum().item()
                class_correct_test[i] += ((predicted_test == i) & (targets[:, i] == 1)).sum().item()
            
            for idx, file in enumerate(key_name):
                test_dataset.df.loc[test_dataset.df['FileNames'] == file, 'Values'].iloc[0].append(torch.argmax(outputs, dim=1)[idx].item())

    test_accuracy = total_correct_test / total_samples_test
    print(f'Test Accuracy: {test_accuracy}')

    unsegmented_average_accuracy = 0
    counter = 0
    for index, row in test_dataset.df.iterrows():
        counter+=1
        filename = row['FileNames']
        values = row['Values']
        predicted_class = max(set(values), key=values.count)
        if filename.startswith('rest'):
            if predicted_class==0:
                unsegmented_average_accuracy += 1
        elif filename.startswith('task_motor'):
            if predicted_class==1:
                unsegmented_average_accuracy += 1
        elif filename.startswith('task_story_math'):
            if predicted_class==2:
                unsegmented_average_accuracy += 1
        elif filename.startswith('task_working_memory'):
            if predicted_class==3:
                unsegmented_average_accuracy += 1
        test_dataset.df.loc[test_dataset.df['FileNames'] == filename, 'Values'].iloc[0] = []


    test_dataset.df = pd.DataFrame({'FileNames': test_dataset.keys, 'Values': [[] for _ in range(len(test_dataset.keys))]})
    average_epoch_accuracy = unsegmented_average_accuracy/counter
    print("REAL SHIT IS HERE (TEST): ", average_epoch_accuracy)
    #test_dataset.df.loc[test_dataset.df['FileNames'] == key_name, 'Values'].iloc[0].append(test_dataset)

    # Print class-wise accuracy for testing
    for i in range(output_size):
        class_accuracy = class_correct_test[i] / class_total_test[i]
        print(f'Test Class {i} Accuracy: {class_accuracy} | Total tests: {class_total_test[i]}')
    
    accuracy_test_history.append(test_accuracy)
    maj_accuracy_test_history.append(average_epoch_accuracy)

# Plot the accuracy improvement
plt.plot(range(1, num_epochs + 1), accuracy_train_history, marker='o', label='Training')
plt.plot(range(1, num_epochs + 1), accuracy_val_history, marker='o', label='Testing')
plt.title('Accuracy Improvement Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
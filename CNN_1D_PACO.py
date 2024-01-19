import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import re


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# AttentionLSTM model with attention mechanism
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Calculate query vector
        query = self.query_layer(lstm_out[:, -1, :])

        # Calculate attention scores
        scores = torch.matmul(lstm_out, query.unsqueeze(2)).squeeze(2)

        # Apply softmax to get attention weights
        attention_weights = self.softmax(scores)

        # Weighted sum of LSTM output
        attended_output = torch.sum(lstm_out * attention_weights.unsqueeze(2), dim=1)

        # Dropout before the fully connected layer
        attended_output = self.dropout(attended_output)

        # Final output
        output = self.fc(attended_output)

        return output

def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('\\')[-1] #If you use windows change / with \\
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

class CustomDataset(Dataset):
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
train_data_dir = 'C:/Users/lazar/OneDrive/Υπολογιστής/test/Final Project data dragon no artifacts/Intra/train'
test_data_dir = 'C:/Users/lazar/OneDrive/Υπολογιστής/test/Final Project data dragon no artifacts/Intra/test'
 
# Create instances of the dataset
train_dataset = CustomDataset(train_data_dir)
test_dataset = CustomDataset(test_data_dir)

batch_size = 32

# Create DataLoader instances
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class MEG_CNN1D(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(MEG_CNN1D, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=247, out_channels=64, kernel_size=15, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.threshold1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Second Convolutional Layer
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=5, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(16)
        self.threshold2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully Connected Layer
        self.fc1 = nn.Linear(576, 128)  # Adjust the input size based on your data dimensions
        self.fc2 = nn.Linear(128, 4)  # Adjust the output size based on your task

    def forward(self, x):
        # Input: (batch_size, channels, length)
        
        # First Convolutional Layer
        # x = self.pool1(self.threshold1(self.batch_norm1(self.dropout1(F.relu(self.conv1(x))))))
        x = self.conv1(x)
        x = self.threshold1(x)
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.pool1(x)

        # Second Convolutional Layer
        # x = self.pool2(self.threshold2(self.batch_norm2(self.dropout2(F.relu(self.conv2(x))))))
        x = self.conv2(x)
        x = self.threshold2(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.pool2(x)

        # Reshape before fully connected layer
        # x = x.view(-1, 16 * 40)  # Adjust the size based on your data dimensions
        x = x.reshape((-1, 16*36))

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x

model = MEG_CNN1D()
'''
# Instantiate the model
input_size = 247  # Number of sensors
hidden_size = 32  # Hidden state size of LSTM
output_size = 4  # Number of classes
num_layers = 2  # Number of LSTM layers
model = AttentionLSTM(input_size, hidden_size, output_size, num_layers, dropout_rate=0.4)
'''
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-6)

# Training
num_epochs = 100    
output_size = 4
accuracy_train_history = []
accuracy_test_history = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    total_correct_train = 0
    total_samples_train = 0
    class_correct_train = [0] * output_size
    class_total_train = [0] * output_size

    for signals, targets, _ in tqdm(train_dataloader):
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

    train_accuracy = total_correct_train / total_samples_train
    print(f'Training Accuracy: {train_accuracy}')
    
    # Print class-wise accuracy
    for i in range(output_size):
        class_accuracy = class_correct_train[i] / class_total_train[i]
        print(f'Training Class {i} Accuracy: {class_accuracy}  | Total tests: {class_total_train[i]}')

    accuracy_train_history.append(train_accuracy)

    # Testing
    model.eval()
    total_correct_test = 0
    total_samples_test = 0
    class_correct_test = [0] * output_size
    class_total_test = [0] * output_size

    with torch.no_grad():
        for signals, targets, key_name in tqdm(test_dataloader):

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
    print("REAL SHIT IS HERE: ", average_epoch_accuracy)
    #test_dataset.df.loc[test_dataset.df['FileNames'] == key_name, 'Values'].iloc[0].append(test_accuracy)

    # Print class-wise accuracy for testing
    for i in range(output_size):
        class_accuracy = class_correct_test[i] / class_total_test[i]
        print(f'Test Class {i} Accuracy: {class_accuracy} | Total tests: {class_total_test[i]}')

    accuracy_test_history.append(test_accuracy)

# Plot the accuracy improvement
plt.plot(range(1, num_epochs + 1), accuracy_train_history, marker='o', label='Training')
plt.plot(range(1, num_epochs + 1), accuracy_test_history, marker='o', label='Testing')
plt.title('Accuracy Improvement Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

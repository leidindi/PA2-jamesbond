import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('/')[-1] #If you use windows change / with \\
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

class CustomDataset(Dataset):
    def __init__(self, root_dir, file_extension='.h5'):
        self.root_dir = root_dir
        self.file_extension = file_extension
        self.file_list = [file for file in os.listdir(root_dir) if file.endswith(file_extension)]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        
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

        return signals, target

# Define your data directories
train_data_dir = '/Users/iacopoermacora/Final Project data global_min_max_scaling segmented/Intra/train'
test_data_dir = '/Users/iacopoermacora/Final Project data global_min_max_scaling segmented/Intra/test'

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
        self.conv1 = nn.Conv1d(in_channels=248, out_channels=64, kernel_size=20, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.threshold1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Second Convolutional Layer
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=10, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.threshold2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully Connected Layer
        self.fc1 = nn.Linear(1024, 128)  # Adjust the input size based on your data dimensions
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
        x = x.reshape((-1, 32*32))

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x

model = MEG_CNN1D()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-6)

# Training
num_epochs = 15
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

    for signals, targets in tqdm(train_dataloader):
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
        for signals, targets in test_dataloader:

            outputs = model(signals)
            _, predicted_test = torch.max(outputs, 1)
            total_samples_test += targets.size(0)
            total_correct_test += (predicted_test == torch.argmax(targets, dim=1)).sum().item()

            # Calculate class-wise accuracy
            for i in range(output_size):
                class_total_test[i] += (targets[:, i] == 1).sum().item()
                class_correct_test[i] += ((predicted_test == i) & (targets[:, i] == 1)).sum().item()

    test_accuracy = total_correct_test / total_samples_test
    print(f'Test Accuracy: {test_accuracy}')

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

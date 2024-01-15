import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import matplotlib.pyplot as plt

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

# Assume you have already defined CustomDataset, AttentionLSTM, and get_dataset_name

# AttentionLSTM model with attention mechanism
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate=0.5):
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

# Instantiate the model
input_size = 248  # Number of sensors
hidden_size = 128  # Hidden state size of LSTM
output_size = 4  # Number of classes
num_layers = 1  # Number of LSTM layers
model = AttentionLSTM(input_size, hidden_size, output_size, num_layers)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training
num_epochs = 50
accuracy_train_history = []
accuracy_test_history = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    total_correct_train = 0
    total_samples_train = 0
    class_correct_train = [0] * output_size
    class_total_train = [0] * output_size

    for signals, targets in train_dataloader:
        signals = signals.permute(0, 2, 1)
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
            signals = signals.permute(0, 2, 1)
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

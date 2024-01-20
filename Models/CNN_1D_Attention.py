import random
import numpy as np
import torch 
from torch import einsum
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
from einops import rearrange 

val_prefixes = [
    'rest_105923_1',
    'task_motor_105923_1',
    'task_story_math_105923_1',
    'task_working_memory_105923_1',
    'rest_113922_1',
    'rest_164636_2',
    'task_motor_113922_1',
    'task_motor_164436_2',
    'task_story_math_113922_1',
    'task_story_math_164636_2',
    'task_working_memory_113922_1',
    'task_working_memory_164636_2'
]

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

class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        #self.apply(self._init_weights)
    '''  
    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.2)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1)
    '''
    def forward(
        self,
        x,
        einops_from,
        einops_to,
        mask=None,
        **einops_dims
    ):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        q = q * self.scale

        # rearrange across time or space
        q_, k_, v_ = map(
            lambda t: rearrange(t, f"{einops_from} -> {einops_to}", **einops_dims),
            (q, k, v),
        )

        # attention
        out = attn(q_, k_, v_, mask=mask)

        # merge back time or space
        out = rearrange(out, f"{einops_to} -> {einops_from}", **einops_dims)

        # merge back the heads
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        # combine heads out
        return self.to_out(out)


def attn(q, k, v, mask=None):
    sim = einsum("b i d, b j d -> b i j", q, k)

    if mask is not None:
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim=-1)
    out = einsum("b i j, b j d -> b i d", attn, v)
    return out


class TrainDataset(Dataset):
    def __init__(self, root_dir, file_extension='.h5'):
        self.root_dir = root_dir
        self.file_extension = file_extension
        self.file_list = [file for file in os.listdir(root_dir) if file.endswith(file_extension)]
        print(len(self.file_list))

        self.file_list = [file for file in self.file_list if not any(file.startswith(prefix) for prefix in val_prefixes)]
        
        print(len(self.file_list))

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
        print(len(self.file_list))

        self.file_list = [file for file in self.file_list if any(file.startswith(prefix) for prefix in val_prefixes)]
        
        print(len(self.file_list))
        
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
train_data_dir = '/Users/iacopoermacora/Final Project data dragon no artifacts/Cross/train'
test_data_dir = '/Users/iacopoermacora/Final Project data dragon no artifacts/Cross/test'

if "Cross" in test_data_dir:
    start = 0
    finish = 3
else:
    start = 0
    finish = 1
    

test_data_dirs = []

# Create instances of the dataset
train_dataset = TrainDataset(train_data_dir)
val_dataset = ValDataset(train_data_dir)
test_datasets = []

batch_size = 32

# Create DataLoader instances
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloaders = []

for n_dataset in range(start, finish): # Skip if intra
    print(n_dataset)
    if finish == 1:
        test_data_dirs.append(test_data_dir)
    else:
        test_data_dirs.append(test_data_dir + str(n_dataset+1))
    print(test_data_dirs[n_dataset])
    test_datasets.append(TestDataset(test_data_dirs[n_dataset]))
    test_dataloaders.append(DataLoader(test_datasets[n_dataset], batch_size=batch_size, shuffle=False))
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

        # Attention Layer
        self.attention_layer = Attention(74, dim_head = 64, heads = 8, dropout = 0.3)

        # Fully Connected Layer
        self.fc1 = nn.Linear(int(((160-((self.layer_one.kernel_size-1)-2*self.layer_one.padding))/2-((self.layer_two.kernel_size-1)-2*self.layer_two.padding))/2*self.layer_two.out_channels), 128) # ((160-((kernel_size_1-1)-2*padding_1))/2-((kernel_size_2-1)-2*padding_2))/2*out_channels_2
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

        x = x + self.attention_layer(x, 'b f d', 'b f d')

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
        x = x.reshape((-1, x.size(1)*x.size(2)))
        #print("After reshape", x.shape)

        # Fully Connected Layers
        x = self.fc1(x)
        #print("After fc1", x.shape)
        x = self.threshold1(x)

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

layer_one = Layer_param(247, 32, 15, 1, 1) # in_channels, out_channels, kernel_size, stride, padding
layer_two = Layer_param(32, 8, 5, 1, 1) # in_channels, out_channels, kernel_size, stride, padding
dropout_rate = 0.5

model = MEG_CNN1D(layer_one, layer_two, dropout_rate)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)

# Training
num_epochs = 150
output_size = 4
# Number of best accuracies to keep track of
num_best_accuracies = 5
best_train_accuracies = np.zeros(num_best_accuracies)
best_val_accuracies = np.zeros(num_best_accuracies)
best_test_accuracies = np.zeros((3, num_best_accuracies))
best_maj_train_accuracies = np.zeros(num_best_accuracies)
best_maj_val_accuracies = np.zeros(num_best_accuracies)
best_maj_test_accuracies = np.zeros((3, num_best_accuracies))
best_epoch = np.zeros(num_best_accuracies)

accuracy_train_history = []
accuracy_val_history = []
accuracy_test_histories = [[] for _ in range(3)]
maj_accuracy_train_history = []
maj_accuracy_val_history = []
maj_accuracy_test_histories = [[] for _ in range(3)]
loss_train_history = []
loss_val_history = []
loss_test_histories = [[] for _ in range(3)]

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    total_correct_train = 0
    total_samples_train = 0
    total_loss_train = 0
    class_correct_train = [0] * output_size
    class_total_train = [0] * output_size
    batches_counter = 0

    for signals, targets, key_name in tqdm(train_dataloader):
        batches_counter += 1
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, predicted_train = torch.max(outputs, 1)
        total_samples_train += targets.size(0)
        total_correct_train += (predicted_train == torch.argmax(targets, dim=1)).sum().item()
        total_loss_train += loss.item()

        # Calculate class-wise accuracy
        for i in range(output_size):
            class_total_train[i] += (targets[:, i] == 1).sum().item()
            class_correct_train[i] += ((predicted_train == i) & (targets[:, i] == 1)).sum().item()
        
        for idx, file in enumerate(key_name):
                train_dataset.df.loc[train_dataset.df['FileNames'] == file, 'Values'].iloc[0].append(torch.argmax(outputs, dim=1)[idx].item())

    train_accuracy = total_correct_train / total_samples_train
    train_loss = total_loss_train / batches_counter
    print(f'Training Accuracy: {train_accuracy}')
    print(f'Training Loss: {train_loss}')

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
    average_epoch_accuracy_train = unsegmented_average_accuracy/counter
    print("MAJORITY (TRAIN): ", average_epoch_accuracy_train)
    #train_dataset.df.loc[train_dataset.df['FileNames'] == key_name, 'Values'].iloc[0].append(train_dataset)

    # Print class-wise accuracy
    for i in range(output_size):
        class_accuracy = class_correct_train[i] / class_total_train[i]
        print(f'Training Class {i} Accuracy: {class_accuracy}  | Total tests: {class_total_train[i]}')

    accuracy_train_history.append(train_accuracy)
    maj_accuracy_train_history.append(average_epoch_accuracy_train)
    loss_train_history.append(train_loss)

    # Validating
    model.eval()
    total_correct_val = 0
    total_samples_val = 0
    total_loss_val = 0
    class_correct_val = [0] * output_size
    class_total_val = [0] * output_size
    batches_counter = 0

    with torch.no_grad():
        for signals, targets, key_name in val_dataloader:
            batches_counter += 1

            outputs = model(signals)
            loss = criterion(outputs, targets)
            _, predicted_val = torch.max(outputs, 1)
            total_samples_val += targets.size(0)
            total_correct_val += (predicted_val == torch.argmax(targets, dim=1)).sum().item()
            total_loss_val += loss.item()

            # Calculate class-wise accuracy
            for i in range(output_size):
                class_total_val[i] += (targets[:, i] == 1).sum().item()
                class_correct_val[i] += ((predicted_val == i) & (targets[:, i] == 1)).sum().item()

            for idx, file in enumerate(key_name):
                val_dataset.df.loc[val_dataset.df['FileNames'] == file, 'Values'].iloc[0].append(torch.argmax(outputs, dim=1)[idx].item())

    val_accuracy = total_correct_val / total_samples_val
    val_loss = total_loss_val / batches_counter
    print(f'Val Accuracy: {val_accuracy}')
    print(f'Val Loss: {val_loss}')

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
    average_epoch_accuracy_val = unsegmented_average_accuracy/counter
    print("MAJORITY (VAL): ", average_epoch_accuracy_val)
    #val_dataset.df.loc[val_dataset.df['FileNames'] == key_name, 'Values'].iloc[0].append(val_accuracy)

    # Print class-wise accuracy for validation
    for i in range(output_size):
        class_accuracy = class_correct_val[i] / class_total_val[i]
        print(f'Val Class {i} Accuracy: {class_accuracy} | Total tests: {class_total_val[i]}')

    accuracy_val_history.append(val_accuracy)
    maj_accuracy_val_history.append(average_epoch_accuracy_val)
    loss_val_history.append(val_loss)
    
    test_accuracy = np.zeros(3)
    average_epoch_accuracy_test = np.zeros(3)

    for n_dataset in range(start, finish):
        # Testing
        model.eval()
        total_correct_test = 0
        total_samples_test = 0
        total_loss_test = 0
        class_correct_test = [0] * output_size
        class_total_test = [0] * output_size
        batches_counter = 0
        with torch.no_grad():
            for signals, targets, key_name in test_dataloaders[n_dataset]:
                batches_counter += 1

                outputs = model(signals)
                loss = criterion(outputs, targets)
                _, predicted_test = torch.max(outputs, 1)
                total_samples_test += targets.size(0)
                total_correct_test += (predicted_test == torch.argmax(targets, dim=1)).sum().item()
                total_loss_test += loss.item()

                # Calculate class-wise accuracy
                for i in range(output_size):
                    class_total_test[i] += (targets[:, i] == 1).sum().item()
                    class_correct_test[i] += ((predicted_test == i) & (targets[:, i] == 1)).sum().item()
                
                for idx, file in enumerate(key_name):
                    test_datasets[n_dataset].df.loc[test_datasets[n_dataset].df['FileNames'] == file, 'Values'].iloc[0].append(torch.argmax(outputs, dim=1)[idx].item())

        test_accuracy[n_dataset] = total_correct_test / total_samples_test
        test_loss = total_loss_test / batches_counter
        print(f'Test {n_dataset+1} Accuracy: {test_accuracy[n_dataset]}')
        print(f'Test {n_dataset+1} Loss: {test_loss}')

        unsegmented_average_accuracy = 0
        counter = 0
        for index, row in test_datasets[n_dataset].df.iterrows():
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
            test_datasets[n_dataset].df.loc[test_datasets[n_dataset].df['FileNames'] == filename, 'Values'].iloc[0] = []


        test_datasets[n_dataset].df = pd.DataFrame({'FileNames': test_datasets[n_dataset].keys, 'Values': [[] for _ in range(len(test_datasets[n_dataset].keys))]})
        average_epoch_accuracy_test[n_dataset] = unsegmented_average_accuracy/counter
        print("MAJORITY (TEST ", n_dataset+1, ": ", average_epoch_accuracy_test[n_dataset])
        #test_dataset.df.loc[test_dataset.df['FileNames'] == key_name, 'Values'].iloc[0].append(test_dataset)

        # Print class-wise accuracy for testing
        for i in range(output_size):
            class_accuracy = class_correct_test[i] / class_total_test[i]
            print(f'Test {n_dataset+1} Class {i} Accuracy: {class_accuracy} | Total tests: {class_total_test[i]}')
        
        accuracy_test_histories[n_dataset].append(test_accuracy[n_dataset])
        maj_accuracy_test_histories[n_dataset].append(average_epoch_accuracy_test[n_dataset])
        loss_test_histories[n_dataset].append(test_loss)

    # Find the index of the lowest value in the array
    lowest_index = np.argmin(best_val_accuracies)

    # Check if the number_to_check is higher than the lowest value
    if train_accuracy > best_val_accuracies[lowest_index]:
        # Update the array with the new value
        best_train_accuracies[lowest_index] = train_accuracy
        best_maj_train_accuracies[lowest_index] = average_epoch_accuracy_train
        best_val_accuracies[lowest_index] = val_accuracy
        best_maj_val_accuracies[lowest_index] = average_epoch_accuracy_val
        for n_dataset in range(start, finish):
            best_test_accuracies[n_dataset][lowest_index] = test_accuracy[n_dataset]
            best_maj_test_accuracies[n_dataset][lowest_index] = average_epoch_accuracy_test[n_dataset]
        best_epoch[lowest_index] = epoch+1

print("The 5 best val accuracies:")
for i in range(0, 5):
    print("Epoch ", best_epoch[i], ":")
    print("Train = ", best_train_accuracies[i])
    print("Val = ", best_val_accuracies[i])
    for n_dataset in range(start, finish):
        print("Test ", n_dataset+1, " = ", best_test_accuracies[n_dataset][i])
    print("Train (maj) = ", best_maj_train_accuracies[i])
    print("Val (maj) = ", best_maj_val_accuracies[i])
    for n_dataset in range(start, finish):
        print("Test ", n_dataset+1, " = (maj) = ", best_maj_test_accuracies[n_dataset][i])
    print("---------------------------------------------------")
print("Average best train accuracy: ", sum(best_train_accuracies) / len(best_train_accuracies))
print("Average best val accuracy: ", sum(best_val_accuracies) / len(best_val_accuracies))
for n_dataset in range(start, finish):
    print("Average best test (", n_dataset+1, ") accuracy: ", sum(best_test_accuracies[n_dataset]) / len(best_test_accuracies[n_dataset]))
print("Average best train (maj) accuracy: ", sum(best_maj_train_accuracies) / len(best_maj_train_accuracies))
print("Average best val (maj) accuracy: ", sum(best_maj_val_accuracies) / len(best_maj_val_accuracies))
for n_dataset in range(start, finish):
    print("Average best test (", n_dataset+1, ") (maj) accuracy: ", sum(best_maj_test_accuracies[n_dataset]) / len(best_maj_test_accuracies[n_dataset]))


# Plot the accuracy improvement
plt.plot(range(1, num_epochs + 1), accuracy_train_history, marker='o', label='Training')
plt.plot(range(1, num_epochs + 1), accuracy_val_history, marker='o', label='Validating')
for n_dataset in range(start, finish):
    plt.plot(range(1, num_epochs + 1), accuracy_test_histories[n_dataset], marker='o', label='Testing '+str(n_dataset+1))
plt.plot(range(1, num_epochs + 1), maj_accuracy_train_history, marker='o', label='Training (Majority label)')
plt.plot(range(1, num_epochs + 1), maj_accuracy_val_history, marker='o', label='Validating (Majority label)')
for n_dataset in range(start, finish):
    plt.plot(range(1, num_epochs + 1), maj_accuracy_test_histories[n_dataset], marker='o', label='Testing '+str(n_dataset+1)+' (Majority label)')
plt.title('Accuracy Improvement Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot the accuracy improvement
plt.figure()  # Create a new figure for the second plot
plt.plot(range(1, num_epochs + 1), maj_accuracy_train_history, marker='o', label='Training (Majority label)')
plt.plot(range(1, num_epochs + 1), maj_accuracy_val_history, marker='o', label='Validating (Majority label)')
for n_dataset in range(start, finish):
    plt.plot(range(1, num_epochs + 1), maj_accuracy_test_histories[n_dataset], marker='o', label='Testing '+str(n_dataset+1)+' (Majority label)')
plt.title('Accuracy Improvement Over Epochs (ONLY MAJORITY)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot the accuracy improvement
plt.figure()  # Create a new figure for the second plot
plt.plot(range(1, num_epochs + 1), accuracy_train_history, marker='o', label='Training')
plt.plot(range(1, num_epochs + 1), accuracy_val_history, marker='o', label='Validating')
for n_dataset in range(start, finish):
    plt.plot(range(1, num_epochs + 1), accuracy_test_histories[n_dataset], marker='o', label='Testing '+str(n_dataset+1))
plt.title('Accuracy Improvement Over Epochs (ONLY SEGMENTS)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.figure()  # Create a new figure for the second plot
plt.plot(range(1, num_epochs + 1), loss_train_history, marker='o', label='Training Loss')
plt.plot(range(1, num_epochs + 1), loss_val_history, marker='o', label='Validation Loss')
for n_dataset in range(start, finish):
    plt.plot(range(1, num_epochs + 1), loss_test_histories[n_dataset], marker='o', label='Testing '+str(n_dataset+1)+' Loss')
plt.title('Loss Improvement Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

print('end')
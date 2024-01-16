from torcheeg.models import EEGNet
import torch
import torch.nn as nn
from torcheeg.models import CCNN
from torcheeg.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from torcheeg.model_selection import KFoldPerSubject
import os
import time
import logging
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
import h5py
from tqdm import tqdm
from torchsummary import summary
import CNN1D as CNN
import matplotlib.pyplot as plt
import pickle

os.makedirs('./examples_vanilla_torch/log', exist_ok=True)
logger = logging.getLogger('Training models with vanilla PyTorch')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
file_handler = logging.FileHandler(
    os.path.join('./examples_vanilla_torch/log', f'{timeticks}.log'))
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

k_fold = KFoldPerSubject(n_splits=10,
                         split_path='./examples_vanilla_torch/split',
                         shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# training process
'''
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        X = batch[0].to(device)
        y = batch[1].to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(X)
            logger.info(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss
'''
# validation process
''


def valid(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)

            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    logger.info(f"Valid Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss:>8f} \n")

    return correct, loss


def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('\\')[-1]  # If you use windows change / with \\
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
            target = torch.Tensor([1, 0, 0, 0])
        elif dataset_name.startswith('task_motor'):
            target = torch.Tensor([0, 1, 0, 0])
        elif dataset_name.startswith('task_story_math'):
            target = torch.Tensor([0, 0, 1, 0])
        elif dataset_name.startswith('task_working_memory'):
            target = torch.Tensor([0, 0, 0, 1])

        return signals, target


loss_fn = nn.CrossEntropyLoss()
batch_size = 64  # Hyperparam?, 1 for conv2d, 64 for conv1d

test_accs = []
test_losses = []

# Define your data directories
train_data_dir = '.\\Final Project data global_min_max_scaling segmented\\Intra\\train'
test_data_dir = '.\\Final Project data global_min_max_scaling segmented\\Intra\\test'

# Create instances of the dataset
train_dataset = CustomDataset(train_data_dir)
test_dataset = CustomDataset(test_data_dir)

# Create DataLoader instances
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# initialize model
dropout = 0.2
kernel1 = 7
kernel2 = 11
model = CNN.CNN1D(kernel_size1=kernel1, kernel_size2=kernel2, dropout1=dropout)
cnn2d = False

if cnn2d:
    model = CNN.CNN2D(kernel_size1=3, kernel_size2=2, dropout1=dropout)

# summary(model)
# input()

"""model = EEGNet(chunk_size=160,
            num_electrodes=248,
            dropout=0.5, # Hyperparam? -> Consider changing it based on results
            kernel_1=8, # Hyperparam 8, 16, 32
            kernel_2=2, # Hyperparam 2, 4
            F1=8, # Default 8, Hyperparam?
            F2=16, # Default 16, Hyperparam?
            D=2, # Default 2, Hyperparam?
            num_classes=4).to(device)"""

# model_summary = summary(model, (batch_size, 1, 248,160))

# initialize optimizer
learning_rate = 1e-5
weight_decay = 5e-6
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=weight_decay)  # , weight_decay=5e-5)  # official: weight_decay=5e-1
# optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)

criterion = torch.nn.CrossEntropyLoss()

averages_train = []
averages_test = []
num_epochs = 200
# Training loop
for epoch in range(num_epochs):
    avg_loss = 0
    avg_acc = 0

    i = 0
    for batch, target in tqdm(train_dataloader):
        # Move the batch to the device (e.g., GPU)

        inputs = batch.to(device)
        """
        # Define new mean and std
        new_mean = 0
        new_std = 1

        # Reshape the tensor for mean and std calculations
        reshaped_tensor = inputs.view(-1, inputs.size(-1))

        # Scale the tensor
        scaled_tensor = (reshaped_tensor - torch.mean(reshaped_tensor)) / torch.std(reshaped_tensor) * new_std + new_mean

        # Reshape the tensor back to its original shape
        inputs = scaled_tensor.view(inputs.size())

        inputs = inputs.unsqueeze(1)"""

        # Forward pass

        if cnn2d:
            # inputs = inputs.squeeze(0)
            inputs = torch.permute(inputs, (0, 2, 1))
            inputs = torch.nn.functional.pad(inputs, (0, 8), value=0)
            # Reshape the padded tensor to 16x16
            inputs = inputs.view(1, 160, 16, 16)
            inputs = torch.permute(inputs, (1, 0, 3, 2))
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, target)  # You need to define 'target' based on your task

        # Step 1: Decode one-hot-encoded predictions and true labels
        predicted_classes = torch.argmax(outputs, axis=1)
        true_classes = torch.argmax(target, axis=1)

        # Step 2: Compare predictions to true labels
        correct_predictions = torch.sum(predicted_classes == true_classes).item()

        # Step 3: Calculate accuracy
        total_predictions = len(predicted_classes)
        accuracy = correct_predictions / total_predictions

        avg_acc += accuracy
        avg_loss += loss
        i += 1
        # print(loss)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        # Print gradients
        '''
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f'Parameter: {name}, Gradient: {param.grad}')
        '''
        optimizer.step()
    print("\n" * 30)
    print("AVERAGE LOSS :", avg_loss / i)
    print("AVERAGE ACC :", avg_acc / i)

    averages_train.append(avg_acc / i)

    # Testing loop
    model.eval()  # Set the model to evaluation mode
    valid_loss = 0
    valid_acc = 0
    j = 0
    with torch.no_grad():
        for batch, target in tqdm(test_dataloader):
            # Move the batch to the device (e.g., GPU)

            inputs = batch.to(device)
            # Define new mean and std
            """
            new_mean = 0
            new_std = 1

            # Reshape the tensor for mean and std calculations
            reshaped_tensor = inputs.view(-1, inputs.size(-1))

            # Scale the tensor
            scaled_tensor = (reshaped_tensor - torch.mean(reshaped_tensor)) / torch.std(reshaped_tensor) * new_std + new_mean

            # Reshape the tensor back to its original shape
            inputs = scaled_tensor.view(inputs.size())

            inputs = inputs.unsqueeze(1)
            """
            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, target)  # You need to define 'target' based on your task

            # Step 1: Decode one-hot-encoded predictions and true labels
            predicted_classes = torch.argmax(outputs, axis=1)
            true_classes = torch.argmax(target, axis=1)

            # Step 2: Compare predictions to true labels
            correct_predictions = torch.sum(predicted_classes == true_classes).item()

            # Step 3: Calculate accuracy
            total_predictions = len(predicted_classes)
            accuracy = correct_predictions / total_predictions

            valid_acc += accuracy

            valid_loss += loss
            j += 1
        print("TEST LOSS: -----", valid_loss / j)
        print("TEST ACC: -----", valid_acc / j)
        averages_test.append(valid_acc / j)
    model.train()

plt.plot(range(0, len(averages_train)), averages_train, label="Training averages")
plt.plot(range(0, len(averages_test)), averages_test, label="Test averages")
plt.legend()
plt.title(
    f'LR:{learning_rate:.02f},WD:{weight_decay:.02f},kernel1:{kernel1},kernel2:{kernel2},Drop:{dropout:.02f},BSz{batch_size}')
plt.show()
print()
'''
for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):
    # initialize model
    model = EEGNet(chunk_size=32,
               num_electrodes=32,
               dropout=0.25, # Hyperparam? -> Consider changing it based on results
               kernel_1=8, # Hyperparam 8, 16, 32
               kernel_2=2, # Hyperparam 2, 4
               F1=8, # Default 8, Hyperparam?
               F2=16, # Default 16, Hyperparam?
               D=2, # Default 2, Hyperparam?
               num_classes=4).to(device)

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-4)  # official: weight_decay=5e-1

    # split train and val
    train_dataset, val_dataset = train_test_split(
        train_dataset,
        test_size=0.2,
        split_path=f'./examples_vanilla_torch/split{i}',
        shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    epochs = 50
    best_val_acc = 0.0
    for t in range(epochs):
        train_loss = train(train_loader, model, loss_fn, optimizer)
        val_acc, val_loss = valid(val_loader, model, loss_fn)
        # save the best model based on val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       f'./examples_vanilla_torch/model{i}.pt')

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # load the best model to test on test set
    model.load_state_dict(torch.load(f'./examples_vanilla_torch/model{i}.pt'))
    test_acc, test_loss = valid(test_loader, model, loss_fn)

    # log the test result
    logger.info(f"Test Error {i}: \n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f}")

    test_accs.append(test_acc)
    test_losses.append(test_loss)

# log the average test result on cross-validation datasets
logger.info(f"Test Error: \n Accuracy: {100*np.mean(test_accs):>0.1f}%, Avg loss: {np.mean(test_losses):>8f}")

'''

if input("save model?") == "y":
    with open('cnn_intra.pkl', 'wb') as f:
        pickle.dump(model, f)
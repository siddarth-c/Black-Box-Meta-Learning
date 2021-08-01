"""
Black-Box Meta-Learning

Aim: To implement and train memory augmented neural networks, a black-box meta-learner that uses a recurrent neural network for few shot classification
Author: C Siddarth
Date: August, 2021
"""

# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import glob
from PIL import Image
import matplotlib.pyplot as plt
import json

# Load data from the config file
with open("config.json") as json_data_file:
    data = json.load(json_data_file)

k_shot = data["k_shot"]
n_way = data["n_way"]
batch_size = data["batch_size"]
size = data["size"]
epochs = data["epochs"]
train_parent_folder = data["train_parent_folder"]
test_test_folder = data["test_parent_folder"]

print('A', k_shot, 'shot', n_way, 'Way classification')

def get_all_paths(why = 'train'):
    """
    Returns a list of all the train/test character folder paths
    Args: 
        why: 'test' or 'train' (default = 'train)
    Returns:
        A list of all characters' path 
    """
    if why == 'train':
        parent_folder = train_parent_folder
    if why == 'test':
        parent_folder = test_test_folder
    sub_folders = glob.glob(parent_folder) # Directories of all languages
    image_paths = [glob.glob(sub_folder + '\*') for sub_folder in sub_folders] # Directories of all characters
    image_paths = sum(image_paths, []) # Flatten out the 2D list to a 1D list 
    return image_paths

def get_image_path_label(all_paths):
    """
    Returns a list of tuples of image path and it's class
    Args: 
        all_paths: A list of all characters' path 
    Returns:
        A list of (k+1 * n) images' path 
    """
    n_folders_int  = random.sample(range(0, len(all_paths)), n_way)
    image_labels = [[(glob.glob(all_paths[n] + '\*')[k], n) 
                    for n in n_folders_int
                    for k in random.sample(range(0, len(glob.glob(all_paths[n] + '\*'))), k_shot+1)
                    ] for b in range(batch_size)] 
    return image_labels

def batch_data(why = 'train'):
    """
    Returns the data required to train / test the modelZ
    Args: 
        why: 'test' or 'train' (default = 'train)
    Returns:
        total_list: A list of input image batches of shape [Batch_size, (K_shot + 1) * N_way, img_shape**2 + N_way]
        true_labels: A list of target label batches of shape [Batch_size, (K_shot + 1) * N_way, 1]
    """
    if why == 'train':
        all_paths = all_train_paths
    if why == 'test':
        all_paths = all_test_paths
    paths_labels = get_image_path_label(all_paths)
    keys = set([path_label[1] for path_label in paths_labels[0]])
    values = [i for i in range(len(keys))]
    label_dict = dict(zip(keys, values))
    total_list = []
    true_labels = []
    for b in range(batch_size):
        dummy_first_set = []
        dummy_second_set = []
        dummy_true_labels = []
        for samp_no, path_label in enumerate(paths_labels[b]):
            path = path_label[0]
            label = path_label[1]
            img = Image.open(path)
            img = img.resize((size, size))
            img = np.array(img).flatten()/ 255.0
            feat_label = torch.zeros([n_way])
            feat_label[label_dict[label]] = 1
            if samp_no % (k_shot + 1) == 0:
                feature = np.concatenate((img,torch.zeros([n_way])))
                dummy_second_set.append(feature)
                dummy_true_labels.append(label_dict[label])
            else:
                feature = np.concatenate((img, feat_label))
                dummy_first_set.append(feature)
        
        dummy_total_list = np.concatenate((dummy_first_set, dummy_second_set))
        total_list.append(torch.tensor(dummy_total_list))
        true_labels.append(torch.tensor(dummy_true_labels))

    total_list = torch.stack(total_list).float()
    true_labels = torch.stack(true_labels).float()
    return total_list, true_labels

class Omniglot_MANN(nn.Module):
    def __init__(self, k_shot, n_way, batch_size, img_size):
        super(Omniglot_MANN, self).__init__()
        self.k_shot = k_shot
        self.n_way = n_way
        self.batch_size = batch_size
        self.img_size = img_size

        self.lstm1 = nn.LSTM(img_size + n_way, 128, batch_first = True)
        self.lstm2 = nn.LSTM(128, n_way, batch_first = True)

    def forward(self, x):
        x1,_ = self.lstm1(x)
        x2,_ = self.lstm2(x1)
        return x2[:,-self.n_way:]

def get_acc(pred, truth):
    """
    Returns the accuracy of the prediction
    Args: 
        pred: A 2D array of predicted values of shape [Batch_size, N_shot]
        truth: A 2D array of target values of shape [Batch_size, N_shot]
    Returns:
        A scalar value
    """
    acc_sum = 0
    for i in range(len(pred)):
        for p, t in zip(pred[i], truth[i]):
            if torch.argmax(p) == t:
                acc_sum += 1
            # print(torch.argmax(p))
    return acc_sum

model = Omniglot_MANN(k_shot = k_shot, n_way = n_way, batch_size = batch_size, img_size = size**2)
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
criterion = nn.CrossEntropyLoss()

all_train_paths = get_all_paths('train')
all_test_paths = get_all_paths('test')

all_accuracy = []

print('Starting training...')
for e in range(epochs):
    
    # Train the model
    optimizer.zero_grad()
    X, Y = batch_data('train')
    pred = model(X)
    loss = criterion(pred, Y.long())
    loss.backward()
    optimizer.step()

    if (e+1) % 100 == 0: 
        # Test the model
        X, Y = batch_data('test')
        pred = model(X)
        val_loss = criterion(pred, Y.long())
        accuracy = ((get_acc(pred, Y))/(batch_size * n_way))*100
        all_accuracy.append(accuracy)

        print('Epoch:', e + 1, '; Loss:', val_loss.item(), '; Acc:', accuracy)

print('Completed Training!')
# To save the training accuracy curve
plt.plot(all_accuracy)
plt.savefig('Acc_curve.png')
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *
from ffn import *

'''

In this file you will write end-to-end code to train two neural networks to categorize fashion-mnist data,
one with a feedforward architecture and the other with a convolutional architecture. You will also write code to
evaluate the models and generate plots.

'''

# Force CPU for training (as instructed).
device = torch.device('cpu')
torch.manual_seed(0)

'''

PART 1:
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted. Please do not change the transforms given below - the autograder assumes these.

'''

transform = transforms.Compose([                            # Use transforms to convert images to tensors and normalize them
    transforms.ToTensor(),                                  # convert images to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])             # Common method for grayscale images
])

batch_size = 128


'''

PART 2:
Load the dataset. Make sure to utilize the transform and batch_size from the last section.

'''

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


'''

PART 3:
Complete the model defintion classes in ffn.py and cnn.py. We instantiate the models below.

'''


feedforward_net = FF_Net().to(device)
conv_net = Conv_Net().to(device)



'''

PART 4:
Choose a good loss function and optimizer - you can use the same loss for both networks.

'''

criterion = nn.CrossEntropyLoss()

optimizer_ffn = optim.Adam(feedforward_net.parameters(), lr=1e-3)
optimizer_cnn = optim.Adam(conv_net.parameters(), lr=1e-3)



'''

PART 5:
Train both your models, one at a time! (You can train them simultaneously if you have a powerful enough computer,
and are using the same number of epochs, but it is not recommended for this assignment.)

'''


num_epochs_ffn = 15
ffn_losses = []

feedforward_net.train()
for epoch in range(num_epochs_ffn):  # loop over the dataset multiple times
    running_loss_ffn = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Flatten inputs for ffn
        inputs = inputs.view(inputs.size(0), -1)

        # zero the parameter gradients
        optimizer_ffn.zero_grad()

        # forward + backward + optimize
        outputs = feedforward_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_ffn.step()
        running_loss_ffn += loss.item()

    ffn_losses.append(running_loss_ffn)
    print(f"[FFN epoch {epoch+1}] Training loss: {running_loss_ffn:.4f}")

print('Finished Training')

torch.save(feedforward_net.state_dict(), 'ffn.pth')  # Saves model file (upload with submission)


num_epochs_cnn = 10
cnn_losses = []

conv_net.train()
for epoch in range(num_epochs_cnn):  # loop over the dataset multiple times
    running_loss_cnn = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer_cnn.zero_grad()

        # forward + backward + optimize
        outputs = conv_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_cnn.step()
        running_loss_cnn += loss.item()

    cnn_losses.append(running_loss_cnn)
    print(f"[CNN epoch {epoch+1}] Training loss: {running_loss_cnn:.4f}")

print('Finished Training')

torch.save(conv_net.state_dict(), 'cnn.pth')  # Saves model file (upload with submission)


'''

PART 6:
Evalute your models! Accuracy should be greater or equal to 80% for both models.

Code to load saved weights commented out below - may be useful for debugging.

'''

# feedforward_net.load_state_dict(torch.load('ffn.pth'))
# conv_net.load_state_dict(torch.load('cnn.pth'))

feedforward_net.eval()
conv_net.eval()

correct_ffn = 0
total_ffn = 0

correct_cnn = 0
total_cnn = 0

all_labels = []
all_preds_ffn = []
all_preds_cnn = []

with torch.no_grad():           # since we're not training, we don't need to calculate the gradients for our outputs
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # FFN
        flat = inputs.view(inputs.size(0), -1)
        out_ffn = feedforward_net(flat)
        _, preds_ffn = torch.max(out_ffn, 1)
        total_ffn += labels.size(0)
        correct_ffn += (preds_ffn == labels).sum().item()

        # CNN
        out_cnn = conv_net(inputs)
        _, preds_cnn = torch.max(out_cnn, 1)
        total_cnn += labels.size(0)
        correct_cnn += (preds_cnn == labels).sum().item()

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds_ffn.extend(preds_ffn.cpu().numpy().tolist())
        all_preds_cnn.extend(preds_cnn.cpu().numpy().tolist())

print('Accuracy for feedforward network: ', correct_ffn/total_ffn)
print('Accuracy for convolutional network: ', correct_cnn/total_cnn)


'''

PART 7:

Check the instructions PDF. You need to generate some plots.

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Save training losses (used by a companion reporting script if needed)
np.save('ffn_losses.npy', np.array(ffn_losses))
np.save('cnn_losses.npy', np.array(cnn_losses))

# Parameter counts
ffn_params = sum(p.numel() for p in feedforward_net.parameters())
cnn_params = sum(p.numel() for p in conv_net.parameters())
print(f'FFN total parameters: {ffn_params}')
print(f'CNN total parameters: {cnn_params}')
with open('param_counts.txt', 'w') as f:
    f.write(f'FFN total parameters: {ffn_params}\n')
    f.write(f'CNN total parameters: {cnn_params}\n')
    f.write(f'FFN test accuracy: {correct_ffn/total_ffn:.4f}\n')
    f.write(f'CNN test accuracy: {correct_cnn/total_cnn:.4f}\n')

# Loss plots
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(range(1, num_epochs_ffn + 1), ffn_losses, marker='o')
ax[0].set_title('FFN Training Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Sum of batch losses')
ax[0].grid(True)
ax[1].plot(range(1, num_epochs_cnn + 1), cnn_losses, marker='o', color='tab:orange')
ax[1].set_title('CNN Training Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Sum of batch losses')
ax[1].grid(True)
plt.tight_layout()
plt.savefig('training_losses.png', dpi=150)
plt.close()

# Correct + incorrect example per network
def find_examples(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    correct_idx = int(np.where(preds == labels)[0][0])
    incorrect_idx = int(np.where(preds != labels)[0][0])
    return correct_idx, incorrect_idx

# Gather images aligned with all_preds
all_images = []
with torch.no_grad():
    for data in testloader:
        inputs, _ = data
        all_images.append(inputs.cpu())
all_images = torch.cat(all_images, dim=0)

for net_name, preds in [('FFN', all_preds_ffn), ('CNN', all_preds_cnn)]:
    c_idx, i_idx = find_examples(preds, all_labels)
    fig, ax = plt.subplots(1, 2, figsize=(6, 3.5))
    img_c = all_images[c_idx].squeeze().numpy()
    img_i = all_images[i_idx].squeeze().numpy()
    ax[0].imshow(img_c, cmap='gray')
    ax[0].set_title(f'{net_name} - Correct\nTrue: {CLASS_NAMES[all_labels[c_idx]]}\nPred: {CLASS_NAMES[preds[c_idx]]}')
    ax[0].axis('off')
    ax[1].imshow(img_i, cmap='gray')
    ax[1].set_title(f'{net_name} - Incorrect\nTrue: {CLASS_NAMES[all_labels[i_idx]]}\nPred: {CLASS_NAMES[preds[i_idx]]}')
    ax[1].axis('off')
    plt.tight_layout()
    plt.savefig(f'{net_name.lower()}_predictions.png', dpi=150)
    plt.close()

# Confusion matrices
for net_name, preds in [('FFN', all_preds_ffn), ('CNN', all_preds_cnn)]:
    cm = confusion_matrix(all_labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{net_name} Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{net_name.lower()}_confusion_matrix.png', dpi=150)
    plt.close()


'''
PART 8:
Compare the performance and characteristics of FFN and CNN models.
'''

print('\n=== Model comparison ===')
print(f'FFN params: {ffn_params} | test accuracy: {correct_ffn/total_ffn:.4f}')
print(f'CNN params: {cnn_params} | test accuracy: {correct_cnn/total_cnn:.4f}')

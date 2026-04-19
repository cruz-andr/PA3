import cv2
import numpy
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from cnn import *


conv_net = Conv_Net()
conv_net.load_state_dict(torch.load('cnn.pth', map_location='cpu'))
conv_net.eval()

# Get the weights of the first convolutional layer of the network
first_conv = conv_net.conv1
kernels = first_conv.weight.detach().cpu().clone()   # shape: (out_channels, in_channels, kH, kW)
num_kernels = kernels.shape[0]


def normalize_to_unit(t):
    t = t - t.min()
    denom = t.max()
    if denom > 0:
        t = t / denom
    return t


# Create a plot that is a grid of images, where each image is one kernel from the conv layer.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels,
# the grid might have 4 rows and 8 columns. Finally, normalize the values in the grid to be
# between 0 and 1 before plotting.

cols = 8
rows = (num_kernels + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
for idx in range(rows * cols):
    ax = axes[idx // cols, idx % cols] if rows > 1 else axes[idx % cols]
    if idx < num_kernels:
        k = kernels[idx, 0]           # grayscale: single input channel
        k = normalize_to_unit(k).numpy()
        ax.imshow(k, cmap='gray')
    ax.axis('off')
fig.suptitle('Learned first-layer kernels')
plt.tight_layout()

# Save the grid to a file named 'kernel_grid.png'. Add the saved image to the PDF report you submit.
plt.savefig('kernel_grid.png', dpi=150)
plt.close()


# Apply the kernel to the provided sample image.

img = cv2.imread('sample_image.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img / 255.0					# Normalize the image
img = torch.tensor(img).float()
img = img.unsqueeze(0).unsqueeze(0)

print(img.shape)

# Apply the kernel to the image
with torch.no_grad():
    output = F.conv2d(img, first_conv.weight, bias=first_conv.bias,
                      stride=first_conv.stride, padding=first_conv.padding)


# convert output from shape (1, num_channels, output_dim_0, output_dim_1) to (num_channels, 1, output_dim_0, output_dim_1) for plotting.
# If not needed for your implementation, you can remove these lines.

output = output.squeeze(0)
output = output.unsqueeze(1)


# Create a plot that is a grid of images, where each image is the result of applying one kernel to the sample image.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels, the grid might have 4 rows and 8 columns.
# Finally, normalize the values in the grid to be between 0 and 1 before plotting.

fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.3))
for idx in range(rows * cols):
    ax = axes[idx // cols, idx % cols] if rows > 1 else axes[idx % cols]
    if idx < num_kernels:
        feat = output[idx, 0].detach().cpu()
        feat = normalize_to_unit(feat).numpy()
        ax.imshow(feat, cmap='gray')
    ax.axis('off')
fig.suptitle('First-layer kernels applied to sample image')
plt.tight_layout()

# Save the grid to a file named 'image_transform_grid.png'. Add the saved image to the PDF report you submit.
plt.savefig('image_transform_grid.png', dpi=150)
plt.close()


# Create a feature map progression. You can manually specify the forward pass order or programatically track each activation through the forward pass of the CNN.

activations = []
labels = []
with torch.no_grad():
    x = img
    activations.append(('Input', x.clone()))

    a1 = F.relu(conv_net.conv1(x))
    activations.append(('Conv1 + ReLU', a1.clone()))
    p1 = conv_net.pool(a1)
    activations.append(('Pool1', p1.clone()))

    a2 = F.relu(conv_net.conv2(p1))
    activations.append(('Conv2 + ReLU', a2.clone()))
    p2 = conv_net.pool(a2)
    activations.append(('Pool2', p2.clone()))

n = len(activations)
fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
for i, (name, tensor) in enumerate(activations):
    t = tensor[0].detach().cpu()              # (C, H, W)
    if t.shape[0] == 1:
        ch_idx = 0
    else:
        # Pick the most-activated channel (highest mean activation) for a more
        # illustrative visualization. See disclaimer in the suptitle.
        ch_idx = int(t.mean(dim=(1, 2)).argmax().item())
    feat = normalize_to_unit(t[ch_idx]).numpy()
    axes[i].imshow(feat, cmap='gray')
    axes[i].set_title(f'{name}\nshape={tuple(tensor.shape[1:])}  ch={ch_idx}')
    axes[i].axis('off')
fig.suptitle('Feature map progression (most-activated channel per stage)',
             fontsize=12)
disclaimer = ('Disclaimer: the assignment suggests showing channel 0. Channel 0 of Conv1 '
              'is near-zero after ReLU for this sample,\nso the most-activated channel '
              '(by mean activation) is shown instead to better illustrate hierarchical '
              'feature extraction.\nChannel index is annotated on each panel.')
fig.text(0.5, 0.02, disclaimer, ha='center', va='bottom', fontsize=9, style='italic')
plt.tight_layout(rect=(0, 0.12, 1, 0.93))

# Save the image as a file named 'feature_progression.png'
plt.savefig('feature_progression.png', dpi=150)
plt.close()

print('Saved: kernel_grid.png, image_transform_grid.png, feature_progression.png')

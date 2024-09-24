
import sklearn
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
import itertools
import time
from functools import partial

import os

import numpy as np
from scipy.special import logsumexp

np.set_printoptions(precision=3)


#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


# image download
from torchvision import datasets

if torch.cuda.is_available() : # 일반 GPU 사용시
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

folder = "data"
MNIST = datasets.MNIST(folder, train=True, download=False)
MNIST_val = datasets.MNIST(folder, train=False, download=False)

class_names = [0,1,2,3,4,5,6,7,8,9]

# image to tensor
from torchvision import transforms

to_tensor = transforms.ToTensor()

# image centerring
MNIST = datasets.MNIST(folder, train=True, download=False, transform=transforms.ToTensor())
imgs = torch.stack([img for img, _ in MNIST], dim=3)
print(imgs.shape)
imgs_flat = imgs.view(1, -1)  # reshape by keeping first 3 channels, but flatten all others
print(imgs_flat.shape)
mu = imgs_flat.mean(dim=1)  # average over second dimension (H*W*N) to get one mean per channel
sigma = imgs_flat.std(dim=1)


MNIST = datasets.MNIST(
    folder,
    train=True,
    download=False,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, sigma)]),
)

MNIST_val = datasets.MNIST(
    folder,
    train=False,
    download=False,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mu, sigma),
        ]
    ),
)

# evaluation func
def compute_accuracy(model, loader,device):
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            outputs = model(imgs.view(imgs.shape[0], -1).to(device))
            _, predicted = torch.max(outputs, dim=1) # max contains the information of the information of the index
            total += labels.shape[0]
            correct += int((predicted == labels.to(device)).sum().item())
    return correct / total


train_loader = torch.utils.data.DataLoader(MNIST, batch_size=64, shuffle=False)
val_loader = torch.utils.data.DataLoader(MNIST_val, batch_size=64, shuffle=False)


# model
img, label = MNIST[0]
img = img.view(-1)#flatten
ninputs = len(img)
nhidden = 512
nclasses = 10

torch.manual_seed(0)
from collections import OrderedDict

# training loop
learning_rate = 1e-2
loss_fn = nn.CrossEntropyLoss()
n_epochs = 20
lambs=[10**(-5),10**(-4),10**(-3),10**(-2),10**(-1),1]
#lambs=[10,10**2,10**3] accuracy loss is large.
n_lambs=len(lambs)
acc_train=np.zeros((n_lambs,n_epochs))
acc_val=np.zeros((n_lambs,n_epochs))

for lamb in range(n_lambs):
    model = nn.Sequential(
        OrderedDict(
            [
                ("hidden_linear", nn.Linear(ninputs, nhidden)),
                ("activation", nn.Tanh()),
                ("output_linear", nn.Linear(nhidden, nclasses)),
                ("softmax", nn.LogSoftmax(dim=1)),
            ]
            )
        ).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    print('model initialized')
    for epoch in range(n_epochs):
        print(f'start epoch {epoch}')
        for imgs, labels in train_loader:
            outputs = model(imgs.view(imgs.shape[0], -1).to(device))
            loss = loss_fn(outputs, labels.to(device))

    #        grads = torch.autograd.grad(loss, [model.hidden_linear], create_graph=True, only_inputs=True)
            grads = torch.autograd.grad(outputs=loss,
                                              inputs=model.parameters(),
                                              create_graph=True)
            grad_norm = 0
            for grad in grads:
                grad_norm += grad.pow(2).sum()
            optimizer.zero_grad()
            loss=loss+grad_norm*lambs[lamb]
            loss.backward()
            optimizer.step()

        # At end of each epoch
        acc_train[lamb,epoch] = compute_accuracy(model, train_loader,device)
        acc_val[lamb,epoch] = compute_accuracy(model, val_loader,device)
        loss_train_batch = float(loss)
        print(f"Epoch {epoch}, Batch Loss {loss_train_batch}, Val acc {acc_val[lamb,epoch]}")

    plt.plot(1-acc_train[lamb,:], label='train_loss')
    plt.plot(1-acc_val[lamb,:],label='val_loss')
    plt.legend()
    plt.savefig(f'gradient penalty lamb {lambs[lamb]}.jpg')
    plt.show()
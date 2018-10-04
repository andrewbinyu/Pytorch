#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 22:42:47 2018

@author: binyu

"""

from torch.autograd import Variable
import torch
import numpy as np
from scipy.io import loadmat
from data import load_mnist, plot_images, save_images

# Load MNIST and Set Up Data
N_data, train_images, train_labels, test_images, test_labels = load_mnist()
train_images = np.round(train_images[0:60000])
train_labels = train_labels[0:60000]
test_images = np.round(test_images[0:10000])
test_labels = test_labels[0:10000]

dim_x = 28*28
dim_h = 200
dim_out = 10

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

x = Variable(torch.from_numpy(train_images), requires_grad=False).type(dtype_float)
y_classes = Variable(torch.from_numpy(np.argmax(train_labels, 1)), requires_grad=False).type(dtype_long)

model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
)
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 0.01

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

l = 0
while l<160:
    y_pred = model(x)
    loss = loss_fn(y_pred, y_classes)
    
    model.zero_grad()  # Zero out the previous gradient computation
    loss.backward()    # Compute the gradient
    optimizer.step()   # Use the gradient information to make a step
    l += 1
    if l%20==0:
        print(l, loss.item())
        y_train_out = model(x).data.numpy()
        eval_train_labels = np.argmax(y_train_out, 1)
        
        x_test_all_var = Variable(torch.from_numpy(test_images), requires_grad=False).type(dtype_float)
        y_test_out = model(x_test_all_var).data.numpy()
        eval_test_labels = np.argmax(y_test_out, 1)
        
        train_acc = np.mean(eval_train_labels == np.argmax(train_labels, axis=1))
        test_acc = np.mean(eval_test_labels == np.argmax(test_labels, axis=1))
        print("accuracy on the train set: {}".format(train_acc))
        print("accuracy on the test set: {}".format(test_acc))

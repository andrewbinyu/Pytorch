#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 12:26:03 2018

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

train_x, train_y = train_images, train_labels
test_x, test_y = test_images, test_labels

dim_x = 28*28
dim_h = 60
dim_out = 10

dtype_float = torch.FloatTensor

x = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
y = Variable(torch.from_numpy(train_y.astype(float)), requires_grad=False).type(dtype_float)
b0 = Variable(torch.randn((1, dim_h)), requires_grad=True)
W0 = Variable(torch.randn((dim_x, dim_h)), requires_grad=True)

b1 = Variable(torch.randn((1, dim_out)), requires_grad=True)
W1 = Variable(torch.randn((dim_h, dim_out)), requires_grad=True)

def model(x, b0, W0, b1, W1):
    h = torch.nn.ReLU()(torch.matmul(x, W0) + b0.repeat(x.data.shape[0], 1))
    out = torch.matmul(h, W1) + b1.repeat(h.data.shape[0], 1)
    return out

learning_rate = 0.1
mini_batch = 100
l = 0
while l<100:
    i = 0
    while i < 600:
        batch_x = x[i*mini_batch:(i+1)*mini_batch]
        batch_y = y[i*mini_batch:(i+1)*mini_batch]
        batch_y_out = model(batch_x, b0, W0, b1, W1)
        logSoftMax = torch.nn.LogSoftmax(dim=1)
        
        loss = -torch.mean(torch.sum(batch_y * logSoftMax(batch_y_out), 1))
        loss.backward()
        
        b0.data -= learning_rate * b0.grad.data
        W0.data -= learning_rate * W0.grad.data
        
        b1.data -= learning_rate * b1.grad.data
        W1.data -= learning_rate * W1.grad.data
        
        b0.grad.data.zero_()
        W0.grad.data.zero_()
        b1.grad.data.zero_()
        W1.grad.data.zero_()
        
        i += 1       
    l += 1
    learning_rate *= 0.9
    if l%20 ==0:
        print(l, loss.item())
        
        y_train_out = model(x, b0, W0, b1, W1).data.numpy()
        eval_train_labels = np.argmax(y_train_out, 1)
        
        x_test_all_var = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
        y_test_out = model(x_test_all_var, b0, W0, b1, W1).data.numpy()
        eval_test_labels = np.argmax(y_test_out, 1)
        
        train_acc = np.mean(eval_train_labels == np.argmax(train_labels, axis=1))
        test_acc = np.mean(eval_test_labels == np.argmax(test_labels, axis=1))
        print("accuracy on the train set: {}".format(train_acc))
        print("accuracy on the test set: {}".format(test_acc))
        

# Pytorch

Learn how to use pytorch

## MNIST data
Vanilla 2 layers(1 hidden) neural network can get much better results than traditional ML such as logistic regression.
Following is some results using 60000 training data with different hyper parameter dim_h and optimizer

60000 training samples
dim_h=400 with RL
tensor(0.9163, grad_fn=<NegBackward>)
tensor(0.7571, grad_fn=<NegBackward>)
tensor(0.7214, grad_fn=<NegBackward>)
tensor(0.7175, grad_fn=<NegBackward>)
tensor(0.7170, grad_fn=<NegBackward>)
accuracy on the train set: 0.98405
accuracy on the test set: 0.9228

dim_h=28*28 with RL
tensor(2.0494, grad_fn=<NegBackward>)
tensor(1.7907, grad_fn=<NegBackward>)
tensor(1.7575, grad_fn=<NegBackward>)
tensor(1.7534, grad_fn=<NegBackward>)
tensor(1.7528, grad_fn=<NegBackward>)
accuracy on the train set: 0.9989333333333333
accuracy on the test set: 0.935
  
dim_h=1500 with Adam
tensor(0.2682, grad_fn=<NllLossBackward>)
accuracy on the train set: 0.9243166666666667
accuracy on the test set: 0.9246
tensor(0.1039, grad_fn=<NllLossBackward>)
accuracy on the train set: 0.9703
accuracy on the test set: 0.9616
tensor(0.0445, grad_fn=<NllLossBackward>)
accuracy on the train set: 0.9889166666666667
accuracy on the test set: 0.9741
tensor(0.0181, grad_fn=<NllLossBackward>)
accuracy on the train set: 0.99705
accuracy on the test set: 0.976
tensor(0.0076, grad_fn=<NllLossBackward>)
accuracy on the train set: 0.9996
accuracy on the test set: 0.9768
tensor(0.0038, grad_fn=<NllLossBackward>)
accuracy on the train set: 0.99995
accuracy on the test set: 0.9761
tensor(0.0023, grad_fn=<NllLossBackward>)
accuracy on the train set: 1.0
accuracy on the test set: 0.9761
tensor(0.0016, grad_fn=<NllLossBackward>)
accuracy on the train set: 1.0
accuracy on the test set: 0.9765


